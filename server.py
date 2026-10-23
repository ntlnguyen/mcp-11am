import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date as Date, datetime, timedelta
from functools import lru_cache
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from mcp.server.fastmcp import FastMCP # builds tool specs from the function name, doc string, and parameter annotations
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

PEERS = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOG', 'SPCX', 'TSM']
hosts = ['localhost', 'localhost:*', '127.0.0.1', '127.0.0.1:*']
origins = ['http://localhost:*', 'http://127.0.0.1:*']
if os.getenv('CODESPACE_NAME'):
    domain = os.getenv('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN', 'app.github.dev')
    public_host = f"{os.environ['CODESPACE_NAME']}-8000.{domain}"
    hosts += [public_host, public_host + ':*']
    origins += ['https://' + public_host]
    print(f'MCP URL: https://{public_host}/mcp', flush=True)

mcp = FastMCP(
    'Classroom Stock Tools', host='0.0.0.0', port=8000,
    stateless_http=True, json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts, allowed_origins=origins,
    ),
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
Company = Annotated[str, Field(description='Company name or Yahoo Finance ticker, preferably a US-listed equity or ADR.', min_length=1, max_length=100)]
Day = Annotated[str, Field(description='Historical date in YYYY-MM-DD format; must be before today in New York.')]


def parse_date(value: str) -> Date:
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Use a date in YYYY-MM-DD format.')
    day = Date.fromisoformat(value)
    if day >= datetime.now(ZoneInfo('America/New_York')).date():
        raise ValueError('Choose a date before today so the daily closing price is final.')
    return day


@lru_cache(maxsize=128)
def resolve_company(company: str) -> dict:
    query = company.strip()
    if not query:
        raise ValueError('Provide a company name or ticker.')
    quotes = yf.Search(query, max_results=8, news_count=0, timeout=10).quotes
    choices = [q for q in quotes if q.get('quoteType') in ('EQUITY', 'ETF') and q.get('symbol')]
    if not choices:
        raise ValueError(f'No equity or ETF found for {query!r}. Try its exact Yahoo ticker.')
    exact = next((q for q in choices if q['symbol'].upper() == query.upper()), None)
    selected = exact or choices[0]
    return {
        'symbol': selected['symbol'],
        'name': selected.get('longname') or selected.get('shortname') or selected['symbol'],
        'resolution': 'Exact ticker match' if exact else 'Top Yahoo Finance equity/ETF search match; verify the company.',
    }


@lru_cache(maxsize=256)
def prices(symbol: str, start: Date, end: Date) -> pd.Series:
    # Yahoo's end is exclusive. Buffer provides a preceding close at the boundary.
    frame = yf.Ticker(symbol).history(
        start=(start - timedelta(days=14)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval='1d', auto_adjust=True, actions=False, timeout=10,
    )
    if frame.empty or 'Close' not in frame:
        raise ValueError(f'No price history for {symbol} in the requested period.')
    series = frame['Close'].dropna().copy()
    series.index = pd.Index([stamp.date() for stamp in series.index])
    series = series[~series.index.duplicated(keep='last')].sort_index()
    series = series[(series > 0) & (series.index <= end)]
    if series.empty:
        raise ValueError(f'No valid closing prices for {symbol}.')
    return series


def context(company: str, date: str):
    requested = parse_date(date)
    stock = resolve_company(company)
    start = requested - relativedelta(months=1)
    close = prices(stock['symbol'], start, requested)
    effective = close.index[-1]
    if (requested - effective).days > 7:
        raise ValueError('Latest available close is more than seven days before the requested date.')
    return stock, requested, effective, close


def failure(exc: Exception) -> dict:
    return {'status': 'error', 'message': str(exc), 'source': 'Yahoo Finance via yfinance'}


@mcp.tool(annotations=READ_ONLY)
def good_day_or_bad_day(company: Company, date: Day) -> dict:
    """Good day or bad day: compare a stock's daily adjusted-close return with
    its arithmetic average daily return in the preceding calendar month,
    excluding the evaluated day. Strictly greater is good; equal or lower is bad.
    Weekends/holidays use the most recent available trading day (up to 7 days back).
    Returns the resolved company, dates, percentage returns, and verdict.
    """
    try:
        stock, requested, effective, close = context(company, date)
        start = effective - relativedelta(months=1)
        # Reload from the effective month boundary if a weekend shifted it.
        close = prices(stock['symbol'], start, effective)
        prior = close[close.index < start]
        if prior.empty or (start - prior.index[-1]).days > 7:
            raise ValueError('Insufficient history for the full preceding month.')
        returns = close.pct_change(fill_method=None).dropna()
        baseline = returns[(returns.index >= start) & (returns.index < effective)]
        if len(baseline) < 10 or effective not in returns.index:
            raise ValueError('Need at least 10 prior daily returns and an evaluated daily return.')
        today_return = float(returns.loc[effective])
        average = float(baseline.mean())
        return {
            'status': 'ok', **stock, 'requested_date': str(requested),
            'effective_date': str(effective), 'baseline_start': str(start),
            'baseline_end': str(baseline.index[-1]), 'baseline_trading_days': len(baseline),
            'daily_return_pct': round(today_return * 100, 6),
            'average_prior_daily_return_pct': round(average * 100, 6),
            'result': 'good day' if today_return > average else 'bad day',
            'source': 'Yahoo Finance via yfinance; dividend/split-adjusted closes',
        }
    except Exception as exc:
        return failure(exc)


@mcp.tool(annotations=READ_ONLY)
def similar_stock(company: Company, date: Day) -> dict:
    """Similar stock: find the closest cumulative one-month adjusted-close
    percentage return among NVDA, MSFT, AAPL, AMZN, GOOG, SPCX, TSM.
    The month ends on the supplied date's latest available trading day.
    Use the input stock's close on/before one calendar month earlier as start;
    peers must have closes on those exact same two dates. Exclude the input
    ticker itself. Minimize absolute return difference, not correlation.
    Report unavailable peers and all successful comparisons.
    """
    try:
        stock, requested, effective, close = context(company, date)
        boundary = effective - relativedelta(months=1)
        close = prices(stock['symbol'], boundary, effective)
        before = close[close.index <= boundary]
        if before.empty or (boundary - before.index[-1]).days > 7:
            raise ValueError('Insufficient history at the one-month starting boundary.')
        start = before.index[-1]
        target_return = float(close.loc[effective] / close.loc[start] - 1)
        candidates = [symbol for symbol in PEERS if symbol != stock['symbol']]

        def compare(symbol):
            try:
                peer = prices(symbol, start, effective)
                if start not in peer.index or effective not in peer.index:
                    raise ValueError('Missing a close on the exact comparison start/end date.')
                value = float(peer.loc[effective] / peer.loc[start] - 1)
                return {'symbol': symbol, 'return_pct': value * 100,
                        'difference_percentage_points': abs(value - target_return) * 100}
            except Exception as exc:
                return {'symbol': symbol, 'reason': str(exc)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(compare, candidates))
        ranked = sorted([r for r in outcomes if 'return_pct' in r],
                        key=lambda r: (r['difference_percentage_points'], r['symbol']))
        skipped = [r for r in outcomes if 'reason' in r]
        if not ranked:
            return {'status': 'error', 'message': 'No peers have matching-date data.', 'skipped': skipped}
        winner = ranked[0]['symbol']  # Compare unrounded numbers; break exact ties alphabetically.
        for row in ranked:
            for key in ('return_pct', 'difference_percentage_points'):
                row[key] = round(row[key], 6)
        return {
            'status': 'ok', **stock, 'requested_date': str(requested),
            'effective_date': str(effective), 'period_start': str(start),
            'period_end': str(effective), 'company_return_pct': round(target_return * 100, 6),
            'result': winner, 'comparisons': ranked, 'skipped': skipped,
            'self_excluded': stock['symbol'] in PEERS,
            'source': 'Yahoo Finance via yfinance; dividend/split-adjusted closes',
        }
    except Exception as exc:
        return failure(exc)


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
