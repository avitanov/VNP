import requests
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

API_URL = 'https://www.alphavantage.co/query'


def fetch_indicator_month(
    symbol: str,
    api_key: str,
    indicator: str,
    year: int,
    month: int,
    interval: str = '5min',
    time_period: int = 14,
    series_type: str = 'close',
    limit: int = 5000,
) -> pd.DataFrame:
    """Fetch a single month of intraday (5‑min) indicator data using the `month` parameter."""
    month_str = f"{year}-{month:02d}"
    params = {
        'function': indicator.upper(),
        'symbol': symbol,
        'apikey': api_key,
        'interval': interval,
        'time_period': time_period,
        'series_type': series_type,
        'month': month_str,
        'limit': limit,  # some indicators respect limit
        'datatype': 'json'
    }
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    key = next((k for k in data if k != 'Meta Data'), None)
    if key is None or not isinstance(data[key], dict):
        return pd.DataFrame()  # empty month

    df = pd.DataFrame.from_dict(data[key], orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'datetime'
    df.columns = [c.split(' ')[1] if ' ' in c else c for c in df.columns]
    return df.sort_index()


def save_month(
    df: pd.DataFrame,
    symbol: str,
    indicator: str,
    year: int,
    month: int,
    out_dir: str
):
    if df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{symbol}_{indicator}_{year}-{month:02d}.csv"
    df.to_csv(os.path.join(out_dir, fname))
    print(f"    → saved {len(df)} rows to {fname}")


if __name__ == '__main__':
    # Configuration
    symbols      = ['AAPL']
    #'EMA', 'RSI','SMA'
    indicators   = ['RSI']
    interval     = '5min'
    time_period  = 14
    series_type  = 'close'
    start_month  = datetime(2022, 3, 1)  # start from 2022‑03
    out_dir      = '../dataset/raw'

    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise RuntimeError('Alpha Vantage API key not set in environment')

    # Iterate month‑by‑month requesting each month separately (requires premium plan)
    current = start_month
    end_month = datetime.today().replace(day=1)

    while current <= end_month:
        year, month = current.year, current.month
        for sym in symbols:
            for ind in indicators:
                print(f"Fetching {ind} {interval} for {sym} {year}-{month:02d}…")
                df = fetch_indicator_month(
                    sym, api_key, ind,
                    year, month,
                    interval=interval,
                    time_period=time_period,
                    series_type=series_type
                )
                save_month(df, sym, ind, year, month, out_dir)
        current += relativedelta(months=1)
