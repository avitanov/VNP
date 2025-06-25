import requests
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_URL = 'https://www.alphavantage.co/query'


def fetch_stock_data(
    symbol: str,
    api_key: str,
    interval: str | None = None,
    outputsize: str = 'full',
    month: str | None = None
) -> pd.DataFrame:
    """
    Fetches stock data for a given month from Alpha Vantage.

    If `interval` is provided, uses the intraday endpoint;
    otherwise uses daily. The optional `month` parameter can
    be passed to limit results to that month (YYYY-MM).
    """
    if interval:
        function = 'TIME_SERIES_INTRADAY'
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'outputsize': outputsize
        }
        if month:
            params['month'] = month
    else:
        function = 'TIME_SERIES_DAILY'
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': outputsize
        }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    ts_key = next((k for k in data if 'Time Series' in k), None)
    if ts_key is None:
        raise ValueError(f"No time series data found for {symbol} {month or ''}: {data}")

    ts = data[ts_key]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'datetime'
    df.columns = [col.split('. ')[1] for col in df.columns]
    return df.sort_index()


if __name__ == '__main__':
    # Configuration
    symbols = ['AAPL']    # tickers to fetch
    interval = '5min'             # intraday interval
    start_month = '2023-10'       # starting month YYYY-MM
    # compute end month as current month
    now = datetime.today()
    end_month = now.strftime('%Y-%m')
    out_dir = '../dataset/raw'

    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise RuntimeError(
            'Alpha Vantage API key not set. '
            'Please set ALPHA_VANTAGE_API_KEY in your environment or .env file.'
        )

    # Iterate over each month from start_month to end_month
    current = datetime.strptime(start_month, '%Y-%m')
    end_dt = datetime.strptime(end_month, '%Y-%m')
    while current <= end_dt:
        month_str = current.strftime('%Y-%m')
        for symbol in symbols:
            df = fetch_stock_data(symbol, api_key, interval=interval, month=month_str)
            # save to CSV per symbol-month
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"{symbol}_{month_str}.csv")
            df.to_csv(filename)
            print(f"Saved {symbol} data for {month_str} to {filename}")
        # move to next month
        current += relativedelta(months=1)
