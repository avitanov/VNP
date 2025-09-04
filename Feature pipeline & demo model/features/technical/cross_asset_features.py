"""
Cross-Asset Features for TFT Model
=================================

Додава cross-asset data како:
- VIX (volatility index) - индекс на страв.
- Treasury yields (2Y, 10Y) каматни стапки на 2 и 10 годишни обврзници
- Dollar Index (DXY) - јачина на доларот
- Sector ETF performance - за одреден сектор како стои цената на берзата.
- Commodity prices -  цена на суровини како злато, нафта, пченица, сребро на светскиот пазар.

Cross-asset features го даваат поширокиот пазарен контекст — дали има зголемена волатилност (VIX),
 дали обврзниците покажуваат инверзија,
 дали доларот е силен, дали одредени сектори се подобри и слично.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class CrossAssetFeatures:
    """
    Cross-asset feature calculator for market context.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://www.alphavantage.co/query"
        
    def fetch_market_data(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Fetch market data for cross-asset symbols."""
        params = {
            'function': 'TIME_SERIES_DAILY' if interval == 'daily' else 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        if interval != 'daily':
            params['interval'] = interval
            
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Find the time series key
            ts_key = next((k for k in data if 'Time Series' in k), None)
            if ts_key is None:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
                
            ts_data = data[ts_key]
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = [col.split('. ')[1] for col in df.columns]
            return df.sort_index()
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_vix_features(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VIX-based features."""
        if vix_data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=vix_data.index)
        close = vix_data['close']
        
        # VIX levels and changes
        features['vix'] = close
        features['vix_change'] = close.pct_change() * 100
        features['vix_ma_5'] = close.rolling(5).mean()
        features['vix_ma_20'] = close.rolling(20).mean()
        
        # VIX relative to moving averages
        features['vix_to_ma5'] = (close / features['vix_ma_5'] - 1) * 100
        features['vix_to_ma20'] = (close / features['vix_ma_20'] - 1) * 100
        
        # VIX volatility
        features['vix_volatility'] = close.rolling(10).std()
        
        # VIX regime indicators
        features['vix_low_regime'] = (close < 20).astype(int)
        features['vix_high_regime'] = (close > 30).astype(int)
        features['vix_spike'] = (close.pct_change() > 0.1).astype(int)
        
        # VIX term structure (if available)
        features['vix_momentum'] = close.diff(5)
        features['vix_acceleration'] = features['vix_momentum'].diff(3)
        
        return features
    
    def calculate_treasury_features(self, treasury_2y: pd.DataFrame, 
                                  treasury_10y: pd.DataFrame) -> pd.DataFrame:
        """Calculate Treasury yield features."""
        features = pd.DataFrame()
        
        if not treasury_2y.empty:
            yield_2y = treasury_2y['close']
            features['yield_2y'] = yield_2y
            features['yield_2y_change'] = yield_2y.diff()
            features['yield_2y_ma_5'] = yield_2y.rolling(5).mean()
            features['yield_2y_ma_20'] = yield_2y.rolling(20).mean()
            
        if not treasury_10y.empty:
            yield_10y = treasury_10y['close']
            features['yield_10y'] = yield_10y
            features['yield_10y_change'] = yield_10y.diff()
            features['yield_10y_ma_5'] = yield_10y.rolling(5).mean()
            features['yield_10y_ma_20'] = yield_10y.rolling(20).mean()
            
        # Yield curve features
        if not treasury_2y.empty and not treasury_10y.empty:
            # Align indices
            common_idx = treasury_2y.index.intersection(treasury_10y.index)
            yield_2y_aligned = treasury_2y.loc[common_idx, 'close']
            yield_10y_aligned = treasury_10y.loc[common_idx, 'close']
            
            curve_features = pd.DataFrame(index=common_idx)
            
            # Yield spread (10Y - 2Y)
            curve_features['yield_spread'] = yield_10y_aligned - yield_2y_aligned
            curve_features['yield_spread_change'] = curve_features['yield_spread'].diff()
            curve_features['yield_spread_ma_5'] = curve_features['yield_spread'].rolling(5).mean()
            
            # Curve steepening/flattening
            curve_features['curve_steepening'] = (curve_features['yield_spread_change'] > 0).astype(int)
            curve_features['curve_flattening'] = (curve_features['yield_spread_change'] < 0).astype(int)
            curve_features['inverted_curve'] = (curve_features['yield_spread'] < 0).astype(int)
            
            # Yield ratio
            curve_features['yield_ratio'] = yield_10y_aligned / (yield_2y_aligned + 0.01)
            
            # Merge with main features
            features = features.join(curve_features, how='outer')
            
        return features
    
    def calculate_dollar_features(self, dxy_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Dollar Index (DXY) features."""
        if dxy_data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=dxy_data.index)
        close = dxy_data['close']
        
        # DXY levels and changes
        features['dxy'] = close
        features['dxy_change'] = close.pct_change() * 100
        features['dxy_ma_5'] = close.rolling(5).mean()
        features['dxy_ma_20'] = close.rolling(20).mean()
        features['dxy_ma_50'] = close.rolling(50).mean()
        
        # DXY relative to moving averages
        features['dxy_to_ma5'] = (close / features['dxy_ma_5'] - 1) * 100
        features['dxy_to_ma20'] = (close / features['dxy_ma_20'] - 1) * 100
        
        # DXY strength indicators
        features['dxy_strong'] = (close > features['dxy_ma_20']).astype(int)
        features['dxy_momentum'] = close.diff(5)
        features['dxy_volatility'] = close.rolling(10).std()
        
        return features
    
    def calculate_sector_features(self, sector_etfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate sector ETF relative performance features."""
        features = pd.DataFrame()
        
        sector_names = list(sector_etfs.keys())
        if not sector_names:
            return features
            
        # Calculate individual sector features
        for sector, data in sector_etfs.items():
            if data.empty:
                continue
                
            close = data['close']
            sector_features = pd.DataFrame(index=data.index)
            
            # Basic features
            sector_features[f'{sector}_return'] = close.pct_change() * 100
            sector_features[f'{sector}_ma_5'] = close.rolling(5).mean()
            sector_features[f'{sector}_ma_20'] = close.rolling(20).mean()
            sector_features[f'{sector}_to_ma20'] = (close / sector_features[f'{sector}_ma_5'] - 1) * 100
            
            # Momentum
            sector_features[f'{sector}_momentum_5'] = close.pct_change(5) * 100
            sector_features[f'{sector}_momentum_20'] = close.pct_change(20) * 100
            
            if features.empty:
                features = sector_features
            else:
                features = features.join(sector_features, how='outer')
        
        # Cross-sector features
        if len(sector_names) > 1:
            # Sector rotation indicators
            return_cols = [col for col in features.columns if '_return' in col]
            momentum_cols = [col for col in features.columns if '_momentum_5' in col]
            
            if len(return_cols) > 1:
                # Sector dispersion
                features['sector_dispersion'] = features[return_cols].std(axis=1)
                
                # Leading/lagging sectors
                features['sector_range'] = (features[return_cols].max(axis=1) - 
                                          features[return_cols].min(axis=1))
            
            if len(momentum_cols) > 1:
                # Momentum dispersion
                features['sector_momentum_dispersion'] = features[momentum_cols].std(axis=1)
        
        return features
    
    def fetch_all_cross_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch all cross-asset data."""
        print("Fetching cross-asset data...")
        
        symbols = {
            'VIX': 'VIX',  # Volatility Index
            'DXY': 'UUP',  # Dollar Index ETF (since DXY might not be available)
            'TNX': 'TNX',  # 10-Year Treasury Yield
            'FVX': 'FVX',  # 5-Year Treasury Yield (as proxy for 2Y)
            # Sector ETFs
            'XLK': 'XLK',  # Technology
            'XLF': 'XLF',  # Financials
            'XLE': 'XLE',  # Energy
            'XLV': 'XLV',  # Healthcare
            'XLI': 'XLI',  # Industrials
            'XLU': 'XLU',  # Utilities
            'XLRE': 'XLRE',  # Real Estate
            'XLP': 'XLP',  # Consumer Staples
            'XLY': 'XLY',  # Consumer Discretionary
            'XLB': 'XLB',  # Materials
        }
        
        data = {}
        for name, symbol in symbols.items():
            print(f"Fetching {name} ({symbol})...")
            df = self.fetch_market_data(symbol)
            if not df.empty:
                data[name] = df
                print(f"  Retrieved {len(df)} records")
            else:
                print(f"  No data for {name}")
                
        return data
    
    def create_cross_asset_features(self, base_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create all cross-asset features aligned to base index."""
        # Fetch all data
        cross_asset_data = self.fetch_all_cross_asset_data()
        
        all_features = pd.DataFrame(index=base_index)
        
        # VIX features
        if 'VIX' in cross_asset_data:
            vix_features = self.calculate_vix_features(cross_asset_data['VIX'])
            vix_features = vix_features.reindex(base_index, method='ffill')
            all_features = all_features.join(vix_features, how='left')
        
        # Treasury features
        treasury_2y = cross_asset_data.get('FVX', pd.DataFrame())
        treasury_10y = cross_asset_data.get('TNX', pd.DataFrame())
        treasury_features = self.calculate_treasury_features(treasury_2y, treasury_10y)
        if not treasury_features.empty:
            treasury_features = treasury_features.reindex(base_index, method='ffill')
            all_features = all_features.join(treasury_features, how='left')
        
        # Dollar features
        if 'DXY' in cross_asset_data:
            dxy_features = self.calculate_dollar_features(cross_asset_data['DXY'])
            dxy_features = dxy_features.reindex(base_index, method='ffill')
            all_features = all_features.join(dxy_features, how='left')
        
        # Sector features
        sector_etfs = {k: v for k, v in cross_asset_data.items() 
                      if k.startswith('XL') and len(k) <= 5}
        sector_features = self.calculate_sector_features(sector_etfs)
        if not sector_features.empty:
            sector_features = sector_features.reindex(base_index, method='ffill')
            all_features = all_features.join(sector_features, how='left')
        
        print(f"Generated {len(all_features.columns)} cross-asset features")
        return all_features


def add_cross_asset_features(main_data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Add cross-asset features to main dataset.
    
    Args:
        main_data_path: Path to main dataset CSV
        output_path: Optional path to save enhanced dataset
        
    Returns:
        DataFrame with cross-asset features added
    """
    # Load main data
    df = pd.read_csv(main_data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Get API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Warning: No Alpha Vantage API key found. Skipping cross-asset features.")
        return df
    
    # Calculate cross-asset features
    cross_asset_calc = CrossAssetFeatures(api_key)
    cross_asset_features = cross_asset_calc.create_cross_asset_features(df.index)
    
    # Combine with main data
    result = pd.concat([df, cross_asset_features], axis=1)
    
    if output_path:
        result.to_csv(output_path)
        print(f"Enhanced dataset with cross-asset features saved to {output_path}")
    
    return result


if __name__ == "__main__":
    # Example usage
    main_data_path = "../../dataset/raw/merged/AAPL_TECHNICAL_FEATURES_ENHANCED.csv"
    output_path = "../../dataset/raw/merged/AAPL_WITH_CROSS_ASSET_FEATURES.csv"
    
    enhanced_df = add_cross_asset_features(main_data_path, output_path)
    print(f"Final dataset shape: {enhanced_df.shape}")
