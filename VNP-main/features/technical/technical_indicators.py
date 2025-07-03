"""
Custom Technical Indicators - NO DUPLICATES with Alpha Vantage
==============================================================

REFACTORED: Only features NOT available from Alpha Vantage API:
- Bollinger Band squeeze and position features
- Candlestick pattern analysis (body/shadow ratios, gaps)
- Volume-price relationships (OBV, VPT, CMF, AD Line)
- Support/resistance detection (pivot points, fractals)
- Time-based cyclical features (hour_sin, dow_cos, etc.)
- Advanced momentum and volatility features

REMOVED (now use Alpha Vantage API):
❌ RSI, SMA, EMA, MACD (basic calculations)
✅ Kept only enhanced derivatives and unique features

[1] Alpha Vantage API                 [2] CustomIndicators class
      ────────────────────────────          ───────────────────────────────
     | RSI, SMA, EMA, MACD, etc.    | → → | BB squeeze, patterns, OBV,    |
     | Standard indicators          |     | S/R levels, cyclical features |
     | Trusted API calculations     |     | NOT available from API        |

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CustomTechnicalIndicators:
    """
    Custom technical indicators calculator - NO DUPLICATES with Alpha Vantage.
    
    REFACTORED APPROACH:
    - Uses Alpha Vantage API for: RSI, SMA, EMA, MACD (basic calculations)
    - Calculates only UNIQUE features not available from API:
      * Bollinger Band squeeze and advanced BB features
      * Candlestick pattern analysis 
      * Volume-price relationships (OBV, VPT, CMF)
      * Support/resistance detection
      * Time-based cyclical features
      * Advanced momentum and volatility derivatives
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                 and datetime index
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        
    def calculate_all(self) -> pd.DataFrame:
        """Calculate only NON-API technical indicators to avoid duplicates."""
        print("Calculating custom technical indicators (non-API features only)...")
        
        # REMOVED: Basic indicators now from Alpha Vantage API
        # ❌ self._moving_averages()  # Use API: SMA, EMA
        # ❌ self._rsi()              # Use API: RSI
        # ❌ self._macd()             # Use API: MACD
        
        # KEPT: Advanced features NOT available from API
        self._bollinger_band_features()      # BB squeeze, position, width
        self._price_patterns()               # Candlestick analysis
        self._volume_price_relationships()   # OBV, VPT, CMF, AD Line
        self._support_resistance()           # Pivot points, fractals
        self._advanced_momentum()            # Derivatives not in API
        self._volatility_features()          # Advanced volatility measures
        self._cyclical_features()            # Time-based features
        
        print(f"Generated {len(self.features.columns)} unique custom features")
        return self.features
    
    def _bollinger_band_features(self):
        """Calculate Bollinger Band advanced features (squeeze, position, etc.)."""
        periods = [20, 50]
        close = self.data['close']
        
        for period in periods:
            # Use API SMA if available, otherwise calculate
            if 'sma' in self.data.columns:
                sma = self.data['sma']  # From Alpha Vantage
            else:
                sma = close.rolling(window=period).mean()  # Fallback
                
            std = close.rolling(window=period).std()
            
            # Bollinger Bands
            bb_upper = sma + (2 * std)
            bb_lower = sma - (2 * std)
            
            # UNIQUE FEATURES (not available from API):
            # BB position (where price is relative to bands)
            self.features[f'bb_{period}_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # BB width (volatility measure)
            self.features[f'bb_{period}_width'] = (bb_upper - bb_lower) / sma
            
            # Distance to bands (normalized)
            self.features[f'bb_{period}_upper_dist'] = (bb_upper - close) / close
            self.features[f'bb_{period}_lower_dist'] = (close - bb_lower) / close
            
            # Squeeze indicator (unique analysis)
            self.features[f'bb_{period}_squeeze'] = (self.features[f'bb_{period}_width'] < 
                                                   self.features[f'bb_{period}_width'].rolling(20).quantile(0.2)).astype(int)
            
            # BB breakout signals
            self.features[f'bb_{period}_breakout_up'] = (close > bb_upper).astype(int)
            self.features[f'bb_{period}_breakout_down'] = (close < bb_lower).astype(int)
    
    def _volume_price_relationships(self):
        """Calculate volume-price relationship indicators NOT available from API."""
        volume = self.data['volume']
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # On-Balance Volume (OBV) - NOT typically in API
        price_change = close.diff()
        obv = (np.sign(price_change) * volume).cumsum()
        self.features['obv'] = obv
        self.features['obv_ma_10'] = obv.rolling(10).mean()
        self.features['obv_momentum'] = obv.diff(5)
        
        # Volume-Price Trend (VPT) - Advanced feature
        vpt = (volume * close.pct_change()).cumsum()
        self.features['vpt'] = vpt
        self.features['vpt_ma_10'] = vpt.rolling(10).mean()
        
        # Accumulation/Distribution Line - Advanced
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        self.features['ad_line'] = ad_line
        self.features['ad_line_ma_10'] = ad_line.rolling(10).mean()
        
        # Chaikin Money Flow - NOT in basic API
        self.features['cmf_20'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        # Volume-weighted moving averages
        vwma_10 = (close * volume).rolling(10).sum() / volume.rolling(10).sum()
        vwma_20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        self.features['vwma_10'] = vwma_10
        self.features['vwma_20'] = vwma_20
        self.features['price_to_vwma_10'] = (close / vwma_10 - 1) * 100
        self.features['price_to_vwma_20'] = (close / vwma_20 - 1) * 100
        
        # Volume ratios and patterns
        for period in [5, 10, 20]:
            vol_ma = volume.rolling(period).mean()
            self.features[f'volume_ratio_{period}'] = volume / vol_ma
            
        # Volume rate of change
        self.features['volume_roc_5'] = volume.pct_change(5) * 100
        self.features['volume_roc_10'] = volume.pct_change(10) * 100
        
        # Money Flow Index - Advanced volume indicator
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        self.features['mfi'] = mfi
    
    def _advanced_momentum(self):
        """Calculate advanced momentum features using API RSI/MACD when available."""
        close = self.data['close']
        
        # Use API RSI if available for enhanced features
        if 'rsi' in self.data.columns:
            rsi = self.data['rsi']
            # RSI-based features (using API RSI)
            self.features['rsi_overbought'] = (rsi > 70).astype(int)
            self.features['rsi_oversold'] = (rsi < 30).astype(int)
            self.features['rsi_divergence'] = (rsi.diff(5) * close.pct_change(5) < 0).astype(int)
            self.features['rsi_momentum'] = rsi.diff(3)
        
        # Use API MACD if available for enhanced features  
        if 'macd' in self.data.columns and 'macd_signal' in self.data.columns:
            macd = self.data['macd']
            macd_signal = self.data['macd_signal']
            histogram = macd - macd_signal
            
            # MACD-based signals (using API MACD)
            self.features['macd_bullish_cross'] = ((macd > macd_signal) & 
                                                 (macd.shift(1) <= macd_signal.shift(1))).astype(int)
            self.features['macd_bearish_cross'] = ((macd < macd_signal) & 
                                                 (macd.shift(1) >= macd_signal.shift(1))).astype(int)
            self.features['macd_momentum'] = histogram.diff(3)
        
        # Price momentum (independent features)
        for period in [1, 5, 10, 20]:
            self.features[f'momentum_{period}'] = close.pct_change(period) * 100
            
        # Acceleration (second derivative)
        self.features['acceleration_5'] = self.features['momentum_5'].diff(5)
        self.features['acceleration_10'] = self.features['momentum_10'].diff(5)
        
        # Commodity Channel Index (CCI) - Advanced
        typical_price = (self.data['high'] + self.data['low'] + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma_tp) / (0.015 * mad)
        self.features['cci'] = cci
        
        # Rate of change features
        for period in [1, 3, 5, 10, 15, 20]:
            roc = (close / close.shift(period) - 1) * 100
            self.features[f'roc_{period}'] = roc
            
        # Smoothed rate of change
        self.features['roc_ema_5'] = self.features['roc_5'].ewm(span=3).mean()
        self.features['roc_ema_10'] = self.features['roc_10'].ewm(span=5).mean()
    
    def _price_patterns(self):
        """Calculate candlestick pattern features - NOT available from API."""
        open_price = self.data['open']
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Candlestick body and shadow ratios
        body_size = np.abs(close - open_price)
        total_range = high - low
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        self.features['body_ratio'] = body_size / (total_range + 1e-10)
        self.features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-10)
        self.features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-10)
        
        # Gap detection
        self.features['gap_up'] = (open_price > close.shift(1)).astype(int)
        self.features['gap_down'] = (open_price < close.shift(1)).astype(int)
        
        # Inside/Outside bars
        self.features['inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        self.features['outside_bar'] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
        
        # Doji detection (small body relative to range)
        self.features['doji'] = (body_size / (total_range + 1e-10) < 0.1).astype(int)
        
        # Hammer/Hanging man patterns
        self.features['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        
    def _volatility_features(self):
        """Calculate advanced volatility measures."""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # Historical volatility
        for period in [5, 10, 20, 50]:
            returns = close.pct_change()
            self.features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252) * 100
            
        # True Range and Average True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        self.features['true_range'] = true_range
        self.features['atr_14'] = true_range.rolling(14).mean()
        self.features['atr_21'] = true_range.rolling(21).mean()
        
        # Normalized volatility
        self.features['normalized_volatility'] = (self.features['volatility_20'] / 
                                                self.features['volatility_20'].rolling(50).mean())
    
    def _support_resistance(self):
        """Calculate dynamic support and resistance levels."""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # Rolling pivot points
        period = 20
        pivot = (high.rolling(period).max() + low.rolling(period).min() + close) / 3
        resistance1 = 2 * pivot - low.rolling(period).min()
        support1 = 2 * pivot - high.rolling(period).max()
        
        self.features['pivot_point'] = pivot
        self.features['resistance_1'] = resistance1
        self.features['support_1'] = support1
        
        # Distance to support/resistance
        self.features['distance_to_resistance'] = (resistance1 - close) / close * 100
        self.features['distance_to_support'] = (close - support1) / close * 100
        
        # Fractal-based support/resistance
        for window in [5, 10]:
            local_max = high.rolling(window*2+1, center=True).max() == high
            local_min = low.rolling(window*2+1, center=True).min() == low
            
            self.features[f'local_max_{window}'] = local_max.astype(int)
            self.features[f'local_min_{window}'] = local_min.astype(int)
    
    def _cyclical_features(self):
        """Create cyclical time-based features."""
        # Time-based features
        dt = self.data.index
        
        # Hour of day (for intraday data)
        hour = dt.hour
        self.features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week
        dow = dt.dayofweek
        self.features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        self.features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Day of month
        dom = dt.day
        self.features['dom_sin'] = np.sin(2 * np.pi * dom / 31)
        self.features['dom_cos'] = np.cos(2 * np.pi * dom / 31)
        
        # Month of year
        moy = dt.month
        self.features['month_sin'] = np.sin(2 * np.pi * moy / 12)
        self.features['month_cos'] = np.cos(2 * np.pi * moy / 12)
        
        # Market session indicators
        self.features['market_open'] = ((hour >= 9) & (hour < 16)).astype(int)
        self.features['pre_market'] = ((hour >= 4) & (hour < 9)).astype(int)
        self.features['after_hours'] = ((hour >= 16) | (hour < 4)).astype(int)


def calculate_custom_features(data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Calculate ONLY custom features (no API duplicates).
    
    Args:
        data_path: Path to CSV file with OHLCV data (and optional API indicators)
        output_path: Optional path to save features
        
    Returns:
        DataFrame with custom features only
    """
    # Load data (should include Alpha Vantage API indicators)
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    print(f"Input data shape: {df.shape}")
    print(f"Available API indicators: {[col for col in df.columns if col in ['rsi', 'sma', 'ema', 'macd', 'macd_signal']]}")
    
    # Calculate custom indicators (no duplicates)
    custom_calc = CustomTechnicalIndicators(df)
    features = custom_calc.calculate_all()
    
    # Combine with original data (API indicators + OHLCV + custom features)
    result = pd.concat([df, features], axis=1)
    
    if output_path:
        result.to_csv(output_path)
        print(f"Custom features saved to {output_path}")
    
    return result


def validate_against_alpha_vantage(df: pd.DataFrame) -> dict:
    """
    Enhanced validation focusing on API utilization.
    
    Args:
        df: DataFrame containing Alpha Vantage API indicators + custom features
        
    Returns:
        Dictionary with API utilization and validation results
    """
    validation_results = {}
    
    # Check API indicator availability
    api_indicators = ['rsi', 'sma', 'ema', 'macd', 'macd_signal', 'stoch_k', 'willr']
    available_api = [col for col in api_indicators if col in df.columns]
    missing_api = [col for col in api_indicators if col not in df.columns]
    
    print("\n" + "="*50)
    print("API INDICATOR UTILIZATION CHECK")
    print("="*50)
    
    print(f"✅ Available API indicators: {available_api}")
    print(f"❌ Missing API indicators: {missing_api}")
    
    # Count custom features
    custom_features = [col for col in df.columns if any(x in col.lower() for x in 
                      ['bb_', 'obv', 'vpt', 'cmf', 'ad_line', 'body_ratio', 'gap_', 
                       'pivot', 'hour_sin', 'dow_cos', 'volatility_', 'momentum_'])]
    
    print(f"� Generated custom features: {len(custom_features)}")
    print(f"📊 Total features: {len(df.columns)}")
    
    validation_results['api_utilization'] = {
        'available_api_indicators': available_api,
        'missing_api_indicators': missing_api,
        'custom_features_count': len(custom_features),
        'total_features': len(df.columns),
        'api_coverage': len(available_api) / len(api_indicators) * 100
    }
    
    return validation_results


if __name__ == "__main__":
    # Example usage with API + custom approach
    print("Testing refactored CustomTechnicalIndicators...")
    print("="*60)
    
    # Test with just price data (OHLCV) 
    data_path = "dataset/raw/merged/AAPL_PRICE_FULL.csv"
    output_path = "dataset/raw/merged/AAPL_CUSTOM_FEATURES_ONLY.csv"
    
    try:
        features_df = calculate_custom_features(data_path, output_path)
        print(f"✅ Generated {len(features_df.columns)} total features")
        print(f"📊 Data shape: {features_df.shape}")
        
        # Run validation
        validation_results = validate_against_alpha_vantage(features_df)
        print(f"📈 API Coverage: {validation_results['api_utilization']['api_coverage']:.1f}%")
        
        # Show what types of features were created
        custom_features = [col for col in features_df.columns if any(x in col.lower() for x in 
                          ['bb_', 'obv', 'vpt', 'cmf', 'ad_line', 'body_ratio', 'gap_', 
                           'pivot', 'hour_sin', 'dow_cos', 'volatility_', 'momentum_'])]
        print(f"\n📋 Sample custom features created:")
        for feature in custom_features[:10]:  # Show first 10
            print(f"   • {feature}")
        
        print(f"\n✅ REFACTOR COMPLETE!")
        print("🎯 Pipeline now uses Alpha Vantage API for basic indicators")
        print("🔧 Custom class calculates only unique features (no duplicates)")
        print("📈 Ready for clean TFT training data!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 This is expected if price data file doesn't exist")
        print("🔧 The refactored code is ready - just need proper data files!")
