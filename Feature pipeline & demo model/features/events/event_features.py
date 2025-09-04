"""
Event features, тука се земаат временски податоци и  додаваат корисни информации што можат да влијаат на цените.

 Гледа каков е сентиментот во вестите, дали е позитивен или негативен, дали има нагли промени, дали има многу или малку вести.
 Проверува дали се ближат важни настани како состаноци на ФЕД, објави за економски податоци или финансиски резултати на компании.
 Гледа дали има празници, половични работни денови и други моменти кога пазарите не функционираат нормално.

На крај, сè тоа го комбинира и ти враќа подобар dataset за предвиување.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


class EventFeatureEngine:
    """
    Event feature engineering for time series forecasting.
    Processes news sentiment and creates event indicators.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data containing sentiment features.
        
        Args:
            data: DataFrame with sentiment columns and datetime index
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        
    def calculate_sentiment_features(self) -> pd.DataFrame:
        """Calculate enhanced sentiment features from news data."""
        print("Calculating enhanced sentiment features...")
        
        # Basic sentiment features (if they exist)
        sentiment_cols = ['sent_mean', 'sent_max_abs', 'bullish_cnt', 'bearish_cnt', 'headline_count']
        available_cols = [col for col in sentiment_cols if col in self.data.columns]
        
        if not available_cols:
            print("No sentiment columns found in data")
            return pd.DataFrame(index=self.data.index)
        
        # Rolling sentiment features
        self._rolling_sentiment_features(available_cols)
        
        # Sentiment momentum and volatility
        self._sentiment_momentum_features()
        
        # News volume analysis
        self._news_volume_features()
        
        # Sentiment extremes and regime detection
        self._sentiment_regime_features()
        
        return self.features
    
    def _rolling_sentiment_features(self, sentiment_cols: List[str]):
        """Calculate rolling sentiment statistics."""
        windows = [3, 7, 14, 30]  # Different time windows
        
        for col in sentiment_cols:
            if col not in self.data.columns:
                continue
                
            values = self.data[col]
            
            for window in windows:
                # Rolling statistics
                self.features[f'{col}_ma_{window}'] = values.rolling(window).mean()
                self.features[f'{col}_std_{window}'] = values.rolling(window).std()
                self.features[f'{col}_min_{window}'] = values.rolling(window).min()
                self.features[f'{col}_max_{window}'] = values.rolling(window).max()
                
                # Rolling quantiles
                self.features[f'{col}_q25_{window}'] = values.rolling(window).quantile(0.25)
                self.features[f'{col}_q75_{window}'] = values.rolling(window).quantile(0.75)
                
                # Z-score (standardized values)
                rolling_mean = values.rolling(window).mean()
                rolling_std = values.rolling(window).std()
                self.features[f'{col}_zscore_{window}'] = (values - rolling_mean) / (rolling_std + 1e-10)
    
    def _sentiment_momentum_features(self):
        """Calculate sentiment momentum and acceleration features."""
        base_cols = ['sent_mean', 'bullish_cnt', 'bearish_cnt', 'headline_count']
        
        for col in base_cols:
            if col not in self.data.columns:
                continue
                
            values = self.data[col]
            
            # First-order momentum (velocity)
            for period in [1, 3, 5, 10]:
                self.features[f'{col}_momentum_{period}'] = values.diff(period)
                
            # Second-order momentum (acceleration)
            for period in [3, 5, 10]:
                momentum = values.diff(period)
                self.features[f'{col}_acceleration_{period}'] = momentum.diff(period)
                
            # Exponential moving average momentum
            ema_short = values.ewm(span=3).mean()
            ema_long = values.ewm(span=10).mean()
            self.features[f'{col}_ema_momentum'] = ema_short - ema_long
            
            # Trend strength
            self.features[f'{col}_trend_strength'] = (
                values.rolling(10).apply(lambda x: np.corrcoef(x, range(len(x)))[0, 1])
            )
    
    def _news_volume_features(self):
        """Calculate news volume and frequency features."""
        if 'headline_count' not in self.data.columns:
            return
            
        headline_count = self.data['headline_count']
        
        # News intensity features
        for window in [6, 12, 24, 48]:  # Hours for intraday data
            self.features[f'news_intensity_{window}h'] = headline_count.rolling(window).sum()
            
        # News frequency patterns
        self.features['news_frequency_hourly'] = headline_count.rolling(12).mean()  # 1 hour
        self.features['news_frequency_daily'] = headline_count.rolling(24*12).mean()  # 1 day
        
        # News burst detection
        news_ma = headline_count.rolling(24).mean()
        news_std = headline_count.rolling(24).std()
        self.features['news_burst'] = (headline_count > news_ma + 2 * news_std).astype(int)
        
        # News drought detection
        self.features['news_drought'] = (headline_count == 0).rolling(12).sum()
        
        # Time since last news
        last_news = (headline_count > 0).cumsum()
        self.features['time_since_news'] = last_news.groupby(last_news).cumcount()
    
    def _sentiment_regime_features(self):
        """Detect sentiment regimes and extreme conditions."""
        if 'sent_mean' not in self.data.columns:
            return
            
        sent_mean = self.data['sent_mean']
        
        # Sentiment regimes based on quantiles
        rolling_q25 = sent_mean.rolling(100).quantile(0.25)
        rolling_q75 = sent_mean.rolling(100).quantile(0.75)
        
        self.features['sentiment_regime_bullish'] = (sent_mean > rolling_q75).astype(int)
        self.features['sentiment_regime_bearish'] = (sent_mean < rolling_q25).astype(int)
        self.features['sentiment_regime_neutral'] = (
            (sent_mean >= rolling_q25) & (sent_mean <= rolling_q75)
        ).astype(int)
        
        # Sentiment volatility regimes
        sent_volatility = sent_mean.rolling(24).std()
        vol_q75 = sent_volatility.rolling(100).quantile(0.75)
        self.features['high_sentiment_volatility'] = (sent_volatility > vol_q75).astype(int)
        
        # Sentiment extremes
        self.features['extreme_positive_sentiment'] = (sent_mean > 0.5).astype(int)
        self.features['extreme_negative_sentiment'] = (sent_mean < -0.5).astype(int)
        
        # Sentiment persistence
        for window in [6, 12, 24]:
            bullish_persistence = (sent_mean > 0).rolling(window).sum() / window
            bearish_persistence = (sent_mean < 0).rolling(window).sum() / window
            
            self.features[f'bullish_persistence_{window}h'] = bullish_persistence
            self.features[f'bearish_persistence_{window}h'] = bearish_persistence
    
    def create_economic_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create economic calendar with major events.
        Note: In production, this should integrate with economic calendar APIs.
        """
        print("Creating economic calendar features...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        calendar = pd.DataFrame(index=date_range)
        
        # Fed meeting dates (approximate - FOMC meets 8 times per year)
        fed_meetings = self._generate_fed_meeting_dates(start_date, end_date)
        calendar['fed_meeting'] = calendar.index.isin(fed_meetings).astype(int)
        
        # Fed meeting proximity
        calendar['days_to_fed_meeting'] = self._days_to_next_event(calendar.index, fed_meetings)
        calendar['days_since_fed_meeting'] = self._days_since_last_event(calendar.index, fed_meetings)
        
        # Earnings season (approximate)
        earnings_dates = self._generate_earnings_season_dates(start_date, end_date)
        calendar['earnings_season'] = calendar.index.isin(earnings_dates).astype(int)
        
        # Major economic data releases (monthly patterns)
        calendar = self._add_economic_data_releases(calendar)
        
        # Market holidays
        calendar = self._add_market_holidays(calendar)
        
        # Options expiration (3rd Friday of each month)
        calendar['options_expiration'] = self._get_options_expiration_dates(calendar.index)
        
        return calendar
    
    def _generate_fed_meeting_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """Generate approximate FOMC meeting dates."""
        # FOMC typically meets 8 times per year, roughly every 6-7 weeks
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Approximate FOMC dates (2nd and 4th Wednesday of specific months)
        fed_dates = []
        current_year = start.year
        
        while current_year <= end.year:
            # Typical FOMC meeting months
            meeting_months = [1, 3, 5, 6, 7, 9, 11, 12]
            
            for month in meeting_months:
                # Find the first Wednesday of the month
                first_day = datetime(current_year, month, 1)
                days_to_wednesday = (2 - first_day.weekday()) % 7
                first_wednesday = first_day + timedelta(days=days_to_wednesday)
                
                # FOMC typically meets on the second Tuesday (day before second Wednesday)
                meeting_date = first_wednesday + timedelta(days=7, hours=-24)
                
                if start <= meeting_date <= end:
                    fed_dates.append(meeting_date.date())
            
            current_year += 1
        
        return fed_dates
    
    def _generate_earnings_season_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """Generate earnings season dates (approximately 3 weeks after quarter end)."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        earnings_dates = []
        current_year = start.year
        
        while current_year <= end.year:
            # Earnings seasons typically start around these dates
            season_starts = [
                datetime(current_year, 1, 15),  # Q4 previous year
                datetime(current_year, 4, 15),  # Q1
                datetime(current_year, 7, 15),  # Q2
                datetime(current_year, 10, 15),  # Q3
            ]
            
            for season_start in season_starts:
                # Earnings season lasts about 3-4 weeks
                for i in range(21):  # 3 weeks
                    earnings_date = season_start + timedelta(days=i)
                    if start <= earnings_date <= end:
                        earnings_dates.append(earnings_date.date())
            
            current_year += 1
        
        return earnings_dates
    
    def _add_economic_data_releases(self, calendar: pd.DataFrame) -> pd.DataFrame:
        """Add major economic data release indicators."""
        # First Friday of month (Employment report)
        calendar['employment_report'] = self._get_first_friday_of_month(calendar.index)
        
        # Mid-month CPI release (typically around 10th-15th)
        calendar['cpi_release'] = calendar.index.day.isin(range(10, 16)).astype(int)
        
        # End of month GDP/PCE releases
        calendar['gdp_release'] = calendar.index.day.isin(range(25, 32)).astype(int)
        
        # Weekly jobless claims (Thursdays)
        calendar['jobless_claims'] = (calendar.index.dayofweek == 3).astype(int)
        
        return calendar
    
    def _add_market_holidays(self, calendar: pd.DataFrame) -> pd.DataFrame:
        """Add market holiday indicators."""
        # US market holidays (simplified)
        calendar['market_holiday'] = 0
        
        for date in calendar.index:
            # New Year's Day
            if date.month == 1 and date.day == 1:
                calendar.loc[date, 'market_holiday'] = 1
            # Independence Day
            elif date.month == 7 and date.day == 4:
                calendar.loc[date, 'market_holiday'] = 1
            # Christmas
            elif date.month == 12 and date.day == 25:
                calendar.loc[date, 'market_holiday'] = 1
            # Thanksgiving (4th Thursday in November)
            elif date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28:
                calendar.loc[date, 'market_holiday'] = 1
        
        # Half-day trading indicators
        calendar['half_day_trading'] = 0
        for date in calendar.index:
            # Day after Thanksgiving
            if date.month == 11 and date.weekday() == 4 and 23 <= date.day <= 29:
                calendar.loc[date, 'half_day_trading'] = 1
            # Christmas Eve
            elif date.month == 12 and date.day == 24:
                calendar.loc[date, 'half_day_trading'] = 1
        
        return calendar
    
    def _get_first_friday_of_month(self, date_index: pd.DatetimeIndex) -> pd.Series:
        """Get first Friday of each month indicator."""
        result = pd.Series(0, index=date_index)
        
        for date in date_index:
            # Check if it's Friday and in the first week
            if date.weekday() == 4 and 1 <= date.day <= 7:
                result[date] = 1
        
        return result
    
    def _get_options_expiration_dates(self, date_index: pd.DatetimeIndex) -> pd.Series:
        """Get options expiration dates (3rd Friday of each month)."""
        result = pd.Series(0, index=date_index)
        
        for date in date_index:
            # Check if it's Friday and in the third week
            if date.weekday() == 4 and 15 <= date.day <= 21:
                result[date] = 1
        
        return result
    
    def _days_to_next_event(self, date_index: pd.DatetimeIndex, 
                           event_dates: List[datetime]) -> pd.Series:
        """Calculate days to next event."""
        result = pd.Series(index=date_index, dtype=float)
        event_dates_pd = pd.to_datetime(event_dates)
        
        for date in date_index:
            future_events = event_dates_pd[event_dates_pd >= date]
            if len(future_events) > 0:
                days_to_next = (future_events[0] - date).days
                result[date] = days_to_next
            else:
                result[date] = np.nan
        
        return result
    
    def _days_since_last_event(self, date_index: pd.DatetimeIndex, 
                              event_dates: List[datetime]) -> pd.Series:
        """Calculate days since last event."""
        result = pd.Series(index=date_index, dtype=float)
        event_dates_pd = pd.to_datetime(event_dates)
        
        for date in date_index:
            past_events = event_dates_pd[event_dates_pd <= date]
            if len(past_events) > 0:
                days_since_last = (date - past_events[-1]).days
                result[date] = days_since_last
            else:
                result[date] = np.nan
        
        return result


def create_event_features(data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Create comprehensive event features for TFT model.
    
    Args:
        data_path: Path to input data CSV
        output_path: Optional path to save enhanced data
        
    Returns:
        DataFrame with event features
    """
    # Load data
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Create event feature engine
    event_engine = EventFeatureEngine(df)
    
    # Calculate sentiment features
    sentiment_features = event_engine.calculate_sentiment_features()
    
    # Create economic calendar
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    calendar = event_engine.create_economic_calendar(start_date, end_date)
    
    # Resample calendar to match data frequency (if intraday)
    if len(df) > len(calendar) * 2:  # Likely intraday data
        calendar_resampled = calendar.reindex(df.index, method='ffill')
    else:
        calendar_resampled = calendar.reindex(df.index, method='nearest')
    
    # Combine all features
    result = pd.concat([df, sentiment_features, calendar_resampled], axis=1)
    
    if output_path:
        result.to_csv(output_path)
        print(f"Event features saved to {output_path}")
    
    print(f"Generated {len(sentiment_features.columns) + len(calendar.columns)} event features")
    return result


if __name__ == "__main__":
    # Example usage
    data_path = "../../dataset/raw/merged/AAPL_WITH_CROSS_ASSET_FEATURES.csv"
    output_path = "../../dataset/raw/merged/AAPL_WITH_EVENT_FEATURES.csv"
    
    enhanced_df = create_event_features(data_path, output_path)
    print(f"Final dataset shape: {enhanced_df.shape}")
