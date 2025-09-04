"""
TFT-Specific Data Preparation and Feature Processing
==================================================

Ги спрема податоците за TFT моделот, категоризација,скалиране,форматиране....



- Povikan od strana na feature_pipeline.py 
- Prava kategorizacija na features.
- presmetuva feature importance i gi zima najdobrite 63 (momentalno, moze da se proba so pomalku ~50, povekje mislam ne.)
- prava skaliranJE na vrednostite, se koriste RobustScaler(najdobar za finanskiski time series podatoci, dobro se spravuva so outlayeri, stabilen e). Moze da se\
  isproba i QuantileUniformTransformer.
- Gi formatira podatocite vo format koj gi ocekuva TFT modelot.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class TFTDataPreprocessor:
   
    
    def __init__(self, target_column: str = 'close', 
                 time_column: str = 'datetime',
                 prediction_horizon: int = 12):
        """
        Initialize TFT preprocessor.
        
        Args:
            target_column: Name of the target variable to predict
            time_column: Name of the datetime column
            prediction_horizon: How many steps ahead to predict
        """
        self.target_column = target_column
        self.time_column = time_column
        self.prediction_horizon = prediction_horizon
        
        # Feature categorization
        self.static_features = []
        self.known_future_features = []
        self.unknown_future_features = []
        self.target_features = []
        
        # Scalers
        self.scalers = {}
        self.label_encoders = {}
        
        # Feature selection
        self.selected_features = {}
        
    def categorize_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features into TFT input types.
        
        Returns:
            Dictionary with feature categories
        """
        print("Categorizing features for TFT...")
        
        all_columns = data.columns.tolist()
        
        # Remove datetime and target from feature lists
        feature_columns = [col for col in all_columns 
                          if col not in [self.time_column, self.target_column]]
        
        # Static covariates (features that don't change over time)
        self.static_features = self._identify_static_features(data, feature_columns)
        
        # Known future inputs (scheduled events, calendar features)
        self.known_future_features = self._identify_known_future_features(feature_columns)
        
        # Unknown future inputs (everything else)
        remaining_features = [col for col in feature_columns 
                            if col not in self.static_features + self.known_future_features]
        self.unknown_future_features = remaining_features
        
        # Target-related features
        self.target_features = [self.target_column]
        
        categorization = {
            'static': self.static_features,
            'known_future': self.known_future_features,
            'unknown_future': self.unknown_future_features,
            'target': self.target_features
        }
        
        print(f"Static features: {len(self.static_features)}")
        print(f"Known future features: {len(self.known_future_features)}")
        print(f"Unknown future features: {len(self.unknown_future_features)}")
        
        return categorization
    
    def _identify_static_features(self, data: pd.DataFrame, 
                                 feature_columns: List[str]) -> List[str]:
        """Identify features that are static (don't change over time)."""
        static_features = []
        
        # Features that are typically static
        static_patterns = [
            'sector', 'market_cap', 'beta', 'industry',
            'exchange', 'currency', 'country'
        ]
        
        for col in feature_columns:
            # Check if column name indicates static feature
            if any(pattern in col.lower() for pattern in static_patterns):
                static_features.append(col)
                continue
                
            # Check if values are constant or nearly constant
            if col in data.columns:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.01:  # Less than 1% unique values
                    static_features.append(col)
        
        return static_features
    
    def _identify_known_future_features(self, feature_columns: List[str]) -> List[str]:
        """Identify features that are known in the future (scheduled events)."""
        known_future_features = []
        
        # Features that are typically known in advance
        known_patterns = [
            'fed_meeting', 'earnings', 'holiday', 'expiration',
            'employment_report', 'cpi_release', 'gdp_release',
            'jobless_claims', 'market_holiday', 'half_day_trading',
            'options_expiration', 'days_to_', 'days_since_',
            # Time-based features
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'dom_sin', 'dom_cos', 'month_sin', 'month_cos',
            'market_open', 'pre_market', 'after_hours'
        ]
        
        for col in feature_columns:
            if any(pattern in col.lower() for pattern in known_patterns):
                known_future_features.append(col)
        
        return known_future_features
    
    def add_static_covariates(self, data: pd.DataFrame, 
                             symbol: str = "AAPL") -> pd.DataFrame:
        """Add static covariates for the stock."""
        data = data.copy()
        
        # Stock-specific static information
        # Note: In production, this should come from a reference data source
        static_info = self._get_stock_static_info(symbol)
        
        for key, value in static_info.items():
            data[key] = value
            
        return data
    
    def _get_stock_static_info(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """Get static information for a stock symbol."""
        # This is a simplified example - in production, use financial data APIs
        stock_info = {
            'AAPL': {
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap_category': 'Large Cap',
                'exchange': 'NASDAQ',
                'beta_category': 'Low Volatility',  # Categorical version of beta
                'dividend_paying': 1,
                'sp500_member': 1,
                'country': 'US'
            },
            # Add more stocks as needed
        }
        
        return stock_info.get(symbol, {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap_category': 'Unknown',
            'exchange': 'Unknown',
            'beta_category': 'Unknown',
            'dividend_paying': 0,
            'sp500_member': 0,
            'country': 'Unknown'
        })
    
    def create_lagged_features(self, data: pd.DataFrame, 
                              lag_periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged features for time series modeling."""
        print("Creating lagged features...")
        
        data = data.copy()
        
        # Features to lag (excluding categorical and already-lagged features)
        lag_candidates = [col for col in self.unknown_future_features 
                         if not any(skip in col.lower() for skip in 
                                  ['_lag_', 'regime', '_sin', '_cos', 'category'])]
        
        # Add target lags
        lag_candidates.append(self.target_column)
        
        for col in lag_candidates:
            if col in data.columns:
                for lag in lag_periods:
                    lag_col_name = f"{col}_lag_{lag}"
                    data[lag_col_name] = data[col].shift(lag)
                    
                    # Add lagged features to unknown future features
                    if col != self.target_column:
                        self.unknown_future_features.append(lag_col_name)
        
        print(f"Created lagged features for {len(lag_candidates)} variables")
        return data
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        print("Creating interaction features...")
        
        data = data.copy()
        
        # Key feature groups for interactions
        price_features = [col for col in data.columns if any(term in col.lower() 
                         for term in ['price', 'close', 'open', 'high', 'low'])]
        volume_features = [col for col in data.columns if 'volume' in col.lower()]
        sentiment_features = [col for col in data.columns if 'sent' in col.lower()]
        volatility_features = [col for col in data.columns if any(term in col.lower() 
                              for term in ['volatility', 'vix', 'atr', 'bb_width'])]
        
        # Price-Volume interactions
        for price_col in price_features[:3]:  # Limit to avoid too many features
            for vol_col in volume_features[:2]:
                if price_col in data.columns and vol_col in data.columns:
                    interaction_name = f"{price_col}_x_{vol_col}"
                    data[interaction_name] = data[price_col] * data[vol_col]
                    self.unknown_future_features.append(interaction_name)
        
        # Sentiment-Volatility interactions
        for sent_col in sentiment_features[:3]:
            for vol_col in volatility_features[:2]:
                if sent_col in data.columns and vol_col in data.columns:
                    interaction_name = f"{sent_col}_x_{vol_col}"
                    data[interaction_name] = data[sent_col] * data[vol_col]
                    self.unknown_future_features.append(interaction_name)
        
        # Technical indicator combinations
        if 'rsi_14' in data.columns and 'macd_histogram' in data.columns:
            data['rsi_macd_combined'] = data['rsi_14'] * data['macd_histogram']
            self.unknown_future_features.append('rsi_macd_combined')
        
        print(f"Created interaction features")
        return data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using robust scaling to handle outliers."""
        print("Scaling features...")
        
        data = data.copy()
        
        # CRITICAL: Preserve original close prices for TFT target
        if self.target_column in data.columns and 'close_original' not in data.columns:
            data['close_original'] = data[self.target_column].copy()
            print(f"✅ Preserved original {self.target_column} prices for TFT target")
        
        # CRITICAL: For TFT, DO NOT scale the target column
        # TFT uses GroupNormalizer internally for target normalization
        
        # Features to scale (numerical features EXCLUDING target)
        scale_features = {
            'unknown_future': [col for col in self.unknown_future_features 
                             if col in data.columns and col != self.target_column 
                             and data[col].dtype in ['float64', 'int64']],
            'known_future': [col for col in self.known_future_features 
                           if col in data.columns and col != self.target_column
                           and data[col].dtype in ['float64', 'int64']]
        }
        
        for feature_type, features in scale_features.items():
            if not features:
                continue
                
            scaler_name = f"{feature_type}_scaler"
            
            if fit:
                # Use RobustScaler for financial data (handles outliers better)
                self.scalers[scaler_name] = RobustScaler()
                data[features] = self.scalers[scaler_name].fit_transform(data[features])
            else:
                if scaler_name in self.scalers:
                    data[features] = self.scalers[scaler_name].transform(data[features])
        
        print(f"✅ Scaled {sum(len(f) for f in scale_features.values())} features (target kept unscaled)")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features for TFT."""
        print("Encoding categorical features...")
        
        data = data.copy()
        
        # Identify categorical features
        categorical_features = [col for col in data.columns 
                              if data[col].dtype == 'object' or 
                              any(term in col.lower() for term in 
                                  ['category', 'regime', 'sector', 'industry', 'exchange'])]
        
        for col in categorical_features:
            if col in data.columns:
                encoder_name = f"{col}_encoder"
                
                if fit:
                    self.label_encoders[encoder_name] = LabelEncoder()
                    # Handle missing values
                    data[col] = data[col].fillna('Unknown')
                    data[col] = self.label_encoders[encoder_name].fit_transform(data[col].astype(str))
                else:
                    if encoder_name in self.label_encoders:
                        data[col] = data[col].fillna('Unknown')
                        # Handle unseen categories
                        known_classes = self.label_encoders[encoder_name].classes_
                        data[col] = data[col].apply(lambda x: x if x in known_classes else 'Unknown')
                        data[col] = self.label_encoders[encoder_name].transform(data[col].astype(str))
        
        return data
    
    def select_features(self, data: pd.DataFrame, 
                       max_features_per_type: Dict[str, int] = None) -> pd.DataFrame:
        """Select most important features using various selection methods."""
        print("Selecting features...")
        
        if max_features_per_type is None:
            max_features_per_type = {
                'unknown_future': 50,
                'known_future': 20,
                'static': 10
            }
        
        data = data.copy()
        
        # Prepare target for feature selection
        if self.target_column not in data.columns:
            print("Target column not found for feature selection")
            return data
        
        target = data[self.target_column].dropna()
        
        # Select features for each category
        for feature_type, max_features in max_features_per_type.items():
            if feature_type == 'unknown_future':
                features = self.unknown_future_features
            elif feature_type == 'known_future':
                features = self.known_future_features
            elif feature_type == 'static':
                features = self.static_features
            else:
                continue
            
            # Filter features that exist in data and have valid values
            available_features = [col for col in features 
                                if col in data.columns and 
                                data[col].notna().sum() > len(data) * 0.1]  # At least 10% non-null
            
            if len(available_features) <= max_features:
                self.selected_features[feature_type] = available_features
                continue
            
            # Feature selection
            feature_data = data[available_features].fillna(0)
            
            # Align with target
            common_idx = feature_data.index.intersection(target.index)
            if len(common_idx) == 0:
                self.selected_features[feature_type] = available_features[:max_features]
                continue
                
            X = feature_data.loc[common_idx]
            y = target.loc[common_idx]
            
            # Use mutual information for feature selection
            try:
                selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
                selector.fit(X, y)
                selected_mask = selector.get_support()
                selected_features = [available_features[i] for i, selected in enumerate(selected_mask) if selected]
                self.selected_features[feature_type] = selected_features
                
                print(f"Selected {len(selected_features)} {feature_type} features")
                
            except Exception as e:
                print(f"Feature selection failed for {feature_type}: {e}")
                self.selected_features[feature_type] = available_features[:max_features]
        
        return data
    
    def prepare_tft_dataset(self, data: pd.DataFrame, 
                           entity_id: str = 'AAPL',
                           min_sequence_length: int = 30) -> pd.DataFrame:
        """
        Prepare final dataset in TFT format.
        
        Args:
            data: Preprocessed data
            entity_id: Identifier for the time series entity
            min_sequence_length: Minimum sequence length for training
            
        Returns:
            TFT-formatted dataset
        """
        print("Preparing TFT dataset format...")
        
        data = data.copy()
        
        # Add entity identifier
        data['entity_id'] = entity_id
        
        # Add time index (needed for TFT)
        data['time_idx'] = range(len(data))
        
        # Create single target column for TFT (shifted by prediction horizon)
        # CRITICAL: Use original unscaled close prices for target
        if 'close_original' in data.columns:
            # Use preserved original prices
            data['target'] = data['close_original'].shift(-self.prediction_horizon)
        else:
            # Use current close column (should be unscaled after our fix)
            data['target'] = data[self.target_column].shift(-self.prediction_horizon)
        
        # Remove rows where target is NaN (caused by the shift operation)
        valid_indices = data['target'].notna()
        data = data[valid_indices].copy()
        
        # Reset time_idx after removing rows
        data['time_idx'] = range(len(data))
        
        # Filter selected features
        final_columns = ['entity_id', 'time_idx', 'target']
        
        # Add the original target column (for reference and features)
        final_columns.append(self.target_column)
        
        # Add selected features
        for feature_type, features in self.selected_features.items():
            final_columns.extend([col for col in features if col in data.columns])
        
        # Remove duplicates while preserving order
        final_columns = list(dict.fromkeys(final_columns))
        
        # Filter data
        tft_data = data[final_columns].copy()
        
        # Handle missing values (but preserve target column integrity)
        # Forward fill then backward fill for non-target columns
        for col in tft_data.columns:
            if col not in ['target', 'entity_id', 'time_idx']:
                tft_data[col] = tft_data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining missing values with median/mode
        for col in tft_data.columns:
            if col not in ['target', 'entity_id', 'time_idx'] and tft_data[col].isna().any():
                if tft_data[col].dtype in ['float64', 'int64']:
                    tft_data[col] = tft_data[col].fillna(tft_data[col].median())
                else:
                    mode_val = tft_data[col].mode()
                    if len(mode_val) > 0:
                        tft_data[col] = tft_data[col].fillna(mode_val.iloc[0])
        
        # Final check - remove any rows where target is still NaN
        tft_data = tft_data[tft_data['target'].notna()]
        
        print(f"Final TFT dataset shape: {tft_data.shape}")
        print(f"Columns: {len(tft_data.columns)}")
        
        return tft_data
    
    def get_feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance scores."""
        if self.target_column not in data.columns:
            return pd.DataFrame()
        
        importance_results = []
        target = data[self.target_column].dropna()
        
        for feature_type, features in self.selected_features.items():
            available_features = [col for col in features if col in data.columns]
            
            if not available_features:
                continue
                
            feature_data = data[available_features].fillna(0)
            common_idx = feature_data.index.intersection(target.index)
            
            if len(common_idx) == 0:
                continue
                
            X = feature_data.loc[common_idx]
            y = target.loc[common_idx]
            
            try:
                # Calculate mutual information scores
                mi_scores = mutual_info_regression(X, y)
                
                for i, feature in enumerate(available_features):
                    importance_results.append({
                        'feature': feature,
                        'feature_type': feature_type,
                        'importance_score': mi_scores[i]
                    })
            except Exception as e:
                print(f"Could not calculate importance for {feature_type}: {e}")
        
        if importance_results:
            importance_df = pd.DataFrame(importance_results)
            importance_df = importance_df.sort_values('importance_score', ascending=False)
            return importance_df
        
        return pd.DataFrame()


def prepare_data_for_tft(data_path: str, output_path: str = None,
                        target_column: str = 'close',
                        prediction_horizon: int = 12) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to prepare data for TFT training.
    
    Args:
        data_path: Path to input data
        output_path: Path to save prepared data
        target_column: Target variable name
        prediction_horizon: Prediction horizon in time steps
        
    Returns:
        Tuple of (prepared_data, metadata)
    """
    print("Starting TFT data preparation...")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Initialize preprocessor
    preprocessor = TFTDataPreprocessor(
        target_column=target_column,
        prediction_horizon=prediction_horizon
    )
    
    # Add static covariates
    df = preprocessor.add_static_covariates(df, symbol="AAPL")
    
    # Categorize features
    feature_categories = preprocessor.categorize_features(df)
    
    # Create lagged features
    df = preprocessor.create_lagged_features(df)
    
    # Create interaction features
    df = preprocessor.create_interaction_features(df)
    
    # Encode categorical features
    df = preprocessor.encode_categorical_features(df, fit=True)
    
    # Select important features
    df = preprocessor.select_features(df)
    
    # Scale features
    df = preprocessor.scale_features(df, fit=True)
    
    # Prepare final TFT dataset
    tft_data = preprocessor.prepare_tft_dataset(df)
    
    # Calculate feature importance
    importance_df = preprocessor.get_feature_importance(df)
    
    # Create metadata
    metadata = {
        'feature_categories': feature_categories,
        'selected_features': preprocessor.selected_features,
        'scalers': preprocessor.scalers,
        'label_encoders': preprocessor.label_encoders,
        'target_column': target_column,
        'prediction_horizon': prediction_horizon,
        'data_shape': tft_data.shape,
        'feature_importance': importance_df.to_dict('records') if not importance_df.empty else []
    }
    
    if output_path:
        tft_data.to_csv(output_path)
        
        # Save metadata
        import json
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert non-serializable objects to strings for JSON
            metadata_for_json = metadata.copy()
            metadata_for_json['scalers'] = list(metadata['scalers'].keys())
            metadata_for_json['label_encoders'] = list(metadata['label_encoders'].keys())
            json.dump(metadata_for_json, f, indent=2)
        
        print(f"TFT data saved to {output_path}")
        print(f"Metadata saved to {metadata_path}")
    
    return tft_data, metadata


if __name__ == "__main__":
    # Example usage
    input_path = "../../dataset/raw/merged/AAPL_WITH_EVENT_FEATURES.csv"
    output_path = "../../dataset/processed/AAPL_TFT_READY.csv"
    
    # Create output directory
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tft_data, metadata = prepare_data_for_tft(
        input_path, 
        output_path,
        target_column='close',
        prediction_horizon=12
    )
    
    print(f"TFT preparation complete!")
    print(f"Dataset shape: {tft_data.shape}")
    print(f"Features by type:")
    for feature_type, features in metadata['selected_features'].items():
        print(f"  {feature_type}: {len(features)} features")
