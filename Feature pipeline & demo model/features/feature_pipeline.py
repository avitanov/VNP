"""
Main Feature Engineering Pipeline for TFT Stock Prediction
=========================================================

- Ги зима цените на акциите,
- им додава технички индикатори,
- зима и други податоци – како VIX, долар индекс, обврзници, сектори,
- плус додава и настани кои можат да влијаат на ценатите,
- ги средува сите features, додава лагови, нормализира, филтрира најважни,
- на крајот прави CSV фајл за тренирање на TFT модел.

- E povikan od strana na run_feature_pipeline.py.
- go loadnuva merge datasetot so tehnickite indikatori i site podatoci od apito (prices + news)
- mu dodava  coustom technical indicators koi gi nema vo apito a mu se potrebni na TFT za podobar rezultat.
- dodava cross-asset features (VIX, dollar index)
- dodava event features (Fed meetings, earnings, holidays)- E povikan od strana na run_feature_pipeline.py.
- go loadnuva merge datasetot so tehnickite indikatori i site podatoci od apito (prices + news)
- mu dodava  coustom technical indicators koi gi nema vo apito a mu se potrebni na TFT za podobar rezultat.
- dodava cross-asset features (VIX, dollar index)
- dodava event features (Fed meetings, earnings, holidays)
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.technical.technical_indicators import calculate_custom_features as calculate_technical_features
from features.technical.cross_asset_features import add_cross_asset_features
from features.events.event_features import create_event_features
from features.tft_prep.tft_preprocessor import prepare_data_for_tft


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for TFT stock prediction.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.pipeline_results = {}
        
    def _get_default_config(self) -> dict:
        """Get default configuration for the pipeline."""
        return {
            'symbol': 'AAPL',
            'prediction_horizon': 12,  # 12 periods ahead (1 hour for 5-min data)
            'target_column': 'close',
            'include_technical': True,
            'include_cross_asset': True,
            'include_events': True,
            'max_features': {
                'unknown_future': 50,
                'known_future': 20,
                'static': 10
            },
            'lag_periods': [1, 5, 10, 20],
            'paths': {
                'input_data': '../dataset/raw/merged/AAPL_PRICE_WITH_NEWS_AND_INDICATORS.csv',
                'output_dir': '../dataset/processed/',
                'intermediate_dir': '../dataset/intermediate/'
            }
        }
    
    def run_complete_pipeline(self) -> dict:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            Dictionary with pipeline results and metadata
        """
        print("="*60)
        print("Starting Complete Feature Engineering Pipeline")
        print("="*60)
        
        start_time = datetime.now()
        
        # Create output directories
        self._create_directories()
        
        # Step 1: Load and validate input data
        print("\nStep 1: Loading and validating input data...")
        input_data = self._load_and_validate_data()
        
        # Step 2: Calculate technical indicators
        if self.config['include_technical']:
            print("\nStep 2: Calculating technical indicators...")
            technical_data = self._calculate_technical_features(input_data)
        else:
            technical_data = input_data
        
        # Step 3: Add cross-asset features
        if self.config['include_cross_asset']:
            print("\nStep 3: Adding cross-asset features...")
            cross_asset_data = self._add_cross_asset_features(technical_data)
        else:
            cross_asset_data = technical_data
        
        # Step 4: Create event features
        if self.config['include_events']:
            print("\nStep 4: Creating event features...")
            event_data = self._create_event_features(cross_asset_data)
        else:
            event_data = cross_asset_data
        
        # Step 5: Prepare data for TFT
        print("\nStep 5: Preparing data for TFT...")
        tft_data, metadata = self._prepare_tft_data(event_data)
        
        # Step 6: Generate pipeline report
        print("\nStep 6: Generating pipeline report...")
        report = self._generate_pipeline_report(tft_data, metadata, start_time)
        
        # Store results
        self.pipeline_results = {
            'final_data': tft_data,
            'metadata': metadata,
            'report': report,
            'config': self.config
        }
        
        print("="*60)
        print("Feature Engineering Pipeline Complete!")
        print("="*60)
        
        return self.pipeline_results
    
    def _create_directories(self):
        """Create necessary directories for output files."""
        for dir_key in ['output_dir', 'intermediate_dir']:
            if dir_key in self.config['paths']:
                os.makedirs(self.config['paths'][dir_key], exist_ok=True)
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the input data."""
        input_path = self.config['paths']['input_data']
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input data file not found: {input_path}")
        
        # Load data
        df = pd.read_csv(input_path, parse_dates=['datetime'], index_col='datetime')
        
        # Basic validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for sufficient data
        if len(df) < 100:
            raise ValueError(f"Insufficient data: only {len(df)} rows")
        
        print(f"Loaded {len(df)} rows of data from {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def _calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Save intermediate data
        intermediate_path = os.path.join(
            self.config['paths']['intermediate_dir'], 
            f"{self.config['symbol']}_technical_features.csv"
        )
        
        # Import and use technical indicators module
        try:
            from features.technical.technical_indicators import CustomTechnicalIndicators
            
            tech_calc = CustomTechnicalIndicators(data)
            tech_features = tech_calc.calculate_all()
            
            # Combine with original data
            result = pd.concat([data, tech_features], axis=1)
            result.to_csv(intermediate_path)
            
            print(f"Generated {len(tech_features.columns)} technical features")
            print(f"Saved to: {intermediate_path}")
            
            return result
            
        except Exception as e:
            print(f"Error calculating technical features: {e}")
            print("Continuing with original data...")
            return data
    
    def _add_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset features."""
        # Save intermediate data
        intermediate_path = os.path.join(
            self.config['paths']['intermediate_dir'], 
            f"{self.config['symbol']}_with_cross_asset.csv"
        )
        
        try:
            from features.technical.cross_asset_features import CrossAssetFeatures
            
            # Check for API key
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                print("No Alpha Vantage API key found. Skipping cross-asset features.")
                return data
            
            cross_asset_calc = CrossAssetFeatures(api_key)
            cross_asset_features = cross_asset_calc.create_cross_asset_features(data.index)
            
            # Combine with existing data
            result = pd.concat([data, cross_asset_features], axis=1)
            result.to_csv(intermediate_path)
            
            print(f"Added {len(cross_asset_features.columns)} cross-asset features")
            print(f"Saved to: {intermediate_path}")
            
            return result
            
        except Exception as e:
            print(f"Error adding cross-asset features: {e}")
            print("Continuing without cross-asset features...")
            return data
    
    def _create_event_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create event-based features."""
        # Save intermediate data
        intermediate_path = os.path.join(
            self.config['paths']['intermediate_dir'], 
            f"{self.config['symbol']}_with_events.csv"
        )
        
        try:
            from features.events.event_features import EventFeatureEngine
            
            event_engine = EventFeatureEngine(data)
            
            # Calculate sentiment features
            sentiment_features = event_engine.calculate_sentiment_features()
            
            # Create economic calendar
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            calendar = event_engine.create_economic_calendar(start_date, end_date)
            
            # Resample calendar to match data frequency
            if len(data) > len(calendar) * 2:  # Likely intraday data
                calendar_resampled = calendar.reindex(data.index, method='ffill')
            else:
                calendar_resampled = calendar.reindex(data.index, method='nearest')
            
            # Combine all features
            result = pd.concat([data, sentiment_features, calendar_resampled], axis=1)
            result.to_csv(intermediate_path)
            
            total_event_features = len(sentiment_features.columns) + len(calendar.columns)
            print(f"Generated {total_event_features} event features")
            print(f"Saved to: {intermediate_path}")
            
            return result
            
        except Exception as e:
            print(f"Error creating event features: {e}")
            print("Continuing without event features...")
            return data
    
    def _prepare_tft_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for TFT training."""
        output_path = os.path.join(
            self.config['paths']['output_dir'], 
            f"{self.config['symbol']}_TFT_READY.csv"
        )
        
        try:
            from features.tft_prep.tft_preprocessor import TFTDataPreprocessor
            
            # Initialize preprocessor
            preprocessor = TFTDataPreprocessor(
                target_column=self.config['target_column'],
                prediction_horizon=self.config['prediction_horizon']
            )
            
            # Add static covariates
            data = preprocessor.add_static_covariates(data, symbol=self.config['symbol'])
            
            # Categorize features
            feature_categories = preprocessor.categorize_features(data)
            
            # Create lagged features
            data = preprocessor.create_lagged_features(data, self.config['lag_periods'])
            
            # Create interaction features
            data = preprocessor.create_interaction_features(data)
            
            # Encode categorical features
            data = preprocessor.encode_categorical_features(data, fit=True)
            
            # Select important features
            data = preprocessor.select_features(data, self.config['max_features'])
            
            # Scale features
            data = preprocessor.scale_features(data, fit=True)
            
            # Prepare final TFT dataset
            tft_data = preprocessor.prepare_tft_dataset(data)
            
            # Calculate feature importance
            importance_df = preprocessor.get_feature_importance(data)
            
            # Create metadata
            metadata = {
                'feature_categories': feature_categories,
                'selected_features': preprocessor.selected_features,
                'target_column': 'target',  # TFT preprocessor creates 'target' column
                'original_target_column': self.config['target_column'],  # Keep reference to original
                'prediction_horizon': self.config['prediction_horizon'],
                'data_shape': tft_data.shape,
                'feature_importance': importance_df.to_dict('records') if not importance_df.empty else []
            }
            
            # Save final data
            tft_data.to_csv(output_path)
            
            # Save metadata
            import json
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"TFT-ready data saved to: {output_path}")
            print(f"Metadata saved to: {metadata_path}")
            
            return tft_data, metadata
            
        except Exception as e:
            print(f"Error preparing TFT data: {e}")
            # Return basic processed data
            return data, {'error': str(e)}
    
    def _generate_pipeline_report(self, tft_data: pd.DataFrame, 
                                 metadata: dict, start_time: datetime) -> dict:
        """Generate a comprehensive pipeline report."""
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        report = {
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time.total_seconds(),
                'status': 'success' if 'error' not in metadata else 'partial_success'
            },
            'data_summary': {
                'final_shape': tft_data.shape,
                'date_range': {
                    'start': str(tft_data.index.min()) if hasattr(tft_data, 'index') else 'unknown',
                    'end': str(tft_data.index.max()) if hasattr(tft_data, 'index') else 'unknown'
                },
                'missing_data_percentage': (tft_data.isnull().sum().sum() / tft_data.size * 100) if hasattr(tft_data, 'isnull') else 0
            },
            'feature_summary': {
                'total_features': len(tft_data.columns) if hasattr(tft_data, 'columns') else 0,
                'features_by_type': {
                    feature_type: len(features) 
                    for feature_type, features in metadata.get('selected_features', {}).items()
                }
            },
            'config_used': self.config
        }
        
        # Add feature importance summary
        if 'feature_importance' in metadata and metadata['feature_importance']:
            top_features = sorted(metadata['feature_importance'], 
                                key=lambda x: x['importance_score'], reverse=True)[:10]
            report['top_features'] = top_features
        
        return report
    
    def save_pipeline_results(self, output_path: str = None):
        """Save pipeline results to files."""
        if not self.pipeline_results:
            print("No pipeline results to save. Run pipeline first.")
            return
        
        if output_path is None:
            output_path = self.config['paths']['output_dir']
        
        # Save report
        report_path = os.path.join(output_path, f"{self.config['symbol']}_pipeline_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.pipeline_results['report'], f, indent=2, default=str)
        
        print(f"Pipeline report saved to: {report_path}")
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline results."""
        if not self.pipeline_results:
            print("No pipeline results available. Run pipeline first.")
            return
        
        report = self.pipeline_results['report']
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Symbol: {self.config['symbol']}")
        print(f"Execution Time: {report['execution_summary']['execution_time_seconds']:.2f} seconds")
        print(f"Status: {report['execution_summary']['status'].upper()}")
        
        print(f"\nData Summary:")
        print(f"  Final Shape: {report['data_summary']['final_shape']}")
        print(f"  Date Range: {report['data_summary']['date_range']['start']} to {report['data_summary']['date_range']['end']}")
        print(f"  Missing Data: {report['data_summary']['missing_data_percentage']:.2f}%")
        
        print(f"\nFeature Summary:")
        print(f"  Total Features: {report['feature_summary']['total_features']}")
        for feature_type, count in report['feature_summary']['features_by_type'].items():
            print(f"  {feature_type.replace('_', ' ').title()}: {count}")
        
        if 'top_features' in report:
            print(f"\nTop 10 Most Important Features:")
            for i, feature in enumerate(report['top_features'], 1):
                print(f"  {i:2d}. {feature['feature']} ({feature['feature_type']}) - {feature['importance_score']:.4f}")


def run_feature_engineering_pipeline(config: dict = None) -> dict:
    """
    Main function to run the complete feature engineering pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Pipeline results dictionary
    """
    pipeline = FeatureEngineeringPipeline(config)
    results = pipeline.run_complete_pipeline()
    pipeline.save_pipeline_results()
    pipeline.print_pipeline_summary()
    
    return results


if __name__ == "__main__":
    # Example configuration
    config = {
        'symbol': 'AAPL',
        'prediction_horizon': 12,  # 1 hour ahead for 5-min data
        'target_column': 'close',
        'include_technical': True,
        'include_cross_asset': True,  # Set to False if no API key
        'include_events': True,
        'max_features': {
            'unknown_future': 40,
            'known_future': 15,
            'static': 8
        },
        'lag_periods': [1, 3, 5, 10, 20],
        'paths': {
            'input_data': '../dataset/raw/merged/AAPL_PRICE_WITH_NEWS_AND_INDICATORS.csv',
            'output_dir': '../dataset/processed/',
            'intermediate_dir': '../dataset/intermediate/'
        }
    }
    
    # Run the pipeline
    results = run_feature_engineering_pipeline(config)
