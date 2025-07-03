"""
Main Feature Pipeline
============================

- go povikuva feature_pipeline.py
- na kraj vrakja dataset so podatocite od apito 
+ uste tehnicki indikatore koi se presmetuvaat lokalno
ne od api i se potrebni za TFT da dade podobar rezultat 
+ cross assets features + event features. (nad 100 features)

"""

import os
import sys
from features.feature_pipeline import run_feature_engineering_pipeline

def main():
    """Run the complete feature pipeline with TFT preprocessing."""
    print("=" * 70)
    print("COMPLETE FEATURE ENGINEERING PIPELINE WITH TFT PREPROCESSING")
    print("=" * 70)
    
    # Configuration for the feature pipeline
    config = {
        'symbol': 'AAPL',
        'prediction_horizon': 12,  # Predict 12 periods ahead
        'target_column': 'close',
        'include_technical': True,
        'include_cross_asset': False,  # Set to True if you have API keys
        'include_events': True,
        'max_features': {
            'unknown_future': 40,
            'known_future': 15,
            'static': 8
        },
        'lag_periods': [1, 3, 5, 10, 20],
        'paths': {
            'input_data': 'dataset/raw/merged/AAPL_PRICE_WITH_NEWS_AND_INDICATORS.csv',
            'output_dir': 'dataset/processed/',
            'intermediate_dir': 'dataset/intermediate/'
        }
    }
    
    print("Configuration:")
    for key, value in config.items():
        if key != 'paths':
            print(f"  {key}: {value}")
    
    print(f"\nInput data: {config['paths']['input_data']}")
    print(f"Output directory: {config['paths']['output_dir']}")
    
    try:
        # Run the complete feature engineering pipeline
        print("\nStarting feature engineering pipeline...")
        results = run_feature_engineering_pipeline(config)
        
        if results and 'final_data' in results:
            print("\n" + "=" * 70)
            print("✅ FEATURE PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            tft_data = results['final_data']
            metadata = results.get('metadata', {})
            
            print(f"Final dataset shape: {tft_data.shape}")
            print(f"Features: {len(tft_data.columns)} columns")
            
            # Show feature categories
            if 'feature_categories' in metadata:
                print("\nFeature categories:")
                for category, features in metadata['feature_categories'].items():
                    if isinstance(features, list):
                        print(f"  {category}: {len(features)} features")
            
            # Output files
            output_file = f"dataset/processed/{config['symbol']}_TFT_READY.csv"
            metadata_file = f"dataset/processed/{config['symbol']}_TFT_READY_metadata.json"
            
            print(f"\n📁 Generated files:")
            print(f"   📊 TFT Data: {output_file}")
            print(f"   📋 Metadata: {metadata_file}")
            
            print(f"\n🚀 Next steps:")
            print(f"1. Train TFT model:")
            print(f"   python run_tft_training_advanced.py")
            print(f"2. Create predictions:")
            print(f"   python create_predictions_advanced.py")
            
            return True
        else:
            print("❌ Feature pipeline failed to generate results")
            return False
            
    except Exception as e:
        print(f"\n❌ Error in feature pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'=' * 70}")
    print(f"Pipeline {'completed successfully' if success else 'failed'}!")
    print(f"{'=' * 70}")
