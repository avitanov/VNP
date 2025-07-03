"""
Production-ready TFT Stock Direction Prediction Script.

This script generates both price predictions and up/down direction classifications,
saving results to files for production use.

Usage:
    python production_predictions.py

Output Files:
    - predictions/tft_predictions_YYYYMMDD_HHMMSS.csv - Full predictions
    - predictions/tft_predictions_summary_YYYYMMDD_HHMMSS.csv - Summary statistics
    - predictions/latest_directions.csv - Latest direction predictions only
"""

import pandas as pd
import numpy as np
import os
import json
import datetime
from models.tft_trainer import TFTStockPredictor

def run_production_predictions(
    steps: int = 12,
    save_files: bool = True,
    output_dir: str = "predictions/"
):
    """
    Run production predictions and save to files.
    
    Args:
        steps: Number of prediction steps
        save_files: Whether to save results to files
        output_dir: Output directory for files
        
    Returns:
        Dictionary with predictions and file paths
    """
    print(" Starting Production TFT Predictions...")
    print("=" * 50)
    
    # Paths
    data_path = 'dataset/processed/AAPL_TFT_READY.csv'
    metadata_path = 'dataset/processed/AAPL_TFT_READY_metadata.json'
    model_dir = 'models/trained/'
    
    # Load configuration
    try:
        with open(os.path.join(model_dir, 'training_results.json'), 'r') as f:
            training_results = json.load(f)
        config = training_results.get('config', {})
        print(f" Loaded model configuration")
    except Exception as e:
        print(f"  Using default configuration: {e}")
        config = {
            'max_prediction_length': 12,
            'max_encoder_length': 50,
            'batch_size': 32,
            'learning_rate': 0.005,
            'hidden_size': 64,
            'attention_head_size': 4,
            'dropout': 0.2,
            'hidden_continuous_size': 32,
            'max_epochs': 20,
            'gradient_clip_val': 0.1,
            'accelerator': 'cpu',
            'devices': 1
        }
    
    # Initialize predictor
    print("🔄 Initializing TFT predictor...")
    predictor = TFTStockPredictor(config)
    
    # Load data and create datasets
    print(" Loading data...")
    try:
        data, metadata = predictor.load_data(data_path, metadata_path)
        training_dataset, validation_dataset = predictor.create_datasets(data, metadata)
        predictor.training_data = training_dataset
        predictor.validation_data = validation_dataset
        print(f" Data loaded: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f" Error loading data: {e}")
        return None
    
    # Load or create model
    print(" Loading model...")
    try:
        model_path = os.path.join(model_dir, 'tft_stock_model')
        predictor.load_model(model_path)
        print(" Model loaded")
    except Exception as e:
        print(f"⚠️  Creating new model: {e}")
        try:
            model = predictor.create_model(predictor.training_data)
            predictor.model = model
            print(" Model created")
        except Exception as e2:
            print(f" Could not create model: {e2}")
            return None
    
    # Generate predictions
    print(f"\n Generating {steps}-step predictions...")
    
    # 1. Price predictions
    print(" Price predictions...")
    price_predictions = predictor.predict_future(steps=steps, return_direction=False)
    
    # 2. Direction predictions
    print("2️ Direction classifications...")
    direction_predictions = predictor.predict_future(steps=steps, return_direction=True)
    
    # 3. Direction-only predictions
    print("3️ Direction-only predictions...")
    direction_only = predictor.predict_direction_only(steps=steps, method='price_based')
    
    # Prepare results
    results = {
        'price_predictions': price_predictions,
        'direction_predictions': direction_predictions,
        'direction_only': direction_only,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config
    }
    
    # Save to files if requested
    file_paths = {}
    if save_files:
        print(f"\n Saving results to {output_dir}...")
        
        # Save full predictions
        full_file = predictor.save_predictions_to_file(
            direction_predictions, 
            output_dir=output_dir,
            filename_prefix="tft_full_predictions"
        )
        file_paths['full_predictions'] = full_file
        
        # Save direction-only predictions
        direction_file = predictor.save_predictions_to_file(
            direction_only,
            output_dir=output_dir, 
            filename_prefix="tft_directions_only"
        )
        file_paths['directions_only'] = direction_file
        
        # Save latest directions (simplified format)
        latest_directions = direction_predictions[['step', 'predicted_price', 'direction', 
                                                 'direction_confident', 'price_change_pct', 
                                                 'up_probability', 'classification']].copy()
        
        latest_file = os.path.join(output_dir, 'latest_directions.csv')
        latest_directions.to_csv(latest_file, index=False)
        print(f" Latest directions saved to: {latest_file}")
        file_paths['latest_directions'] = latest_file
        
        # Save trading signals (UP/DOWN only)
        trading_signals = pd.DataFrame({
            'step': direction_predictions['step'],
            'signal': direction_predictions['direction'],
            'confidence': direction_predictions['up_probability'].apply(lambda x: max(x, 1-x)),
            'strength': direction_predictions['magnitude'],
            'price_target': direction_predictions['predicted_price'],
            'change_percent': direction_predictions['price_change_pct']
        })
        
        signals_file = os.path.join(output_dir, 'trading_signals.csv')
        trading_signals.to_csv(signals_file, index=False)
        print(f" Trading signals saved to: {signals_file}")
        file_paths['trading_signals'] = signals_file
    
    # Print summary
    print(f"\n PREDICTION SUMMARY:")
    print(f"   Steps predicted: {steps}")
    print(f"   Current price: ${direction_predictions['current_price'].iloc[0]:.2f}")
    print(f"   Avg predicted price: ${direction_predictions['predicted_price'].mean():.2f}")
    print(f"   Price range: ${direction_predictions['predicted_price'].max() - direction_predictions['predicted_price'].min():.2f}")
    
    direction_counts = direction_predictions['direction'].value_counts()
    print(f"   Direction distribution: {dict(direction_counts)}")
    
    avg_confidence = direction_predictions['up_probability'].apply(lambda x: max(x, 1-x)).mean()
    print(f"   Average confidence: {avg_confidence:.1%}")
    
    if file_paths:
        print(f"\n FILES CREATED:")
        for name, path in file_paths.items():
            print(f"   {name}: {path}")
    
    results['file_paths'] = file_paths
    return results

def main():
    """Main function to run production predictions."""
    
    # Create output directory
    output_dir = "predictions/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run predictions
    results = run_production_predictions(
        steps=12,  # Predict 12 steps ahead
        save_files=True,
        output_dir=output_dir
    )
    
    if results:
        print("\n Production predictions completed successfully!")
        print("\n KEY OUTPUT FILES:")
        print(f"    Trading signals: {results['file_paths']['trading_signals']}")
        print(f"   Latest directions: {results['file_paths']['latest_directions']}")
        print(f"    Full predictions: {results['file_paths']['full_predictions']}")
        
        print("\n💡 USAGE:")
        print("   - Check 'trading_signals.csv' for UP/DOWN signals")
        print("   - Check 'latest_directions.csv' for detailed predictions")
        print("   - Use confidence scores to filter high-quality signals")
        
    else:
        print(" Prediction failed. Check the error messages above.")

if __name__ == "__main__":
    main()
