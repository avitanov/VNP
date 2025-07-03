"""
Run TFT training with the demo pipeline output.

- Go zima gotoviot dataset od pipelinot so momentalno 63 features.
- Gi deli podatocite na 80 / 20
- parametrite na modelot (momentalno e staven na 20 epohi za brzo da se trenira, pak mu treba 30+ min, ama idealno
  bi bilo da se proba so okolu 70 i ke trae sigurno nad 2h.
 - modelot treba da se prilagoduva, momentalniot dropout e 0.2, mozno e da e visok, batch sizeot moze da se proba so 16.
 - Parametrite na modelot se pisuvat tuka i istite se ovveridnuvat vo tft_trainer.py
"""

from models.tft_trainer import train_tft_model
import os

def main():
    # Define paths to our advanced feature pipeline output
    data_path = 'dataset/processed/AAPL_TFT_READY.csv'
    metadata_path = 'dataset/processed/AAPL_TFT_READY_metadata.json'
    output_dir = 'models/trained/'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configuration optimized for advanced feature pipeline (63 features)
    config = {
        'max_prediction_length': 12,   # Match the feature pipeline prediction horizon
        'max_encoder_length': 50,      # More history for better patterns
        'batch_size': 32,              # Larger batch for stable training
        'learning_rate': 0.005,        # Lower learning rate for stability
        'hidden_size': 64,             # Larger model for more features
        'attention_head_size': 4,      # More attention heads
        'dropout': 0.2,                # More dropout to prevent overfitting
        'hidden_continuous_size': 32,  # Larger continuous hidden size
        'max_epochs': 20,              # More epochs for convergence
        'gradient_clip_val': 0.1,
        'accelerator': 'cpu',          # Use CPU for training
        'devices': 1                   # Number of devices to use
    }

    print('Starting TFT model training...')
    print(f'Data path: {data_path}')
    print(f'Metadata path: {metadata_path}')
    print(f'Output directory: {output_dir}')
    print(f'Configuration: {config}')
    print('-' * 50)
    
    try:
        results = train_tft_model(data_path, metadata_path, output_dir, config)
        
        print('\n' + '='*50)
        print('Training completed successfully!')
        print(f'Model saved to: {results["model_path"]}')
        print('\nFinal evaluation metrics:')
        for metric, value in results['metrics'].items():
            print(f'  {metric}: {value:.4f}')
        
        if results['feature_importance']:
            print('\nTop 5 most important features:')
            for i, feat in enumerate(results['feature_importance'][:5]):
                print(f'  {i+1}. {feat["feature"]}: {feat["importance"]:.4f}')
        
    except Exception as e:
        print(f'\nError during training: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
