"""
View latest trading signals and up/down predictions.

This script reads the latest prediction files and displays
the trading signals in a clean format.
"""

import pandas as pd
import os
from datetime import datetime

def view_latest_signals(predictions_dir: str = "predictions/"):
    """
    View the latest trading signals.
    
    Args:
        predictions_dir: Directory containing prediction files
    """
    print(" TFT TRADING SIGNALS VIEWER")
    print("=" * 50)
    
    # Check if files exist
    signals_file = os.path.join(predictions_dir, 'trading_signals.csv')
    directions_file = os.path.join(predictions_dir, 'latest_directions.csv')
    
    if not os.path.exists(signals_file):
        print(f" No signals file found at: {signals_file}")
        print("   Run 'python production_predictions.py' first")
        return
    
    # Load signals
    try:
        signals = pd.read_csv(signals_file)
        print(f"✅ Loaded {len(signals)} trading signals")
        
        # Display trading signals
        print(f"\n TRADING SIGNALS:")
        print("-" * 30)
        
        for _, row in signals.iterrows():
            step = int(row['step'])
            signal = row['signal']
            confidence = row['confidence']
            strength = row['strength']
            price = row['price_target']
            change = row['change_percent']
            
            # Format signal with emoji
            signal_emoji = "📈" if signal == "UP" else "📉"
            confidence_stars = "⭐" * int(confidence * 5)  # 1-5 stars based on confidence
            
            print(f"Step {step:2d}: {signal_emoji} {signal:4s} | "
                  f"${price:7.2f} | {change:+6.2f}% | "
                  f"{strength:8s} | {confidence_stars} ({confidence:.1%})")
        
        # Summary statistics
        up_count = (signals['signal'] == 'UP').sum()
        down_count = (signals['signal'] == 'DOWN').sum()
        avg_confidence = signals['confidence'].mean()
        high_confidence = (signals['confidence'] > 0.8).sum()
        
        print(f"\n📈 SUMMARY:")
        print(f"   UP signals: {up_count}")
        print(f"   DOWN signals: {down_count}")
        print(f"   Average confidence: {avg_confidence:.1%}")
        print(f"   High confidence (>80%): {high_confidence}")
        
        # Show high confidence signals
        high_conf_signals = signals[signals['confidence'] > 0.8]
        if len(high_conf_signals) > 0:
            print(f"\n⭐ HIGH CONFIDENCE SIGNALS (>80%):")
            for _, row in high_conf_signals.iterrows():
                print(f"   Step {int(row['step'])}: {row['signal']} "
                      f"${row['price_target']:.2f} ({row['confidence']:.1%})")
        
    except Exception as e:
        print(f" Error reading signals: {e}")
    
    # Also show detailed directions if available
    if os.path.exists(directions_file):
        try:
            directions = pd.read_csv(directions_file)
            print(f"\n📋 DETAILED DIRECTIONS (first 5 steps):")
            print("-" * 40)
            
            display_cols = ['step', 'predicted_price', 'direction', 
                          'direction_confident', 'price_change_pct']
            print(directions[display_cols].head().to_string(index=False))
            
        except Exception as e:
            print(f"⚠️  Could not load detailed directions: {e}")

def watch_signals(predictions_dir: str = "predictions/", refresh_seconds: int = 60):
    """
    Watch for updated signals and display them.
    
    Args:
        predictions_dir: Directory to watch
        refresh_seconds: How often to check for updates
    """
    import time
    
    print(f"👀 Watching for signal updates every {refresh_seconds} seconds...")
    print("Press Ctrl+C to stop")
    
    last_modified = 0
    
    try:
        while True:
            signals_file = os.path.join(predictions_dir, 'trading_signals.csv')
            
            if os.path.exists(signals_file):
                current_modified = os.path.getmtime(signals_file)
                
                if current_modified > last_modified:
                    print(f"\n🔄 Updated signals detected at {datetime.now().strftime('%H:%M:%S')}")
                    view_latest_signals(predictions_dir)
                    last_modified = current_modified
                    print(f"\n⏰ Next check in {refresh_seconds} seconds...")
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n👋 Stopped watching signals")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # Watch mode
        watch_signals()
    else:
        # Single view mode
        view_latest_signals()

if __name__ == "__main__":
    main()
