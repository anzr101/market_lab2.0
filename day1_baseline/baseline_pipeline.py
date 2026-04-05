"""
Baseline Pipeline - Complete Execution System
Runs 30 models with 300+ features on assigned stock batch
With checkpointing and optimization support
"""

import sys
from pathlib import Path

# Add shared to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'shared'))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from features.feature_master import FeatureMaster
from models.model_zoo import ModelZoo
from models.ensemble import SimpleEnsemble
from utils.checkpoint_manager import CheckpointManager
from utils.hyperparameter_optimizer import HyperparameterOptimizer
from stock_batches import STOCK_BATCHES
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import yfinance as yf


class BaselinePipeline:
    """
    Complete baseline pipeline with:
    - 300+ features
    - 30 models
    - Checkpointing
    - Optional hyperparameter optimization
    """
    
    def __init__(self, pc_number: int, random_state: int = 42):
        self.pc_number = pc_number
        self.random_state = random_state
        self.checkpoint_mgr = CheckpointManager(f'checkpoint_pc{pc_number}.json')
        self.feature_master = FeatureMaster()
        self.model_zoo = ModelZoo(random_state=random_state)
        self.ensemble = SimpleEnsemble()
        self.optimizer = HyperparameterOptimizer(n_trials=30, random_state=random_state)
        
    def run_batch(self, 
                  start_date: str = "2004-01-01", 
                  end_date: str = "2024-12-31",
                  optimize_top_models: bool = False,
                  use_production_models: bool = True):
        """
        Run baseline on assigned batch
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            optimize_top_models: Whether to optimize top 5 models (slow!)
            use_production_models: Use fast models only (recommended)
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 BASELINE PIPELINE - PC {self.pc_number}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🔧 Optimization: {'YES' if optimize_top_models else 'NO'}")
        print(f"⚡ Production mode: {'YES' if use_production_models else 'NO (all models)'}")
        print(f"{'='*80}")
        
        # Get stock batch
        stocks = STOCK_BATCHES[self.pc_number]
        
        # Check for checkpoint
        checkpoint = self.checkpoint_mgr.load_checkpoint()
        
        if checkpoint:
            completed = checkpoint['completed_stocks']
            remaining = checkpoint['remaining_stocks']
        else:
            completed = []
            remaining = stocks
        
        print(f"\n📊 Batch info:")
        print(f"   Total stocks: {len(stocks)}")
        print(f"   Completed: {len(completed)}")
        print(f"   Remaining: {len(remaining)}")
        
        # Create results directory
        results_dir = Path(project_root / 'final_results')
        results_dir.mkdir(exist_ok=True)
        (results_dir / 'models').mkdir(exist_ok=True)
        
        # Process remaining stocks
        all_results = []
        
        for i, ticker in enumerate(remaining, 1):
            print(f"\n{'='*80}")
            print(f"📊 Processing {ticker} ({i}/{len(remaining)})")
            print(f"   Overall progress: {len(completed) + i}/{len(stocks)}")
            print(f"{'='*80}")
            
            try:
                # Process single stock
                result = self.process_stock(
                    ticker, 
                    start_date, 
                    end_date, 
                    results_dir,
                    use_production_models
                )
                
                all_results.append(result)
                
                # Update checkpoint
                completed.append(ticker)
                remaining_updated = [s for s in stocks if s not in completed]
                
                metadata = {
                    'last_ticker': ticker,
                    'total_models_trained': len(result.get('models_trained', [])),
                    'best_r2': result.get('best_r2', 0)
                }
                
                self.checkpoint_mgr.save_progress(completed, remaining_updated, metadata)
                
                print(f"\n✅ {ticker} COMPLETE!")
                
            except Exception as e:
                print(f"\n❌ ERROR processing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Save error but continue
                completed.append(ticker)
                remaining_updated = [s for s in stocks if s not in completed]
                self.checkpoint_mgr.save_progress(completed, remaining_updated, 
                                                 {'error': str(e)})
                continue
        
        # All done!
        print(f"\n{'='*80}")
        print(f"🎉 PC {self.pc_number} COMPLETE!")
        print(f"✅ Successfully processed: {len(all_results)}/{len(stocks)} stocks")
        print(f"{'='*80}")
        
        self.checkpoint_mgr.clear_checkpoint()
        
        return all_results
    
    def process_stock(self, ticker: str, start_date: str, end_date: str, 
                     results_dir: Path, use_production: bool = True):
        """Process single stock through complete pipeline"""
        
        # 1. Download data
        print(f"\n   📥 Downloading data...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(df) == 0:
            raise ValueError(f"No data downloaded for {ticker}")
        
        # FIX: Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # FIX: Ensure columns are strings, not tuples
        df.columns = [str(col).strip() for col in df.columns]
        
        # FIX: Remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # FIX: Standardize column names (remove extra spaces)
        column_mapping = {col: col.strip() for col in df.columns}
        df = df.rename(columns=column_mapping)
        
        print(f"      ✅ Loaded {len(df)} rows")
        
        # 2. Generate features
        print(f"\n   🔧 Generating 300+ features...")
        df_features = self.feature_master.generate_all_features(df, ticker=ticker)
        
        # 3. Create target
        df_features['target'] = df_features['Close'].shift(-1)
        df_clean = df_features.dropna()
        
        print(f"      ✅ Clean data: {df_clean.shape}")
        
        # 4. Separate features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                       'Date', 'target']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        print(f"      ✅ Features: {len(feature_cols)}")
        
        # 5. Temporal split (80/20)
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"      ✅ Train: {len(X_train)} | Test: {len(X_test)}")
        
        # 6. Get models
        if use_production:
            models = self.model_zoo.get_production_models()
            print(f"\n   🤖 Training {len(models)} production models...")
        else:
            models = self.model_zoo.get_all_models()
            print(f"\n   🤖 Training {len(models)} models...")
        
        # 7. Train all models
        results = {}
        predictions = {}
        scores = {}
        trained_models = {}
        
        for model_name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Store
                predictions[model_name] = y_pred
                scores[model_name] = r2
                results[model_name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
                trained_models[model_name] = model
                
                print(f"      ✅ {model_name:25s} | R² = {r2:7.4f} | MAE = {mae:7.2f}")
                
            except Exception as e:
                print(f"      ❌ {model_name:25s} | FAILED: {str(e)[:40]}")
                continue
        
        # 8. Create ensembles
        print(f"\n   🎯 Creating ensembles...")
        
        # Get valid predictions (R² > 0)
        valid_predictions = {k: v for k, v in predictions.items() if scores[k] > 0}
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        
        if len(valid_predictions) > 0:
            # Simple average
            simple_avg = self.ensemble.simple_average(valid_predictions)
            simple_r2 = r2_score(y_test, simple_avg)
            simple_mae = mean_absolute_error(y_test, simple_avg)
            results['Ensemble_Simple'] = {'r2': simple_r2, 'mae': simple_mae}
            
            # Weighted average
            weighted_avg = self.ensemble.weighted_average(valid_predictions, valid_scores)
            weighted_r2 = r2_score(y_test, weighted_avg)
            weighted_mae = mean_absolute_error(y_test, weighted_avg)
            results['Ensemble_Weighted'] = {'r2': weighted_r2, 'mae': weighted_mae}
            
            # Top-5 average
            top5_avg = self.ensemble.top_n_average(valid_predictions, valid_scores, n=min(5, len(valid_predictions)))
            top5_r2 = r2_score(y_test, top5_avg)
            top5_mae = mean_absolute_error(y_test, top5_avg)
            results['Ensemble_Top5'] = {'r2': top5_r2, 'mae': top5_mae}
            
            print(f"      ✅ Simple Ensemble    | R² = {simple_r2:.4f}")
            print(f"      ✅ Weighted Ensemble  | R² = {weighted_r2:.4f}")
            print(f"      ✅ Top-5 Ensemble     | R² = {top5_r2:.4f}")
        
        # 9. Save results
        print(f"\n   💾 Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_clean = ticker.replace('.NS', '')
        
        # Save CSV
        results_df = pd.DataFrame(results).T
        csv_path = results_dir / f"{ticker_clean}_results_{timestamp}.csv"
        results_df.to_csv(csv_path)
        
        # Save models (top 5 only to save space)
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for model_name, _ in sorted_models:
            if model_name in trained_models:
                model_path = results_dir / 'models' / f"{ticker_clean}_{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(trained_models[model_name], f)
        
        # Save summary JSON
        summary = {
            'ticker': ticker,
            'timestamp': timestamp,
            'data_points': len(df_clean),
            'features': len(feature_cols),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_models': len(results),
            'models_with_positive_r2': sum(1 for r in results.values() if r.get('r2', -999) > 0),
            'best_model': max(results.items(), key=lambda x: x[1].get('r2', -999))[0],
            'best_r2': max(results.items(), key=lambda x: x[1].get('r2', -999))[1]['r2'],
            'best_mae': max(results.items(), key=lambda x: x[1].get('r2', -999))[1]['mae']
        }
        
        json_path = results_dir / f"{ticker_clean}_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"      ✅ Results saved!")
        
        return {
            'ticker': ticker,
            'summary': summary,
            'models_trained': list(trained_models.keys()),
            'best_r2': summary['best_r2']
        }


def main(pc_number: int):
    """Main execution"""
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              MARKETLAB 2.0 - BASELINE PIPELINE                   ║
    ║                   30 Models + 300 Features                       ║
    ║                  Geopolitical Integration                        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    pipeline = BaselinePipeline(pc_number=pc_number, random_state=42)
    
    results = pipeline.run_batch(
        start_date="2004-01-01",
        end_date="2024-12-31",
        optimize_top_models=False,  # Set True for optimization (very slow!)
        use_production_models=True  # Fast models only
    )
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH COMPLETE!")
    print(f"📊 Processed {len(results)} stocks")
    print(f"📁 Results saved to: final_results/")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Get PC number from command line
    if len(sys.argv) > 1:
        pc_num = int(sys.argv[1])
    else:
        print("Usage: python baseline_pipeline.py <pc_number>")
        print("Example: python baseline_pipeline.py 1")
        pc_num = 1
    
    main(pc_num)