"""
Ensemble Methods
Combines multiple model predictions
"""

import numpy as np
import pandas as pd
from typing import Dict

class SimpleEnsemble:
    """
    Simple ensemble methods for combining predictions
    """
    
    def simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple average of all predictions
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            
        Returns:
            Averaged predictions
        """
        pred_arrays = list(predictions.values())
        return np.mean(pred_arrays, axis=0)
    
    def weighted_average(self, predictions: Dict[str, np.ndarray], 
                        scores: Dict[str, float]) -> np.ndarray:
        """
        Weighted average based on model scores (R²)
        Better models get more weight
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            scores: Dict of {model_name: r2_score}
            
        Returns:
            Weighted average predictions
        """
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in scores:
                weight = max(scores[model_name], 0)  # Use R² as weight (clip negative)
                weighted_sum += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self.simple_average(predictions)
    
    def top_n_average(self, predictions: Dict[str, np.ndarray],
                      scores: Dict[str, float],
                      n: int = 5) -> np.ndarray:
        """
        Average of top N best-performing models
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            scores: Dict of {model_name: r2_score}
            n: Number of top models to use
            
        Returns:
            Top-N averaged predictions
        """
        # Sort by score
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_n_models = [name for name, score in sorted_models[:n]]
        
        # Get predictions for top N
        top_predictions = {name: predictions[name] for name in top_n_models 
                          if name in predictions}
        
        return self.simple_average(top_predictions)
    
    def median_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Median of all predictions (robust to outliers)
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            
        Returns:
            Median predictions
        """
        pred_arrays = list(predictions.values())
        return np.median(pred_arrays, axis=0)
    
    def trimmed_mean(self, predictions: Dict[str, np.ndarray],
                     trim_percent: float = 0.2) -> np.ndarray:
        """
        Trimmed mean (removes extreme predictions)
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            trim_percent: Percentage to trim from each end (0-0.5)
            
        Returns:
            Trimmed mean predictions
        """
        from scipy import stats
        pred_arrays = list(predictions.values())
        return stats.trim_mean(pred_arrays, trim_percent, axis=0)


# Test
if __name__ == "__main__":
    print("Testing Ensemble Methods...")
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 100
    
    predictions = {
        'Model1': np.random.randn(n_samples) + 100,
        'Model2': np.random.randn(n_samples) + 100.5,
        'Model3': np.random.randn(n_samples) + 99.5,
        'Model4': np.random.randn(n_samples) + 100.2,
        'Model5': np.random.randn(n_samples) + 99.8,
    }
    
    scores = {
        'Model1': 0.75,
        'Model2': 0.82,  # Best
        'Model3': 0.65,
        'Model4': 0.71,
        'Model5': 0.68,
    }
    
    ensemble = SimpleEnsemble()
    
    # Test all methods
    simple_avg = ensemble.simple_average(predictions)
    weighted_avg = ensemble.weighted_average(predictions, scores)
    top3_avg = ensemble.top_n_average(predictions, scores, n=3)
    median_pred = ensemble.median_ensemble(predictions)
    
    print(f"✅ Simple Average: Mean = {simple_avg.mean():.2f}")
    print(f"✅ Weighted Average: Mean = {weighted_avg.mean():.2f}")
    print(f"✅ Top-3 Average: Mean = {top3_avg.mean():.2f}")
    print(f"✅ Median Ensemble: Mean = {median_pred.mean():.2f}")
    
    print("\n✅ Ensemble methods working!")