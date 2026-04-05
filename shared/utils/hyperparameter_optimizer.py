"""
Hyperparameter Optimization using Optuna
Bayesian optimization for model tuning
"""

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna
    """
    
    def __init__(self, n_trials: int = 50, cv: int = 3, random_state: int = 42):
        """
        Args:
            n_trials: Number of optimization trials
            cv: Cross-validation folds
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        
    def optimize_ridge(self, X, y):
        """Optimize Ridge regression"""
        from sklearn.linear_model import Ridge
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
            
            model = Ridge(alpha=alpha, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                    scoring=make_scorer(r2_score), n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_lasso(self, X, y):
        """Optimize Lasso regression"""
        from sklearn.linear_model import Lasso
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.0001, 10.0, log=True)
            
            model = Lasso(alpha=alpha, max_iter=2000, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                    scoring=make_scorer(r2_score), n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_random_forest(self, X, y):
        """Optimize Random Forest"""
        from sklearn.ensemble import RandomForestRegressor
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                    scoring=make_scorer(r2_score), n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_lightgbm(self, X, y):
        """Optimize LightGBM"""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None, None
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            num_leaves = trial.suggest_int('num_leaves', 10, 100)
            min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
            
            model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1
            )
            
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                    scoring=make_scorer(r2_score), n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_xgboost(self, X, y):
        """Optimize XGBoost"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None, None
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1
            )
            
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                    scoring=make_scorer(r2_score), n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_model(self, model_name: str, X, y):
        """
        Optimize any supported model
        
        Args:
            model_name: Name of model to optimize
            X: Features
            y: Target
            
        Returns:
            best_params, best_score
        """
        optimizers = {
            'Ridge': self.optimize_ridge,
            'Lasso': self.optimize_lasso,
            'RandomForest': self.optimize_random_forest,
            'LightGBM': self.optimize_lightgbm,
            'XGBoost': self.optimize_xgboost,
        }
        
        if model_name in optimizers:
            return optimizers[model_name](X, y)
        else:
            print(f"⚠️  No optimizer for {model_name}")
            return None, None


# Test
if __name__ == "__main__":
    print("Testing Hyperparameter Optimizer...")
    
    from sklearn.datasets import make_regression
    
    # Create sample data
    X, y = make_regression(n_samples=500, n_features=20, random_state=42)
    
    optimizer = HyperparameterOptimizer(n_trials=10, cv=3)  # Small for testing
    
    # Test Ridge optimization
    print("\n🔧 Optimizing Ridge...")
    best_params, best_score = optimizer.optimize_ridge(X, y)
    print(f"   Best params: {best_params}")
    print(f"   Best R²: {best_score:.4f}")
    
    # Test Random Forest
    print("\n🔧 Optimizing Random Forest...")
    best_params, best_score = optimizer.optimize_random_forest(X, y)
    print(f"   Best params: {best_params}")
    print(f"   Best R²: {best_score:.4f}")
    
    print("\n✅ Hyperparameter Optimizer working!")