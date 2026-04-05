"""
Model Zoo - 30 Machine Learning Models
Complete collection of regression models
"""

import numpy as np
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    BayesianRidge, HuberRegressor, RANSACRegressor, TheilSenRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Gradient boosting libraries
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")


class ModelZoo:
    """
    Collection of 30 regression models
    Organized by family
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def get_all_models(self):
        """
        Get all 30 models with default configurations
        Returns dict of {model_name: model_instance}
        """
        models = {}
        
        # ========== LINEAR MODELS (6) ==========
        models['LinearRegression'] = LinearRegression()
        
        models['Ridge'] = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        models['Lasso'] = Lasso(
            alpha=0.1,
            random_state=self.random_state,
            max_iter=2000
        )
        
        models['ElasticNet'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=self.random_state,
            max_iter=2000
        )
        
        models['BayesianRidge'] = BayesianRidge(
            max_iter=500
        )
        
        models['HuberRegressor'] = HuberRegressor(
            epsilon=1.35,
            max_iter=200
        )
        
        # ========== TREE-BASED MODELS (8) ==========
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state
        )
        
        models['HistGradientBoosting'] = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=self.random_state
        )
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                random_state=self.random_state,
                verbose=0
            )
        
        models['RandomForestDeep'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,  # Deeper trees
            min_samples_split=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # ========== BOOSTING VARIANTS (3) ==========
        models['AdaBoost'] = AdaBoostRegressor(
            n_estimators=50,
            learning_rate=1.0,
            random_state=self.random_state
        )
        
        models['AdaBoost_Linear'] = AdaBoostRegressor(
            estimator=LinearRegression(),
            n_estimators=50,
            learning_rate=1.0,
            random_state=self.random_state
        )
        
        models['GradientBoosting_Huber'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            loss='huber',  # Robust loss
            random_state=self.random_state
        )
        
        # ========== KERNEL METHODS (5) ==========
        models['SVR_RBF'] = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            gamma='scale'
        )
        
        models['SVR_Poly'] = SVR(
            kernel='poly',
            degree=3,
            C=1.0,
            epsilon=0.1,
            gamma='scale'
        )
        
        models['SVR_Linear'] = SVR(
            kernel='linear',
            C=1.0,
            epsilon=0.1
        )
        
        models['SVR_Sigmoid'] = SVR(
            kernel='sigmoid',
            C=1.0,
            epsilon=0.1,
            gamma='scale'
        )
        
        models['KernelRidge'] = KernelRidge(
            alpha=1.0,
            kernel='rbf',
            gamma=0.1
        )
        
        # ========== INSTANCE-BASED (1) ==========
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        
        # ========== ENSEMBLE METHODS (4) ==========
        models['Bagging'] = BaggingRegressor(
            estimator=LinearRegression(),
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Simple voting ensemble
        voting_estimators = [
            ('ridge', Ridge(random_state=self.random_state)),
            ('lasso', Lasso(random_state=self.random_state, max_iter=2000)),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1))
        ]
        models['VotingRegressor'] = VotingRegressor(
            estimators=voting_estimators
        )
        
        # Stacking ensemble
        stacking_estimators = [
            ('ridge', Ridge(random_state=self.random_state)),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=self.random_state, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=self.random_state))
        ]
        models['StackingRegressor'] = StackingRegressor(
            estimators=stacking_estimators,
            final_estimator=Ridge(random_state=self.random_state),
            n_jobs=-1
        )
        
        # Weighted voting (custom weights based on typical performance)
        models['VotingRegressor_Weighted'] = VotingRegressor(
            estimators=voting_estimators,
            weights=[2, 1, 3]  # Give RF more weight
        )
        
        # ========== SPECIALIZED MODELS (3) ==========
        # Gaussian Process (small kernel for speed)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        models['GaussianProcess'] = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self.random_state,
            n_restarts_optimizer=0  # Fast
        )
        
        models['RANSACRegressor'] = RANSACRegressor(
            random_state=self.random_state,
            max_trials=100
        )
        
        models['TheilSenRegressor'] = TheilSenRegressor(
            random_state=self.random_state,
            max_iter=300,
            n_jobs=-1
        )
        
        return models
    
    def get_production_models(self):
        """
        Get fast models for production (excludes slow ones)
        Returns ~18-20 models that train quickly
        """
        all_models = self.get_all_models()
        
        # Exclude very slow models for production speed
        exclude = [
            'GaussianProcess',  # Very slow on large datasets
            'SVR_RBF',          # Slow on large datasets
            'SVR_Poly',         # Slow on large datasets
            'SVR_Sigmoid',      # Slow on large datasets
            'RANSACRegressor',  # Can be slow
            'TheilSenRegressor', # Slow on large datasets
            'StackingRegressor', # Slower (trains multiple models)
        ]
        
        production_models = {k: v for k, v in all_models.items() if k not in exclude}
        
        return production_models
    
    def get_model_families(self):
        """Get models organized by family"""
        return {
            'Linear': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 'HuberRegressor'],
            'Tree': ['RandomForest', 'ExtraTrees', 'RandomForestDeep'],
            'Boosting': ['GradientBoosting', 'HistGradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost',
                        'AdaBoost', 'AdaBoost_Linear', 'GradientBoosting_Huber'],
            'Kernel': ['SVR_RBF', 'SVR_Poly', 'SVR_Linear', 'SVR_Sigmoid', 'KernelRidge'],
            'Instance': ['KNN'],
            'Ensemble': ['Bagging', 'VotingRegressor', 'StackingRegressor', 'VotingRegressor_Weighted'],
            'Specialized': ['GaussianProcess', 'RANSACRegressor', 'TheilSenRegressor']
        }


# Test
if __name__ == "__main__":
    print("="*80)
    print("TESTING MODEL ZOO - 30 MODELS")
    print("="*80)
    
    zoo = ModelZoo(random_state=42)
    
    # Get all models
    all_models = zoo.get_all_models()
    
    print(f"\n📊 Total models: {len(all_models)}")
    
    # Show by family
    families = zoo.get_model_families()
    
    print("\n🤖 MODELS BY FAMILY:")
    print("-"*80)
    
    total_count = 0
    for family, model_names in families.items():
        available = [name for name in model_names if name in all_models]
        print(f"\n{family} ({len(available)} models):")
        for name in available:
            print(f"  ✅ {name}")
        total_count += len(available)
    
    print(f"\n{'='*80}")
    print(f"✅ {total_count} models ready!")
    print(f"{'='*80}")
    
    # Test one model
    print("\n🧪 Testing one model (Ridge)...")
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    
    model = all_models['Ridge']
    model.fit(X[:80], y[:80])
    score = model.score(X[80:], y[80:])
    
    print(f"✅ Ridge R² score: {score:.4f}")
    print("\n✅ Model Zoo working perfectly!")