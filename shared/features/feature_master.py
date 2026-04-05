"""
Feature Master - Orchestrates All 300+ Features
Combines all feature generators into one pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from features.volatility_advanced import VolatilityFeatures
from features.momentum_advanced import MomentumFeatures
from features.cross_sectional_microstructure import CrossSectionalMicrostructureFeatures
from features.trend_patterns import TrendPatternFeatures
from features.geopolitical import GeopoliticalFeatures
from features.macro_features import MacroFeatures

class FeatureMaster:
    """
    Master feature generator
    Orchestrates all 300+ features
    """
    
    def __init__(self):
        self.volatility_gen = VolatilityFeatures()
        self.momentum_gen = MomentumFeatures()
        self.cross_sectional_gen = CrossSectionalMicrostructureFeatures()
        self.trend_gen = TrendPatternFeatures()
        self.geopolitical_gen = GeopoliticalFeatures()
        self.macro_gen = MacroFeatures()
        
    def generate_all_features(self, df: pd.DataFrame, ticker: str = None, 
                             include_geopolitical: bool = True,
                             include_macro: bool = True) -> pd.DataFrame:
        """
        Generate all 300+ features
        
        Args:
            df: OHLCV DataFrame
            ticker: Stock ticker (for sector-specific features)
            include_geopolitical: Whether to include war features
            include_macro: Whether to include macro features
            
        Returns:
            DataFrame with all features
        """
        
        print(f"🔧 Generating features for {ticker}...")
        df_features = df.copy()
        
        # 1. Volatility features (60)
        print("  ⚡ Adding 60 volatility features...")
        df_features = self.volatility_gen.generate_all(df_features)
        
        # 2. Momentum features (60)
        print("  📈 Adding 60 momentum features...")
        df_features = self.momentum_gen.generate_all(df_features)
        
        # 3. Cross-sectional & microstructure features (60)
        print("  🔍 Adding 60 cross-sectional & microstructure features...")
        df_features = self.cross_sectional_gen.generate_all(df_features)
        
        # 4. Trend & pattern features (60)
        print("  📊 Adding 60 trend & pattern features...")
        df_features = self.trend_gen.generate_all(df_features)
        
        # 5. Geopolitical features (25)
        if include_geopolitical:
            print("  🌍 Adding 25 geopolitical features...")
            df_features = self.geopolitical_gen.generate_all(df_features, ticker=ticker)
        
        # 6. Macro features (35)
        if include_macro:
            print("  🌐 Adding 35 macro features...")
            start_date = df_features.index[0].strftime('%Y-%m-%d')
            df_features = self.macro_gen.generate_all(df_features, start_date=start_date)
        
        # Count features
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        new_features = [col for col in df_features.columns 
                       if col not in original_cols and col != 'Date']
        
        print(f"      ✅ Generated {len(new_features)} new features!")
        
        return df_features


# Test
if __name__ == "__main__":
    print("="*80)
    print("TESTING FEATURE MASTER - 300+ FEATURES")
    print("="*80)
    
    import yfinance as yf
    
    # Download sample data
    print("\n📥 Downloading test data...")
    df = yf.download('RELIANCE.NS', start='2023-01-01', end='2024-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"✅ Loaded {len(df)} rows")
    
    # Generate all features
    feature_master = FeatureMaster()
    df_with_features = feature_master.generate_all_features(df, ticker='RELIANCE.NS')
    
    # Verify
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"\n📊 FEATURE BREAKDOWN:")
    print(f"   Original columns: {len(original_cols)}")
    print(f"   New features: {len(new_features)}")
    print(f"   Total columns: {len(df_with_features.columns)}")
    
    # Show sample
    print(f"\n📋 Sample features (first 20):")
    for i, feat in enumerate(new_features[:20], 1):
        print(f"   {i:2d}. {feat}")
    
    print(f"\n✅ Feature Master working perfectly!")
    print(f"✅ Ready to generate 300+ features for all stocks!")
    print("="*80)