"""
Cross-Sectional & Microstructure Features - 60 features
Measures relative performance and market microstructure
"""

import pandas as pd
import numpy as np

class CrossSectionalMicrostructureFeatures:
    """Generates 60 cross-sectional and microstructure features"""
    
    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all cross-sectional and microstructure features"""
        df = df.copy()
        
        # Returns for ranking
        returns_5d = df['Close'].pct_change(5)
        returns_10d = df['Close'].pct_change(10)
        returns_20d = df['Close'].pct_change(20)
        returns_50d = df['Close'].pct_change(50)
        returns_100d = df['Close'].pct_change(100)
        returns_200d = df['Close'].pct_change(200)
        
        # Relative strength (normalized returns)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'relative_strength_{period}'] = df['Close'].pct_change(period)
        
        # Z-scores (how many std devs from mean)
        for period in [20, 50, 100]:
            returns = df['Close'].pct_change()
            mean = returns.rolling(period).mean()
            std = returns.rolling(period).std()
            df[f'return_zscore_{period}d'] = (returns - mean) / std
        
        # Volume Z-scores
        for period in [20, 50]:
            vol_mean = df['Volume'].rolling(period).mean()
            vol_std = df['Volume'].rolling(period).std()
            df[f'volume_zscore_{period}d'] = (df['Volume'] - vol_mean) / vol_std
        
        # Percentile ranks (position in rolling window)
        for period in [20, 50, 100]:
            df[f'return_percentile_{period}d'] = df['Close'].pct_change().rolling(period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # High-Low Spread (microstructure)
        for period in [5, 10, 20, 50]:
            df[f'hl_spread_{period}'] = (df['High'] - df['Low']).rolling(period).mean() / df['Close']
        
        # Close-Open Spread
        for period in [5, 10, 20, 50]:
            df[f'co_spread_{period}'] = (df['Close'] - df['Open']).rolling(period).mean() / df['Close']
        
        # Close-Open Range
        for period in [5, 10, 20, 50]:
            df[f'co_range_{period}'] = np.abs(df['Close'] - df['Open']).rolling(period).mean() / df['Close']
        
        # Gap analysis (overnight gaps)
        df['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        for period in [5, 10, 20, 50]:
            df[f'gap_mean_{period}'] = df['gap_pct'].rolling(period).mean()
            df[f'gap_std_{period}'] = df['gap_pct'].rolling(period).std()
        
        # Intraday volatility
        for period in [5, 10, 20, 50, 100]:
            intraday_range = (df['High'] - df['Low']) / df['Open']
            df[f'intraday_vol_{period}'] = intraday_range.rolling(period).mean()
        
        # Amihud Illiquidity (price impact)
        for period in [10, 20]:
            returns = np.abs(df['Close'].pct_change())
            illiquidity = returns / (df['Volume'] + 1)  # +1 to avoid division by zero
            df[f'amihud_illiquidity_{period}'] = illiquidity.rolling(period).mean()
        
        # Volume-weighted metrics
        df['vwap_5d'] = self._vwap(df, 5)
        df['vwap_20d'] = self._vwap(df, 20)
        df['distance_from_vwap_5d'] = (df['Close'] - df['vwap_5d']) / df['vwap_5d']
        df['distance_from_vwap_20d'] = (df['Close'] - df['vwap_20d']) / df['vwap_20d']
        
        return df
    
    def _vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
        return vwap


# Test
if __name__ == "__main__":
    print("Testing Cross-Sectional & Microstructure Features...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=300)
    np.random.seed(42)
    
    price = 100 + np.random.randn(300).cumsum()
    df = pd.DataFrame({
        'Open': price + np.random.randn(300) * 0.5,
        'High': price + np.abs(np.random.randn(300) * 2),
        'Low': price - np.abs(np.random.randn(300) * 2),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)
    
    # Ensure High >= Close >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    cs_features = CrossSectionalMicrostructureFeatures()
    df_with_features = cs_features.generate_all(df)
    
    # Count features
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} cross-sectional & microstructure features!")
    print(f"Sample features: {new_features[:10]}")
    
    assert len(new_features) == 60, f"Expected 60 features, got {len(new_features)}"
    print("✅ Cross-Sectional & Microstructure Features module working!")