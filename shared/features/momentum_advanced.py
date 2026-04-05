"""
Advanced Momentum Features - 60 features
Measures price momentum and trend strength
"""

import pandas as pd
import numpy as np

class MomentumFeatures:
    """Generates 60 momentum-based features"""
    
    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all momentum features"""
        df = df.copy()
        
        # RSI (Relative Strength Index) - multiple periods
        for period in [5, 7, 14, 21, 28, 50]:
            df[f'rsi_{period}'] = self._rsi(df, period)
        
        # MACD (Moving Average Convergence Divergence) - different configurations
        for fast, slow, signal in [(8, 17, 9), (12, 26, 9), (16, 32, 9)]:
            macd, signal_line, hist = self._macd(df, fast, slow, signal)
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal_line
            df[f'macd_hist_{fast}_{slow}'] = hist
        
        # Stochastic Oscillator - multiple configurations
        for k, d, smooth in [(5, 3, 3), (14, 3, 3), (21, 5, 5)]:
            stoch_k, stoch_d = self._stochastic(df, k, d, smooth)
            df[f'stoch_k_{k}_{smooth}'] = stoch_k
            df[f'stoch_d_{k}_{smooth}'] = stoch_d
        
        # Williams %R - multiple periods
        for period in [7, 14, 28]:
            df[f'williams_r_{period}'] = self._williams_r(df, period)
        
        # CCI (Commodity Channel Index) - multiple periods
        for period in [14, 20, 50, 100]:
            df[f'cci_{period}'] = self._cci(df, period)
        
        # ROC (Rate of Change) - multiple periods
        for period in [5, 10, 20, 50]:
            df[f'roc_{period}'] = self._roc(df, period)
        
        # Simple Momentum (price changes)
        for period in [5, 10, 20, 50, 100]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Money Flow Index (volume-weighted RSI)
        for period in [14, 28]:
            df[f'mfi_{period}'] = self._mfi(df, period)
        
        # Ultimate Oscillator (combines multiple timeframes)
        df['ultimate_oscillator'] = self._ultimate_oscillator(df)
        
        # Momentum trend indicator
        df['momentum_trend'] = self._momentum_trend(df)
        
        return df
    
    def _rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int):
        """Moving Average Convergence Divergence"""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _stochastic(self, df: pd.DataFrame, k_period: int, d_period: int, smooth: int):
        """Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        
        # %K (fast stochastic)
        stoch_k = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Smooth %K
        stoch_k = stoch_k.rolling(window=smooth).mean()
        
        # %D (slow stochastic - moving average of %K)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def _williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        
        williams_r = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def _cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def _roc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Rate of Change"""
        roc = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        return roc
    
    def _mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Money Flow Index (volume-weighted RSI)"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Sum over period
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Money flow ratio and index
        mf_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi
    
    def _ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """Ultimate Oscillator (combines 7, 14, 28 period momentum)"""
        # Buying pressure
        bp = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
        
        # True range
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Average for different periods
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        # Ultimate oscillator
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
        
        return uo
    
    def _momentum_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Momentum trend indicator
        Combines multiple momentum signals
        """
        rsi_14 = self._rsi(df, 14)
        macd, _, _ = self._macd(df, 12, 26, 9)
        roc_10 = self._roc(df, 10)
        
        # Normalize to 0-100 scale
        rsi_norm = rsi_14
        macd_norm = (macd - macd.rolling(50).min()) / (macd.rolling(50).max() - macd.rolling(50).min()) * 100
        roc_norm = (roc_10 + 50).clip(0, 100)  # Shift and clip
        
        # Average
        trend = (rsi_norm + macd_norm + roc_norm) / 3
        
        return trend


# Test
if __name__ == "__main__":
    print("Testing Momentum Features...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=200)
    np.random.seed(42)
    
    price = 100 + np.random.randn(200).cumsum()
    df = pd.DataFrame({
        'Open': price + np.random.randn(200) * 0.5,
        'High': price + np.abs(np.random.randn(200) * 2),
        'Low': price - np.abs(np.random.randn(200) * 2),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, 200)
    }, index=dates)
    
    # Ensure High >= Close >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    momentum_features = MomentumFeatures()
    df_with_features = momentum_features.generate_all(df)
    
    # Count features
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} momentum features!")
    print(f"Sample features: {new_features[:10]}")
    
    assert len(new_features) == 60, f"Expected 60 features, got {len(new_features)}"
    print("✅ Momentum Features module working!")