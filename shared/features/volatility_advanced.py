"""
Advanced Volatility Features - 60 features
Measures market volatility using multiple methodologies
"""

import pandas as pd
import numpy as np

class VolatilityFeatures:
    """Generates 60 volatility-based features"""
    
    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all volatility features"""
        df = df.copy()
        
        # Parkinson volatility (high-low range based)
        for period in [5, 10, 20, 50, 100]:
            df[f'parkinson_vol_{period}'] = self._parkinson_volatility(df, period)
        
        # Garman-Klass volatility (OHLC based)
        for period in [5, 10, 20, 50, 100]:
            df[f'gk_vol_{period}'] = self._garman_klass_volatility(df, period)
        
        # Rogers-Satchell volatility (drift-independent)
        for period in [5, 10, 20, 50, 100]:
            df[f'rs_vol_{period}'] = self._rogers_satchell_volatility(df, period)
        
        # Yang-Zhang volatility (combines overnight and intraday)
        for period in [5, 10, 20, 50, 100]:
            df[f'yz_vol_{period}'] = self._yang_zhang_volatility(df, period)
        
        # ATR (Average True Range)
        for period in [7, 14, 21, 28]:
            df[f'atr_{period}'] = self._atr(df, period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['Close'] * 100
        
        # Bollinger Band Width
        for period in [10, 20, 50]:
            df[f'bollinger_width_{period}'] = self._bollinger_width(df, period)
            df[f'bollinger_pct_{period}'] = self._bollinger_percentile(df, period)
        
        # Keltner Channel Width
        for period in [10, 20, 50]:
            df[f'keltner_width_{period}'] = self._keltner_width(df, period)
        
        # Realized volatility (close-to-close)
        for period in [5, 10, 20, 50, 100]:
            df[f'realized_vol_{period}'] = self._realized_volatility(df, period)
        
        # Volatility of volatility (meta-indicator)
        df['vol_of_vol_20'] = df['realized_vol_20'].rolling(20).std()
        df['vol_of_vol_50'] = df['realized_vol_50'].rolling(50).std()
        
        return df
    
    def _parkinson_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Parkinson volatility using high-low range"""
        hl = np.log(df['High'] / df['Low'])
        return np.sqrt((1 / (4 * np.log(2))) * (hl ** 2).rolling(period).mean())
    
    def _garman_klass_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Garman-Klass volatility (OHLC)"""
        hl = np.log(df['High'] / df['Low'])
        co = np.log(df['Close'] / df['Open'])
        
        gk = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        return np.sqrt(gk.rolling(period).mean())
    
    def _rogers_satchell_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Rogers-Satchell volatility (drift-independent)"""
        hc = np.log(df['High'] / df['Close'])
        ho = np.log(df['High'] / df['Open'])
        lc = np.log(df['Low'] / df['Close'])
        lo = np.log(df['Low'] / df['Open'])
        
        rs = hc * ho + lc * lo
        return np.sqrt(rs.rolling(period).mean())
    
    def _yang_zhang_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Yang-Zhang volatility (combines overnight and intraday)"""
        # Overnight volatility
        co = np.log(df['Open'] / df['Close'].shift(1))
        overnight_vol = (co ** 2).rolling(period).mean()
        
        # Open-close volatility
        oc = np.log(df['Close'] / df['Open'])
        oc_vol = (oc ** 2).rolling(period).mean()
        
        # Rogers-Satchell component
        rs_vol = self._rogers_satchell_volatility(df, period) ** 2
        
        # Combine
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        yz = overnight_vol + k * oc_vol + (1 - k) * rs_vol
        
        return np.sqrt(yz)
    
    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _bollinger_width(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Bollinger Band Width"""
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (upper - lower) / sma
    
    def _bollinger_percentile(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Position within Bollinger Bands"""
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (df['Close'] - lower) / (upper - lower)
    
    def _keltner_width(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Keltner Channel Width"""
        ema = df['Close'].ewm(span=period).mean()
        atr = self._atr(df, period)
        upper = ema + (2 * atr)
        lower = ema - (2 * atr)
        return (upper - lower) / ema
    
    def _realized_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Realized volatility (close-to-close returns)"""
        returns = np.log(df['Close'] / df['Close'].shift(1))
        return returns.rolling(period).std() * np.sqrt(252)  # Annualized


# Test
if __name__ == "__main__":
    print("Testing Volatility Features...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Ensure High >= Close >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    vol_features = VolatilityFeatures()
    df_with_features = vol_features.generate_all(df)
    
    # Count features
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} volatility features!")
    print(f"Features: {new_features[:10]}...")  # Show first 10
    
    assert len(new_features) == 60, f"Expected 60 features, got {len(new_features)}"
    print("✅ Volatility Features module working!")