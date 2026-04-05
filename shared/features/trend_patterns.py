"""
Trend & Pattern Features - 60 features
Measures trend strength and pattern recognition
"""

import pandas as pd
import numpy as np

class TrendPatternFeatures:
    """Generates 60 trend and pattern features"""
    
    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all trend and pattern features"""
        df = df.copy()
        
        # ADX (Average Directional Index) - multiple periods
        for period in [7, 14, 20, 30, 50, 100]:
            df[f'adx_{period}'] = self._adx(df, period)
        
        # DMI (Directional Movement Index)
        for period in [14, 20, 30, 50]:
            plus_di, minus_di = self._dmi(df, period)
            df[f'dmi_plus_{period}'] = plus_di
            df[f'dmi_minus_{period}'] = minus_di
            df[f'dmi_{period}'] = plus_di - minus_di  # Directional difference
        
        # Parabolic SAR - different acceleration factors
        for af in [0.01, 0.02, 0.05, 0.1, 0.15]:
            df[f'psar_af_{af}'] = self._parabolic_sar(df, af)
            df[f'psar_distance_{af}'] = (df['Close'] - df[f'psar_af_{af}']) / df['Close']
        
        # TRIX (Triple Exponential Average)
        for period in [14, 20, 30]:
            df[f'trix_{period}'] = self._trix(df, period)
        
        # Aroon Indicator
        for period in [14, 25, 50]:
            aroon_up, aroon_down = self._aroon(df, period)
            df[f'aroon_up_{period}'] = aroon_up
            df[f'aroon_down_{period}'] = aroon_down
            df[f'aroon_oscillator_{period}'] = aroon_up - aroon_down
        
        # Vortex Indicator
        for period in [14, 20, 50, 100]:
            vi_plus, vi_minus = self._vortex(df, period)
            df[f'vortex_plus_{period}'] = vi_plus
            df[f'vortex_minus_{period}'] = vi_minus
        
        # SuperTrend - different multipliers
        for period, multiplier in [(10, 2), (10, 3), (20, 2), (20, 3)]:
            df[f'supertrend_{period}_{multiplier}'] = self._supertrend(df, period, multiplier)
        
        # Ichimoku Cloud components
        df['ichimoku_tenkan'] = self._ichimoku_tenkan(df, 9)
        df['ichimoku_kijun'] = self._ichimoku_kijun(df, 26)
        df['ichimoku_senkou_a'] = self._ichimoku_senkou_a(df)
        df['ichimoku_senkou_b'] = self._ichimoku_senkou_b(df, 52)
        
        # Donchian Channel
        for period in [20, 50]:
            upper, lower, mid = self._donchian_channel(df, period)
            df[f'donchian_upper_{period}'] = upper
            df[f'donchian_lower_{period}'] = lower
            df[f'donchian_width_{period}'] = (upper - lower) / mid
        
        # Linear regression slope (trend strength)
        for period in [10, 20, 50]:
            df[f'lr_slope_{period}'] = self._linear_regression_slope(df, period)
        
        # Trend consistency score
        df['trend_consistency_20'] = self._trend_consistency(df, 20)
        df['trend_consistency_50'] = self._trend_consistency(df, 50)
        
        return df
    
    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average Directional Index"""
        plus_dm, minus_dm, tr = self._directional_movement(df)
        
        # Smooth the values
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        tr_smooth = tr.ewm(span=period, adjust=False).mean()
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX is smoothed DX
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def _dmi(self, df: pd.DataFrame, period: int):
        """Directional Movement Index"""
        plus_dm, minus_dm, tr = self._directional_movement(df)
        
        # Smooth
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        tr_smooth = tr.ewm(span=period, adjust=False).mean()
        
        # Directional indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        return plus_di, minus_di
    
    def _directional_movement(self, df: pd.DataFrame):
        """Helper for ADX/DMI calculation"""
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        
        # Plus DM
        plus_dm = high_diff.copy()
        plus_dm[~((high_diff > low_diff) & (high_diff > 0))] = 0
        
        # Minus DM
        minus_dm = low_diff.copy()
        minus_dm[~((low_diff > high_diff) & (low_diff > 0))] = 0
        
        # True Range
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        return plus_dm, minus_dm, tr
    
    def _parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        sar = df['Close'].copy()
        af = af_start
        ep = df['High'].iloc[0]
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df)):
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            if trend == 1:
                if df['Low'].iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = df['Low'].iloc[i]
                    af = af_start
                else:
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af = min(af + af_start, af_max)
            else:
                if df['High'].iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = df['High'].iloc[i]
                    af = af_start
                else:
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af = min(af + af_start, af_max)
        
        return sar
    
    def _trix(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Triple Exponential Average"""
        ema1 = df['Close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        trix = ema3.pct_change() * 100
        return trix
    
    def _aroon(self, df: pd.DataFrame, period: int):
        """Aroon Indicator"""
        aroon_up = df['High'].rolling(period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100
        )
        aroon_down = df['Low'].rolling(period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100
        )
        
        return aroon_up, aroon_down
    
    def _vortex(self, df: pd.DataFrame, period: int):
        """Vortex Indicator"""
        # Vortex Movement
        vm_plus = np.abs(df['High'] - df['Low'].shift(1))
        vm_minus = np.abs(df['Low'] - df['High'].shift(1))
        
        # True Range
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Vortex Indicators
        vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
        vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
        
        return vi_plus, vi_minus
    
    def _supertrend(self, df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
        """SuperTrend Indicator"""
        # ATR
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        # Basic bands
        hl_avg = (df['High'] + df['Low']) / 2
        basic_ub = hl_avg + (multiplier * atr)
        basic_lb = hl_avg - (multiplier * atr)
        
        # Final bands
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        
        for i in range(period, len(df)):
            if basic_ub.iloc[i] < final_ub.iloc[i-1] or df['Close'].iloc[i-1] > final_ub.iloc[i-1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i-1]
                
            if basic_lb.iloc[i] > final_lb.iloc[i-1] or df['Close'].iloc[i-1] < final_lb.iloc[i-1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i-1]
        
        # SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = final_ub.iloc[i]
            elif df['Close'].iloc[i] <= final_ub.iloc[i]:
                supertrend.iloc[i] = final_ub.iloc[i]
            else:
                supertrend.iloc[i] = final_lb.iloc[i]
        
        return supertrend
    
    def _ichimoku_tenkan(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Ichimoku Tenkan-sen (Conversion Line)"""
        high = df['High'].rolling(period).max()
        low = df['Low'].rolling(period).min()
        return (high + low) / 2
    
    def _ichimoku_kijun(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Ichimoku Kijun-sen (Base Line)"""
        high = df['High'].rolling(period).max()
        low = df['Low'].rolling(period).min()
        return (high + low) / 2
    
    def _ichimoku_senkou_a(self, df: pd.DataFrame) -> pd.Series:
        """Ichimoku Senkou Span A (Leading Span A)"""
        tenkan = self._ichimoku_tenkan(df, 9)
        kijun = self._ichimoku_kijun(df, 26)
        return ((tenkan + kijun) / 2).shift(26)
    
    def _ichimoku_senkou_b(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Ichimoku Senkou Span B (Leading Span B)"""
        high = df['High'].rolling(period).max()
        low = df['Low'].rolling(period).min()
        return ((high + low) / 2).shift(26)
    
    def _donchian_channel(self, df: pd.DataFrame, period: int):
        """Donchian Channel"""
        upper = df['High'].rolling(period).max()
        lower = df['Low'].rolling(period).min()
        mid = (upper + lower) / 2
        return upper, lower, mid
    
    def _linear_regression_slope(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Linear regression slope (trend strength)"""
        def calculate_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        slope = df['Close'].rolling(period).apply(calculate_slope, raw=True)
        return slope
    
    def _trend_consistency(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Trend consistency score
        Measures how consistently price moves in one direction
        """
        returns = df['Close'].pct_change()
        
        # Count consecutive positive/negative days
        consistency = returns.rolling(period).apply(
            lambda x: np.abs(np.sum(np.sign(x))) / len(x)
        )
        
        return consistency


# Test
if __name__ == "__main__":
    print("Testing Trend & Pattern Features...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=300)
    np.random.seed(42)
    
    # Create trending data
    trend = np.linspace(0, 50, 300)
    noise = np.random.randn(300) * 5
    price = 100 + trend + noise
    
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
    
    trend_features = TrendPatternFeatures()
    df_with_features = trend_features.generate_all(df)
    
    # Count features
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} trend & pattern features!")
    print(f"Sample features: {new_features[:10]}")
    
    assert len(new_features) == 60, f"Expected 60 features, got {len(new_features)}"
    print("✅ Trend & Pattern Features module working!")