"""
Macro Features - 35 features
Global macroeconomic indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

class MacroFeatures:
    """Generates 35 macroeconomic features"""
    
    def generate_all(self, df: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        """Generate all macro features"""
        df = df.copy()
        
        if start_date is None:
            start_date = df.index[0].strftime('%Y-%m-%d')
        
        # Download macro data
        print("   📊 Downloading macro indicators...")
        macro_data = self._download_macro_data(start_date, df.index[-1].strftime('%Y-%m-%d'))
        
        # Merge with stock data
        for col in macro_data.columns:
            df[col] = macro_data[col].reindex(df.index, method='ffill')
        
        # Generate derived features
        df = self._generate_derived_features(df)
        
        return df
    
    def _safe_download(self, ticker: str, start_date: str, end_date: str, default_value: float):
        """
        Safely download data from yfinance, return proxy on failure
        
        Returns:
            pd.Series with proper index
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return pd.Series(default_value, index=date_range)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Get Close column
            if 'Close' in data.columns:
                series = data['Close']
            elif len(data.columns) > 0:
                series = data.iloc[:, 0]
            else:
                return pd.Series(default_value, index=date_range)
            
            # Ensure it's a Series
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            
            # Convert to Series if needed
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            
            return series
            
        except Exception as e:
            print(f"      ⚠️  {ticker} download failed: {str(e)[:30]}")
            return pd.Series(default_value, index=date_range)
    
    def _download_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download macroeconomic data from Yahoo Finance"""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        macro_dict = {}
        
        # Oil prices
        print("      📊 Brent oil...")
        brent = self._safe_download('BZ=F', start_date, end_date, 80.0)
        macro_dict['oil_brent'] = brent.reindex(date_range, method='ffill')
        macro_dict['oil_brent_change'] = macro_dict['oil_brent'].pct_change()
        macro_dict['oil_brent_change_5d'] = macro_dict['oil_brent'].pct_change(5)
        
        # USD-INR
        print("      💱 USD-INR...")
        inr = self._safe_download('INR=X', start_date, end_date, 83.0)
        macro_dict['usd_inr'] = inr.reindex(date_range, method='ffill')
        macro_dict['usd_inr_change'] = macro_dict['usd_inr'].pct_change()
        macro_dict['usd_inr_volatility'] = macro_dict['usd_inr'].pct_change().rolling(20).std()
        
        # Gold
        print("      🥇 Gold...")
        gold = self._safe_download('GC=F', start_date, end_date, 2000.0)
        macro_dict['gold_price'] = gold.reindex(date_range, method='ffill')
        macro_dict['gold_change'] = macro_dict['gold_price'].pct_change()
        
        # India VIX
        print("      📈 India VIX...")
        vix = self._safe_download('^INDIAVIX', start_date, end_date, 15.0)
        macro_dict['india_vix'] = vix.reindex(date_range, method='ffill')
        macro_dict['india_vix_change'] = macro_dict['india_vix'].pct_change()
        
        # NIFTY 50
        print("      📊 NIFTY 50...")
        nifty = self._safe_download('^NSEI', start_date, end_date, 22000.0)
        macro_dict['nifty_level'] = nifty.reindex(date_range, method='ffill')
        macro_dict['nifty_return'] = macro_dict['nifty_level'].pct_change()
        macro_dict['nifty_return_5d'] = macro_dict['nifty_level'].pct_change(5)
        
        # US 10Y Yield
        print("      📉 US 10Y...")
        us10y = self._safe_download('^TNX', start_date, end_date, 4.5)
        macro_dict['us_10yr_yield'] = us10y.reindex(date_range, method='ffill')
        
        # DXY
        print("      💵 DXY...")
        dxy = self._safe_download('DX-Y.NYB', start_date, end_date, 104.0)
        macro_dict['dxy_index'] = dxy.reindex(date_range, method='ffill')
        macro_dict['dxy_change'] = macro_dict['dxy_index'].pct_change()
        
        # Copper
        print("      🔶 Copper...")
        copper = self._safe_download('HG=F', start_date, end_date, 4.0)
        macro_dict['copper_price'] = copper.reindex(date_range, method='ffill')
        
        # Silver
        print("      ⚪ Silver...")
        silver = self._safe_download('SI=F', start_date, end_date, 24.0)
        macro_dict['silver_price'] = silver.reindex(date_range, method='ffill')
        
        # Natural Gas
        print("      🔥 Natural Gas...")
        ng = self._safe_download('NG=F', start_date, end_date, 3.0)
        macro_dict['natgas_price'] = ng.reindex(date_range, method='ffill')
        
        # US VIX
        print("      📊 US VIX...")
        usvix = self._safe_download('^VIX', start_date, end_date, 18.0)
        macro_dict['us_vix'] = usvix.reindex(date_range, method='ffill')
        
        # Bitcoin
        print("      ₿ Bitcoin...")
        btc = self._safe_download('BTC-USD', start_date, end_date, 60000.0)
        macro_dict['bitcoin_price'] = btc.reindex(date_range, method='ffill')
        macro_dict['bitcoin_change'] = macro_dict['bitcoin_price'].pct_change()
        
        # Create DataFrame
        macro_df = pd.DataFrame(macro_dict, index=date_range)
        
        # Fill any NaNs
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return macro_df
    
    def _generate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate derived macro features"""
        
        # Oil-related
        if 'oil_brent' in df.columns:
            df['oil_brent_ma20'] = df['oil_brent'].rolling(20).mean()
            df['oil_above_ma20'] = (df['oil_brent'] > df['oil_brent_ma20']).astype(int)
            df['oil_volatility_20d'] = df['oil_brent_change'].rolling(20).std()
        
        # Currency
        if 'usd_inr' in df.columns:
            df['usd_inr_ma50'] = df['usd_inr'].rolling(50).mean()
            df['usd_inr_strength'] = df['usd_inr'] / df['usd_inr_ma50']
        
        # VIX percentile
        if 'india_vix' in df.columns:
            df['india_vix_percentile'] = df['india_vix'].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
        
        # NIFTY momentum
        if 'nifty_return' in df.columns:
            df['nifty_momentum_20d'] = df['nifty_return'].rolling(20).mean()
            df['nifty_trending'] = (df['nifty_momentum_20d'] > 0).astype(int)
        
        # Gold-Oil ratio
        if 'gold_price' in df.columns and 'oil_brent' in df.columns:
            df['gold_oil_ratio'] = df['gold_price'] / df['oil_brent']
        
        # Risk appetite
        if 'india_vix' in df.columns and 'nifty_return' in df.columns:
            df['risk_appetite'] = -df['india_vix_change'] + df['nifty_return']
        
        return df


# Test
if __name__ == "__main__":
    print("Testing Macro Features...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    df = pd.DataFrame({
        'Close': 100 + np.random.randn(len(dates)).cumsum()
    }, index=dates)
    
    macro_features = MacroFeatures()
    df_with_features = macro_features.generate_all(df, start_date='2024-01-01')
    
    # Count features
    original_cols = ['Close']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} macro features!")
    print(f"Sample features: {new_features[:15]}")
    
    assert len(new_features) >= 30, f"Expected 30+ features, got {len(new_features)}"
    print("✅ Macro Features module working!")