"""
Geopolitical Features - 25 features
US-Iran War 2026 impact on Indian markets
YOUR UNIQUE CONTRIBUTION!
"""

import pandas as pd
import numpy as np
from datetime import datetime

class GeopoliticalFeatures:
    """
    Generates 25 geopolitical features
    Models US-Iran conflict impact (Feb-Mar 2026)
    """
    
    # War timeline (key dates)
    WAR_START = pd.Timestamp('2026-02-15')
    WAR_ESCALATION = pd.Timestamp('2026-03-01')
    WAR_PEAK = pd.Timestamp('2026-03-15')
    
    def generate_all(self, df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """Generate all geopolitical features"""
        df = df.copy()
        
        # Event timeline features (10)
        df['conflict_active'] = (df.index >= self.WAR_START).astype(int)
        df['days_since_conflict'] = (df.index - self.WAR_START).days
        df['days_since_conflict'] = df['days_since_conflict'].clip(lower=0)
        
        df['days_to_escalation'] = (self.WAR_ESCALATION - df.index).days
        df['days_to_peak'] = (self.WAR_PEAK - df.index).days
        
        # Conflict intensity (0 to 1 scale)
        df['conflict_intensity'] = self._conflict_intensity(df.index)
        
        # Tension scores (rolling)
        df['tension_7d'] = df['conflict_intensity'].rolling(7).mean()
        df['tension_30d'] = df['conflict_intensity'].rolling(30).mean()
        
        # Phase encoding
        df['pre_war_phase'] = (df.index < self.WAR_START).astype(int)
        df['escalation_phase'] = ((df.index >= self.WAR_START) & (df.index < self.WAR_PEAK)).astype(int)
        df['peak_phase'] = (df.index >= self.WAR_PEAK).astype(int)
        
        # Sector-specific impact features (10)
        sector = self._get_sector(ticker) if ticker else 'Other'
        
        # Defense sector boost
        df['defense_boost'] = (df['conflict_active'] * (1 if sector == 'Defense' else 0))
        df['defense_intensity'] = df['conflict_intensity'] * (1 if sector == 'Defense' else 0)
        
        # Energy sector impact (oil price proxy)
        df['energy_impact'] = df['conflict_intensity'] * (1 if sector == 'Energy' else 0)
        df['oil_spike_proxy'] = df['conflict_intensity'] * 0.3  # Assume 30% oil impact
        
        # IT/Export sector (currency volatility)
        df['it_currency_risk'] = df['conflict_intensity'] * (1 if sector == 'IT' else 0)
        
        # Aviation sector (oil cost impact)
        df['aviation_impact'] = df['conflict_intensity'] * -1 * (1 if sector == 'Auto' else 0)
        
        # Safe haven sectors (FMCG, Pharma)
        is_safe_haven = 1 if sector in ['FMCG', 'Pharma'] else 0
        df['safe_haven_premium'] = df['conflict_intensity'] * is_safe_haven
        
        # Banking sector (risk aversion)
        df['banking_risk'] = df['conflict_intensity'] * (1 if sector == 'Banking' else 0)
        
        # Metals/Commodities (war demand)
        df['commodity_demand'] = df['conflict_intensity'] * (1 if sector == 'Metals' else 0)
        
        # Infrastructure (government spending)
        df['infra_spending'] = df['conflict_intensity'] * 0.5 * (1 if sector == 'Infrastructure' else 0)
        
        # Market regime features (5)
        df['volatility_regime'] = self._volatility_regime(df)
        df['correlation_breakdown'] = df['conflict_intensity'] * 0.8  # Markets decorrelate
        df['flight_to_quality'] = df['conflict_intensity'] * 0.6
        df['market_efficiency_drop'] = df['conflict_intensity'] * 0.4
        df['liquidity_shock'] = (df['conflict_intensity'] > 0.7).astype(int)
        
        return df
    
    def _conflict_intensity(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate conflict intensity over time
        0 = peace, 1 = peak conflict
        """
        intensity = pd.Series(0.0, index=dates)
        
        for date in dates:
            if date < self.WAR_START:
                intensity[date] = 0.0
            elif date < self.WAR_ESCALATION:
                # Ramp up from 0 to 0.5
                days_elapsed = (date - self.WAR_START).days
                intensity[date] = min(0.5, days_elapsed / 14 * 0.5)
            elif date < self.WAR_PEAK:
                # Ramp from 0.5 to 1.0
                days_since_escalation = (date - self.WAR_ESCALATION).days
                intensity[date] = 0.5 + min(0.5, days_since_escalation / 14 * 0.5)
            else:
                # Peak and maintain
                intensity[date] = 1.0
        
        return intensity
    
    def _volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Market volatility regime (increases during conflict)"""
        if 'Close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        returns = df['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        
        # Amplify during conflict
        regime = rolling_vol * (1 + df.get('conflict_intensity', 0))
        
        return regime
    
    def _get_sector(self, ticker: str) -> str:
        """Map ticker to sector"""
        if not ticker:
            return 'Other'
        
        sector_map = {
            'Defense': ['HAL.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'COALINDIA.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS'],
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 
                       'KOTAKBANK.NS', 'INDUSINDBK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'],
            'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS'],
            'Infrastructure': ['LT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'ADANIPORTS.NS']
        }
        
        for sector, tickers in sector_map.items():
            if ticker in tickers:
                return sector
        
        return 'Other'


# Test
if __name__ == "__main__":
    print("Testing Geopolitical Features...")
    
    # Create sample data covering war period
    dates = pd.date_range('2026-01-01', '2026-04-30', freq='D')
    np.random.seed(42)
    
    price = 100 + np.random.randn(len(dates)).cumsum()
    df = pd.DataFrame({
        'Open': price + np.random.randn(len(dates)) * 0.5,
        'High': price + np.abs(np.random.randn(len(dates)) * 2),
        'Low': price - np.abs(np.random.randn(len(dates)) * 2),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    geo_features = GeopoliticalFeatures()
    df_with_features = geo_features.generate_all(df, ticker='HAL.NS')
    
    # Count features
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in df_with_features.columns if col not in original_cols]
    
    print(f"✅ Generated {len(new_features)} geopolitical features!")
    print(f"Sample features: {new_features[:10]}")
    
    # Check key dates
    print(f"\nConflict intensity on key dates:")
    print(f"  Before war (Feb 14): {df_with_features.loc['2026-02-14', 'conflict_intensity']:.2f}")
    print(f"  War start (Feb 15): {df_with_features.loc['2026-02-15', 'conflict_intensity']:.2f}")
    print(f"  Escalation (Mar 01): {df_with_features.loc['2026-03-01', 'conflict_intensity']:.2f}")
    print(f"  Peak (Mar 15): {df_with_features.loc['2026-03-15', 'conflict_intensity']:.2f}")
    
    assert len(new_features) == 25, f"Expected 25 features, got {len(new_features)}"
    print("✅ Geopolitical Features module working!")