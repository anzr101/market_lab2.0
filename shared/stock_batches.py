"""
Stock Batches for 3-Computer Parallel Execution
50 NIFTY 50 stocks divided into 3 equal batches
"""

# Complete NIFTY 50 list (as of 2026)
NIFTY_50_ALL = [
    # Batch 1 stocks (17)
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
    'SUNPHARMA.NS', 'TITAN.NS',
    
    # Batch 2 stocks (17)
    'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'WIPRO.NS', 
    'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS', 'TATAMOTORS.NS',
    'TECHM.NS', 'ADANIENT.NS', 'INDUSINDBK.NS', 'TATASTEEL.NS',
    'BAJAJFINSV.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'HINDALCO.NS',
    
    # Batch 3 stocks (16)
    'GRASIM.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'CIPLA.NS',
    'HEROMOTOCO.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS', 'TATACONSUM.NS',
    'JSWSTEEL.NS', 'BPCL.NS', 'IOC.NS', 'UPL.NS', 'ADANIPORTS.NS',
    'SBILIFE.NS', 'LTIM.NS', 'HAL.NS'
]

# Sector mapping (for analysis)
SECTORS = {
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'COALINDIA.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTIM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 
                'INDUSINDBK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
    'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS'],
    'Infrastructure': ['LT.NS', 'ULTRACEMCO.NS', 'GRASIM.NS', 'ADANIPORTS.NS'],
    'Telecom': ['BHARTIARTL.NS'],
    'Utilities': ['NTPC.NS', 'POWERGRID.NS'],
    'Consumer': ['TITAN.NS', 'ASIANPAINT.NS'],
    'Diversified': ['ADANIENT.NS', 'UPL.NS'],
    'Defense': ['HAL.NS']
}

# Divide into 3 batches for parallel processing
BATCH_1 = NIFTY_50_ALL[:17]   # PC 1: 17 stocks
BATCH_2 = NIFTY_50_ALL[17:34] # PC 2: 17 stocks  
BATCH_3 = NIFTY_50_ALL[34:]   # PC 3: 16 stocks

STOCK_BATCHES = {
    1: BATCH_1,
    2: BATCH_2,
    3: BATCH_3
}

def get_sector(ticker):
    """Get sector for a given ticker"""
    for sector, tickers in SECTORS.items():
        if ticker in tickers:
            return sector
    return 'Other'

# Verify counts
assert len(NIFTY_50_ALL) == 50, "Should have 50 stocks"
assert len(BATCH_1) == 17, "Batch 1 should have 17 stocks"
assert len(BATCH_2) == 17, "Batch 2 should have 17 stocks"
assert len(BATCH_3) == 16, "Batch 3 should have 16 stocks"

if __name__ == "__main__":
    print("="*80)
    print("NIFTY 50 STOCK BATCHES")
    print("="*80)
    print(f"\nTotal stocks: {len(NIFTY_50_ALL)}")
    print(f"Batch 1 (PC 1): {len(BATCH_1)} stocks")
    print(f"Batch 2 (PC 2): {len(BATCH_2)} stocks")
    print(f"Batch 3 (PC 3): {len(BATCH_3)} stocks")
    print("\nSector distribution:")
    for sector, tickers in SECTORS.items():
        print(f"  {sector:15s}: {len(tickers)} stocks")
    print("\n✅ Stock batches configured!")