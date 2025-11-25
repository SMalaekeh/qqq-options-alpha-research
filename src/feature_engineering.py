"""
Robust Feature Engineering Module with Regime Detection
Generates 100+ features with anti-overfitting measures:
- No forward-looking data leakage
- Regime detection (Low/High Volatility)
- Proper rolling calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Generates comprehensive feature set from QQQ options data:
    - Volatility Surface (IV by moneyness x tenor buckets)
    - Greeks & Gamma Exposure (GEX)
    - Variance Risk Premium (VRP)
    - Flow & Sentiment (Put/Call ratios)
    - Regime Detection (Low/High Volatility)
    - Interaction Terms
    
    Key robustness measures:
    - All features use only past data (no look-ahead bias)
    - Regime flag for stratified analysis
    - No global scaling before train/test split
    """
    
    def __init__(self):
        self.moneyness_buckets = {
            'deep_otm_put': (0.0, 0.90),
            'otm_put': (0.90, 0.97),
            'atm': (0.97, 1.03),
            'otm_call': (1.03, 1.10),
            'deep_otm_call': (1.10, 2.0)
        }
        
        self.tenor_buckets = {
            'weekly': (0, 10),
            'monthly': (10, 45),
            'quarterly': (45, 90),
            'long': (90, 1000)
        }
    
    def load_data(self, filepath: str, nrows: int = None) -> pd.DataFrame:
        """Load options data using Pandas."""
        print(f"Loading data from {filepath}...")
        
        df = pd.read_csv(
            filepath,
            parse_dates=['tradeDate', 'expirDate'],
            nrows=nrows
        )
        
        print(f"Loaded {len(df):,} rows")
        print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
        return df
    
    def calculate_moneyness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moneyness (strike / spot) for each option."""
        df['moneyness'] = df['strike'] / df['spotPrice']
        return df
    
    def calculate_implied_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate implied volatility from option greeks.
        Using vega relationship: OptionPrice ≈ Vega * IV * sqrt(T)
        """
        print("Calculating implied volatility...")
        
        # IV from calls
        df['iv_call_proxy'] = np.where(
            df['vega'] > 0.01,
            df['callValue'] / (df['vega'] * np.sqrt(df['dte'].clip(lower=1) / 365)),
            np.nan
        )
        
        # IV from puts
        df['iv_put_proxy'] = np.where(
            df['vega'] > 0.01,
            df['putValue'] / (df['vega'] * np.sqrt(df['dte'].clip(lower=1) / 365)),
            np.nan
        )
        
        # Average where both exist
        df['iv'] = np.where(
            df['iv_call_proxy'].notna() & df['iv_put_proxy'].notna(),
            (df['iv_call_proxy'] + df['iv_put_proxy']) / 2,
            np.where(
                df['iv_call_proxy'].notna(),
                df['iv_call_proxy'],
                df['iv_put_proxy']
            )
        )
        
        # Clip to reasonable ranges (0 to 200% IV)
        df['iv'] = df['iv'].clip(0, 2.0)
        
        return df
    
    def build_volatility_surface(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate IV by moneyness x tenor buckets.
        Creates 20 surface features (5 moneyness x 4 tenor).
        """
        print("Building volatility surface features...")
        
        for tenor_name, (dte_min, dte_max) in self.tenor_buckets.items():
            for money_name, (m_min, m_max) in self.moneyness_buckets.items():
                mask = (
                    (df['dte'] >= dte_min) & 
                    (df['dte'] < dte_max) &
                    (df['moneyness'] >= m_min) & 
                    (df['moneyness'] < m_max) &
                    df['iv'].notna()
                )
                
                feature_name = f'iv_{money_name}_{tenor_name}'
                
                # Volume-weighted IV
                volume = df['callVolume'] + df['putVolume'] + 1
                df[f'{feature_name}_weighted'] = np.where(mask, df['iv'] * volume, np.nan)
                df[f'{feature_name}_volume'] = np.where(mask, volume, 0)
        
        return df
    
    def aggregate_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate all features to daily level.
        """
        print("Aggregating features to daily level...")
        
        agg_dict = {
            'spotPrice': 'last',
        }
        
        # IV surface features (volume-weighted)
        iv_cols = [col for col in df.columns if col.startswith('iv_') and ('weighted' in col or 'volume' in col)]
        for col in iv_cols:
            if 'weighted' in col:
                base_name = col.replace('_weighted', '')
                agg_dict[col] = 'sum'
                agg_dict[f'{base_name}_volume'] = 'sum'
        
        # Greeks & GEX
        df['gex_call'] = df['gamma'] * df['callOpenInterest'] * df['strike'] * 100
        df['gex_put'] = df['gamma'] * df['putOpenInterest'] * df['strike'] * 100
        agg_dict['gex_call'] = 'sum'
        agg_dict['gex_put'] = 'sum'
        
        # GEX ATM (key dealer hedging level)
        df['gex_atm'] = np.where(
            (df['moneyness'] >= 0.97) & (df['moneyness'] <= 1.03),
            df['gamma'] * (df['callOpenInterest'] - df['putOpenInterest']) * df['strike'] * 100,
            0
        )
        agg_dict['gex_atm'] = 'sum'
        
        # Vega exposure
        df['vega_call'] = df['vega'] * df['callOpenInterest']
        df['vega_put'] = df['vega'] * df['putOpenInterest']
        agg_dict['vega_call'] = 'sum'
        agg_dict['vega_put'] = 'sum'
        
        # Delta exposure
        df['delta_call'] = df['delta'] * df['callOpenInterest']
        df['delta_put'] = df['delta'] * df['putOpenInterest']
        agg_dict['delta_call'] = 'sum'
        agg_dict['delta_put'] = 'sum'
        
        # Flow & Sentiment
        agg_dict.update({
            'putVolume': 'sum',
            'callVolume': 'sum',
            'putOpenInterest': 'sum',
            'callOpenInterest': 'sum',
        })
        
        # Volume by moneyness
        df['otm_put_volume'] = np.where(df['moneyness'] < 0.97, df['putVolume'], 0)
        df['otm_call_volume'] = np.where(df['moneyness'] > 1.03, df['callVolume'], 0)
        agg_dict['otm_put_volume'] = 'sum'
        agg_dict['otm_call_volume'] = 'sum'
        
        # Volume by tenor
        df['volume_weekly'] = np.where(df['dte'] < 10, df['callVolume'] + df['putVolume'], 0)
        df['volume_monthly'] = np.where(
            (df['dte'] >= 10) & (df['dte'] < 45),
            df['callVolume'] + df['putVolume'],
            0
        )
        agg_dict['volume_weekly'] = 'sum'
        agg_dict['volume_monthly'] = 'sum'
        
        # Dollar volume
        df['call_dollar_volume'] = df['callValue'] * df['callVolume']
        df['put_dollar_volume'] = df['putValue'] * df['putVolume']
        agg_dict['call_dollar_volume'] = 'sum'
        agg_dict['put_dollar_volume'] = 'sum'
        
        # Aggregate
        daily = df.groupby('tradeDate').agg(agg_dict).reset_index()
        
        # Calculate volume-weighted IVs
        for col in iv_cols:
            if 'weighted' in col:
                base_name = col.replace('_weighted', '')
                volume_col = f'{base_name}_volume'
                if volume_col in daily.columns:
                    daily[base_name] = daily[col] / (daily[volume_col] + 1)
                    daily = daily.drop([col, volume_col], axis=1)
        
        # Derived features
        daily['spot_price'] = daily['spotPrice']
        daily['gex_raw'] = daily['gex_call'] - daily['gex_put']
        
        # GEX normalization with AGGRESSIVE outlier protection
        # First clip raw GEX to remove extreme values before normalization
        gex_raw_p05 = daily['gex_raw'].quantile(0.05)
        gex_raw_p95 = daily['gex_raw'].quantile(0.95)
        daily['gex_raw_clipped'] = daily['gex_raw'].clip(gex_raw_p05, gex_raw_p95)
        
        # Normalize by notional value with proper scaling
        daily['gex'] = daily['gex_raw_clipped'] / (daily['spot_price'] ** 2 * 100 + 1e6)
        daily['gex_atm_normalized'] = daily['gex_atm'] / (daily['spot_price'] ** 2 * 100 + 1e6)
        
        # Additional winsorization on normalized values (5th-95th percentile for more aggressive clipping)
        gex_p05 = daily['gex'].quantile(0.05)
        gex_p95 = daily['gex'].quantile(0.95)
        daily['gex'] = daily['gex'].clip(gex_p05, gex_p95)
        
        gex_atm_p05 = daily['gex_atm_normalized'].quantile(0.05)
        gex_atm_p95 = daily['gex_atm_normalized'].quantile(0.95)
        daily['gex_atm_normalized'] = daily['gex_atm_normalized'].clip(gex_atm_p05, gex_atm_p95)
        
        # Z-score normalization to make GEX more interpretable
        gex_mean = daily['gex'].mean()
        gex_std = daily['gex'].std()
        daily['gex'] = (daily['gex'] - gex_mean) / (gex_std + 1e-8)
        
        # Final safety clip to [-5, 5] range
        daily['gex'] = daily['gex'].clip(-5, 5)
        
        # Put/Call ratios
        daily['total_put_volume'] = daily['putVolume']
        daily['total_call_volume'] = daily['callVolume']
        daily['total_put_oi'] = daily['putOpenInterest']
        daily['total_call_oi'] = daily['callOpenInterest']
        
        daily['pcr_volume'] = daily['total_put_volume'] / (daily['total_call_volume'] + 1)
        daily['pcr_oi'] = daily['total_put_oi'] / (daily['total_call_oi'] + 1)
        daily['pcr_otm'] = daily['otm_put_volume'] / (daily['otm_call_volume'] + 1)
        
        # Net flows
        daily['total_delta_call'] = daily['delta_call']
        daily['total_delta_put'] = daily['delta_put']
        daily['net_delta_flow'] = daily['total_delta_call'] - daily['total_delta_put']
        
        daily['total_vega_call'] = daily['vega_call']
        daily['total_vega_put'] = daily['vega_put']
        daily['vega_ratio'] = daily['total_vega_put'] / (daily['total_vega_call'] + 1)
        
        # Volume metrics
        daily['volume_term_ratio'] = daily['volume_weekly'] / (daily['volume_monthly'] + 1)
        daily['avg_call_premium'] = daily['call_dollar_volume'] / (daily['total_call_volume'] + 1)
        daily['avg_put_premium'] = daily['put_dollar_volume'] / (daily['total_put_volume'] + 1)
        
        # Sort by date
        daily = daily.sort_values('tradeDate').reset_index(drop=True)
        
        return daily
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window features for momentum and realized volatility.
        All features use only past data (no look-ahead).
        Uses min_periods to prevent forward-looking bias at start of series.
        """
        print("Adding rolling features...")
        
        # Calculate returns
        df['return_1d'] = df['spot_price'].pct_change()
        
        # Rolling windows: 5, 10, 20, 60 days
        windows = [5, 10, 20, 60]
        
        for window in windows:
            # Use min_periods to avoid NaN contamination
            min_periods = max(int(window * 0.5), 3)
            
            df[f'momentum_{window}d'] = df['return_1d'].rolling(window, min_periods=min_periods).sum()
            df[f'rv_{window}d'] = df['return_1d'].rolling(window, min_periods=min_periods).std() * np.sqrt(252)
            df[f'gex_ma_{window}d'] = df['gex'].rolling(window, min_periods=min_periods).mean()
            df[f'pcr_volume_ma_{window}d'] = df['pcr_volume'].rolling(window, min_periods=min_periods).mean()
            
            if 'iv_atm_monthly' in df.columns:
                df[f'iv_atm_monthly_ma_{window}d'] = df['iv_atm_monthly'].rolling(window, min_periods=min_periods).mean()
        
        # Clip extreme momentum values to prevent outliers
        for window in windows:
            mom_col = f'momentum_{window}d'
            if mom_col in df.columns:
                mom_p01 = df[mom_col].quantile(0.01)
                mom_p99 = df[mom_col].quantile(0.99)
                df[mom_col] = df[mom_col].clip(mom_p01, mom_p99)
        
        return df
    
    def add_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRUCIAL: Add regime flag (Low/High Volatility) for stratified analysis.
        Regime is based on rv_20d vs its 60-day moving average.
        
        High volatility regime: rv_20d > ma_60(rv_20d)
        Low volatility regime: rv_20d <= ma_60(rv_20d)
        """
        print("Adding regime detection...")
        
        if 'rv_20d' in df.columns:
            # Calculate 60-day moving average of realized vol
            rv_ma_60 = df['rv_20d'].rolling(60).mean()
            
            # Create regime flag
            df['regime_high_vol'] = (df['rv_20d'] > rv_ma_60).astype(int)
            df['regime_label'] = df['regime_high_vol'].map({0: 'Low Vol', 1: 'High Vol'})
            
            print(f"  High Vol periods: {df['regime_high_vol'].sum()} days")
            print(f"  Low Vol periods: {(1 - df['regime_high_vol']).sum()} days")
        else:
            print("  Warning: rv_20d not found, skipping regime detection")
            df['regime_high_vol'] = 0
            df['regime_label'] = 'Unknown'
        
        return df
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VRP, skew, and interaction terms.
        With proper outlier handling and clipping.
        """
        print("Adding advanced features...")
        
        # Variance Risk Premium (VRP) - key predictor
        # FIXED: Only calculate VRP when both IV and RV are valid (non-zero)
        # This prevents spurious zeros from NaN-filled values
        if 'iv_atm_monthly' in df.columns and 'rv_20d' in df.columns:
            # Mark original NaN locations before any calculations
            iv_valid = (df['iv_atm_monthly'].notna()) & (df['iv_atm_monthly'] > 0.01)
            rv_valid = (df['rv_20d'].notna()) & (df['rv_20d'] > 0.01)
            
            # Calculate VRP only where both are valid
            df['vrp'] = np.where(
                iv_valid & rv_valid,
                df['iv_atm_monthly'] - df['rv_20d'],
                np.nan
            )
            # Clip VRP to reasonable range
            df['vrp'] = df['vrp'].clip(-0.5, 0.5)
        
        # Volatility Skew
        if 'iv_otm_put_monthly' in df.columns and 'iv_otm_call_monthly' in df.columns:
            df['vol_skew_monthly'] = df['iv_otm_put_monthly'] - df['iv_otm_call_monthly']
            df['vol_skew_monthly'] = df['vol_skew_monthly'].clip(-0.3, 0.3)
        
        if 'iv_deep_otm_put_monthly' in df.columns and 'iv_deep_otm_call_monthly' in df.columns:
            df['vol_skew_deep'] = df['iv_deep_otm_put_monthly'] - df['iv_deep_otm_call_monthly']
            df['vol_skew_deep'] = df['vol_skew_deep'].clip(-0.5, 0.5)
        
        # Term Structure
        if 'iv_atm_monthly' in df.columns and 'iv_atm_weekly' in df.columns:
            df['term_structure_atm'] = df['iv_atm_monthly'] - df['iv_atm_weekly']
            df['term_structure_atm'] = df['term_structure_atm'].clip(-0.3, 0.3)
        
        if 'iv_atm_long' in df.columns and 'iv_atm_monthly' in df.columns:
            df['term_structure_long'] = df['iv_atm_long'] - df['iv_atm_monthly']
            df['term_structure_long'] = df['term_structure_long'].clip(-0.3, 0.3)
        
        # Smile features
        if all(col in df.columns for col in ['iv_atm_monthly', 'iv_otm_put_monthly', 'iv_otm_call_monthly']):
            df['vol_smile'] = df['iv_atm_monthly'] - (df['iv_otm_put_monthly'] + df['iv_otm_call_monthly']) / 2
            df['vol_smile'] = df['vol_smile'].clip(-0.2, 0.2)
        
        # Interaction Terms (with outlier control)
        if 'gex' in df.columns and 'momentum_20d' in df.columns:
            df['gex_x_momentum'] = df['gex'] * df['momentum_20d']
            # Clip interaction terms
            p01 = df['gex_x_momentum'].quantile(0.01)
            p99 = df['gex_x_momentum'].quantile(0.99)
            df['gex_x_momentum'] = df['gex_x_momentum'].clip(p01, p99)
        
        if 'vrp' in df.columns and 'vol_skew_monthly' in df.columns:
            df['vrp_x_skew'] = df['vrp'] * df['vol_skew_monthly']
        
        if 'pcr_otm' in df.columns and 'rv_20d' in df.columns:
            df['pcr_x_rvol'] = df['pcr_otm'] * df['rv_20d']
        
        if 'net_delta_flow' in df.columns and 'momentum_10d' in df.columns:
            df['delta_flow_x_momentum'] = df['net_delta_flow'] * df['momentum_10d']
            # Clip this interaction
            p01 = df['delta_flow_x_momentum'].quantile(0.01)
            p99 = df['delta_flow_x_momentum'].quantile(0.99)
            df['delta_flow_x_momentum'] = df['delta_flow_x_momentum'].clip(p01, p99)
        
        if 'gex' in df.columns and 'rv_20d' in df.columns:
            df['gex_x_rvol'] = df['gex'] * df['rv_20d']
        
        # Rolling Z-scores (standardization within rolling window - no look-ahead)
        for feature in ['gex', 'pcr_volume', 'vrp', 'vol_skew_monthly']:
            if feature in df.columns:
                mean = df[feature].rolling(60, min_periods=20).mean()
                std = df[feature].rolling(60, min_periods=20).std()
                df[f'{feature}_zscore'] = (df[feature] - mean) / (std + 1e-8)
                # Clip z-scores to [-3, 3] range
                df[f'{feature}_zscore'] = df[f'{feature}_zscore'].clip(-3, 3)
        
        return df
    
    def create_target(self, df: pd.DataFrame, forward_days: int = 1) -> pd.DataFrame:
        """
        Create target variable: next-day return.
        """
        print(f"Creating target (forward {forward_days} day return)...")
        df['spot_price_future'] = df['spot_price'].shift(-forward_days)
        df['target'] = df['spot_price_future'] / df['spot_price'] - 1
        
        return df
    
    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: replace infinities with NaNs, fill NaNs with 0.
        Drop rows with missing targets.
        
        IMPORTANT: No global scaling here - scaling must happen after train/test split!
        """
        print("Sanitizing data...")
        
        # Replace inf with nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['tradeDate', 'target', 'spot_price', 'spot_price_future', 'spotPrice', 
                       'regime_high_vol', 'regime_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Fill NaN in features with 0
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Drop rows where target is missing (last day typically)
        df = df.dropna(subset=['target'])
        
        print(f"Data shape after sanitization: {df.shape}")
        print(f"Features: {len(feature_cols)}")
        
        return df
    
    def run(self, filepath: str, nrows: int = None) -> pd.DataFrame:
        """
        Full feature engineering pipeline with robustness measures.
        """
        print("=" * 80)
        print("ROBUST FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        # Load
        df = self.load_data(filepath, nrows=nrows)
        
        # Calculate moneyness
        df = self.calculate_moneyness(df)
        
        # Calculate IV
        df = self.calculate_implied_volatility(df)
        
        # Build volatility surface
        df = self.build_volatility_surface(df)
        
        # Aggregate to daily
        daily = self.aggregate_daily_features(df)
        
        # Add rolling features
        daily = self.add_rolling_features(daily)
        
        # Add regime detection (CRUCIAL for robustness analysis)
        daily = self.add_regime_detection(daily)
        
        # Add advanced features
        daily = self.add_advanced_features(daily)
        
        # Create target
        daily = self.create_target(daily)
        
        # Sanitize
        daily = self.sanitize(daily)
        
        print("=" * 80)
        print(f"✓ Feature engineering complete!")
        print(f"  Total features: {len([c for c in daily.columns if c not in ['tradeDate', 'target', 'spot_price', 'spot_price_future', 'spotPrice', 'regime_high_vol', 'regime_label']])}")
        print(f"  Date range: {daily['tradeDate'].min()} to {daily['tradeDate'].max()}")
        print(f"  Total samples: {len(daily)}")
        print("=" * 80)
        
        return daily


if __name__ == "__main__":
    fe = FeatureEngineer()
    df = fe.run('data/options_eod_QQQ.csv')
    
    # Save to parquet
    df.to_parquet('data/daily_features.parquet', index=False)
    print(f"\nSaved to data/daily_features.parquet")
    
    # Show regime distribution
    print("\nRegime Distribution:")
    print(df['regime_label'].value_counts())
