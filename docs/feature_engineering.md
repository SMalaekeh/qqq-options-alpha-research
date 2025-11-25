# Feature Engineering Documentation

This document describes all features generated from QQQ options data, organized by category.

## Overview

We transform raw options data (5M+ rows) into **100+ daily features** representing:
- Volatility surface dynamics
- Dealer positioning (Greeks)
- Market sentiment (flows)
- Term structure & skew
- Momentum & realized volatility

**Anti-Overfitting Measures:**
- No forward-looking data
- All features use only past information
- Outlier clipping (winsorization)
- Regime detection for stratified analysis

---

## Feature Categories

### 1. Volatility Surface (20 features)

**Concept:** Implied Volatility (IV) varies by **moneyness** (strike/spot) and **tenor** (time to expiration). We create a grid capturing this entire surface.

**Moneyness Buckets:**
- `deep_otm_put`: 0.0 - 0.90 (far out-of-the-money puts)
- `otm_put`: 0.90 - 0.97 (out-of-the-money puts)
- `atm`: 0.97 - 1.03 (at-the-money)
- `otm_call`: 1.03 - 1.10 (out-of-the-money calls)
- `deep_otm_call`: 1.10 - 2.0 (far out-of-the-money calls)

**Tenor Buckets:**
- `weekly`: 0-10 days to expiration
- `monthly`: 10-45 days to expiration
- `quarterly`: 45-90 days to expiration
- `long`: 90+ days to expiration

**Feature Format:** `iv_{moneyness}_{tenor}`

**Examples:**
- `iv_atm_monthly`: IV for ATM options with 10-45 DTE (most liquid, representative)
- `iv_otm_put_weekly`: IV for OTM puts expiring in 0-10 days (hedging activity)
- `iv_deep_otm_call_monthly`: IV for deep OTM calls (tail risk speculation)

**Calculation:**
- Volume-weighted IV within each bucket
- Aggregated to daily level
- Missing values filled with 0 (no activity in that bucket)

**Why These Buckets?**
- **ATM options** are most liquid → best price discovery
- **OTM puts** reveal hedging demand
- **OTM calls** reveal speculative demand
- **Weekly vs. Monthly** separates short-term vs. long-term views

---

### 2. Gamma Exposure (GEX) Features (14 features)

**Core Features:**
- `gex_call`: Total gamma from call options weighted by OI
- `gex_put`: Total gamma from put options weighted by OI
- `gex_raw`: Net gamma (calls - puts)
- `gex`: Normalized and winsorized net gamma (Z-scored)
- `gex_atm`: Gamma exposure for ATM options only (key dealer level)
- `gex_atm_normalized`: ATM gamma normalized by notional

**Rolling Averages:**
- `gex_ma_5d`, `gex_ma_10d`, `gex_ma_20d`, `gex_ma_60d`
- Smoothed GEX over different windows

**Interaction Terms:**
- `gex_x_momentum`: GEX × 20-day momentum
- `gex_x_rvol`: GEX × 20-day realized vol
- `gex_zscore`: Rolling Z-score of GEX (60-day window)

**Calculation Details:**
```python
gex_call = Σ (gamma × call_OI × strike × 100)
gex_put = Σ (gamma × put_OI × strike × 100)
gex_raw = gex_call - gex_put

# Aggressive outlier protection
gex_clipped = clip(gex_raw, p05, p95)  # 5th-95th percentile
gex_normalized = gex_clipped / (spot_price^2 × 100 + 1e6)
gex = zscore(gex_normalized)  # Z-score for interpretability
gex = clip(gex, -5, 5)  # Final safety clip
```

**Why This Normalization?**
- Raw GEX grows with QQQ price (higher notional)
- Normalization by `spot^2` accounts for this
- Z-scoring makes values interpretable (-3 to +3 range)
- Multiple clipping layers prevent extreme outliers

---

### 3. Variance Risk Premium (VRP) Features (3 features)

**Core Feature:**
- `vrp`: IV_ATM_Monthly - RV_20d

**Interaction Terms:**
- `vrp_x_skew`: VRP × Volatility Skew
- `vrp_zscore`: Rolling Z-score of VRP

**Calculation:**
```python
# Only calculate VRP when both IV and RV are valid (non-zero)
iv_valid = (iv_atm_monthly > 0.01) & (iv_atm_monthly is not NaN)
rv_valid = (rv_20d > 0.01) & (rv_20d is not NaN)

vrp = where(iv_valid & rv_valid, 
            iv_atm_monthly - rv_20d, 
            NaN)
vrp = clip(vrp, -0.5, 0.5)  # Reasonable range
```

**Why This Fix?**
- **Original Issue:** Early in time series, rolling features have NaN values
- When NaN filled with 0, VRP = 0 - 0 = 0 (spurious zeros)
- **New Approach:** Only calculate VRP when both components are valid
- Results in cleaner signal, no artificial flat periods

**Interpretation:**
- VRP > 0.1: High insurance premium (overpriced volatility)
- VRP near 0: Fair pricing
- VRP < -0.1: Underpriced volatility (danger signal)

---

### 4. Volatility Skew Features (4 features)

**Core Features:**
- `vol_skew_monthly`: IV_OTM_Put_Monthly - IV_OTM_Call_Monthly
- `vol_skew_deep`: IV_Deep_OTM_Put_Monthly - IV_Deep_OTM_Call_Monthly

**Interaction Term:**
- `vrp_x_skew`: VRP × Skew (combining two vol signals)
- `vol_skew_monthly_zscore`: Rolling Z-score

**Calculation:**
```python
vol_skew = iv_otm_put - iv_otm_call
vol_skew = clip(vol_skew, -0.3, 0.3)  # Typical range: -10% to +30%
```

**Why Two Skew Measures?**
- **OTM Skew:** Near-the-money, high liquidity (hedging)
- **Deep OTM Skew:** Tail risk pricing, less liquid (tail hedging)

**Typical Values:**
- **Positive Skew (0.05 to 0.15):** Normal market (puts more expensive)
- **Steep Skew (>0.20):** High fear, peak hedging demand
- **Flat Skew (<0.05):** Complacency, low hedging

---

### 5. Put/Call Ratio (PCR) Features (9 features)

**Core Ratios:**
- `pcr_volume`: Total put volume / Total call volume
- `pcr_oi`: Total put OI / Total call OI
- `pcr_otm`: OTM put volume / OTM call volume

**Rolling Averages:**
- `pcr_volume_ma_5d`, `pcr_volume_ma_10d`, `pcr_volume_ma_20d`, `pcr_volume_ma_60d`

**Interaction Terms:**
- `pcr_x_rvol`: PCR × Realized volatility
- `pcr_volume_zscore`: Rolling Z-score

**Calculation:**
```python
pcr_volume = total_put_volume / (total_call_volume + 1)  # +1 prevents div/0
pcr_oi = total_put_oi / (total_call_oi + 1)
pcr_otm = otm_put_volume / (otm_call_volume + 1)
```

**Why Three PCR Versions?**
- **Volume:** Intraday activity (faster signal, more noise)
- **Open Interest:** Accumulated positioning (slower signal, more stable)
- **OTM:** Speculative/hedging extremes (best for reversals)

**Typical Values:**
- PCR < 0.7: Greed (more calls than puts)
- PCR 0.7-1.3: Neutral
- PCR > 1.3: Fear (more puts than calls)
- PCR > 2.0: Panic (contrarian buy signal)

---

### 6. Greeks & Flows (10 features)

**Delta Exposure:**
- `total_delta_call`: Aggregate delta from all call options
- `total_delta_put`: Aggregate delta from all put options
- `net_delta_flow`: Delta_call - Delta_put (directional bias)
- `delta_flow_x_momentum`: Delta flow × Momentum (interaction)

**Vega Exposure:**
- `total_vega_call`: Aggregate vega from calls
- `total_vega_put`: Aggregate vega from puts
- `vega_ratio`: Vega_put / Vega_call (vol sensitivity bias)

**Volume Metrics:**
- `volume_term_ratio`: Weekly volume / Monthly volume (short-term vs. long-term)
- `avg_call_premium`: Dollar volume / Contract volume (average premium paid)
- `avg_put_premium`: Dollar volume / Contract volume

**Calculation:**
```python
delta_call = Σ (delta × call_OI)
delta_put = Σ (delta × put_OI)
net_delta_flow = delta_call - delta_put
```

**Interpretation:**
- **High net delta call exposure:** Bullish positioning
- **High vega put/call ratio:** More vol sensitivity on downside (bearish)
- **High avg premiums:** Willingness to pay up (conviction)

---

### 7. Momentum Features (6 features)

**Core Features:**
- `momentum_5d`, `momentum_10d`, `momentum_20d`, `momentum_60d`
- Cumulative return over X days

**Interaction Terms:**
- `gex_x_momentum`: Gamma exposure × Momentum
- `delta_flow_x_momentum`: Delta flow × Momentum

**Calculation:**
```python
momentum_Xd = rolling_sum(daily_returns, window=X)
# Clip extreme values (1st-99th percentile) to prevent outliers
momentum_Xd = clip(momentum_Xd, p01, p99)
```

**Why Multiple Windows?**
- **5d:** Very short-term trend
- **10d:** Short-term trend
- **20d:** Medium-term trend (monthly)
- **60d:** Long-term trend (quarterly)

Model can weight different horizons based on predictive power.

---

### 8. Realized Volatility Features (4 features)

**Core Features:**
- `rv_5d`, `rv_10d`, `rv_20d`, `rv_60d`
- Standard deviation of returns (annualized)

**Calculation:**
```python
rv_Xd = rolling_std(daily_returns, window=X) × sqrt(252)
```

**Interpretation:**
- RV < 10%: Extremely calm market
- RV 10-20%: Normal market
- RV 20-30%: Elevated volatility
- RV > 30%: High stress (crisis mode)

**Why These Windows?**
- **5d:** Captures very recent volatility spikes
- **20d:** Standard "monthly" volatility (used in VRP)
- **60d:** Longer-term regime (used in regime detection)

---

### 9. Term Structure Features (2 features)

**Core Features:**
- `term_structure_atm`: IV_ATM_Monthly - IV_ATM_Weekly
- `term_structure_long`: IV_ATM_Long - IV_ATM_Monthly

**Calculation:**
```python
term_structure_atm = iv_atm_monthly - iv_atm_weekly
term_structure_atm = clip(term_structure_atm, -0.3, 0.3)
```

**Interpretation:**
- **Positive (Contango):** Normal market (longer-dated options more expensive)
- **Flat:** Uncertainty across all horizons
- **Negative (Backwardation):** Crisis (near-term vol expected to spike)

**Why It Matters:**
- **Steep contango:** Complacency (VIX carry trades profitable)
- **Backwardation:** Stress (market pricing near-term event)

---

### 10. Volume by Moneyness & Tenor (6 features)

**Moneyness:**
- `otm_put_volume`: Volume in OTM puts (hedging)
- `otm_call_volume`: Volume in OTM calls (speculation)

**Tenor:**
- `volume_weekly`: Total volume in 0-10 DTE (short-term speculation)
- `volume_monthly`: Total volume in 10-45 DTE (standard options)

**Dollar Volume:**
- `call_dollar_volume`: Premium paid for calls
- `put_dollar_volume`: Premium paid for puts

**Why Track This?**
- **OTM volume** reveals speculative vs. hedging flows
- **Weekly volume** spikes during events (earnings, FOMC)
- **Dollar volume** shows conviction (large traders pay more)

---

### 11. Interaction Terms (5 features)

**Why Interactions?**
Single features may not capture non-linear relationships. Interactions allow model to learn:
- "GEX only matters when momentum is strong"
- "VRP signal is stronger when skew is steep"

**Implemented Interactions:**
1. `gex_x_momentum`: GEX × 20d momentum
   - High GEX caps momentum (resistance)
   
2. `vrp_x_skew`: VRP × Vol skew
   - High VRP + steep skew = maximum fear
   
3. `pcr_x_rvol`: PCR × Realized vol
   - High PCR + high vol = panic selling
   
4. `delta_flow_x_momentum`: Delta flow × 10d momentum
   - Flow aligned with momentum = persistence
   
5. `gex_x_rvol`: GEX × Realized vol
   - GEX effect varies by volatility regime

**Calculation:**
```python
interaction = feature_A × feature_B
interaction = clip(interaction, p01, p99)  # Outlier control
```

---

### 12. Rolling Z-Scores (4 features)

**Purpose:** Convert features to **relative** values (stationarity).

**Features:**
- `gex_zscore`
- `pcr_volume_zscore`
- `vrp_zscore`
- `vol_skew_monthly_zscore`

**Calculation:**
```python
z_score = (value - rolling_mean(value, 60)) / (rolling_std(value, 60) + 1e-8)
z_score = clip(z_score, -3, 3)  # 3-sigma bounds
```

**Why This Helps:**
- Raw GEX of "100" means nothing (depends on QQQ price level)
- Z-score of "+2" means "2 standard deviations above 60-day average" (interpretable)
- Helps model generalize to new market regimes

---

### 13. Regime Detection (2 features)

**Purpose:** Tag each day as High Vol or Low Vol regime for stratified analysis.

**Features:**
- `regime_high_vol`: Binary flag (0 = Low Vol, 1 = High Vol)
- `regime_label`: Text label ('Low Vol' or 'High Vol')

**Calculation:**
```python
rv_ma_60 = rolling_mean(rv_20d, 60)
regime_high_vol = (rv_20d > rv_ma_60).astype(int)
```

**Interpretation:**
- **High Vol Regime:** RV above 60-day average
  - Strategies: Mean reversion, reduce leverage
  
- **Low Vol Regime:** RV below 60-day average
  - Strategies: Trend-following, increase leverage

**Why 60-day MA?**
- Captures "normal" volatility for recent period
- Responsive enough to catch regime shifts
- Not so fast that it whipsaws

---

## Feature Selection Strategy

**Total Features Generated:** ~100

**Used by Model:** 15 (top features selected via F-statistics)

**Why Select Features?**
- Reduces overfitting (curse of dimensionality)
- Faster training
- More interpretable results

**Selection Method:**
- F-regression (correlation with target returns)
- SelectKBest from scikit-learn
- Performed **after** train/test split (no data leakage)

**Typical Selected Features:**
1. Z-scored sentiment features (PCR, GEX)
2. VRP and volatility features
3. Momentum indicators
4. Key interaction terms

---

## Anti-Overfitting Safeguards

1. **No Look-Ahead Bias:**
   - All rolling calculations use `min_periods` parameter
   - No data from future used in past calculations

2. **Outlier Protection:**
   - Winsorization (5th-95th percentile clipping)
   - Z-score bounds (-3 to +3)
   - Inf/NaN replacement

3. **Regime Awareness:**
   - Features labeled by volatility regime
   - Model can learn regime-dependent patterns

4. **Proper Scaling:**
   - No global scaling before train/test split
   - Scaling fit on train set only, applied to test

5. **Feature Validation:**
   - Sanity checks (no values > 100% IV, etc.)
   - Missing data handling (forward fill where appropriate)

---

## Feature Engineering Pipeline Summary

**Input:** Raw options data (5M+ rows)
- Columns: tradeDate, spotPrice, strike, dte, Greeks, volume, OI, etc.

**Output:** Daily feature matrix (1400+ rows × 100 columns)
- Each row = one trading day
- Each column = one feature
- Target = next-day return

**Process:**
1. Calculate moneyness (strike/spot)
2. Estimate implied volatility from Greeks
3. Aggregate IV by moneyness × tenor buckets
4. Calculate Greeks & GEX (daily aggregation)
5. Add rolling features (momentum, RV)
6. Add advanced features (VRP, skew, term structure)
7. Add interaction terms
8. Add regime detection
9. Sanitize (handle NaN, Inf, outliers)

**Runtime:** ~2-5 minutes for 5M rows (single-threaded Python)

---

## Conclusion

This feature engineering approach balances:
- **Comprehensiveness:** 100+ features covering all aspects of options market
- **Robustness:** Extensive outlier handling and validation
- **Interpretability:** Features map to financial concepts (GEX, VRP, skew)
- **Anti-Overfitting:** Proper temporal structure, no look-ahead bias

The result is a clean, predictive feature set ready for machine learning.

