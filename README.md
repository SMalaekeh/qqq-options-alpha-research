# QQQ Options Alpha Research

> **Forecasting QQQ with its Own Options Data: A Ensemble Machine Learning Approach**

This repository contains a robust, production-ready trading strategy that uses end-of-day QQQ options data to forecast next-day directional movement and generate daily trading signals with leverage between -1.0x and +1.5x.

## 🎯 Objective

Design a model that systematically deciphers **sentiment, risk appetite, and positioning** embedded within the QQQ options market to gain an edge on future price action.

**Key Performance Target:**
- **Calmar Ratio > 2.0** (Risk-adjusted returns)
- **Robustness:** Strategy stable to ±10% parameter variations
- **Leverage Range:** -1.0x (full short) to +1.5x (leveraged long)

## 📊 Results Summary

### Test Set Performance (2024-07-26 to 2025-09-17)

| Metric | Value |
|--------|-------|
| **Calmar Ratio** | **2.14** ✅ |
| **Sharpe Ratio** | **1.92** |
| Total Return | 23.5% |
| Max Drawdown | -13.7% |
| Win Rate | ~55% |

**Robustness Check:** Strategy maintains Calmar > 1.5 across all parameter variations (±10%).

## 🏗️ Repository Structure

```
qqq-options-alpha-research/
├── data/
│   ├── options_eod_QQQ.csv          # Raw options data (>5M rows)
│   └── daily_features.parquet        # Preprocessed features (generated)
├── notebooks/
│   ├── eda.ipynb                     # Feature engineering & EDA (NEW - Modular)
│   └── model.ipynb                   # Model training & evaluation (NEW - Modular)

├── src/
│   ├── feature_engineering.py        # Robust feature generation (100+ features)
│   ├── ensemble_model.py             # Ensemble ML model (LightGBM + XGBoost + RF + Ridge)
│   ├── visualization.py              # Plotting utilities (NEW)
│   └── backtesting.py                # Performance metrics & robustness testing (NEW)
├── docs/
│   ├── strategy_logic.md             # Detailed strategy rationale
│   ├── feature_engineering.md        # Feature descriptions
│   └── methodology.md                # Model methodology & assumptions
├── outputs/                          # Generated results & plots
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd qqq-options-alpha-research

# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

### 2. Run Feature Engineering

```bash
# Open and run eda.ipynb
jupyter notebook notebooks/eda.ipynb
```

This will:
- Load raw options data
- Generate 100+ features including:
  - Volatility surface (IV by moneyness × tenor)
  - Greeks & GEX (Gamma Exposure)
  - Variance Risk Premium (VRP)
  - Put/Call ratios & flow metrics
  - Regime detection (High/Low volatility)
- Save features to `data/daily_features.parquet`

### 3. Train & Evaluate Model

```bash
# Open and run model.ipynb
jupyter notebook notebooks/model.ipynb
```

This will:
- Train ensemble model (LightGBM + XGBoost + RF + Ridge)
- Generate trading signals with volatility targeting
- Evaluate performance (Sharpe, Calmar, drawdown)
- Run robustness analysis

## 📈 Strategy Overview

### Core Philosophy

The options market is a **sentiment barometer** and **positioning indicator**. Large institutional traders must hedge their positions, creating predictable flows. By analyzing these flows and implied volatility dynamics, we can forecast short-term QQQ movements.

### Key Signals

1. **Variance Risk Premium (VRP)**
   - Spread between implied volatility and realized volatility
   - High VRP → Market overpricing risk → Mean reversion opportunity

2. **Gamma Exposure (GEX)**
   - Measures dealer hedging needs
   - High GEX → Price suppression (dealers hedging)
   - Low/Negative GEX → Increased volatility

3. **Put/Call Ratios**
   - Sentiment indicator (fear vs. greed)
   - Elevated put buying → Potential reversal signal

4. **Volatility Skew**
   - Difference between OTM put and call IVs
   - Steepening skew → Rising hedging demand → Bearish signal

### Model Architecture

**Ensemble Approach:**
- **LightGBM** (30%): Fast gradient boosting for feature interactions
- **XGBoost** (30%): Robust gradient boosting with regularization
- **Random Forest** (30%): Bagging for stability
- **Ridge Regression** (10%): Linear anchor to prevent overfitting

**Signal Generation:**
1. Raw predictions → EMA smoothing (reduce noise)
2. Volatility targeting (scale positions by realized vol)
3. Regime filter (reduce/eliminate positions during crises)

**Risk Management:**
- Leverage capped at -1.0x to +1.5x
- Kill switch: Go to cash if vol > 50% (crisis mode)
- Reduce size by 50% if vol > 35% (elevated risk)

## 📚 Documentation

Detailed documentation in `/docs/`:

- **[Strategy Logic](docs/strategy_logic.md)**: Why the model works (financial rationale)
- **[Feature Engineering](docs/feature_engineering.md)**: Description of all 100+ features
- **[Methodology](docs/methodology.md)**: Model training, validation, and robustness testing

## 🔬 Robustness & Anti-Overfitting Measures

1. **Chronological Splitting**: 60/20/20 train/val/test (no look-ahead)
2. **Rolling Z-scores**: Features converted to relative values (stationarity)
3. **Feature Selection**: Top 15 features selected via F-statistics
4. **Regime Detection**: Model aware of market volatility state
5. **Outlier Clipping**: All features winsorized to prevent extreme values
6. **Parameter Stability**: Calmar > 1.5 across all tested variations

## 📊 Key Features Generated

### Volatility Surface (20 features)
- IV by moneyness: Deep OTM Put, OTM Put, ATM, OTM Call, Deep OTM Call
- IV by tenor: Weekly (0-10d), Monthly (10-45d), Quarterly (45-90d), Long (90d+)

### Greeks & Positioning (15 features)
- GEX (Gamma Exposure): Total, ATM, Call, Put
- Vega exposure by call/put
- Delta exposure & net flow

### Sentiment & Flow (10 features)
- Put/Call ratios: Volume, OI, OTM
- Volume by moneyness and tenor
- Dollar volume flows

### Momentum & Volatility (8 features)
- Realized volatility: 5d, 10d, 20d, 60d
- Price momentum: 5d, 10d, 20d, 60d

### Advanced (20+ features)
- VRP (Variance Risk Premium)
- Volatility skew (put premium over calls)
- Term structure (short-term vs. long-term IV)
- Interaction terms (GEX × momentum, VRP × skew, etc.)

## 🎓 References & Inspiration

- **Variance Risk Premium**: Carr & Wu (2009), Bollerslev et al. (2009)
- **Gamma Exposure**: SqueezeMetrics, SpotGamma research
- **Volatility Skew**: Bates (1991), Rubinstein (1994)
- **Options Market Microstructure**: Gârleanu, Pedersen, Poteshman (2009)

## 🛠️ Technical Details

**Dependencies:**
- Python 3.8+
- pandas, numpy, scikit-learn
- lightgbm, xgboost
- matplotlib, seaborn

**Data Requirements:**
- QQQ end-of-day options data (strike, IV, Greeks, volume, OI)
- Minimum 2+ years of history for proper training

**Computational Requirements:**
- Feature engineering: ~2-5 minutes (5M rows)
- Model training: ~1-2 minutes
- Can run on laptop (no GPU required)

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

Developed as part of the Quanta Options Big Data Challenge. Special thanks to the quantitative finance community for open research on options market microstructure.

---

**Disclaimer:** This is a research project. Past performance does not guarantee future results. Trade at your own risk.

