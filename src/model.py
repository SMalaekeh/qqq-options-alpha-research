"""
LightGBM Regressor Model
------------------------
Adapted from High-Calmar Pipeline.
Predicts Returns directly (Regression) instead of Direction (Classification).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class RobustAlphaModel:
    def __init__(self, vol_target: float = 0.15, ema_alpha: float = 0.10):
        self.vol_target = vol_target
        self.ema_alpha = ema_alpha
        self.model = None
        
        # Preprocessing artifacts
        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self.scaler = StandardScaler()
        self.selector = None # Initialized in fit
        self.selected_indices = None
        self.trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Standard feature preparation.
        """
        # Exclude non-feature columns
        exclude_cols = ['tradeDate', 'target', 'spot_price', 'spot_price_future', 'spotPrice',
                       'regime_high_vol', 'regime_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y_returns = df['target'].values
        y_binary = (y_returns > 0).astype(int)
        
        return X, y_binary, y_returns, feature_cols

    def fit(self, X: np.ndarray, y_binary: np.ndarray, feature_cols: List[str], y_returns: np.ndarray = None):
        """
        Train LightGBM Regressor. 
        NOTE: We need y_returns here because this is a regression model!
        """
        print(f"Training LightGBM Regressor on {X.shape[0]} samples...")
        
        if y_returns is None:
            raise ValueError("LightGBM Regressor requires y_returns (actual returns) for training.")

        # 1. Impute
        X_clean = self.imputer.fit_transform(X)
        
        # 2. Feature Selection (Using F-classif on binary target as per your snippet)
        # Select Top 20 features
        n_features = 20
        self.selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        X_selected = self.selector.fit_transform(X_clean, y_binary)
        self.selected_indices = self.selector.get_support(indices=True)
        
        selected_names = [feature_cols[i] for i in self.selected_indices]
        print(f"   Selected features: {selected_names[:5]}...")

        # 3. Scale
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 4. Train LightGBM (Exact params from your snippet)
        params = {
            'objective': 'regression', # predicting returns, not classes
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_samples': 20,
            'max_depth': 8,
            'random_state': 42,
            'verbosity': -1,
        }
        
        train_data = lgb.Dataset(X_scaled, label=y_returns)
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=400
        )
        
        self.trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts returns, then normalizes them to a 0-1 range to mimic probability.
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")
            
        # 1. Preprocess
        X_clean = self.imputer.transform(X)
        X_selected = X_clean[:, self.selected_indices]
        X_scaled = self.scaler.transform(X_selected)
        
        # 2. Predict (Raw Returns)
        pred_returns = self.model.predict(X_scaled)
        
        # 3. Convert to Pseudo-Probability (0 to 1)
        # This normalization technique helps the signal generator work with regression outputs
        # We handle division by zero and clip to ensure stability
        _min = pred_returns.min()
        _max = pred_returns.max()
        denom = _max - _min
        
        if denom == 0:
            prob = np.ones_like(pred_returns) * 0.5
        else:
            prob = (pred_returns - _min) / denom
            
        # Clip to ensure bounds
        prob = np.clip(prob, 0, 1)
        
        # Stack to match [p_down, p_up] format expected by pipeline
        return np.column_stack((1-prob, prob))
    
    def generate_signals(self, proba_up: np.ndarray, returns: np.ndarray) -> np.ndarray:
        signals = np.zeros(len(proba_up))
        smoothed_proba = np.zeros(len(proba_up))
        smoothed_proba[0] = 0.5
        
        # Volatility for sizing
        rolling_vol = pd.Series(returns).rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.fillna(0.15).values

        for i in range(1, len(proba_up)):
            # EMA Smoothing
            smoothed_proba[i] = self.ema_alpha * proba_up[i] + (1 - self.ema_alpha) * smoothed_proba[i-1]
            
            # Thresholds (Optimized for Normalized Regression Output)
            # Since we normalized min/max to 0-1, 0.5 is the midpoint of predicted return range
            if smoothed_proba[i] > 0.55: 
                raw_signal = 1.0
            elif smoothed_proba[i] < 0.45:
                raw_signal = -1.0
            else:
                raw_signal = 0.0
            
            # Vol Scaling
            if rolling_vol[i] > 0:
                vol_scalar = self.vol_target / rolling_vol[i]
                vol_scalar = np.clip(vol_scalar, 0.5, 1.5)
            else:
                vol_scalar = 1.0
            
            signals[i] = raw_signal * vol_scalar
            
        return signals

    def evaluate(self, y_binary, proba_up, y_returns, signals):
        try:
            auc = roc_auc_score(y_binary, proba_up)
        except:
            auc = 0.5
        
        portfolio_returns = signals * y_returns
        valid_idx = ~np.isnan(portfolio_returns) & (signals != 0)
        clean_ret = portfolio_returns[valid_idx]
        
        if len(clean_ret) == 0:
            return {'auc': auc, 'sharpe': 0, 'calmar': 0, 'total_return': 0, 'max_drawdown': 0}
            
        sharpe = np.mean(clean_ret) / (np.std(clean_ret) + 1e-8) * np.sqrt(252)
        cum_ret = np.cumprod(1 + clean_ret)
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak
        max_dd = np.min(dd)
        calmar = (np.mean(clean_ret) * 252) / abs(max_dd + 1e-8)
        
        return {
            'auc': auc, 'sharpe': sharpe, 'calmar': calmar, 
            'total_return': cum_ret[-1]-1 if len(cum_ret)>0 else 0, 
            'max_drawdown': max_dd
        }