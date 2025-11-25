"""
Ensemble Alpha Model for QQQ Options Trading
Combines LightGBM, XGBoost, Random Forest, and Ridge regression.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class EnsembleAlphaModel:
    """
    Ensemble model combining multiple ML algorithms:
    - LightGBM (Gradient Boosting)
    - XGBoost (Gradient Boosting - different algorithm)
    - Random Forest (Bagging - stabilizer)
    - Ridge Regression (Linear anchor)
    
    Features:
    - Automatic feature selection
    - Z-score transformation for stationarity
    - Volatility-based position sizing
    - EMA smoothing for signal stability
    """
    
    def __init__(self, vol_target: float = 0.15, ema_alpha: float = 0.15):
        """
        Initialize ensemble model.
        
        Args:
            vol_target: Target volatility for position sizing (default 15%)
            ema_alpha: EMA smoothing factor for signals (default 0.15)
        """
        self.vol_target = vol_target
        self.ema_alpha = ema_alpha
        
        # Preprocessing artifacts
        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self.scaler = RobustScaler()  # Better for financial outliers
        self.selected_features = None
        self.selected_indices = None
        
        # Ensemble members
        self.models = {
            'lgbm': None,
            'xgb': None,
            'rf': None,
            'ridge': None
        }
        self.trained = False
    
    def _to_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Convert raw values to rolling Z-scores for stationarity.
        
        Args:
            series: Input series
            window: Rolling window size
            
        Returns:
            Z-score transformed series
        """
        roll_mean = series.rolling(window, min_periods=5).mean()
        roll_std = series.rolling(window, min_periods=5).std()
        zscore = (series - roll_mean) / (roll_std + 1e-8)
        return zscore.clip(-4, 4).fillna(0)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features with Z-score transformation.
        
        Args:
            df: DataFrame with raw features and 'target' column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        data = df.copy()
        
        # Create stationary features (Z-scores)
        if 'pcr_volume' in data.columns:
            data['z_pcr_vol'] = self._to_zscore(data['pcr_volume'])
        if 'pcr_otm' in data.columns:
            data['z_pcr_otm'] = self._to_zscore(data['pcr_otm'])
        if 'vrp' in data.columns:
            data['z_vrp'] = self._to_zscore(data['vrp'])
        if 'vol_skew_monthly' in data.columns:
            data['z_skew'] = self._to_zscore(data['vol_skew_monthly'])
        if 'gex' in data.columns:
            data['z_gex'] = self._to_zscore(data['gex'])
        if 'momentum_20d' in data.columns:
            data['z_mom'] = self._to_zscore(data['momentum_20d'])
        if 'rv_20d' in data.columns:
            data['z_vol'] = self._to_zscore(data['rv_20d'])
        
        # Collect Z-score features
        feature_cols = [c for c in data.columns if c.startswith('z_')]
        
        # Fallback to raw features if not enough Z-scores
        if len(feature_cols) < 3:
            exclude = ['tradeDate', 'target', 'spot_price', 'spotPrice', 
                      'spot_price_future', 'regime_high_vol', 'regime_label']
            feature_cols = [c for c in data.columns if c not in exclude]
        
        X = data[feature_cols].values
        y = data['target'].values
        
        return X, y, feature_cols
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]):
        """
        Train the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target returns
            feature_cols: Feature names
        """
        print(f"Training Ensemble on {X.shape[0]} samples...")
        
        # 1. Impute missing values
        X_clean = self.imputer.fit_transform(X)
        
        # 2. Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 3. Feature selection (top 15 features)
        n_features = min(15, X.shape[1])
        selector = SelectKBest(f_regression, k=n_features)
        X_sel = selector.fit_transform(X_scaled, y)
        self.selected_indices = selector.get_support(indices=True)
        
        self.selected_features = [feature_cols[i] for i in self.selected_indices]
        print(f"   Selected features: {self.selected_features}")
        
        # 4. Train models
        print("   -> Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'learning_rate': 0.01,
            'num_leaves': 16,
            'max_depth': 4,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'n_estimators': 500
        }
        train_ds = lgb.Dataset(X_sel, label=y)
        self.models['lgbm'] = lgb.train(lgb_params, train_ds)
        
        print("   -> Training XGBoost...")
        self.models['xgb'] = xgb.XGBRegressor(
            max_depth=3,
            learning_rate=0.01,
            n_estimators=500,
            reg_alpha=1.0,
            reg_lambda=5.0,
            n_jobs=1,
            random_state=42
        )
        self.models['xgb'].fit(X_sel, y)
        
        print("   -> Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            max_features='sqrt',
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        self.models['rf'].fit(X_sel, y)
        
        print("   -> Training Ridge Regression...")
        self.models['ridge'] = Ridge(alpha=10.0)
        self.models['ridge'].fit(X_sel, y)
        
        self.trained = True
        print("   ✓ Ensemble training complete!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted returns
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        X_sel = X_scaled[:, self.selected_indices]
        
        # Get predictions from each model
        p_lgb = self.models['lgbm'].predict(X_sel)
        p_xgb = self.models['xgb'].predict(X_sel)
        p_rf = self.models['rf'].predict(X_sel)
        p_ridge = self.models['ridge'].predict(X_sel)
        
        # Ensemble average (weights: 30%, 30%, 30%, 10%)
        predictions = (0.3 * p_lgb) + (0.3 * p_xgb) + (0.3 * p_rf) + (0.1 * p_ridge)
        
        return predictions
    
    def generate_signals(self, raw_predictions: np.ndarray, 
                        actual_returns: np.ndarray) -> np.ndarray:
        """
        Convert raw predictions into position signals with:
        - EMA smoothing
        - Volatility targeting
        - Regime filtering
        
        Args:
            raw_predictions: Predicted returns from model
            actual_returns: Actual historical returns (for vol calculation)
            
        Returns:
            Array of position signals (-1.0 to +1.5)
        """
        signals = np.zeros(len(raw_predictions))
        smoothed_pred = np.zeros(len(raw_predictions))
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(actual_returns).rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.fillna(0.15).values
        
        for i in range(1, len(raw_predictions)):
            # 1. EMA smoothing
            smoothed_pred[i] = (self.ema_alpha * raw_predictions[i] + 
                              (1 - self.ema_alpha) * smoothed_pred[i-1])
            
            # 2. Direction signal with threshold
            threshold = 0.0005  # 5 basis points
            if smoothed_pred[i] > threshold:
                dir_signal = 1.0
            elif smoothed_pred[i] < -threshold:
                dir_signal = -1.0
            else:
                dir_signal = 0.0
            
            # 3. Volatility targeting
            if rolling_vol[i] > 0:
                vol_scalar = self.vol_target / rolling_vol[i]
                vol_scalar = np.clip(vol_scalar, 0.5, 1.5)
            else:
                vol_scalar = 1.0
            
            # 4. Regime filter (kill switch)
            if rolling_vol[i] > 0.50:  # >50% vol: go to cash
                final_signal = 0.0
            elif rolling_vol[i] > 0.35:  # >35% vol: cut size in half
                final_signal = dir_signal * vol_scalar * 0.5
            else:
                final_signal = dir_signal * vol_scalar
            
            signals[i] = final_signal
        
        return signals

