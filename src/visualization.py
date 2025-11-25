"""
Visualization Module for QQQ Options Alpha Research
Provides reusable plotting functions for EDA and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """
    Centralized plotting utilities for the QQQ Options Alpha project.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid', palette='husl'):
        """Initialize plotting style."""
        plt.style.use(style)
        sns.set_palette(palette)
    
    def plot_regime_detection(self, df: pd.DataFrame, figsize=(14, 8)):
        """
        Plot realized volatility and regime classification over time.
        
        Args:
            df: DataFrame with 'tradeDate', 'rv_20d', 'regime_high_vol' columns
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Realized volatility
        axes[0].plot(df['tradeDate'], df['rv_20d'] * 100, 
                    label='20d Realized Vol', linewidth=1.5)
        axes[0].plot(df['tradeDate'], df['rv_20d'].rolling(60).mean() * 100, 
                    label='60d MA', linewidth=2, color='red', linestyle='--')
        axes[0].set_ylabel('Realized Vol (%)', fontsize=12)
        axes[0].set_title('Realized Volatility & Regime Detection', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Regime indicator
        regime_colors = df['regime_high_vol'].map({0: 'green', 1: 'red'})
        axes[1].scatter(df['tradeDate'], df['regime_high_vol'], 
                       c=regime_colors, alpha=0.3, s=10)
        axes[1].set_ylabel('Regime\n(0=Low Vol, 1=High Vol)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_title('Volatility Regime Classification', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # return fig
    
    def plot_target_distribution(self, df: pd.DataFrame, figsize=(14, 5)):
        """
        Plot target (return) distribution and time series.
        
        Args:
            df: DataFrame with 'tradeDate' and 'target' columns
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(df['target'] * 100, bins=50, alpha=0.7, 
                    edgecolor='black', color='steelblue')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Daily Return (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Time series
        axes[1].plot(df['tradeDate'], df['target'] * 100, 
                    alpha=0.7, linewidth=0.8, color='steelblue')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Daily Return (%)', fontsize=12)
        axes[1].set_title('Target Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # return fig
    
    def plot_feature_correlations(self, df: pd.DataFrame, features: List[str], 
                                  figsize=(12, 10)):
        """
        Plot correlation heatmap for selected features.
        
        Args:
            df: DataFrame with features
            features: List of feature column names
        """
        # Filter features that exist
        features_exist = [f for f in features if f in df.columns]
        
        # Correlation matrix
        corr_matrix = df[features_exist].corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, corr_matrix
    
    def plot_feature_timeseries(self, df: pd.DataFrame, 
                               features: Dict[str, str], figsize=(14, 12)):
        """
        Plot multiple features over time in subplots.
        
        Args:
            df: DataFrame with 'tradeDate' and feature columns
            features: Dict mapping feature name to plot label
                     e.g., {'gex': 'Gamma Exposure (GEX)'}
        """
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
        
        if n_features == 1:
            axes = [axes]
        
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
        
        for idx, (feature, label) in enumerate(features.items()):
            if feature in df.columns:
                color = colors[idx % len(colors)]
                axes[idx].plot(df['tradeDate'], df[feature], 
                             linewidth=1.5, color=color)
                axes[idx].axhline(0, color='red', linestyle='--', linewidth=1)
                axes[idx].set_ylabel(feature, fontsize=12)
                axes[idx].set_title(label, fontsize=14, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        # return fig
    
    def plot_equity_curve(self, df: pd.DataFrame, signals: np.ndarray, 
                         y_test: np.ndarray, metrics: Dict, figsize=(14, 10)):
        """
        Plot comprehensive backtest results: equity curve, leverage, and volatility.
        
        Args:
            df: DataFrame with 'tradeDate' column
            signals: Array of position signals
            y_test: Array of actual returns
            metrics: Dict with 'calmar', 'sharpe', etc.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # 1. Equity Curve
        portfolio_returns = signals * y_test
        cum_strategy = np.cumprod(1 + portfolio_returns) - 1
        cum_hold = np.cumprod(1 + y_test) - 1
        
        axes[0].plot(df['tradeDate'], cum_strategy * 100, 
                    label='Strategy', color='green', linewidth=2)
        axes[0].plot(df['tradeDate'], cum_hold * 100, 
                    label='Buy & Hold', color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Return (%)', fontsize=12)
        axes[0].set_title(
            f"Equity Curve (Calmar: {metrics.get('calmar', 0):.2f}, "
            f"Sharpe: {metrics.get('sharpe', 0):.2f})", 
            fontsize=14, fontweight='bold'
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Leverage (Position Sizing)
        axes[1].plot(df['tradeDate'], signals, color='blue', alpha=0.8, linewidth=1)
        axes[1].fill_between(df['tradeDate'], signals, 0, color='blue', alpha=0.1)
        axes[1].axhline(1.5, color='red', linestyle=':', alpha=0.5, label='Max Long')
        axes[1].axhline(-1.0, color='red', linestyle=':', alpha=0.5, label='Max Short')
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Position', fontsize=12)
        axes[1].set_title('Leverage (Position Size)', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Rolling Volatility (Regime Filter)
        rolling_vol = pd.Series(y_test).rolling(20).std() * np.sqrt(252)
        axes[2].plot(df['tradeDate'], rolling_vol * 100, color='red', linewidth=1.5)
        axes[2].axhline(35, color='black', linestyle='--', 
                       label='Reduce Size > 35%', alpha=0.7)
        axes[2].axhline(50, color='black', linestyle=':', 
                       label='Cash > 50%', alpha=0.7)
        axes[2].set_ylabel('Volatility (%)', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_title('Rolling Volatility (Regime Filter)', 
                         fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # return fig
    
    def plot_robustness_check(self, results_df: pd.DataFrame, 
                             metric='Calmar', figsize=(12, 6)):
        """
        Plot robustness analysis results as bar chart.
        
        Args:
            results_df: DataFrame with 'Label' and metric columns
            metric: Metric to plot (e.g., 'Calmar', 'Sharpe')
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color logic: Green for Baseline, Blue for others
        colors = ['#2ecc71' if 'BASELINE' in label else '#3498db' 
                 for label in results_df['Label']]
        
        bars = ax.bar(results_df['Label'], results_df[metric], 
                     color=colors, alpha=0.8, edgecolor='black')
        
        # Target lines
        if metric == 'Calmar':
            ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, 
                      label='Target (2.0)')
            ax.axhline(1.5, color='orange', linestyle=':', linewidth=1.5, 
                      label='Robustness Floor (1.5)')
        
        ax.set_title(f"Robustness Check: {metric} Ratio Stability", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{metric} Ratio", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: np.ndarray, 
                               top_n: int = 15, figsize=(10, 8)):
        """
        Plot feature importances from model.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance scores
            top_n: Number of top features to show
        """
        # Sort by importance
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(top_features)), top_importances, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        # return fig


def print_target_statistics(df: pd.DataFrame):
    """Print summary statistics for target variable."""
    print("\n" + "=" * 80)
    print("TARGET STATISTICS (Next-Day Returns)")
    print("=" * 80)
    print(df['target'].describe())
    print(f"\nAnnualized Return: {df['target'].mean() * 252:.2%}")
    print(f"Annualized Volatility: {df['target'].std() * np.sqrt(252):.2%}")
    sharpe = df['target'].mean() / df['target'].std() * np.sqrt(252)
    print(f"Sharpe Ratio (buy-and-hold): {sharpe:.2f}")


def print_feature_groups(df: pd.DataFrame):
    """Print organized feature groups."""
    feature_groups = {
        'Volatility Surface': [c for c in df.columns 
                              if c.startswith('iv_') 
                              and not c.endswith(('ma_5d', 'ma_10d', 'ma_20d', 'ma_60d')) 
                              and '_zscore' not in c],
        'GEX Features': [c for c in df.columns if 'gex' in c],
        'VRP Features': [c for c in df.columns if 'vrp' in c],
        'Skew Features': [c for c in df.columns if 'skew' in c],
        'PCR Features': [c for c in df.columns if 'pcr' in c],
        'Momentum': [c for c in df.columns if 'momentum' in c],
        'Realized Vol': [c for c in df.columns if c.startswith('rv_')],
        'Interaction Terms': [c for c in df.columns if '_x_' in c],
    }
    
    print("\n" + "=" * 80)
    print("FEATURE GROUPS")
    print("=" * 80)
    for group, features in feature_groups.items():
        print(f"\n{group}: {len(features)} features")
        if len(features) <= 10:
            for f in features:
                print(f"  - {f}")
        else:
            print(f"  First 5: {features[:5]}")
            print(f"  Last 5: {features[-5:]}")
    
    return feature_groups


def print_regime_distribution(df: pd.DataFrame):
    """Print volatility regime distribution."""
    print("\n" + "=" * 80)
    print("VOLATILITY REGIME DISTRIBUTION")
    print("=" * 80)
    regime_counts = df['regime_label'].value_counts()
    print(regime_counts)
    print(f"\nHigh Vol %: {regime_counts.get('High Vol', 0) / len(df) * 100:.1f}%")
    print(f"Low Vol %: {regime_counts.get('Low Vol', 0) / len(df) * 100:.1f}%")

