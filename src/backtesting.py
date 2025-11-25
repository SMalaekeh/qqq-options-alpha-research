"""
Backtesting & Evaluation Module
Provides utilities for strategy backtesting and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_metrics(returns: np.ndarray, 
                         signals: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Array of daily returns
            signals: Optional array of position signals (for filtering)
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter out zero-position days if signals provided
        if signals is not None:
            valid_mask = (signals != 0) & ~np.isnan(returns)
            clean_returns = returns[valid_mask]
        else:
            clean_returns = returns[~np.isnan(returns)]
        
        if len(clean_returns) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': 0
            }
        
        # Basic metrics
        total_return = np.prod(1 + clean_returns) - 1
        annual_return = np.mean(clean_returns) * 252
        annual_volatility = np.std(clean_returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annual_return / (annual_volatility + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = clean_returns[clean_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = annual_return / (downside_std * np.sqrt(252) + 1e-8)
        
        # Drawdown analysis
        cum_returns = np.cumprod(1 + clean_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar = annual_return / (abs(max_drawdown) + 1e-8)
        
        # Win rate
        win_rate = np.mean(clean_returns > 0)
        
        # Profit factor
        gross_profit = np.sum(clean_returns[clean_returns > 0])
        gross_loss = abs(np.sum(clean_returns[clean_returns < 0]))
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(clean_returns)
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "PERFORMANCE METRICS"):
        """Pretty print performance metrics."""
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(f"Total Return:        {metrics['total_return']:>10.2%}")
        print(f"Annual Return:       {metrics['annual_return']:>10.2%}")
        print(f"Annual Volatility:   {metrics['annual_volatility']:>10.2%}")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"Number of Trades:    {metrics['num_trades']:>10}")
        print("=" * 60)


class RobustnessAnalyzer:
    """
    Perform robustness checks on trading strategies.
    """
    
    @staticmethod
    def parameter_sweep(model, raw_predictions: np.ndarray, 
                       returns: np.ndarray, 
                       param_grid: list) -> pd.DataFrame:
        """
        Test strategy across parameter variations.
        
        Args:
            model: Model instance with generate_signals method
            raw_predictions: Raw model predictions
            returns: Actual returns
            param_grid: List of parameter dictionaries
            
        Returns:
            DataFrame with robustness results
        """
        results = []
        
        for params in param_grid:
            # Update model parameters
            for key, value in params.items():
                if key != 'label' and hasattr(model, key):
                    setattr(model, key, value)
            
            # Generate signals
            signals = model.generate_signals(raw_predictions, returns)
            
            # Calculate performance
            portfolio_returns = signals * returns
            metrics = PerformanceMetrics.calculate_metrics(portfolio_returns, signals)
            
            result = {
                'Label': params.get('label', 'Unnamed'),
                **{k: v for k, v in params.items() if k != 'label'},
                'Calmar': metrics['calmar_ratio'],
                'Sharpe': metrics['sharpe_ratio'],
                'Max DD': metrics['max_drawdown'],
                'Total Return': metrics['total_return']
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def print_robustness_summary(results_df: pd.DataFrame, 
                                baseline_label: str = 'BASELINE'):
        """Print robustness analysis summary."""
        print("\n" + "=" * 70)
        print("ROBUSTNESS ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Find baseline
        baseline = results_df[results_df['Label'].str.contains(baseline_label)]
        if len(baseline) > 0:
            baseline_calmar = baseline['Calmar'].values[0]
            print(f"Baseline Calmar: {baseline_calmar:.2f}")
            
            # Calculate deviations
            results_df['Calmar_Deviation'] = (
                (results_df['Calmar'] - baseline_calmar) / baseline_calmar * 100
            )
            
            print(f"\nCalmar Range: {results_df['Calmar'].min():.2f} to "
                  f"{results_df['Calmar'].max():.2f}")
            print(f"Average Deviation: {results_df['Calmar_Deviation'].mean():.1f}%")
            print(f"Max Positive Deviation: {results_df['Calmar_Deviation'].max():.1f}%")
            print(f"Max Negative Deviation: {results_df['Calmar_Deviation'].min():.1f}%")
        
        print("\n" + "-" * 70)
        print("All Configurations:")
        print("-" * 70)
        for _, row in results_df.iterrows():
            print(f"{row['Label']:<30} -> Calmar: {row['Calmar']:>6.2f} | "
                  f"Sharpe: {row['Sharpe']:>6.2f}")


class TrainTestSplitter:
    """
    Utility for chronological train/validation/test splits.
    """
    
    @staticmethod
    def split_data(df: pd.DataFrame, 
                   train_pct: float = 0.6, 
                   val_pct: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform chronological split.
        
        Args:
            df: DataFrame sorted by date
            train_pct: Training set percentage
            val_pct: Validation set percentage
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    @staticmethod
    def print_split_info(train_df: pd.DataFrame, 
                        val_df: pd.DataFrame, 
                        test_df: pd.DataFrame):
        """Print split information."""
        print("\n" + "=" * 60)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 60)
        print(f"Train: {len(train_df):>4} samples | "
              f"{train_df['tradeDate'].min().date()} to "
              f"{train_df['tradeDate'].max().date()}")
        print(f"Val:   {len(val_df):>4} samples | "
              f"{val_df['tradeDate'].min().date()} to "
              f"{val_df['tradeDate'].max().date()}")
        print(f"Test:  {len(test_df):>4} samples | "
              f"{test_df['tradeDate'].min().date()} to "
              f"{test_df['tradeDate'].max().date()}")
        print("=" * 60)


def calculate_signal_statistics(signals: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about trading signals.
    
    Args:
        signals: Array of position signals
        
    Returns:
        Dictionary with signal statistics
    """
    long_days = np.sum(signals > 0.1)
    short_days = np.sum(signals < -0.1)
    neutral_days = np.sum(np.abs(signals) <= 0.1)
    
    avg_long_exposure = np.mean(signals[signals > 0.1]) if long_days > 0 else 0
    avg_short_exposure = np.mean(signals[signals < -0.1]) if short_days > 0 else 0
    
    return {
        'total_days': len(signals),
        'long_days': long_days,
        'short_days': short_days,
        'neutral_days': neutral_days,
        'long_pct': long_days / len(signals) * 100,
        'short_pct': short_days / len(signals) * 100,
        'neutral_pct': neutral_days / len(signals) * 100,
        'avg_long_exposure': avg_long_exposure,
        'avg_short_exposure': avg_short_exposure,
        'max_long': np.max(signals),
        'max_short': np.min(signals)
    }


def print_signal_statistics(stats: Dict[str, float]):
    """Pretty print signal statistics."""
    print("\n" + "=" * 60)
    print("SIGNAL STATISTICS")
    print("=" * 60)
    print(f"Total Days:          {stats['total_days']:>6}")
    print(f"Long Days:           {stats['long_days']:>6} ({stats['long_pct']:>5.1f}%)")
    print(f"Short Days:          {stats['short_days']:>6} ({stats['short_pct']:>5.1f}%)")
    print(f"Neutral Days:        {stats['neutral_days']:>6} ({stats['neutral_pct']:>5.1f}%)")
    print(f"\nAvg Long Exposure:   {stats['avg_long_exposure']:>6.2f}x")
    print(f"Avg Short Exposure:  {stats['avg_short_exposure']:>6.2f}x")
    print(f"Max Long:            {stats['max_long']:>6.2f}x")
    print(f"Max Short:           {stats['max_short']:>6.2f}x")
    print("=" * 60)

