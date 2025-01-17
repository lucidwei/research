# coding=gbk
# Time Created: 2025/1/14 14:11
# Author  : Lucid
# FileName: optimization.py
# Software: PyCharm
from performance_evaluator import PerformanceEvaluator
from itertools import product

def grid_search(signal_generator, strategy_num, param_grid, performance_evaluator, start_date='2001-12'):
    """
    Performs grid search to find the best parameters for a given strategy.

    Parameters:
        signal_generator (SignalGenerator): Instance of SignalGenerator.
        strategy_num (int): Strategy number to optimize.
        param_grid (dict): Parameter grid for the strategy.
        performance_evaluator (PerformanceEvaluator): Instance of PerformanceEvaluator.
        start_date (str): Start date for backtesting.

    Returns:
        dict: Best parameters and performance metrics.
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_metric = -np.inf
    best_params = None

    for params in param_combinations:
        # Generate signals with current parameters
        signals = signal_generator.generate_strategy_zhaoshang_signals(strategy_num, **params)
        temp_df = signal_generator.df.copy()
        temp_df[f'strategy{strategy_num}_signal'] = signals

        # Initialize a new PerformanceEvaluator for this parameter set
        evaluator = PerformanceEvaluator(temp_df, [f'strategy{strategy_num}_signal'])
        evaluator.backtest_all_strategies(start_date=start_date)
        evaluator.calculate_metrics_all_strategies()

        # Assume we are maximizing the Sharpe Ratio
        sharpe_ratio = evaluator.metrics_df.loc[f'策略{strategy_num}', '夏普比率']

        if sharpe_ratio > best_metric:
            best_metric = sharpe_ratio
            best_params = params

    return {'best_params': best_params, 'best_metric': best_metric}