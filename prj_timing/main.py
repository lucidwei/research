# coding=gbk
# Time Created: 2025/1/14 14:09
# Author  : Lucid
# FileName: main.py
# Software: PyCharm
from data_handler import DataHandler
from signal_generator import SignalGenerator
from performance_evaluator import PerformanceEvaluator
from optimization import grid_search

def main():
    # Initialize DataHandler with desired frequency ('D' for daily, 'M' for monthly)
    data_handler = DataHandler(file_path='path_to_your_file.xlsx', frequency='M')
    df = data_handler.get_data()

    # Initialize SignalGenerator
    signal_generator = SignalGenerator(df)

    # Define parameter grid for Strategy 1 (example)
    param_grid_strategy1 = {
        'ma_window': [2, 3, 4, 5]
    }

    # Initialize PerformanceEvaluator (initially without any signals)
    evaluator = PerformanceEvaluator(df, [])

    # Perform grid search to find best parameters for Strategy 1
    optimization_result = grid_search(
        signal_generator=signal_generator,
        strategy_num=1,
        param_grid=param_grid_strategy1,
        performance_evaluator=evaluator,
        start_date='2001-12'
    )

    print(f"最佳参数: {optimization_result['best_params']}")
    print(f"最佳夏普比率: {optimization_result['best_metric']:.2f}")

    # Generate signals with best parameters
    best_params = optimization_result['best_params']
    signal_generator.generate_signals_for_all_strategies(strategies_params={1: best_params})

    # Initialize PerformanceEvaluator with the new signals
    signals_columns = ['strategy1_signal']
    evaluator = PerformanceEvaluator(signal_generator.df, signals_columns)

    # Backtest and evaluate
    evaluator.backtest_all_strategies(start_date='2001-12')
    evaluator.calculate_metrics_all_strategies()

    # Generate Excel reports
    evaluator.generate_excel_reports(
        metrics_output='strategy_metrics.xlsx',
        annual_output='strategy6_annual.xlsx'
    )

if __name__ == "__main__":
    main()