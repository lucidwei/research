# coding=gbk
# Time Created: 2025/1/14 14:09
# Author  : Lucid
# FileName: main.py
# Software: PyCharm
import sys
from data_handler import DataHandler
from signal_generator import SignalGenerator
from performance_evaluator import PerformanceEvaluator
from optimization import grid_search

def run_without_optimization():
    """
    运行不包含参数优化的回测流程。
    功能与原始的 main() 函数相同。
    """
    FREQ = 'D'
    # FREQ = 'M'
    INDEX_NAME = '上证指数'
    # INDEX_NAME = '沪深300'

    # 指定Excel文件路径和输出路径
    file_path = r"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\招商择时快速复现.xlsx"
    output_file = r"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\策略回测结果_当日开.xlsx"

    # 实例化 DataHandler 类，加载并预处理数据（默认按月频处理）
    data_handler = DataHandler(file_path=file_path, frequency=FREQ)
    macro_data = data_handler.get_macro_data()  # 分离的宏观数据
    indices_data = data_handler.get_indices_data()  # 分离的指数数据

    # 定义策略名称映射
    strategy_names = {}
    strategy_names[f"{INDEX_NAME}_strategy_turnover"] = f'{INDEX_NAME}_strategy_turnover'
    # 为策略1-6命名
    # for num in range(1, 7):
    #     strategy_id = f"{INDEX_NAME}_strategy_{num}"
    #     strategy_names[strategy_id] = strategy_id

    # 实例化 SignalGenerator 类，生成策略信号
    signal_generator = SignalGenerator(indices_data, macro_data)
    df_with_signals = signal_generator.generate_signals_for_all_strategies(strategy_names=strategy_names)

    # 获取所有策略的信号列名
    signal_columns = [f"{name}_signal" for name in strategy_names.values()]

    # 实例化 PerformanceEvaluator 类，进行回测和绩效评估
    performance_evaluator = PerformanceEvaluator(df_with_signals, signal_columns, FREQ)
    performance_evaluator.backtest_all_strategies(start_date='2001-12')
    performance_evaluator.calculate_metrics_all_strategies()
    # 针对个别策略进行按年份统计
    # annual_metrics_strategy_name = f'{INDEX_NAME}_strategy_6'
    annual_metrics_strategy_name = f'{INDEX_NAME}_strategy_turnover'
    performance_evaluator.calculate_annual_metrics_for(annual_metrics_strategy_name)

    # 生成并保存Excel报告
    performance_evaluator.generate_excel_reports(output_file, annual_metrics_strategy_name)

    print(f'回测完成，结果已保存到 {output_file}')

def run_with_optimization():
    """
    运行包含参数优化的回测流程。
    """
    # 指定Excel文件路径和输出路径
    file_path = r"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\招商择时快速复现.xlsx"
    output_file = r"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\策略回测结果_优化.xlsx"

    # 实例化 DataHandler 类，加载并预处理数据（按月频处理）
    data_handler = DataHandler(file_path=file_path, frequency='M')
    df = data_handler.get_data()

    # 定义策略名称映射
    strategy_names = {
        1: 'strategy1',
        2: 'strategy2',
        3: 'strategy3',
        4: 'strategy4',
        5: 'strategy5',
        6: 'strategy6'
    }

    # 实例化 SignalGenerator 类
    signal_generator = SignalGenerator(df)

    # 定义需要优化的策略及其参数网格
    # 例如：优化策略1的 'ma_window' 参数
    param_grid = {
        1: {  # 策略1
            'ma_window': [2, 3, 4, 5]
        },
        # 如需优化更多策略，按相同格式添加
        # 2: { 'param_name': [values], ... },
        # 3: { 'param_name': [values], ... },
    }

    # 定义优化的策略数量，这里假设仅策略1需要优化
    strategies_to_optimize = [1]

    # 执行参数优化
    for strategy_num in strategies_to_optimize:
        print(f"\n开始优化策略{strategy_num}的参数...")
        grid_result = grid_search(
            signal_generator=signal_generator,
            strategy_num=strategy_num,
            param_grid=param_grid[strategy_num],
            performance_evaluator=None,  # PerformanceEvaluator将在 grid_search 中处理
            df=df,
            start_date='2001-12'
        )
        best_params = grid_result['best_params']
        best_metric = grid_result['best_metric']
        print(f"策略{strategy_num}的最佳参数: {best_params}")
        print(f"策略{strategy_num}的最佳绩效指标（例如夏普比率）: {best_metric:.4f}")

        # 使用最佳参数生成信号
        signal_generator.generate_strategy_signals(strategy_num, **best_params)
        # 如策略6依赖于策略1的信号，确保策略6信号重新生成
        if strategy_num in [1, 2, 3, 4, 5]:
            signal_generator.generate_strategy6_signals()

    # 获取所有策略的信号列名
    signal_columns = [f"{name}_signal" for name in strategy_names.values()]

    # 合并生成的信号到DataFrame
    df_with_signals = signal_generator.indices_data

    # 实例化 PerformanceEvaluator 类，进行回测和绩效评估
    performance_evaluator = PerformanceEvaluator(df_with_signals, signal_columns)
    performance_evaluator.backtest_all_strategies(start_date='2001-12')
    performance_evaluator.calculate_metrics_all_strategies()
    performance_evaluator.calculate_annual_metrics_all_strategies()

    # 生成并保存Excel报告
    performance_evaluator.generate_excel_reports(output_file)

    print(f'回测（含参数优化）完成，结果已保存到 {output_file}')

# 5. Execute the main function
if __name__ == "__main__":
    # 通过命令行参数选择运行模式
    # 示例：python main.py no_optimization 或 python main.py optimization
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'no_optimization':
            run_without_optimization()
        elif mode == 'optimization':
            run_with_optimization()
        else:
            print("未知的运行模式。请使用 'no_optimization' 或 'optimization'。")
    else:
        # 默认运行不包含参数优化的流程
        run_without_optimization()