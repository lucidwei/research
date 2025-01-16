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
    ���в����������Ż��Ļز����̡�
    ������ԭʼ�� main() ������ͬ��
    """
    FREQ = 'D'
    # FREQ = 'M'
    INDEX_NAME = '��ָ֤��'
    # INDEX_NAME = '����300'

    # ָ��Excel�ļ�·�������·��
    file_path = r"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\������ʱ���ٸ���.xlsx"
    output_file = r"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\���Իز���_���տ�.xlsx"

    # ʵ���� DataHandler �࣬���ز�Ԥ�������ݣ�Ĭ�ϰ���Ƶ����
    data_handler = DataHandler(file_path=file_path, frequency=FREQ)
    macro_data = data_handler.get_macro_data()  # ����ĺ������
    indices_data = data_handler.get_indices_data()  # �����ָ������

    # �����������ӳ��
    strategy_names = {}
    strategy_names[f"{INDEX_NAME}_strategy_turnover"] = f'{INDEX_NAME}_strategy_turnover'
    # Ϊ����1-6����
    # for num in range(1, 7):
    #     strategy_id = f"{INDEX_NAME}_strategy_{num}"
    #     strategy_names[strategy_id] = strategy_id

    # ʵ���� SignalGenerator �࣬���ɲ����ź�
    signal_generator = SignalGenerator(indices_data, macro_data)
    df_with_signals = signal_generator.generate_signals_for_all_strategies(strategy_names=strategy_names)

    # ��ȡ���в��Ե��ź�����
    signal_columns = [f"{name}_signal" for name in strategy_names.values()]

    # ʵ���� PerformanceEvaluator �࣬���лز�ͼ�Ч����
    performance_evaluator = PerformanceEvaluator(df_with_signals, signal_columns, FREQ)
    performance_evaluator.backtest_all_strategies(start_date='2001-12')
    performance_evaluator.calculate_metrics_all_strategies()
    # ��Ը�����Խ��а����ͳ��
    # annual_metrics_strategy_name = f'{INDEX_NAME}_strategy_6'
    annual_metrics_strategy_name = f'{INDEX_NAME}_strategy_turnover'
    performance_evaluator.calculate_annual_metrics_for(annual_metrics_strategy_name)

    # ���ɲ�����Excel����
    performance_evaluator.generate_excel_reports(output_file, annual_metrics_strategy_name)

    print(f'�ز���ɣ�����ѱ��浽 {output_file}')

def run_with_optimization():
    """
    ���а��������Ż��Ļز����̡�
    """
    # ָ��Excel�ļ�·�������·��
    file_path = r"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\������ʱ���ٸ���.xlsx"
    output_file = r"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\���Իز���_�Ż�.xlsx"

    # ʵ���� DataHandler �࣬���ز�Ԥ�������ݣ�����Ƶ����
    data_handler = DataHandler(file_path=file_path, frequency='M')
    df = data_handler.get_data()

    # �����������ӳ��
    strategy_names = {
        1: 'strategy1',
        2: 'strategy2',
        3: 'strategy3',
        4: 'strategy4',
        5: 'strategy5',
        6: 'strategy6'
    }

    # ʵ���� SignalGenerator ��
    signal_generator = SignalGenerator(df)

    # ������Ҫ�Ż��Ĳ��Լ����������
    # ���磺�Ż�����1�� 'ma_window' ����
    param_grid = {
        1: {  # ����1
            'ma_window': [2, 3, 4, 5]
        },
        # �����Ż�������ԣ�����ͬ��ʽ���
        # 2: { 'param_name': [values], ... },
        # 3: { 'param_name': [values], ... },
    }

    # �����Ż��Ĳ���������������������1��Ҫ�Ż�
    strategies_to_optimize = [1]

    # ִ�в����Ż�
    for strategy_num in strategies_to_optimize:
        print(f"\n��ʼ�Ż�����{strategy_num}�Ĳ���...")
        grid_result = grid_search(
            signal_generator=signal_generator,
            strategy_num=strategy_num,
            param_grid=param_grid[strategy_num],
            performance_evaluator=None,  # PerformanceEvaluator���� grid_search �д���
            df=df,
            start_date='2001-12'
        )
        best_params = grid_result['best_params']
        best_metric = grid_result['best_metric']
        print(f"����{strategy_num}����Ѳ���: {best_params}")
        print(f"����{strategy_num}����Ѽ�Чָ�꣨�������ձ��ʣ�: {best_metric:.4f}")

        # ʹ����Ѳ��������ź�
        signal_generator.generate_strategy_signals(strategy_num, **best_params)
        # �����6�����ڲ���1���źţ�ȷ������6�ź���������
        if strategy_num in [1, 2, 3, 4, 5]:
            signal_generator.generate_strategy6_signals()

    # ��ȡ���в��Ե��ź�����
    signal_columns = [f"{name}_signal" for name in strategy_names.values()]

    # �ϲ����ɵ��źŵ�DataFrame
    df_with_signals = signal_generator.indices_data

    # ʵ���� PerformanceEvaluator �࣬���лز�ͼ�Ч����
    performance_evaluator = PerformanceEvaluator(df_with_signals, signal_columns)
    performance_evaluator.backtest_all_strategies(start_date='2001-12')
    performance_evaluator.calculate_metrics_all_strategies()
    performance_evaluator.calculate_annual_metrics_all_strategies()

    # ���ɲ�����Excel����
    performance_evaluator.generate_excel_reports(output_file)

    print(f'�ز⣨�������Ż�����ɣ�����ѱ��浽 {output_file}')

# 5. Execute the main function
if __name__ == "__main__":
    # ͨ�������в���ѡ������ģʽ
    # ʾ����python main.py no_optimization �� python main.py optimization
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'no_optimization':
            run_without_optimization()
        elif mode == 'optimization':
            run_with_optimization()
        else:
            print("δ֪������ģʽ����ʹ�� 'no_optimization' �� 'optimization'��")
    else:
        # Ĭ�����в����������Ż�������
        run_without_optimization()