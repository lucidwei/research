# coding=gbk
# Time Created: 2024/10/30 14:59
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from data_loader import ExcelDataLoader, ResultsUploader
import importlib
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False  # �������ͼ���Ǹ���'-'��ʾΪ���������

# �� Excel �ļ���ȡ����
data_path = rf"D:\WPS����\WPS����\����-���\ר���о�\ר��-����Ԥ����ʲ����ò���\�ʲ��۸�.xlsx"
data_loader = ExcelDataLoader(data_path)
data_dict = data_loader.get_data()
asset_class_mapping = data_loader.asset_class_mapping

# ������������б�
strategies_config = [
    {
        'name': 'FixedWeight',
        'module': 'strategy_pure_passive',
        'class': 'FixedWeightStrategy',
        'parameters': {
            'weights': {'��֤800': 0.2, 'SHFE�ƽ�': 0.1, '��ծ-�ܲƸ�(��ֵ)ָ��': 0.7},
            'asset_class_mapping': asset_class_mapping,
        }
    },
    {
        'name': 'RiskParity',
        'module': 'strategy_pure_passive',
        'class': 'RiskParityStrategy',
        'parameters': {
            'selected_assets': ['��֤800', 'SHFE�ƽ�', '��ծ-�ܲƸ�(��ֵ)ָ��',],
            'asset_class_mapping': asset_class_mapping,
            'risk_budget': {'��֤800': 0.8, '��ծ-�ܲƸ�(��ֵ)ָ��': 0.18, 'SHFE�ƽ�': 0.02},
            'rebalance_frequency': 'M',
            'lookback_periods': [63, 126, 252],
        }
    },
    {
        'name': 'MomentumStrategy',
        'module': 'strategy_momentum',
        'class': 'MomentumStrategy',
        'parameters': {
            'asset_class_mapping': asset_class_mapping,  # �����Ҫ
            'rebalance_frequency': 'M',
            'lookback_periods': [252, 126, 63],
            'top_n_assets': 8,
            'risk_budget': {'Equity': 0.8, 'Bond': 0.18, 'Commodity': 0.02},  # �����ʲ����ķ���Ԥ��
            # ������ṩ risk_budget����ʹ�÷���ƽ��
        }
    },
    {
        'name': 'SignalBasedStrategy',
        'module': 'strategy_signal_based',
        'class': 'SignalBasedStrategy',
        'parameters': {
            'asset_class_mapping': asset_class_mapping,
            'risk_budget': {'��֤800': 0.8, '��ծ-�ܲƸ�(��ֵ)ָ��': 0.18, 'SHFE�ƽ�': 0.02},
            'rebalance_frequency': 'M',
        }
    },
]

# ����ͨ�ò���
start_date = '2010-12-31'
end_date = None

# �洢���в��Ե��������
all_evaluations = {}
all_strategies = {}

for config in strategies_config:
    strategy_name = config['name']
    module_name = config['module']
    class_name = config['class']
    parameters = config['parameters']
    print(f"Running strategy: {strategy_name}")

    # Dynamically import strategy module
    strategy_module = importlib.import_module(module_name)
    strategy_class = getattr(strategy_module, class_name)

    # Initialize strategy
    strategy = strategy_class()

    # Run strategy
    evaluator = strategy.run_strategy(data_dict, start_date, end_date, parameters)

    # Store evaluation and strategy instance
    all_evaluations[strategy_name] = evaluator
    all_strategies[strategy_name] = strategy

# Output performance metrics
for strategy_name, evaluator in all_evaluations.items():
    print(f"=== {strategy_name} ===")
    performance = evaluator.performance
    for key, value in performance.items():
        print(f'{key}: {value:.4f}')
    print("\n")

# ������ƾ�ֵ����
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for strategy_name, evaluator in all_evaluations.items():
    evaluator.net_value.plot(label=strategy_name)
plt.title('���Ծ�ֵ���߶Ա�')
plt.xlabel('����')
plt.ylabel('��ֵ')
plt.legend()
plt.show()

# After running all strategies, upload results to the database
base_config = BaseConfig('multi-asset')
for strategy_name in all_evaluations.keys():
    evaluator = all_evaluations[strategy_name]
    strategy = all_strategies[strategy_name]

    # Initialize ResultsUploader and upload results
    results_uploader = ResultsUploader(strategy_name, strategy, evaluator, base_config)
    results_uploader.upload_results()
