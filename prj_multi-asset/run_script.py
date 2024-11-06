# coding=gbk
# Time Created: 2024/10/30 14:59
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from data_loader import ExcelDataLoader, ResultsUploader
import importlib
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 从 Excel 文件读取数据
data_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\专题-风险预算的资产配置策略\资产价格.xlsx"
data_loader = ExcelDataLoader(data_path)
data = data_loader.get_data()
price_data = data['close_prices']
asset_class_mapping = data_loader.asset_class_mapping

# 定义策略配置列表
strategies_config = [
    {
        'name': 'FixedWeight',
        'module': 'strategy_pure_passive',
        'class': 'FixedWeightStrategy',
        'parameters': {
            'weights': {'中证800': 0.2, 'SHFE黄金': 0.1, '中债-总财富(总值)指数': 0.7},
            'asset_class_mapping': asset_class_mapping,
        }
    },
    {
        'name': 'RiskParity',
        'module': 'strategy_pure_passive',
        'class': 'RiskParityStrategy',
        'parameters': {
            'selected_assets': ['中证800', 'SHFE黄金', '中债-总财富(总值)指数',],
            'asset_class_mapping': {
                '中证800': 'Equity',
                'SHFE黄金': 'Commodity',
                '中债-总财富(总值)指数': 'Bond',
            },
            'risk_budget': {'Equity': 0.5, 'Bond': 0.3, 'Commodity': 0.2},
            'rebalance_frequency': 'M',
            'lookback_periods': [21, 63, 126],
        }
    },
    # {
    #     'name': 'Momentum',
    #     'module': 'strategy_momentum',
    #     'class': 'MomentumStrategy',
    #     'parameters': {
    #         'asset_class_mapping': asset_class_mapping,
    #         'risk_prefs': {'Equity': 0.5, 'Bond': 0.3, 'Commodity': 0.2},
    #         'macro_adj': {'Commodity': 0.3},
    #         'subjective_adj': {'Equity': 0.4},
    #     }
    # },
    # {
    #     'name': 'SignalBased',
    #     'module': 'strategy_signal_based',
    #     'class': 'SignalBasedStrategy',
    #     'parameters': {
    #         'asset_class_mapping': asset_class_mapping,
    #         'signal_parameters': {
    #             # Signal-specific parameters, if any
    #         },
    #         'risk_budget': [0.4, 0.4, 0.2],
    #         # Other strategy-specific parameters
    #     }
    # }
]

# 设置通用参数
start_date = '2021-12-31'
end_date = '2024-10-31'

# 存储所有策略的评估结果
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
    evaluator = strategy.run_strategy(price_data, start_date, end_date, parameters)

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

# After running all strategies, upload results to the database
base_config = BaseConfig('multi-asset')
for strategy_name in all_evaluations.keys():
    evaluator = all_evaluations[strategy_name]
    strategy = all_strategies[strategy_name]

    # Initialize ResultsUploader and upload results
    results_uploader = ResultsUploader(strategy_name, strategy, evaluator, base_config)
    results_uploader.upload_results()

# 如需绘制净值曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for strategy_name, evaluator in all_evaluations.items():
    evaluator.net_value.plot(label=strategy_name)
plt.title('策略净值曲线对比')
plt.xlabel('日期')
plt.ylabel('净值')
plt.legend()
plt.show()