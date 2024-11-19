# coding=gbk
# Time Created: 2024/10/30 14:59
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from data_loader import ExcelDataLoader, ResultsUploader
import importlib
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 从 Excel 文件读取数据
data_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\专题-风险预算的资产配置策略\资产价格.xlsx"
data_loader = ExcelDataLoader(data_path)
data_dict = data_loader.get_data()
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
            'asset_class_mapping': asset_class_mapping,
            'risk_budget': {'中证800': 0.8, '中债-总财富(总值)指数': 0.18, 'SHFE黄金': 0.02},
            'rebalance_frequency': 'M',
            'lookback_periods': [63, 126, 252],
        }
    },
    {
        'name': 'MomentumStrategy',
        'module': 'strategy_momentum',
        'class': 'MomentumStrategy',
        'parameters': {
            'asset_class_mapping': asset_class_mapping,  # 如果需要
            'rebalance_frequency': 'M',
            'lookback_periods': [252, 126, 63],
            'top_n_assets': 8,
            'risk_budget': {'Equity': 0.8, 'Bond': 0.18, 'Commodity': 0.02},  # 或者资产类别的风险预算
            # 如果不提供 risk_budget，则使用风险平价
        }
    },
    {
        'name': 'SignalBasedStrategy',
        'module': 'strategy_signal_based',
        'class': 'SignalBasedStrategy',
        'parameters': {
            'asset_class_mapping': asset_class_mapping,
            'risk_budget': {'中证800': 0.8, '中债-总财富(总值)指数': 0.18, 'SHFE黄金': 0.02},
            'rebalance_frequency': 'M',
        }
    },
]

# 设置通用参数
start_date = '2010-12-31'
end_date = None

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

# After running all strategies, upload results to the database
base_config = BaseConfig('multi-asset')
for strategy_name in all_evaluations.keys():
    evaluator = all_evaluations[strategy_name]
    strategy = all_strategies[strategy_name]

    # Initialize ResultsUploader and upload results
    results_uploader = ResultsUploader(strategy_name, strategy, evaluator, base_config)
    results_uploader.upload_results()
