# coding=gbk
# Time Created: 2023/4/26 10:48
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from prj_risk_parity.db_updater import DatabaseUpdater
from prj_risk_parity.db_reader import DatabaseReader
from prj_risk_parity.asset_allocator import AssetAllocator, StrategicAllocator, TacticalAllocator
from prj_risk_parity.evaluator import Evaluator
from prj_risk_parity.visualizer import PerformanceVisualizer
from prj_risk_parity.results_uploader import ResultsUploader
from datetime import datetime

today_str = datetime.now().strftime('%Y-%m-%d')

base_config = BaseConfig('risk_parity')
data_updater = DatabaseUpdater(base_config)
read_data = DatabaseReader(base_config)
parameters = {
    # stk_bond_gold
    'risk_budget': [0.8, 0.18, 0.02],
    'benchmark_weight': [0.2, 0.7, 0.1],
    'start_date': '2012-01-01',
    'end_date': today_str,
}
# strategic_allocator = StrategicAllocator(read_data, stk_bond_gold_risk_budget)
# tactical_allocator = TacticalAllocator(read_data)
asset_allocator = AssetAllocator(read_data, parameters)
evaluator = Evaluator(asset_allocator)
results_uploader = ResultsUploader(evaluator, base_config)

performance_visualizer = PerformanceVisualizer(evaluator)
performance_visualizer.visualize()


