# coding=gbk
# Time Created: 2023/6/27 10:23
# Author  : Lucid
# FileName: run.py
# Software: PyCharm
import pickle
from base_config import BaseConfig
from prj_equity_sentiment.db_updater import DatabaseUpdater
from prj_equity_sentiment.processor import Processor
from prj_equity_sentiment.plotter import Plotter

base_config = BaseConfig('equity_sentiment')
# 只需要更新数据时启用
DatabaseUpdater(base_config)


#############
processor = Processor(base_config)
processor.upload_indicators()
# processed_data = processor.wide_results
# with open('processed_data.pkl', 'wb') as f:
#     pickle.dump(processed_data, f)

# 从硬盘上加载Processor对象
# with open('processed_data.pkl', 'rb') as f:
#     processed_data = pickle.load(f)
# ###########
# plotter = Plotter(base_config, processed_data)
