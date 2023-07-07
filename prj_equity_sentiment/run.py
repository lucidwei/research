# coding=gbk
# Time Created: 2023/6/27 10:23
# Author  : Lucid
# FileName: run.py
# Software: PyCharm
from base_config import BaseConfig
from prj_equity_sentiment.db_updater import DatabaseUpdater
from prj_equity_sentiment.processor import Processor
from prj_equity_sentiment.plotter import Plotter

base_config = BaseConfig('equity_sentiment')
# 只需要更新数据时启用
# data_updater = DatabaseUpdater(base_config)
processed_data = Processor(base_config)
# processed_data.upload_indicators()
plotter = Plotter(base_config, processed_data)