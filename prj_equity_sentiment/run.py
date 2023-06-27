# coding=gbk
# Time Created: 2023/6/27 10:23
# Author  : Lucid
# FileName: run.py
# Software: PyCharm
from base_config import BaseConfig
from prj_equity_liquidity.db_updater import DatabaseUpdater
from prj_equity_liquidity.plotter import Plotter

base_config = BaseConfig('equity_sentiment')
# data_updater = DatabaseUpdater(base_config)
plotter = Plotter(base_config)