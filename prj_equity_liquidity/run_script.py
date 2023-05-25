# coding=gbk
# Time Created: 2023/5/25 9:43
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from prj_equity_liquidity.db_updater import DatabaseUpdater

base_config = BaseConfig('equity_liquidity')
data_updater = DatabaseUpdater(base_config)