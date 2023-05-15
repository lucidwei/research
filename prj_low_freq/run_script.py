# coding=gbk
# Time Created: 2023/4/25 13:22
# Author  : Lucid
# FileName: run_script.py
# Software: PyCharm
from base_config import BaseConfig
from prj_low_freq.db_updater import DatabaseUpdater

base_config = BaseConfig('low_freq')
data_updater = DatabaseUpdater(base_config, if_rename=False)
