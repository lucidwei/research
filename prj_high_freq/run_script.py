# coding=gbk
# Time Created: 2023/3/10 18:33
# Author  : Lucid
# FileName: basis_study.py
# Software: PyCharm
from base_config import BaseConfig
from prj_high_freq.db_updater import DatabaseUpdater

base_config = BaseConfig('high_freq')
data_updater = DatabaseUpdater(base_config)


