# coding=gbk
# Time Created: 2024/2/27 13:13
# Author  : Lucid
# Software: PyCharm
from base_config import BaseConfig
from prj_quarterly.db_updater import DatabaseUpdater

base_config = BaseConfig('quarterly')
data_updater = DatabaseUpdater(base_config)
data_updater.run_all_updater()