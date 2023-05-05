# coding=gbk
# Time Created: 2023/3/10 18:33
# Author  : Lucid
# FileName: basis_study.py
# Software: PyCharm
from base_config import BaseConfig
from prj_T_basis.db_reader import DatabaseReader
from prj_T_basis.db_updater import DatabaseUpdater
from prj_T_basis.processor import Processor
from prj_T_basis.plotter import Plotter

base_config = BaseConfig(project='T_basis', auto_save_fig=True)
data_updater = DatabaseUpdater(base_config)
data_reader = DatabaseReader(base_config)
data_processed = Processor(data_reader)
plotter = Plotter(data_processed)


