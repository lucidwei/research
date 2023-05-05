# coding=gbk
# Time Created: 2023/4/26 8:10
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re, math
from WindPy import w
import pandas as pd
from utils import timeit, get_nearest_dates_from_contract, check_wind
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text, MetaData, Table
from pypinyin import lazy_pinyin


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_edb_by_id_to_high_freq('S0059749')
        self.update_edb_by_id_to_high_freq('G0005428')
        self.update_wsd("VIX.GI", "close")
        self.update_wsd("CBA00301.CS", "close")
        self.update_wsd("000906.SH", "close,volume,pe_ttm")
        self.update_wsd("AU9999.SGE", "close")
        # self.set_all_nan_to_null()
        self.close()






