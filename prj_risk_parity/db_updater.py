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


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_high_freq_by_edb_id('S0059749')
        self.update_high_freq_by_edb_id('G0005428')
        self.update_markets_daily_by_wsd_id_fields("VIX.GI", "close")
        self.update_markets_daily_by_wsd_id_fields("CBA00301.CS", "close")
        self.update_markets_daily_by_wsd_id_fields("000906.SH", "close,volume,pe_ttm")
        self.update_markets_daily_by_wsd_id_fields("AU9999.SGE", "close")
        # self.set_all_nan_to_null()
        self.close()






