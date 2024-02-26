# coding=gbk
# Time Created: 2024/2/26 13:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime
import os
import re
import numpy as np
import pandas as pd
from datetime import timedelta
from sqlalchemy import text
from WindPy import w

from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from utils import split_tradedays_into_weekly_ranges


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)

    def run_all_updater(self):
        pass


class StkShareholdersUpdater:
    """
    个股前十大股东：记录股东名称和持股数量
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass


class StkWSDUpdater:
    """
    个股实际控制人：wset是死的，要用WSD动态季度调整记录
    上市板：WSD-mkt
    市值、pe（区分风格）
    # 证券存续状态-sec_status
    # 上市日期
    # 摘牌日期
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass


class FundHoldsUpdater:
    """
    注意区分基金类别：主动管理、ETF
    基金重仓：前十大持股
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass


class FundPositionUpdater:
    """
    基金仓位：WSD 股票、债券、现金、其它占基金资产总值
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass


class IndicesComponentUpdater:
    """
    要设计新表还是记录在quarterly_long？
    各指数成分占比，行业占比
    热门指数，宽基、概念（红利）
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass


class OldFundsPurchaseUpdater:
    """
    基金仓位：WSD 股票、债券、现金、其它占基金资产总值
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass

