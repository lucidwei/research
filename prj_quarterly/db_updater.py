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

    @property
    def quarterly_dates(self):
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
    基金规模：用于加权
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
    wset-板块与指数-板块成分
    基金：主动管理（普通股票、偏股混合、灵活配置）
    wset-板块与指数-指数权重
    指数：沪深300、万德全A、中证红利等热门的宽基、概念（红利）
    各指数个股成分，行业占比
    不设计新表，记录在quarterly_long
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
    老基金申购赎回
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass

