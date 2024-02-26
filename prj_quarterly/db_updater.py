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
    ����ǰʮ��ɶ�����¼�ɶ����ƺͳֹ�����
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
    ����ʵ�ʿ����ˣ�wset�����ģ�Ҫ��WSD��̬���ȵ�����¼
    ���а壺WSD-mkt
    ��ֵ��pe�����ַ��
    # ֤ȯ����״̬-sec_status
    # ��������
    # ժ������
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
    ע�����ֻ��������������ETF

    �����ز֣�ǰʮ��ֹ�
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
    �����λ��WSD ��Ʊ��ծȯ���ֽ�����ռ�����ʲ���ֵ
    �����ģ�����ڼ�Ȩ
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
    wset-�����ָ��-���ɷ�
    ��������������ͨ��Ʊ��ƫ�ɻ�ϡ�������ã�
    wset-�����ָ��-ָ��Ȩ��
    ָ��������300�����ȫA����֤���������ŵĿ�������������
    ��ָ�����ɳɷ֣���ҵռ��
    ������±���¼��quarterly_long
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
    �ϻ����깺���
    """
    def __init__(self, db_updater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def _check_meta_table(self):
        pass

    def _check_data_table(self):
        pass

