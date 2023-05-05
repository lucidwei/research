# coding=gbk
# Time Created: 2023/4/25 9:45
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re, math
from WindPy import w
import pandas as pd
from utils import timeit, get_nearest_dates_from_contract, check_wind
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text, MetaData, Table
from pypinyin import lazy_pinyin


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_pmi()
        self.update_gdp()
        self.set_all_nan_to_null()
        self.close()

    def update_gdp(self):

        pass

    def update_pmi(self):
        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ�ֶ�ӳ���Ӣ������ (����chatgpt���)
        map_name_to_english = {
            "ŷԪ��:�ۺ�PMI": "pmi_comprehensive_eurozone",
            "�ձ�:�ۺ�PMI": "pmi_comprehensive_japan",
            "����:�ۺ�PMI": "pmi_comprehensive_usa",
            "����:��Ӧ����Э��(ISM):����ҵPMI": "pmi_manufacturing_ism_usa",
            "ŷԪ��:����ҵPMI": "pmi_manufacturing_eurozone",
            "�ձ�:����ҵPMI": "pmi_manufacturing_japan"
        }

        self.update_low_freq_from_excel_meta('��ʿPMI.xlsx', map_name_to_english)
