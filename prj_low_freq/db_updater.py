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
        # 创建一个字典，键为指标 ID，值为手动映射的英文列名 (交给chatgpt完成)
        map_name_to_english = {
            "欧元区:综合PMI": "pmi_comprehensive_eurozone",
            "日本:综合PMI": "pmi_comprehensive_japan",
            "美国:综合PMI": "pmi_comprehensive_usa",
            "美国:供应管理协会(ISM):制造业PMI": "pmi_manufacturing_ism_usa",
            "欧元区:制造业PMI": "pmi_manufacturing_eurozone",
            "日本:制造业PMI": "pmi_manufacturing_japan"
        }

        self.update_low_freq_from_excel_meta('博士PMI.xlsx', map_name_to_english)
