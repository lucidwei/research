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
        self.conn = self.alch_engine.connect()
        self.config_quarterly_dates()
        self.indices_component_updater = IndicesComponentUpdater(self)

    def run_all_updater(self):
        try:
            self.indices_component_updater.update_indices_component()
        finally:
            self.conn.close()  # 确保无论如何都会关闭连接

    def config_quarterly_dates(self):
        self.quarterly_dates = w.tdays(self.tradedays_str[0], self.tradedays_str[150], "Period=Q").Data[0]
        self.quarterly_dates_str = [str(x) for x in self.quarterly_dates]

    def insert_quarterly_long(self, row: pd.Series):
        query = text("""
                    INSERT INTO quarterly_long (report_period, field, value_number, value_str, product_code)
                    VALUES (:report_period, :field, :value_number, :value_str, :product_code)
                    ON CONFLICT (report_period, field, product_code) DO UPDATE 
                    SET value_number = EXCLUDED.value_number,
                        value_str = EXCLUDED.value_str
                    RETURNING internal_id;
                    """)
        result = self.conn.execute(query,
                              {
                                  'report_period': row['report_period'],
                                  'field': row['field'],
                                  'product_code': row['product_code'],
                                  'value_number': None if pd.isnull(row['value_number']) else row['value_number'],
                                  'value_str': None if pd.isnull(row['value_str']) else row['value_str'],
                              })
        internal_id = result.fetchone()[0]
        self.conn.commit()


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
    基金：主动管理（普通股票、偏股混合、灵活配置）。因为基金清盘退市和新增比较频繁。
    wset-板块与指数-指数权重
    指数：沪深300、万德全A、中证红利等热门的宽基、概念（红利）
    各指数个股成分，行业占比（须利用product_static_info计算）
    不设计新表，记录在quarterly_long
    """
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def update_indices_component(self):
        """
        从2000年开始按季度更新
        """
        active_funds = {
            '普通股票型': '2001010101000000',
            '偏股混合型': '2001010201000000',
            '灵活配置型': '1000011486000000',
        }
        major_indices = {
            '沪深300': '000300.SH',
            '万德全A': '881001.WI',
            '中证红利': '000922.CSI',
        }
        for date in self.db_updater.quarterly_dates_str:
            for fund_type in active_funds.keys():
                print(f'Downloading active funds sectorconstituent on {date} for update_indices_component')
                downloaded_df = w.wset("sectorconstituent", f"date={date};sectorid={active_funds[fund_type]};"
                                                            f"field=wind_code,sec_name", usedf=True)[1]
                if downloaded_df.empty:
                    print(f"No sectorconstituent data on {date}, no data downloaded for update_indices_component")
                    continue

                # 解析下载的数据并上传至quarterly_long
                # TODO sec_name该存到psi中
                funds_upload_df = downloaded_df.rename(
                    columns={'windcode': 'value_str', 'sec_name': 'chinese_name'})
                funds_upload_df['report_period'] = date
                funds_upload_df['field'] = f'{fund_type}component'

                for _, row in funds_upload_df.iterrows():
                    self.db_updater.insert_quarterly_long(row)

            for stk_index in major_indices.keys():
                print(f'Downloading stock index indexconstituent on {date} for update_indices_component')
                downloaded_df = w.wset("indexconstituent", f"date={date};"
                                        f"windcode={major_indices[stk_index]};field=wind_code,i_weight", usedf=True)[1]
                if downloaded_df.empty:
                    print(f"No indexconstituent data on {date}, no data downloaded for update_indices_component")
                    continue

                # 解析下载的数据并上传至quarterly_long
                funds_upload_df = downloaded_df.rename(
                    columns={'windcode': 'value_str', 'i_weight': 'value_number'})
                funds_upload_df['report_period'] = date
                funds_upload_df['field'] = f'{stk_index}component and weight'

                for _, row in funds_upload_df.iterrows():
                    self.db_updater.insert_quarterly_long(row)

    def update_missing_info_in_psi(self):
        """
        利用quarterly_long中的基金和股票代码，在psi table中进行基本数据更新
        注意已存在的数据就不要浪费quota、internal_id的处理
        """
        pass

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

