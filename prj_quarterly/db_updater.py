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
            self.conn.close()  # ȷ��������ζ���ر�����

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
    ��������������ͨ��Ʊ��ƫ�ɻ�ϡ�������ã�����Ϊ�����������к������Ƚ�Ƶ����
    wset-�����ָ��-ָ��Ȩ��
    ָ��������300�����ȫA����֤���������ŵĿ�������������
    ��ָ�����ɳɷ֣���ҵռ�ȣ�������product_static_info���㣩
    ������±���¼��quarterly_long
    """
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater
        self._check_meta_table()
        self._check_data_table()

    def update_indices_component(self):
        """
        ��2000�꿪ʼ�����ȸ���
        """
        active_funds = {
            '��ͨ��Ʊ��': '2001010101000000',
            'ƫ�ɻ����': '2001010201000000',
            '���������': '1000011486000000',
        }
        major_indices = {
            '����300': '000300.SH',
            '���ȫA': '881001.WI',
            '��֤����': '000922.CSI',
        }
        for date in self.db_updater.quarterly_dates_str:
            for fund_type in active_funds.keys():
                print(f'Downloading active funds sectorconstituent on {date} for update_indices_component')
                downloaded_df = w.wset("sectorconstituent", f"date={date};sectorid={active_funds[fund_type]};"
                                                            f"field=wind_code,sec_name", usedf=True)[1]
                if downloaded_df.empty:
                    print(f"No sectorconstituent data on {date}, no data downloaded for update_indices_component")
                    continue

                # �������ص����ݲ��ϴ���quarterly_long
                # TODO sec_name�ô浽psi��
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

                # �������ص����ݲ��ϴ���quarterly_long
                funds_upload_df = downloaded_df.rename(
                    columns={'windcode': 'value_str', 'i_weight': 'value_number'})
                funds_upload_df['report_period'] = date
                funds_upload_df['field'] = f'{stk_index}component and weight'

                for _, row in funds_upload_df.iterrows():
                    self.db_updater.insert_quarterly_long(row)

    def update_missing_info_in_psi(self):
        """
        ����quarterly_long�еĻ���͹�Ʊ���룬��psi table�н��л������ݸ���
        ע���Ѵ��ڵ����ݾͲ�Ҫ�˷�quota��internal_id�Ĵ���
        """
        pass

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

