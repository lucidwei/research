# coding=gbk
# Time Created: 2023/4/13 10:35
# Author  : Lucid
# FileName: db_reader.py
# Software: PyCharm
from pgdb_manager import PgDbManager
from base_config import BaseConfig
import pandas as pd
import numpy as np
from utils import timeit
from sqlalchemy import text


class DatabaseReader(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.get_active_contracts_timeseries()
        self.get_delivery_dates()
        self.get_rates_timeseries()
        self.close()

    @property
    def active_contracts_timeseries(self):
        return self._active_contracts_timeseries

    @property
    def delivery_dates_dict(self):
        return self._delivery_dates_dict

    @property
    def contracts_set(self):
        return self._contracts_set

    @property
    def rates_timeseries(self):
        return self._rates_timeseries

    @timeit
    def get_active_contracts_timeseries(self):
        self._active_contracts_timeseries = pd.DataFrame(index=self.base_config.tradedays_str, columns=[
            'T_active1', 'TF_active1', 'TS_active1', 'T_active2', 'TF_active2', 'TS_active2'])
        for contract_type in ['T', 'TF', 'TS']:
            for act_num in ['1', '2']:
                for date in self.base_config.tradedays_str:
                    sql = f"SELECT contract_code FROM contract_stats_ts WHERE date = '{date}' AND contract_prefix = '{contract_type}' AND active_num = {act_num};"
                    df = pd.read_sql_query(text(sql), con=self.alch_conn)
                    if df.empty:
                        continue
                    contract = df.values[0][0]
                    self._active_contracts_timeseries.loc[date, f"{contract_type}_active{act_num}"] = contract

    @timeit
    def get_delivery_dates(self):
        # 整理合约交割日 : key contract_code, value delivery_date
        self._delivery_dates_dict = {}
        self._contracts_set = np.array([])
        for label, content in self._active_contracts_timeseries.items():
            self._contracts_set = np.append(self._contracts_set, pd.unique(content))
        self._contracts_set = {x for x in self._contracts_set if type(x) == str}
        for contract in self._contracts_set:
            sql = f"SELECT DISTINCT deliver_date FROM contract_stats_ts WHERE contract_code = '{contract}';"
            df = pd.read_sql_query(text(sql), con=self.alch_conn)
            if df.empty:
                continue
            deli_date = df.values[0][0]
            self._delivery_dates_dict[contract] = deli_date

    @timeit
    def get_rates_timeseries(self):
        sql = f"SELECT * FROM rates_ts;"
        df = pd.read_sql_query(text(sql), con=self.alch_conn)
        self._rates_timeseries = df.set_index('date')