# coding=gbk
# Time Created: 2023/6/21 8:53
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime
import os
import numpy as np
import pandas as pd
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from WindPy import w
from sqlalchemy import text


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)


    def _check_meta_table(self, table_name, check_column, type_identifier):
        match type_identifier:
            case 'margin_by_industry':
                print(f'Wind downloading tradingstatisticsbyindustry industryname for _check_meta_table')
                self.today_industries_df = w.wset("tradingstatisticsbyindustry",
                                                  f"exchange=citic;startdate={self.tradedays_str[-2]};enddate={self.tradedays_str[-2]};"
                                                  "field=industryname", usedf=True)[1]
                industry_list = self.today_industries_df['industryname'].tolist()
                required_value = ['融资融券行业交易统计_' + str(value) for value in industry_list]

            case _:
                raise Exception(f'type_identifier {type_identifier} not supported')

        existing_value = self.select_existing_values_in_target_column(table_name, check_column,
                                                                      ('type_identifier', type_identifier))
        missing_value = set(required_value) - set(existing_value)
        if missing_value:
            return True
        else:
            return False

    def _check_data_table(self, table_name, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # 获取需要更新的日期区间
        match type_identifier:
            case 'price_valuation':
                filter_condition = f"product_static_info.type_identifier = '{type_identifier}'"
                if additional_filter:
                    filter_condition += f" AND {additional_filter}"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name=table_name,
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case '...':
                filter_condition = f"metric_static_info.type_identifier = '{type_identifier}'"
                if additional_filter:
                    filter_condition += f" AND {additional_filter}"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='metric_static_info',
                    target_join_column='internal_id',
                    join_table_name=table_name,
                    join_column='metric_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case _:
                raise Exception(f'type_identifier: {type_identifier} not supported.')

        if len(existing_dates) == 0:
            missing_dates = self.tradedays
        else:
            missing_dates = self.get_missing_dates(all_dates=self.tradedays, existing_dates=existing_dates)

        if not missing_dates:
            print(f"No missing dates for check_data_table, type_identifier={type_identifier}")
            return []
        return missing_dates

    def logic_volume(self):
        """
        用wind.py更新行业和全A的成交量，用来构建拥挤度。
        """
        # 不需检查或更新meta_table，因为行业指数和全A已经存在于product_static_info
        # 直接检查或更新data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation',
                                               additional_filter=f"field=''")
        if missing_dates:
            self._upload_missing_data_price_valuation(missing_dates)

    def logic_stk_industry(self):
        """
        用tushare更新各行业包含的股票（及其量价），用来构建拥挤度。
        """
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code',
                                                        type_identifier='price_valuation')
        # 检查或更新data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation')
        if need_update_meta_table:
            self._upload_missing_meta_price_valuation()
        if missing_dates:
            self._upload_missing_data_price_valuation(missing_dates)

