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
        self.logic_volume()
        self.logic_stk_industry_price_volume()


    def _check_meta_table(self, type_identifier):
        match type_identifier:
            case 'price_volume':
                # 剔除掉万德全A
                sector_codes = self.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
                                                                            "product_type='index'")
                sector_codes = sector_codes[sector_codes['chinese_name'] != '万德全A']
                print(f'Wind downloading sectorconstituent for _check_meta_table')
                for _, row in sector_codes:
                    sector_code = row['code']
                    industry_name = row['chinese_name']
                    # 先判断是否需要更新，以防浪费quota
                    sector_update_date = self.select_existing_values_in_target_column('product_static_info', 'update_date',
                                                                            f"product_type='stock' AND stk_industry_cs={industry_name}")
                    # 如果现在没有数据，或最近更新的时间超过了30天
                    if not sector_update_date or sector_update_date < self.tradedays[-1] - pd.Timedelta(days=30):
                        return True
            case _:
                raise Exception(f'type_identifier {type_identifier} not supported in _check_meta_table')


    def _check_data_table(self, table_name, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # 获取需要更新的日期区间
        match type_identifier:
            case 'price_volume':
                filter_condition = f"product_static_info.product_type='stock' AND {table_name}.field='close'"
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
        用wind.py更新行业和全A的成交额，用来构建拥挤度。
        """
        # 不需检查或更新meta_table，因为行业指数和全A已经存在于product_static_info
        # 直接检查或更新data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation',
                                               additional_filter=f"field='成交额'")
        if missing_dates:
            self._upload_missing_data_price_valuation(missing_dates)

    def _upload_missing_data_price_valuation(self, missing_dates):
        industry_codes = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('type_identifier', 'price_valuation')
        )

        for date in missing_dates:
            for code in industry_codes:
                print(
                    f'Downloading and uploading volume,amt,turn for {code} {date} _upload_missing_data_price_valuation')
                df = w.wsd(code, "volume,amt,turn", date,
                           date, "unit=1", usedf=True)[1]
                if df.empty:
                    print(
                        f"Missing data for {date} {code}, no data downloaded for _upload_missing_data_price_valuation")
                    continue
                df_upload = df.rename(
                    columns={'WindCodes': 'code',
                             'VOLUME': '成交量',
                             'AMT': '成交额',
                             'TURN': '换手率',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                           value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_stk_industry_price_volume(self):
        """
        用tushare更新各行业包含的股票（及其量价），用来构建拥挤度。
        """
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code',
                                                        type_identifier='price_volume')
        # 检查或更新data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_volume')
        if need_update_meta_table:
            self._upload_missing_meta_stk_industry_price_volume()
        if missing_dates:
            self._upload_missing_data_stk_industry_price_volume(missing_dates)

    def _upload_missing_meta_stk_industry_price_volume(self):
        sector_codes = self.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
                                                                    "product_type='index'")
        sector_codes = sector_codes[sector_codes['chinese_name'] != '万德全A']

        for _, row in sector_codes.iterrows():
            sector_code = row['code']
            industry_name = row['chinese_name']
            print(f'Wind downloading {industry_name} sectorconstituent for _check_meta_table')
            today_industry_stocks_df = w.wset("sectorconstituent",
                                               f"date={self.tradedays_str[-2]};"
                                               f"windcode={sector_code}", usedf=True)[1]
            sector_stk_codes = today_industry_stocks_df['wind_code'].tolist()
            existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                          f"product_type='stock' AND stk_industry_cs={industry_name}")
            new_stk_codes = sector_stk_codes - existing_codes
            selected_rows = today_industry_stocks_df[today_industry_stocks_df['code'].isin(new_stk_codes)]

            # 构建df，列包含code, chinese_name, source, stk_industry_cs, product_type, update_date
            df_upload = selected_rows['wind_code', 'sec_name'].rename(columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
            df_upload['source'] = 'wind and tushare'
            df_upload['product_type'] = 'stock'
            df_upload['update_date'] = self.tradedays_str[-2]
            for i, row_stk in df_upload.iterrows():
                code = row_stk['code']
                date = self.tradedays[-2]
                print(f'Wind downloading industry_citic for {code} on {date}')
                info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
                                usedf=True)[1]
                if info_df.empty:
                    print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
                    continue
                industry = info_df.iloc[0]['INDUSTRY_CITIC']
                row_stk['stk_industry_cs'] = industry
                self.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
                self.insert_product_static_info(row_stk)
    def _upload_missing_data_stk_industry_price_volume(self, missing_dates):
        existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        for code in existing_codes:
            print(f"tushare downloading 行情 for {code}")
            df = self.pro.daily(**{
                "ts_code": code,
                "trade_date": "",
                "start_date": missing_dates[0],
                "end_date": missing_dates[-1],
                "offset": "",
                "limit": ""
            }, fields=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pct_chg",
                "vol",
                "amount"
            ])
            df = df.rename(columns={
                "ts_code": 'product_name',
                "trade_date": 'date',
                "open": '开盘价',
                "high": '最高价',
                "low": '最低价',
                "close": '收盘价',
                "pct_chg": '当日涨跌幅',
                "vol": '成交量',
                "amount": '成交额'
            })
            # 转换为长格式数据框
            df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)


    def logic_analyst(self):
        pass