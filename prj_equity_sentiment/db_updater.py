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
                # �޳������ȫA
                sector_codes = self.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
                                                                            "product_type='index'")
                sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']
                print(f'Wind downloading sectorconstituent for _check_meta_table')
                for _, row in sector_codes:
                    sector_code = row['code']
                    industry_name = row['chinese_name']
                    # ���ж��Ƿ���Ҫ���£��Է��˷�quota
                    sector_update_date = self.select_existing_values_in_target_column('product_static_info', 'update_date',
                                                                            f"product_type='stock' AND stk_industry_cs={industry_name}")
                    # �������û�����ݣ���������µ�ʱ�䳬����30��
                    if not sector_update_date or sector_update_date < self.tradedays[-1] - pd.Timedelta(days=30):
                        return True
            case _:
                raise Exception(f'type_identifier {type_identifier} not supported in _check_meta_table')


    def _check_data_table(self, table_name, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # ��ȡ��Ҫ���µ���������
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
        ��wind.py������ҵ��ȫA�ĳɽ����������ӵ���ȡ�
        """
        # ����������meta_table����Ϊ��ҵָ����ȫA�Ѿ�������product_static_info
        # ֱ�Ӽ������data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation',
                                               additional_filter=f"field='�ɽ���'")
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
                             'VOLUME': '�ɽ���',
                             'AMT': '�ɽ���',
                             'TURN': '������',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                           value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_stk_industry_price_volume(self):
        """
        ��tushare���¸���ҵ�����Ĺ�Ʊ���������ۣ�����������ӵ���ȡ�
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code',
                                                        type_identifier='price_volume')
        # �������data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_volume')
        if need_update_meta_table:
            self._upload_missing_meta_stk_industry_price_volume()
        if missing_dates:
            self._upload_missing_data_stk_industry_price_volume(missing_dates)

    def _upload_missing_meta_stk_industry_price_volume(self):
        sector_codes = self.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
                                                                    "product_type='index'")
        sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']

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

            # ����df���а���code, chinese_name, source, stk_industry_cs, product_type, update_date
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
            print(f"tushare downloading ���� for {code}")
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
                "open": '���̼�',
                "high": '��߼�',
                "low": '��ͼ�',
                "close": '���̼�',
                "pct_chg": '�����ǵ���',
                "vol": '�ɽ���',
                "amount": '�ɽ���'
            })
            # ת��Ϊ����ʽ���ݿ�
            df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)


    def logic_analyst(self):
        pass