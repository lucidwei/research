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
from utils import has_large_date_gap, match_recent_tradedays


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.logic_industry_volume()
        self.logic_industry_large_order()
        self.logic_industry_stk_price_volume()
        # self.logic_analyst()

    def _check_meta_table(self, type_identifier):
        match type_identifier:
            case 'price_volume':
                # �޳������ȫA
                sector_codes = self.select_existing_values_in_target_column('product_static_info',
                                                                            ['code', 'chinese_name'],
                                                                            "product_type='index'")
                sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']
                for _, row in sector_codes.iterrows():
                    industry_name = row['chinese_name']
                    # ���ж��Ƿ���Ҫ���£��Է��˷�quota
                    sector_update_date = self.select_existing_values_in_target_column('product_static_info',
                                                                                      'update_date',
                                                                                      f"product_type='stock' AND stk_industry_cs='{industry_name}'")
                    # �������û�����ݣ���������µ�ʱ�䳬����30��
                    if not sector_update_date:
                        print('No sector_update_date, logic_industry_stk_price_volume need update')
                        return True
                    elif len(sector_update_date) == 1:
                        sector_update_datetime = datetime.datetime.strptime(sector_update_date[0], '%Y-%m-%d').date()
                        if sector_update_datetime < self.tradedays[-1] - pd.Timedelta(days=30):
                            print('sector_update_date last updated over a month ago, '
                                  'logic_industry_stk_price_volume need update')
                            return True
                        else:
                            print('sector_update_date last updated less than a month ago, '
                                  'logic_industry_stk_price_volume not need update')
                            return False
                    else:
                        raise Exception('len(sector_update_date) should be either 0 or 1')
            case _:
                raise Exception(f'type_identifier {type_identifier} not supported in _check_meta_table')

    def _check_data_table(self, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # ��ȡ��Ҫ���µ���������
        match type_identifier:
            case 'industry_volume':
                filter_condition = f"product_static_info.product_type='index' AND markets_daily_long.field='�ɽ���'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'price_volume':
                filter_condition = f"product_static_info.product_type='stock' AND markets_daily_long.field='���̼�'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'stk_price_volume':
                stock_code = kwargs.get('stock_code')
                filter_condition = f"stocks_daily_long.product_name='{stock_code}' AND stocks_daily_long.field='���̼�'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='stocks_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'industry_large_order':
                filter_condition = f"markets_daily_long.field='�����������'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'analyst':
                stock_code = kwargs.get('stock_code')
                filter_condition = f"markets_daily_long.product_name='{stock_code}' AND markets_daily_long.field='ȯ��-�б�����'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
                # �б����ݷ������������Ҫ�ر�����
                if len(existing_dates) == 0:
                    missing_dates = self.tradedays
                else:
                    missing_dates = []
                    missing_dates[0] = existing_dates[-1] + datetime.timedelta(days=1)
                    missing_dates[1] = self.tradedays[-1]
                return missing_dates

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

    def logic_industry_volume(self):
        """
        ��wind.py������ҵ��ȫA�ĳɽ����������ӵ���ȡ�
        """
        # ����������meta_table����Ϊ��ҵָ����ȫA�Ѿ�������product_static_info
        # ֱ�Ӽ������data_table
        missing_dates = self._check_data_table(type_identifier='industry_volume',
                                               additional_filter=f"field='�ɽ���'")
        if missing_dates:
            self._upload_missing_data_industry_volume(missing_dates)

    def _upload_missing_data_industry_volume(self, missing_dates):
        industry_codes = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        if missing_dates[0] == self.tradedays[-1]:
            print('Only today is missing, skipping update _upload_missing_data_industry_volume')
            return
        for date in missing_dates:
            for code in industry_codes:
                print(
                    f'Wind downloading and upload volume,amt,turn for {code} {date} _upload_missing_data_industry_volume')
                df = w.wsd(code, "volume,amt,turn", date,
                           date, "unit=1", usedf=True)[1]
                if df.empty:
                    print(
                        f"Missing data for {date} {code}, _upload_missing_data_industry_volume")
                    continue
                df_upload = df.rename(
                    columns={'VOLUME': '�ɽ���',
                             'AMT': '�ɽ���',
                             'TURN': '������',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                           value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_industry_large_order(self):
        """
        ��wind.py������ҵ��ȫA��������������������ӵ���ȡ�
        """
        # ����������meta_table����Ϊ��ҵָ����ȫA�Ѿ�������product_static_info
        missing_dates = self._check_data_table(type_identifier='industry_large_order',
                                               additional_filter=f"field='�����������'")
        if missing_dates:
            self._upload_missing_data_industry_large_order(missing_dates)

    def _upload_missing_data_industry_large_order(self, missing_dates):
        industry_codes = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        if missing_dates[0] == self.tradedays[-1]:
            print('Only today is missing, skipping update _upload_missing_data_industry_large_order')
            return

        for code in industry_codes:
            print(
                f'Wind downloading and upload mfd_inflow_m for {code} {missing_dates[0]}~{missing_dates[-1]} '
                f'_upload_missing_data_industry_large_order')
            df = w.wsd(code, "mfd_inflow_m", missing_dates[0],
                       missing_dates[-1], "unit=1", usedf=True)[1]
            if df.empty:
                print(
                    f"Missing data for {code} {missing_dates[0]}~{missing_dates[-1]}, "
                    f"_upload_missing_data_industry_large_order")
                continue
            df_upload = df.reset_index().rename(
                columns={'index': 'date',
                         'MFD_INFLOW_M': '�����������',
                         })
            df_upload['product_name'] = code
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                       value_name='value').dropna()

            df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_industry_stk_price_volume(self):
        """
        ��tushare���¸���ҵ�����Ĺ�Ʊ���������ۣ�����������ӵ���ȡ�
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table(type_identifier='price_volume')
        # �״�run��forceΪTRUE
        # need_update_meta_table = True
        # �������data_table
        missing_dates = self._check_data_table(type_identifier='price_volume')
        if need_update_meta_table:
            self._upload_missing_meta_stk_industry_price_volume()
        if missing_dates:
            self._upload_missing_data_stk_price_volume()

    def _upload_missing_meta_stk_industry_price_volume(self):
        sector_codes = self.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
                                                                    "product_type='index'")
        sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']

        for _, row in sector_codes.iterrows():
            sector_code = row['code']
            industry_name = row['chinese_name']
            print(f'Wind downloading {industry_name} sectorconstituent for _upload_missing_meta_stk_industry_price_volume')
            today_industry_stocks_df = w.wset("sectorconstituent",
                                              f"date={self.tradedays_str[-2]};"
                                              f"windcode={sector_code}", usedf=True)[1]
            sector_stk_codes = today_industry_stocks_df['wind_code'].tolist()
            existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                          f"product_type='stock' AND stk_industry_cs='{industry_name}'")
            new_stk_codes = set(sector_stk_codes) - set(existing_codes)
            selected_rows = today_industry_stocks_df[today_industry_stocks_df['wind_code'].isin(new_stk_codes)]

            # ����df���а���code, chinese_name, source, stk_industry_cs, product_type, update_date
            df_upload = selected_rows[['wind_code', 'sec_name']].rename(
                columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
            df_upload['source'] = 'wind and tushare'
            df_upload['product_type'] = 'stock'
            df_upload['update_date'] = self.tradedays_str[-2]
            for i, row_stk in df_upload.iterrows():
                code = row_stk['code']
                date = self.tradedays[-2]
                print(f'Wind downloading industry_citic for {code} on {date} for _upload_missing_meta_stk_industry_price_volume')
                info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
                                usedf=True)[1]
                if info_df.empty:
                    print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
                    continue
                industry = info_df.iloc[0]['INDUSTRY_CITIC']
                row_stk['stk_industry_cs'] = industry
                self.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
                self.insert_product_static_info(row_stk)

    def _upload_missing_data_stk_price_volume(self):
        existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        for code in existing_codes[3810:]:
            missing_dates = self._check_data_table(type_identifier='stk_price_volume',
                                                   stock_code=code)
            missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]
            if missing_dates[0] == self.tradedays[-1]:
                print(f'{code} Only today is missing, skipping update _upload_missing_data_stk_price_volume')
                continue
            elif 2 <= len(missing_dates) <= 5:
                # missing_dates������5�������պ���������
                missing_dates_recent = [date for date in missing_dates if date in self.tradedays[-5:]]
                missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates_recent]
                # ֻȱ�������ݣ�����
                if missing_dates_recent[0] == self.tradedays[-1]:
                    print(f'{code} ֻȱһ�������ݲ�����, skipping update _upload_missing_data_stk_price_volume')
                    continue
            elif has_large_date_gap(missing_dates) and missing_dates[-1] == self.tradedays[-1]:
                print(f'{code} �ڼ���ͣ��, ���Ѿ����¹���ֻȱ���죬 skipping update _upload_missing_data_stk_price_volume')
                continue
            elif not has_large_date_gap(missing_dates) and not match_recent_tradedays(missing_dates, self.tradedays):
                print(f'{code} ����ͣ��, ���������¹��ˣ� skipping update _upload_missing_data_stk_price_volume')
                continue
            elif self.tradedays[-1] - missing_dates[-6] > datetime.timedelta(days=15):
                print(f'{code}���ڼ�������Ϊ�¹ɣ��ҷ��к�������Ѿ����¹��ˣ� skipping update _upload_missing_data_stk_price_volume')
            elif code == '900936.SH':
                # B��û������
                continue

            print(f"tushare downloading ���� for {code} {missing_dates[0]}~{missing_dates[-2]}")
            df = self.pro.daily(**{
                "ts_code": code,
                "trade_date": "",
                "start_date": missing_dates_str_list[0],
                "end_date": missing_dates_str_list[-2],
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
            df_long.to_sql('stocks_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_analyst(self):
        existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        for code in existing_codes:
            missing_dates = self._check_data_table(type_identifier='analyst',
                                                   stock_code=code)
            missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]
            print(f"tushare downloading �б���ʷ for {code} {missing_dates[0]}-{missing_dates[1]}")
            df = self.pro.report_rc(**{
                "ts_code": code,
                "report_date": "",
                "start_date": missing_dates_str_list[0],
                "end_date": missing_dates_str_list[-1],
                "limit": "",
                "offset": ""
            }, fields=[
                "ts_code",
                "report_date",
                "report_title",
                "report_type",
                "classify",
                "org_name",
                "rating"
            ])
            df = df.rename(columns={
                "ts_code": 'product_name',
                "report_date": 'date',
                "report_title": '�������',
                "report_type": '��������',
                "classify": '�������',
                "org_name": '��������',
                "rating": '��������',
            }).drop_duplicates()
            df['ȯ��-�б�����'] = df['��������'] + '-' + df['�������']
            df_selected = df[['date', 'product_name', 'ȯ��-�б�����']].drop_duplicates()
            df_report_counts = df_selected.groupby(['product_name', 'date'])['ȯ��-�б�����'].nunique().reset_index()
            df_report_counts.columns = ['product_name', 'date', '��������']

            #TODO: ��Ϊ�Ƕ������ݣ���Ҫ�޳��Ѵ��ڵ�

            # ת��Ϊ����ʽ���ݿ�
            df_long = pd.melt(df_report_counts, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)