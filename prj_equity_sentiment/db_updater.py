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
        self.industry_data_updater = IndustryDataUpdater(self)
        self.industry_stk_updater = IndustryStkUpdater(self)

    def run_all_updater(self):
        self.industry_data_updater.logic_industry_volume()
        self.industry_data_updater.logic_industry_large_order()
        self.industry_data_updater.logic_industry_order_inflows()
        # self.industry_stk_updater.logic_industry_stk_price_volume()
        # self.logic_analyst()

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
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='stocks_daily_partitioned_close',
                )
            case 'stk_price_volume':
                stock_code = kwargs.get('stock_code')
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='stocks_daily_partitioned_close',
                    product_name=stock_code
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
            case 'industry_order_inflows':
                field = kwargs.get('field')
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='markets_daily_long',
                    field=field,
                    filter_condition=additional_filter
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
                # �б����ݷ������������Ҫ�ر���
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


class IndustryDataUpdater:
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater
        self.excel_data = None

    def logic_industry_volume(self):
        """
        ��wind.py������ҵ��ȫA�ĳɽ����������ӵ���ȡ�
        """
        # ����������meta_table����Ϊ��ҵָ����ȫA�Ѿ�������product_static_info
        # ֱ�Ӽ������data_table
        missing_dates = self.db_updater._check_data_table(type_identifier='industry_volume',
                                               additional_filter=f"field='�ɽ���'")
        missing_dates_filtered = self.db_updater.remove_today_if_trading_day(missing_dates)
        if missing_dates_filtered:
            self._upload_missing_data_industry_volume(missing_dates_filtered)

    def _upload_missing_data_industry_volume(self, missing_dates):
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        if missing_dates[0] == self.db_updater.tradedays[-1]:
            print('Only today is missing, skipping update _upload_missing_data_industry_volume')
            return
        for date in missing_dates[::-1]:
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

                df_upload.to_sql('markets_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def logic_industry_large_order(self):
        """
        ��wind.py������ҵ��ȫA��������������������ӵ���ȡ�
        """
        # ����������meta_table����Ϊ��ҵָ����ȫA�Ѿ�������product_static_info
        missing_dates = self.db_updater._check_data_table(type_identifier='industry_large_order',
                                               additional_filter=f"field='�����������'")
        # ÿ�����ĵ��ͳ��˽�������õȵ��ڶ����ٸ���
        # missing_dates_filtered = self.db_updater.remove_today_if_trading_day(missing_dates)

        if missing_dates:
            self._upload_missing_data_industry_large_order(missing_dates)

    def _upload_missing_data_industry_large_order(self, missing_dates):
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        # if missing_dates[0] == self.db_updater.tradedays[-1]:
        #     print('Only today is missing, skipping update _upload_missing_data_industry_large_order')
        #     return

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
            if len(missing_dates) == 1:
                df_upload = df.reset_index().rename(
                    columns={'index': 'product_name',
                             'MFD_INFLOW_M': '�����������',
                             })
                df_upload['date'] = missing_dates[0]
            else:
                df_upload = df.reset_index().rename(
                    columns={'index': 'date',
                             'MFD_INFLOW_M': '�����������',
                             })
                df_upload['product_name'] = code
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                       value_name='value').dropna()
            # ȥ���Ѿ����ڵ�����
            existing_dates = self.db_updater.select_existing_dates_from_long_table('markets_daily_long',
                                                                        product_name=code,
                                                                        field='�����������')
            downloaded_filtered = df_upload[~df_upload['date'].isin(existing_dates)]

            downloaded_filtered.to_sql('markets_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def logic_industry_order_inflows(self):
        """
        1���ҵ���С��4��Ԫ��С����
        2���ҵ���4��Ԫ��20��Ԫ֮�䣬�е���
        3���ҵ���20��Ԫ��100��Ԫ֮�䣬�󵥣�
        4���ҵ������100��Ԫ�����󵥡�
        �����ù�ģ��������ɽ����
        �������ù�ģ��������-�������
        �����������ù�ģ��������ȥ�ɽ����ǹҵ��ɽ��Ľ��
        large_orderΪ�󵥺ͳ��󵥵ľ������֮�ͣ�������
        ���ֹ�ģ������ھ����ͬ��ģ�Ӻ�Ӧ�ӽ�0�����Ǻ����
        """
        # ��ȡ������ҵ����
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('internal_id BETWEEN 69000 AND 69030')
        )

        trader_type_dict = {
            1: '����',
            2: '��',
            3: '�л�',
            4: 'ɢ��',
        }
        self.trader_type_dict = trader_type_dict

        # self._preprocess_excel(rf"D:\WPS����\WPS����\����-���\�о�trial\��ʷorder_inflows.xlsx")

        for code in industry_codes:
            for trader_type in [1, 2, 3, 4]:
                trader_type_name = trader_type_dict.get(trader_type, f"Type{trader_type}")
                field = f'�����_{trader_type_name}'  # ����ɸѡ������
                additional_filter = f"product_name='{code}'"

                # ��ѯ��ǰ code �� trader_type ��ȱʧ����
                missing_dates = self.db_updater._check_data_table(
                    type_identifier='industry_order_inflows',
                    field=field,
                    additional_filter=additional_filter
                )

                if missing_dates:
                    print(
                        f"��ʼ�ϴ�ȱʧ���� - ��ҵ����: {code}, ��������: {trader_type_name}, ȱʧ������: {len(missing_dates)}")
                    self._upload_missing_data_industry_order_inflows(code, trader_type, missing_dates)
                else:
                    print(f"No missing dates for ��ҵ����: {code}, ��������: {trader_type_name}")


    def _upload_missing_data_industry_order_inflows(self, code, trader_type, missing_dates):
        if code == 'CI005030.WI' and len(missing_dates)>1: # ����ҵȱ��������
            missing_dates = [date for date in missing_dates if date > datetime.date(2019, 11, 29)]
        # ��ȡȱʧ���ڵ���ֹʱ��
        start_date = missing_dates[0]
        end_date = missing_dates[-1]

        # ���Դ�Excel�л�ȡ����
        df_upload = self._get_data_from_excel(code, trader_type, start_date, end_date)

        if df_upload is None:
        # ���Excel��û�����ݣ���ͨ��w.wsd��������
            print(
                f'Wind downloading and upload order_inflows for {code} trader_type {self.trader_type_dict[trader_type]} {start_date}~{end_date} '
                f'_upload_missing_data_industry_order_inflows')
            df = w.wsd(code, "mfd_buyamt_d,mfd_netbuyamt,mfd_netbuyamt_a", start_date,
                       missing_dates[-1], "unit=1", traderType=trader_type, usedf=True)[1]
            if df.empty:
                print(
                    f"Missing data for {code} {start_date}~{end_date}, "
                    f"_upload_missing_data_industry_order_order_inflows")
                return
            if len(missing_dates) == 1:
                df_upload = df.reset_index().rename(
                    columns={'index': 'product_name',
                             'MFD_BUYAMT_D': f'�����_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT': f'�������_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT_A': f'�����������_{self.trader_type_dict[trader_type]}',
                             })
                df_upload['date'] = missing_dates[0]
            else:
                df_upload = df.reset_index().rename(
                    columns={'index': 'date',
                             'MFD_BUYAMT_D': f'�����_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT': f'�������_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT_A': f'�����������_{self.trader_type_dict[trader_type]}',
                             })
                df_upload['product_name'] = code

        df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                   value_name='value').dropna()
        self.db_updater.upsert_dataframe_to_postgresql(df_upload, 'markets_daily_long',
                                            ['date', 'product_name', 'field', 'code'])

    def _preprocess_excel(self, excel_path):
        """
        Ԥ����Excel�ļ�������ת��Ϊ�ṹ����DataFrame��
        �����û������ĸ��ӽṹ���������ÿ�����ݿ顣
        """
        try:
            # ��ȡExcel�ļ����ޱ�ͷ
            raw_df = pd.read_excel(excel_path, sheet_name='Sheet3', header=None, dtype=str)

            # ����
            total_cols = raw_df.shape[1]

            # �ҵ��������ݿ����ʼ�У���0�а���"�����"
            data_block_cols = raw_df.columns[
                raw_df.iloc[0].astype(str).str.contains('��', na=False)
            ].tolist()

            # ��ʼ��һ���б�洢��������
            data_records = []

            for start_col in data_block_cols:
                # ��ȡ�ֶ�Ԫ���ݣ����� "����� [��λ]Ԫ [����]����"
                field_metadata = raw_df.iloc[0, start_col]

                if pd.isna(field_metadata):
                    continue

                # �����ֶ��������� "����� [��λ]Ԫ [����]����" -> "�����_����"
                if '[����]' in field_metadata:
                    parts = field_metadata.split('[����]')
                    field_name = (
                            parts[0].strip().replace('\n', '').replace('[��λ]Ԫ', '') +
                            '_' +
                            parts[1].strip(']').strip()
                    )
                else:
                    raise Exception('Not expected in splitting metadata str')

                # ȷ�����ݿ�Ľ����У���һ�����ݿ����ʼ��֮ǰ���У�
                current_index = data_block_cols.index(start_col)
                if current_index < len(data_block_cols) - 1:
                    end_col = data_block_cols[current_index + 1] - 1
                else:
                    end_col = total_cols - 1

                # ��ȡ���ı��⣨��2�У�����Ϊ2��
                chinese_headers = raw_df.iloc[2, start_col:end_col + 1].tolist()

                # ��ȡӢ�Ĵ��루��3�У�����Ϊ3��
                codes = raw_df.iloc[3, start_col:end_col + 1].tolist()

                # ������ǰ���ݿ��е�ÿ����ҵ�У�������һ�С����ڡ���
                date_col = start_col -1
                for col_idx in range(start_col, end_col + 1):
                    code = codes[col_idx - start_col]
                    if pd.isna(code):
                        # �����ҵ����Ϊ�գ�����
                        continue

                    # ���������У��ӵ�4�п�ʼ������Ϊ4��
                    for row in range(4, len(raw_df)):
                        date_str = raw_df.iloc[row, date_col]  # ���������������ǰһ��
                        if pd.isna(date_str):
                            continue

                        try:
                            date = pd.to_datetime(date_str).date()
                        except:
                            # �������ת��ʧ�ܣ���������
                            continue

                        value_str = raw_df.iloc[row, col_idx]
                        if pd.isna(value_str):
                            continue

                        try:
                            value = float(value_str)
                        except ValueError:
                            # ���ֵת��ʧ�ܣ�����
                            continue

                        data_records.append({
                            'date': date,
                            'code': code,
                            'field': field_name,
                            'value': value
                        })

            # ����DataFrame
            self.excel_data = pd.DataFrame(data_records)

            # ��¼Ԥ����ɹ�����Ϣ
            print("Successfully preprocessed Excel data.")

        except Exception as e:
            # ���񲢼�¼�����쳣
            print(f"Error preprocessing Excel file: {e}")
            self.excel_data = None

    def _get_data_from_excel(self, code, trader_type, start_date, end_date):
        """
        ��Ԥ�����Excel�����л�ȡָ������ҵ���롢�������ͺ����ڷ�Χ�����ݡ�
        ������ݴ��ڣ��򷵻�DataFrame�����򣬷���None��

        :param code: ��ҵ����
        :param trader_type: �������ͣ�������1-4��
        :param start_date: ��ʼ���ڣ��ַ�����ʽ��
        :param end_date: �������ڣ��ַ�����ʽ��
        :return: DataFrame��None
        """
        if self.excel_data is None:
            print("Excel data not loaded or preprocessing failed.")
            return None

        trader_type_dict = {
            1: '����',
            2: '��',
            3: '�л�',
            4: 'ɢ��',
        }
        trader_type_name = trader_type_dict.get(trader_type, f"Type{trader_type}")

        # ת�����ڸ�ʽ
        try:
            start_date_obj = pd.to_datetime(start_date).date()
            end_date_obj = pd.to_datetime(end_date).date()
        except Exception as e:
            print(f"Error converting dates: {e}")
            return None

        # ����DataFrame
        filtered_df = self.excel_data[
            (self.excel_data['code'] == code) &
            (self.excel_data['field'].str.contains(trader_type_name)) &
            (self.excel_data['date'] >= start_date_obj) &
            (self.excel_data['date'] <= end_date_obj)
            ]

        if filtered_df.empty:
            print(f"No Excel data found for {code} {trader_type_name} from {start_date} to {end_date}.")
            return None

        # ������ת��Ϊԭʼ��ʽ���Ա��ϴ�
        try:
            # Pivot to wide format
            pivot_df = filtered_df.pivot(index='date', columns='field', values='value').reset_index()
            pivot_df['product_name'] = code

            if pivot_df.empty:
                return None

            return pivot_df
        except Exception as e:
            print(f"Error processing Excel data for {code} {trader_type_name}: {e}")
            return None


class IndustryStkUpdater:
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater

    def _check_meta_table(self):
        # �޳������ȫA
        sector_codes = self.db_updater.select_existing_values_in_target_column('product_static_info',
                                                                    ['code', 'chinese_name'],
                                                                    "product_type='index'")
        sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']
        for _, row in sector_codes.iterrows():
            industry_name = row['chinese_name']
            # ���ж��Ƿ���Ҫ���£��Է��˷�quota
            sector_update_date = self.db_updater.select_existing_values_in_target_column('product_static_info',
                                                                              'update_date',
                                                                              f"product_type='stock' AND stk_industry_cs='{industry_name}'")
            # �������û�����ݣ���������µ�ʱ�䳬����30��
            if not sector_update_date:
                print('No sector_update_date, logic_industry_stk_price_volume need update')
                return True
            elif len(sector_update_date) == 1:
                sector_update_datetime = datetime.datetime.strptime(sector_update_date[0], '%Y-%m-%d').date()
                if sector_update_datetime < self.db_updater.tradedays[-1] - pd.Timedelta(days=30):
                    print('sector_update_date last updated over a month ago, '
                          'logic_industry_stk_price_volume need update')
                    return True
                else:
                    print('sector_update_date last updated less than a month ago, '
                          'logic_industry_stk_price_volume not need update')
                    return False
            else:
                raise Exception('len(sector_update_date) should be either 0 or 1')

    def logic_industry_stk_price_volume(self):
        """
        ��tushare���¸���ҵ�����Ĺ�Ʊ���������ۣ�����������ӵ���ȡ�
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table()
        # �������data_table
        missing_dates = self.db_updater._check_data_table(type_identifier='price_volume')
        missing_dates_filtered = self.db_updater.remove_today_if_trading_time(missing_dates)

        if need_update_meta_table:
            self._upload_missing_meta_stk_industry_price_volume()
        if missing_dates_filtered:
            self._upload_whole_market_data_by_missing_dates(missing_dates_filtered)

        self._upload_missing_data_stk_price_volume()

    def _upload_missing_meta_stk_industry_price_volume(self):
        print(f"Wind downloading 'ȫ��A��' sectorconstituent for _upload_missing_meta_stk_industry_price_volume")
        today_stocks_df = w.wset("sectorconstituent",
                                          f"date={self.db_updater.tradedays_str[-2]};"
                                          f"sectorid=a001010100000000", usedf=True)[1]
        sector_stk_codes = today_stocks_df['wind_code'].tolist()
        existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                                 f"product_type='stock'")
        existing_codes_missing_industry = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                                 f"product_type='stock' AND (stk_industry_cs='NaN' OR stk_industry_cs is null)")
        new_stk_codes = set(sector_stk_codes) - set(existing_codes)
        stk_needing_update = new_stk_codes.union(set(existing_codes_missing_industry))
        selected_rows = today_stocks_df[today_stocks_df['wind_code'].isin(stk_needing_update)]

        # ����df���а���code, chinese_name, source, stk_industry_cs, product_type, update_date
        df_upload = selected_rows[['wind_code', 'sec_name']].rename(
            columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
        df_upload['source'] = 'wind'
        df_upload['product_type'] = 'stock'
        df_upload['update_date'] = self.db_updater.all_dates_str[-1]
        for i, row_stk in df_upload.iterrows():
            code = row_stk['code']
            date = self.db_updater.tradedays[-2]
            print(
                f'Wind downloading industry_citic for {code} on {date} for _upload_missing_meta_stk_industry_price_volume')
            info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
                            usedf=True)[1]
            if info_df.empty:
                print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
                continue
            industry = info_df.iloc[0]['INDUSTRY_CITIC']
            row_stk['stk_industry_cs'] = industry
            self.db_updater.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
            self.db_updater.insert_product_static_info(row_stk)

        # ����ҵ��ɽǶȸ��¸�����ҵ�����������ࡣ
        # sector_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
        #                                                             "product_type='index'")
        # sector_codes = sector_codes[sector_codes['chinese_name'] != '���ȫA']
        #
        # for _, row in sector_codes.iterrows():
        #     sector_code = row['code']
        #     industry_name = row['chinese_name']
        #     print(f'Wind downloading {industry_name} sectorconstituent for _upload_missing_meta_stk_industry_price_volume')
        #     today_industry_stocks_df = w.wset("sectorconstituent",
        #                                       f"date={self.db_updater.tradedays_str[-2]};"
        #                                       f"windcode={sector_code}", usedf=True)[1]
        #     sector_stk_codes = today_industry_stocks_df['wind_code'].tolist()
        #     existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
        #                                                                   f"product_type='stock' AND stk_industry_cs='{industry_name}'")
        #     new_stk_codes = set(sector_stk_codes) - set(existing_codes)
        #     selected_rows = today_industry_stocks_df[today_industry_stocks_df['wind_code'].isin(new_stk_codes)]
        #
        #     # ����df���а���code, chinese_name, source, stk_industry_cs, product_type, update_date
        #     df_upload = selected_rows[['wind_code', 'sec_name']].rename(
        #         columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
        #     df_upload['source'] = 'wind and tushare'
        #     df_upload['product_type'] = 'stock'
        #     df_upload['update_date'] = self.db_updater.tradedays_str[-2]
        #     for i, row_stk in df_upload.iterrows():
        #         code = row_stk['code']
        #         date = self.db_updater.tradedays[-2]
        #         print(f'Wind downloading industry_citic for {code} on {date} for _upload_missing_meta_stk_industry_price_volume')
        #         info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
        #                         usedf=True)[1]
        #         if info_df.empty:
        #             print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
        #             continue
        #         industry = info_df.iloc[0]['INDUSTRY_CITIC']
        #         row_stk['stk_industry_cs'] = industry
        #         self.db_updater.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
        #         self.db_updater.insert_product_static_info(row_stk)

    def _upload_whole_market_data_by_missing_dates(self, missing_dates: list):
        for date in missing_dates:
            date_str = date.strftime('%Y%m%d')
            print(f"tushare downloading ȫ�г����� for {date}")
            df = self.db_updater.pro.daily(**{
                "ts_code": "",
                "trade_date": date_str,
                "start_date": "",
                "end_date": "",
                "offset": "",
                "limit": ""
            }, fields=[
                "ts_code",
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
                "open": '���̼�',
                "high": '��߼�',
                "low": '��ͼ�',
                "close": '���̼�',
                "pct_chg": '�����ǵ���',
                "vol": '�ɽ���',
                "amount": '�ɽ���'
            })
            df['date'] = date
            # ת��Ϊ����ʽ���ݿ�
            print('_upload_whole_market_data_by_missing_dates�ϴ��С���')
            df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def _upload_missing_data_stk_price_volume(self):
        """
        ����ʱ�䣺ÿ�����̺�
        """
        current_time = datetime.datetime.now()
        close_time = datetime.datetime.now().replace(hour=15, minute=10, second=0, microsecond=0)
        if current_time<close_time:
            print("��δ���̣��նȹ�Ʊ���ݲ�������")
            return

        existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        empty_request_count = 0
        for code in existing_codes:
            print(f"processing {code}")
            missing_dates = self.db_updater._check_data_table(type_identifier='stk_price_volume',
                                                   stock_code=code)
            missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]

            if code.startswith("900") or code.startswith("200"):
                # B��tushareû������
                continue
            if not missing_dates:
                continue

            if 2 <= len(missing_dates) <= 150:
                # ̫��Զ�Ĳ�����
                # �����������missing_dates���и���
                missing_dates_recent = [date for date in missing_dates if date in self.db_updater.tradedays[-150:]]
                missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates_recent]
            elif has_large_date_gap(missing_dates):
                print(missing_dates) # TODO ����̫Ƶ����Ҫdebug
                print(f'{code} �ڼ���ͣ��, ���Ѿ����¹��ˣ� skipping update _upload_missing_data_stk_price_volume')
                continue
            elif not has_large_date_gap(missing_dates) and not match_recent_tradedays(missing_dates, self.db_updater.tradedays):
                print(f'{code} ����ͣ��, ���������¹��ˣ� skipping update _upload_missing_data_stk_price_volume')
                continue
            elif self.db_updater.tradedays[-1] - missing_dates[-6] > datetime.timedelta(days=15):
                print(f'{code}���ڼ�������Ϊ�¹ɣ��ҷ��к�������Ѿ����¹��ˣ� skipping update _upload_missing_data_stk_price_volume')
                continue

            if not missing_dates_str_list:
                continue
            # ͣ�����ݻ᷵�ؿ�df����˲��ذ��ո���
            print(f"tushare downloading ���� for {code} {missing_dates_str_list[0]}~{missing_dates_str_list[-1]}")
            df = self.db_updater.pro.daily(**{
                "ts_code": code,
                "trade_date": "",
                "start_date": missing_dates_str_list[0],
                "end_date": missing_dates_str_list[-1],
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
            if df.empty:
                print(f"�����ϴ� {code} {missing_dates_str_list[0]}~{missing_dates_str_list[-1]}��Ϊtushare���ؿ�����(ͣ��/δ����)")
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
            df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

        # �в�ͨ����Ǹ����ÿ�������ʸýӿ�20000�Σ�Ȩ�޵ľ���������ʣ�https://tushare.pro/document/1?doc_id=108
        #     for date in missing_dates_str_list:
        #         print(f"tushare downloading ���� for {code} on {date}")
        #         df = self.db_updater.pro.daily(**{
        #             "ts_code": code,
        #             "trade_date": date,
        #             "start_date": "",
        #             "end_date": "",
        #             "offset": "",
        #             "limit": ""
        #         }, fields=[
        #             "ts_code",
        #             "trade_date",
        #             "open",
        #             "high",
        #             "low",
        #             "close",
        #             "pct_chg",
        #             "vol",
        #             "amount"
        #         ])
        #         if df.empty:
        #             print(f"���� {code} on {date} ��Ϊ������")
        #             empty_request_count += 1
        #             continue
        #         df = df.rename(columns={
        #             "ts_code": 'product_name',
        #             "trade_date": 'date',
        #             "open": '���̼�',
        #             "high": '��߼�',
        #             "low": '��ͼ�',
        #             "close": '���̼�',
        #             "pct_chg": '�����ǵ���',
        #             "vol": '�ɽ���',
        #             "amount": '�ɽ���'
        #         })
        #         # ת��Ϊ����ʽ���ݿ�
        #         df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
        #         df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)
        # print(f"empty_request_count={empty_request_count}")
