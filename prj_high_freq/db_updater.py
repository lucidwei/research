# coding=gbk
# Time Created: 2023/3/24 20:25
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re, math
from WindPy import w
import pandas as pd
from utils import timeit, get_nearest_dates_from_contract, check_wind
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from sqlalchemy import text, MetaData, Table
from pypinyin import lazy_pinyin


class DatabaseUpdater(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_dates()
        self.update_estate_area()
        self.process_estate_area()
        self.set_all_nan_to_null()
        self.close()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str

    @timeit
    def update_estate_area(self):
        """
        �Ȼ�ȡȱʧ�������б�,��Ҫ���µ����������ǣ�
        - all_dates������һ�쵽���ݿ�������ݵ�����һ��
        - ���ݿ�������ݵ����һ�쵽all_dates�����һ�죬Ҳ���ǽ���
        """
        # ����DataFrame�������ͱ��������Ķ�Ӧ��ϵ
        self.metadata = self.base_config.process_wind_metadata('30��metadata.xlsx')
        indicator_id_col = self.metadata.loc[:, 'ָ��ID']
        indicator_name_col = self.metadata.loc[:, 'ָ������']

        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ���ַ���
        id_to_new_string = {}
        for ind_id, ind_name in zip(indicator_id_col, indicator_name_col):
            # ��ȡָ�����Ƶ�ǰ��������
            first_two_chars = ind_name[:2]

            # ת��Ϊƴ��
            pinyin_str = ''.join(lazy_pinyin(first_two_chars))

            # �������ַ���
            new_string = f'estate_new_{pinyin_str}'

            # ��ָ�� ID �����ַ�����ӵ��ֵ���
            id_to_new_string[ind_id] = new_string

            # ����source_code
            source_code = f"wind_{ind_id}"

            # ע�⣬��λת�����ϴ����ݲ������
            unit = '��ƽ����'

            # ��metric_static_info���в���source_code, chinese_name, ��unit
            with self.alch_engine.begin() as connection:
                connection.execute(text(f"""
                    INSERT INTO metric_static_info (source_code, chinese_name, unit)
                    VALUES ('{source_code}', '{ind_name}', '{unit}')
                    ON CONFLICT (source_code) DO UPDATE
                    SET chinese_name = EXCLUDED.chinese_name,
                        unit = EXCLUDED.unit;
                """))

        self.cities_col = id_to_new_string.values()

        # ��������
        dates_missing = self.get_missing_dates(self.all_dates, "estate_new_wide")
        ids = self.metadata['ָ��ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print('Wind downloading for high_freq ���ز���������')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(dates_missing[-1]), usedf=True)[1]
            # wind��ʱ�᷵��Ī��������У��ѷǿ�С��2������ɾ��
            downloaded_df.dropna(thresh=2, inplace=True)
            # wind���ص�df������Ϊһ��Ͷ���ĸ�ʽ��һ��
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing

            # ��������Ϊ���ݿ�����
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)
            downloaded_df.rename(columns=id_to_new_string, inplace=True)

            # ��ȡ��λ�У�ת����λ
            unit_col = self.metadata.loc[:, '��λ']
            # �ҵ���λΪ '��ƽ����' ��ָ�� ID
            indicator_ids_to_adjust = [ind_id for ind_id, unit in zip(indicator_id_col, unit_col) if unit == '��ƽ����']
            # ������Щָ�� ID�������ݳ��� 10000
            for ind_id in indicator_ids_to_adjust:
                new_col_name = id_to_new_string[ind_id]
                downloaded_df[new_col_name] = downloaded_df[new_col_name] * 10000

            # �����в������ݿ���
            long_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            long_df.dropna(inplace=True)
            long_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    def process_estate_area(self):
        """
        ���ȣ��������������һ��һ�δ����������µ�����������5��1�ź�5��10�ź�5��17�ţ����Ȱ�5��11~5��17��ֵ��Ϊ5��17�ŵ�ֵ����7��
        Ȼ���5��2�ŵ�5��10�ŵ�ֵ��Ϊ5��10�ŵ�ֵ����9��������Լ���ʵӦ��ȡ�������ݱ��еĿ�ֵ����+1��ע�⣬ÿһ��֮��ļ���ǲ��㶨�ġ�
        """

        # Read data from the estate_new_wide view
        df = pd.read_sql(text("SELECT * FROM estate_new_wide"), self.alch_conn)

        # Define a function to process the columns
        def process_column(column):
            non_null_indices = df[column].dropna().index
            start_idx = df.index.min()

            # Iterate through the non-null indices
            for idx in non_null_indices:
                # Fill missing values between start_idx and idx (inclusive) with the value at idx divided by the number of missing values + 1
                df.loc[start_idx:idx, column] = df.loc[idx, column] / (idx - start_idx + 1)
                start_idx = idx + 1

            # Fill remaining missing values after the last non-null index
            if start_idx < len(df):
                df.loc[start_idx:, column] = 0

        # Process estate_new_xiamen and estate_new_wuxi
        for column in ['estate_new_xiamen', 'estate_new_wuxi']:
            process_column(column)

        # Fill remaining NaNs with 0
        # Fill remaining NaNs with 0, excluding 'estate_new_xiamen' and 'estate_new_wuxi'
        cols_to_fill = [col for col in df.columns if col not in ['estate_new_xiamen', 'estate_new_wuxi']]
        df.loc[:, cols_to_fill] = df.loc[:, cols_to_fill].fillna(0)

        # Create a new table for processed data in the processed_data schema
        df.to_sql('estate_new_processed', self.alch_engine, if_exists='replace', index=False, schema='processed_data')

        # Compute the 7-day moving average for all columns except 'estate_new_xiamen' and 'estate_new_wuxi'
        ma_columns = [col for col in df.columns if col not in ['date', 'estate_new_xiamen', 'estate_new_wuxi']]
        ma_df = df[ma_columns].rolling(window=7, min_periods=1).mean()

        # Combine the moving average columns with the original date, estate_new_xiamen, and estate_new_wuxi columns
        estate_new_processed_ma7 = pd.concat([df[['date', 'estate_new_xiamen', 'estate_new_wuxi']], ma_df], axis=1)

        # Create a new table for processed data in the processed_data schema
        estate_new_processed_ma7.to_sql('estate_new_processed_ma7', self.alch_engine, if_exists='replace', index=False,
                                        schema='processed_data')
