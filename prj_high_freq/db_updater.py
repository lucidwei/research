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
        self.calc_estate_area_ma()
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
        self.cities_col = id_to_new_string.values()

        # ��������
        dates_missing = self.get_missing_dates(self.all_dates, "high_freq_wide")
        ids = self.metadata['ָ��ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print('Wind downloading for high_freq ���ز���������')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(dates_missing[-1]), usedf=True)[1]
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
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    def calc_estate_area_ma(self):
        """
        �����ݱ�����һЩ�������ȣ������Щ�������ǲ����նȸ��£�
        �նȸ��µ��б��׼�ǣ������������ݣ��ǿշ�0�����ݵ��Ƿ�ռ90%���ϡ�
        ���ڷ��նȸ��µ���������Ҫ���������ƶ�ƽ��MA������������ƶ�ƽ����Ҫ�������ݵĸ���Ƶ�ʾ�����MA(n)��nȡ����ƽ���������һ�ηǿշ�0���ݵ㡣
        """
        # Read data from the database
        query = text("SELECT * FROM high_freq_wide")
        df_wide = pd.read_sql_query(query, self.alch_conn)

        # Define a function to calculate the moving average based on a given n
        def calculate_moving_average(column, n):
            return column.rolling(window=n).mean()

        # Filter the last 6 months of data
        last_six_months = df_wide[df_wide['date'] > pd.Timestamp.now() - pd.DateOffset(months=6)]

        # Iterate through columns to process
        for column_name in self.cities_col:

            # Check if the column is daily updated
            non_zero_count = last_six_months[column_name].notna().astype(int).sum()
            total_count = len(last_six_months)
            daily_updated = (non_zero_count / total_count) > 0.9

            if not daily_updated:
                # Calculate the average number of days between non-zero data points
                non_zero_dates = last_six_months.loc[last_six_months[column_name].notna(), 'date']
                days_between_non_zero = (non_zero_dates.diff().dropna() / pd.Timedelta(days=1)).mean()

                # Calculate the moving average
                df_wide[column_name] = calculate_moving_average(df_wide[column_name], math.ceil(days_between_non_zero))

        # Write the results back to the database
        df_wide = df_wide.melt(id_vars=['date'], var_name='metric_name', value_name='value')
        df_wide = df_wide.sort_values(by='date', ascending=False)
        df_wide.to_sql('high_freq_long', self.alch_engine, if_exists='replace', index=False)