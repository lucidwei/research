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
        先获取缺失的日期列表,需要更新的两段日期是：
        - all_dates的最早一天到数据库存在数据的最早一天
        - 数据库存在数据的最后一天到all_dates的最后一天，也就是今天
        """
        # 定义DataFrame中列名和表中列名的对应关系
        self.metadata = self.base_config.process_wind_metadata('30城metadata.xlsx')
        indicator_id_col = self.metadata.loc[:, '指标ID']
        indicator_name_col = self.metadata.loc[:, '指标名称']

        # 创建一个字典，键为指标 ID，值为新字符串
        id_to_new_string = {}
        for ind_id, ind_name in zip(indicator_id_col, indicator_name_col):
            # 获取指标名称的前两个汉字
            first_two_chars = ind_name[:2]

            # 转换为拼音
            pinyin_str = ''.join(lazy_pinyin(first_two_chars))

            # 构造新字符串
            new_string = f'estate_new_{pinyin_str}'

            # 将指标 ID 与新字符串添加到字典中
            id_to_new_string[ind_id] = new_string
        self.cities_col = id_to_new_string.values()

        # 更新数据
        dates_missing = self.get_missing_dates(self.all_dates, "high_freq_wide")
        ids = self.metadata['指标ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print('Wind downloading for high_freq 房地产销售数据')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(dates_missing[-1]), usedf=True)[1]
            # wind返回的df，日期为一天和多天的格式不一样
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing

            # 重命名列为数据库中列
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)
            downloaded_df.rename(columns=id_to_new_string, inplace=True)

            # 获取单位列，转换单位
            unit_col = self.metadata.loc[:, '单位']
            # 找到单位为 '万平方米' 的指标 ID
            indicator_ids_to_adjust = [ind_id for ind_id, unit in zip(indicator_id_col, unit_col) if unit == '万平方米']
            # 对于这些指标 ID，将数据乘以 10000
            for ind_id in indicator_ids_to_adjust:
                new_col_name = id_to_new_string[ind_id]
                downloaded_df[new_col_name] = downloaded_df[new_col_name] * 10000

            # 将新行插入数据库中
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    def calc_estate_area_ma(self):
        """
        对数据本身做一些处理。首先，检查这些列数据是不是日度更新：
        日度更新的判别标准是，最近半年的数据，非空非0的数据点是否占90%以上。
        对于非日度更新的数据我们要对它们求移动平均MA，具体求几天的移动平均需要根据数据的更新频率决定，MA(n)的n取决于平均几天出现一次非空非0数据点。
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