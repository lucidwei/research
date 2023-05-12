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

            # 构建source_code
            source_code = f"wind_{ind_id}"

            # 注意，单位转换在上传数据部分完成
            unit = '万平方米'

            # 向metric_static_info表中插入source_code, chinese_name, 和unit
            with self.alch_engine.begin() as connection:
                connection.execute(text(f"""
                    INSERT INTO metric_static_info (source_code, chinese_name, unit)
                    VALUES ('{source_code}', '{ind_name}', '{unit}')
                    ON CONFLICT (source_code) DO UPDATE
                    SET chinese_name = EXCLUDED.chinese_name,
                        unit = EXCLUDED.unit;
                """))

        self.cities_col = id_to_new_string.values()

        # 更新数据
        dates_missing = self.get_missing_dates(self.all_dates, "estate_new_wide")
        ids = self.metadata['指标ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print('Wind downloading for high_freq 房地产销售数据')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(dates_missing[-1]), usedf=True)[1]
            # wind有时会返回莫名其妙的行，把非空小于2个的行删除
            downloaded_df.dropna(thresh=2, inplace=True)
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
            long_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            long_df.dropna(inplace=True)
            long_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    def process_estate_area(self):
        """
        首先，针对无锡和厦门一段一段处理，比如最新的三个数据在5月1号和5月10号和5月17号，首先把5月11~5月17的值设为5月17号的值除以7，
        然后把5月2号到5月10号的值设为5月10号的值除以9，具体除以几其实应该取决于数据表中的空值数量+1。注意，每一段之间的间隔是不恒定的。
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
