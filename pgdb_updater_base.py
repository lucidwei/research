# coding=gbk
# Time Created: 2023/4/26 15:32
# Author  : Lucid
# FileName: pgdb_updater_base.py
# Software: PyCharm
import datetime
from typing import List
import pandas as pd
from WindPy import w
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from utils import check_wind, timeit
from sqlalchemy import Table, MetaData, text, select
import tushare as ts


class PgDbUpdaterBase(PgDbManager):
    connection_checked = False

    def __init__(self, base_config: BaseConfig):
        if not PgDbUpdaterBase.connection_checked:
            check_wind()
            PgDbUpdaterBase.connection_checked = True
        super().__init__(base_config)
        self.pro = ts.pro_api('3c0eb978b70236184bebf8378aab04fa29867f6ddb0dc1c578e1f9d1')
        self.set_dates()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str
        self.months_ends = self.base_config.month_ends
        self.months_ends_str = self.base_config.month_ends_str

    def remove_today_if_trading_day(self, dates):
        today = datetime.date.today()

        if today in self.tradedays:
            filtered_dates = [date for date in dates if date != today]
        else:
            filtered_dates = dates

        return filtered_dates

    def remove_today_if_trading_time(self, dates):
        today = datetime.date.today()
        now = datetime.datetime.now().time()
        closing_time = datetime.datetime.strptime("17:00", "%H:%M").time()

        if today in self.tradedays and now < closing_time:
            filtered_dates = [date for date in dates if date != today]
        else:
            filtered_dates = dates

        return filtered_dates

    @property
    def conversion_dicts(self):
        """
        存在的原因：WSD和EDB不同，不会返回id的中英文名，所以需要转换字典。
        保存所有ID->english以及english->chinese的转换。
        TODO: 不完美，这个步骤需要手工。最好通过wind API自动获取
        """
        # ID到英文列名
        id_to_english = {
            '000906.SH': 'csi_g800_index',
            'AU9999.SGE': 'sge_gold_9999',
            'CBA00301.CS': 'cba_total_return_index',
            'S0059749': 'china_t_yield',
            'G0005428': 'us_tips_10y',
            'VIX.GI': 'cboe_vix',
        }

        # 英文列名到中文名
        english_to_chinese = {
            'csi_g800_index': '中证800指数',
            'sge_gold_9999': '国内现货黄金',
            'cba_total_return_index': '中债总财富指数',
            'china_t_yield': '中债国债到期收益率:10年',
            'us_tips_10y': '美国通胀指数国债收益率:10年',
            'cboe_vix': '芝加哥期权交易所波动率指数',
        }

        return {'id_to_english': id_to_english, 'english_to_chinese': english_to_chinese}

    def update_high_freq_by_edb_id(self, code: str):
        """
        TODO: 这种函数不能给上传的数据加description列，但后续描述列是要放在外键的。因此不用担心
        注意：
        1. self.get_missing_dates(self.all_dates导致这个方法只针对日频数据，低频数据需重新定义函数
        2. 如需对原始数据做处理，不应使用此方法
        先获取缺失的日期列表,需要更新的两段日期是：
        - all_dates的最早一天到数据库存在数据的最早一天
        - 数据库存在数据的最后一天到all_dates的最后一天，也就是今天
        """
        existing_dates = self.select_existing_dates_from_long_table('high_freq_long',
                                                                    metric_name=self.conversion_dicts['id_to_english'][
                                                                        code])
        dates_missing = self.get_missing_dates(self.tradedays, existing_dates=existing_dates)
        if len(dates_missing) == 0:
            return
        old_dates = [date for date in dates_missing if date.year < 2023]
        new_dates = [date for date in dates_missing if date.year >= 2023]
        for dates_update in [old_dates, new_dates]:
            if not dates_update:
                continue
            print(
                f'Wind downloading {code} for table high_freq_long between {str(dates_update[0])} and {str(dates_update[-1])}')
            downloaded_df = w.edb(code, str(dates_update[0]), str(dates_update[-1]), usedf=True)[1]
            # wind返回的df有毛病，日期为一天和多天的格式不一样
            try:
                # 尝试将索引转换为日期时间，如果失败则是有毛病df
                pd.to_datetime(downloaded_df.index)
            except:
                if dates_update[0] == dates_update[-1]:
                    downloaded_df = downloaded_df.T
                    downloaded_df.index = dates_update
                else:
                    return
            # 重命名列为数据库中列
            downloaded_df.columns = [self.conversion_dicts['id_to_english'][code]]
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)

            # 将新行插入数据库中
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_markets_daily_by_wsd_id_fields(self, code: str, fields: str):
        """
        """
        fields_list = fields.split(',')

        for field in fields_list:
            existing_dates = self.select_existing_dates_from_long_table("markets_daily_long",
                                                                        metric_name=
                                                                        self.conversion_dicts['id_to_english'][
                                                                            code],
                                                                        field=field.lower())
            dates_missing = self.get_missing_dates(self.tradedays, existing_dates)
            if len(dates_missing) == 0:
                print(f'No missing data for {code} {field} in markets_daily_long, skipping download')
                return

            print(
                f'Wind downloading {code} {field} for markets_daily_long between {str(dates_missing[0])} and {str(dates_missing[-1])}')
            # 这里Days=Weekdays简化设置，为所有product设定交易所有点难度。
            downloaded_df = \
                w.wsd(code, field, str(dates_missing[0]), str(dates_missing[-1]), "Days=Weekdays", usedf=True)[1]
            # 转换下载的数据框为长格式
            downloaded_df.index.name = 'date'
            downloaded_df.reset_index(inplace=True)
            downloaded_df.columns = [col.lower() for col in downloaded_df.columns]
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='field', value_name='value')
            downloaded_df.dropna(subset=['value'], inplace=True)

            # 添加其他所需列
            product_name = self.conversion_dicts['id_to_english'][code]
            source_code = f"wind_{code}"
            chinese_name = self.conversion_dicts['english_to_chinese'][product_name]

            # 确保metric_static_info表中存在对应的source_code和chinese_name
            # 获取metric_static_info表中对应记录的internal_id
            with self.alch_engine.connect() as conn:
                query = f"""
                INSERT INTO metric_static_info (source_code, chinese_name)
                VALUES ('{source_code}', '{chinese_name}')
                ON CONFLICT (source_code) DO UPDATE
                SET chinese_name = EXCLUDED.chinese_name
                RETURNING internal_id;
                """
                internal_id = conn.execute(text(query)).fetchone()[0]

            # 将新行插入数据库中, df要非空
            if downloaded_df.iloc[0, 0] != 0:
                print('Uploading data to database...')
                downloaded_df['product_name'] = product_name
                downloaded_df['metric_static_info_id'] = internal_id
                downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, sheet_name=None, if_rename=False):
        """
        Updates the data based on the metadata from an Excel file downloaded from Wind.

        Args:
            excel_file (str): The path to the Excel file containing the metadata and data.
            name_mapping (dict): A dictionary mapping indicator names from the Excel file to their corresponding names in the database.
            sheet_name (str, optional): The name of the sheet in the Excel file to read. Defaults to None.
            if_rename (bool, optional): Whether to rename the columns in the database. Defaults to False.

        Performs the following steps:
        1. Loads the metadata and data from the Excel file using the base_config's process_wind_excel method.
        2. Filters the metadata to include only the rows with indicator names present in the name_mapping dictionary.
        3. Extracts the necessary columns from the filtered metadata, such as indicator ID, name, unit, and start month end.
        4. Creates dictionaries mapping indicator IDs to their respective Chinese names, units, English names, and start month ends.
        5. Optionally deletes the columns corresponding to the names in names_to_delete if if_rename is True.
        6. Updates the data for each indicator by calling the update_low_freq_by_excel_indicator_id method.

        Note:
        - The update_low_freq_by_excel_indicator_id method is called to update the data for each indicator using the metadata.
        - The update process involves checking for missing data, preparing the data for upload, updating the metric_static_info table,
          and inserting the new data into the low_freq_long table.
        """

        def get_start_month_end(s):
            """
            Extracts the start month end from a string.

            Args:
                s (str): The string containing the start date.

            Returns:
                pd.Timestamp: The start month end as a pandas Timestamp object.
            """
            start_date_str = s.split(':')[0]  # 提取开始日期字符串
            start_date = pd.to_datetime(start_date_str, format='%Y-%m')  # 转换成日期格式
            start_month_end = start_date + pd.offsets.MonthEnd(1)  # 获取月末日期
            return start_month_end

        # '指标ID' 列字符串长度小于 5 的行是EDB中计算后数值。因为有些指标的基础处理还是需要靠万得，比如货币转换，所以这些指标
        # 后续也会用到，但这些指标ID用不到。
        self.metadata, data = self.base_config.process_wind_excel(excel_file, sheet_name)

        # excel中可能有多余的列。故Select the rows where '指标ID' is in keys
        filtered_metadata = self.metadata[self.metadata['指标名称'].isin(name_mapping.keys())]

        # 定义 DataFrame 中列名和表中列名的对应关系
        col_indicator_id = filtered_metadata.loc[:, '指标ID']
        col_indicator_name = filtered_metadata.loc[:, '指标名称']
        col_unit = filtered_metadata.loc[:, '单位']
        col_earlist_date = filtered_metadata['时间区间'].apply(get_start_month_end)
        # maps
        map_id_to_chinese = dict(zip(col_indicator_id, col_indicator_name))
        map_id_to_unit = dict(zip(col_indicator_id, col_unit))
        map_id_to_english = {id: name_mapping[name] for id, name in map_id_to_chinese.items()}
        map_id_to_earlist_date = dict(zip(col_indicator_id, col_earlist_date))

        if if_rename:
            # renaming中会用到,其实是删除了excel_file对应的所有列，没有进行筛选。但因为很少需要rename，因此不做优化
            names_to_delete = list(map_id_to_chinese.values())
            self.delete_for_renaming(names_to_delete)

        maps_tuple = (map_id_to_chinese, map_id_to_unit, map_id_to_english)
        # 更新数据
        for id in col_indicator_id:
            self.update_low_freq_by_excel_indicator_id(map_id_to_earlist_date[id], id, maps_tuple, data)

    def update_low_freq_by_excel_indicator_id(self, earliest_available_date, code, maps: tuple, data):
        """
        Updates the low_freq_long table with data from an Excel file based on the indicator ID.

        Args:
            earliest_available_date (datetime.date): The earliest date for which data is available *from wind data source*, read from wind excel file.
            code (str): The indicator ID code.
            maps (tuple): A tuple containing dictionaries that map indicator IDs to their respective names, units, and English names.
            data (pd.DataFrame): The data from the Excel file.

        Performs the following steps:
        1. Checks for missing data by comparing existing dates in the low_freq_long table with the available dates.
        2. Prepares the data for upload by filtering out zero values and selecting only the missing dates.
        3. Updates the metric_static_info table by inserting or updating the metadata for the indicator.
        4. Inserts the new data into the low_freq_long table with the corresponding metadata.
        """
        map_id_to_name, map_id_to_unit, map_id_to_english = maps

        # Check for missing data
        existing_dates = self.select_existing_dates_from_long_table('low_freq_long', map_id_to_english[code])
        dates_missing = self.get_missing_months_ends(self.months_ends, earliest_available_date, 'low_freq_long',
                                                     map_id_to_english[code])
        if len(dates_missing) == 0:
            print(f'No missing data for low_freq_long {map_id_to_name[code]}, skipping download')
            return

        # Prepare data for upload
        df_to_upload = self._prepare_data_for_upload(code, dates_missing, data, map_id_to_name, existing_dates)
        if df_to_upload is None:
            return

        # Update metric_static_info table
        (internal_id, update_freq) = self._update_metric_static_info(df_to_upload, existing_dates, map_id_to_unit, code,
                                                                     map_id_to_name, map_id_to_english)

        # Insert new data into low_freq_long
        self._insert_into_low_freq_long(df_to_upload, internal_id, map_id_to_english, update_freq)

    def _prepare_data_for_upload(self, code, dates_missing, data, map_id_to_name, existing_dates):
        """
        Prepares the data for upload to the low_freq_long table.

        Args:
            code (str): The indicator code.
            dates_missing (list): The list of missing dates.
            data (DataFrame): The data to be uploaded.
            map_id_to_name (dict): A dictionary mapping indicator IDs to names.
            existing_dates (list): The list of existing dates in the database.

        Returns:
            DataFrame: The prepared data to be uploaded.

        Performs the following steps:
        1. If the code length is less than or equal to 5:
            - Load the data for the code from the 'data' DataFrame.
            - Set the index of the loaded DataFrame to datetime format.
            - Remove the rows with 0 values from the loaded DataFrame.
            - Keep only the new data for the missing dates.
            - If the loaded DataFrame is empty, print a message and return None.
            - Assign the loaded DataFrame to 'df_to_upload'.
        2. If the code length is greater than 5:
            - Print a message indicating wind downloading for the code and the missing dates range.
            - Download the data for the code using the Wind API.
            - Remove the rows with dates that already exist in the database.
            - If the downloaded DataFrame is empty, print a message and return None.
            - If only one row is returned by Wind API, print a message and return None.
            - If only one date is requested, transpose the downloaded DataFrame, assign the missing date as the index,
              and assign the transposed DataFrame to 'df_to_upload'.
            - Assign the downloaded DataFrame to 'df_to_upload'.

        Returns the prepared data to be uploaded as a DataFrame.
        """
        if len(code) <= 5:
            loaded_df = data[code]
            loaded_df.index = pd.to_datetime(loaded_df.index)
            # excel中许多0值要去掉
            loaded_df = loaded_df[loaded_df != 0]
            # 只保留新数据
            loaded_df = pd.DataFrame(loaded_df)
            loaded_df = loaded_df.reindex(dates_missing).dropna()

            if loaded_df.empty:
                print(f'No missing data for low_freq_long的{map_id_to_name[code]}, will not upload')
                return
            df_to_upload = loaded_df
        else:
            print(
                f'Wind downloading for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}')
            downloaded_df = w.edb(code, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
            downloaded_df.columns = [code]

            # 删除与 existing_dates 中日期相同的行
            existing_dates_set = set(existing_dates)
            downloaded_df = downloaded_df.loc[~downloaded_df.index.isin(existing_dates_set)]

            if downloaded_df.empty:
                print(f'No missing data for low_freq_long的{map_id_to_name[code]}, downloaded but will not upload')
                return

            # wind返回的df如果只有一行数据，index会被设为wind_id，我们要转换回日期。
            ## 第一种情况，我们只请求了一个日期
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing
            ## 第二种情况，我们请求了多个日期，但wind只返回一个日期，这种直接丢弃掉，因为不知道是哪个日期的数据
            elif downloaded_df.shape[0] == 1:
                print(f'''
                We asked for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}, 
                but wind return only one data point, wind is actively updating this metric. Skipping...''')
                return

            df_to_upload = downloaded_df

        return df_to_upload

    def _insert_into_low_freq_long(self, df_to_upload, internal_id, map_id_to_english, update_freq):
        """
        Inserts the new data into the low_freq_long table.

        Args:
            df_to_upload (DataFrame): The prepared data to be uploaded.
            internal_id (int): The internal ID of the metric in the metric_static_info table.
            map_id_to_english (dict): A dictionary mapping indicator IDs to English column names.
            update_freq (float): The update frequency of the data.

        Performs the following steps:
        1. Rename the columns of 'df_to_upload' using the 'map_id_to_english' dictionary.
        2. Melt the DataFrame to transform it from wide to long format, using 'date' as the id variable, 'metric_name' as
           the variable name, and 'value' as the value name.
        3. Drop rows with missing values in the 'value' column.
        4. Add the 'update_freq' and 'metric_static_info_id' columns to the DataFrame.
        5. Insert the DataFrame into the 'low_freq_long' table in the database using the SQLAlchemy 'to_sql' method.
        """
        # 将新数据插入low_freq_long中
        df_upload = df_to_upload.rename(columns=map_id_to_english)
        df_upload = df_upload.melt(id_vars=['date'], var_name='metric_name', value_name='value')
        df_upload.dropna(subset=['value'], inplace=True)
        # 添加 additional_info:update_freq 和 metric_static_info_id 列
        df_upload['update_freq'] = update_freq
        df_upload['metric_static_info_id'] = internal_id
        df_upload.to_sql('low_freq_long', self.alch_engine, if_exists='append', index=False)

    def _update_metric_static_info(self, df_to_upload, existing_dates, map_id_to_unit, code, map_id_to_name,
                                   map_id_to_english):
        """
        Updates the metric_static_info table with the metadata for the uploaded data.

        Args:
            df_to_upload (pandas.DataFrame): The DataFrame containing the data to be uploaded.
            existing_dates (list): The list of existing dates in the database for the metric.
            map_id_to_unit (dict): A dictionary mapping metric IDs to their units of measurement.
            code (str): The code of the metric, excel_indicator_id code.
            map_id_to_name (dict): A dictionary mapping metric IDs to their Chinese names.
            map_id_to_english (dict): A dictionary mapping metric IDs to their English names.

        Returns:
            tuple: A tuple containing the internal ID and the update frequency.

        Performs the following steps:
        1. Defines a nested function `get_update_freq()` to calculate the update frequency based on the number of non-null data points in the past six months.
        2. 将日期index作为一列.
        3. Calls `get_update_freq()` to calculate the update frequency.
        4. Updates the metric_static_info table with the metadata.
           - If the metric is wind-calculated, the code is not used, and the internal ID is auto-incremented.
           - If the metric is wind-transformed, the code is used to identify the metric, and the internal ID is obtained from the insert_metric_static_info method.
        5. Returns the internal ID and the update frequency as a tuple.
        """

        def get_update_freq():
            """
            Calculates the update frequency based on the number of non-null data points in the past six months.

            Returns:
                float: The update frequency.

            Performs the following steps:
            1. Calculates the start date as six months ago from the current date.
            2. Combines the existing dates and the dates from the DataFrame.
            3. Selects the dates from the past six months.
            4. Counts the number of non-null data points in the selected dates.
            5. Calculates the update frequency as 180 divided by the number of non-null data points.
            6. Returns the update frequency as a float.
            """
            # 计算近半年起始日期
            six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
            # 计算近半年的数据点个数并计算更新频率
            existing_dates_series = pd.Series(existing_dates, dtype='datetime64[D]').dt.date
            combined_dates = pd.to_datetime(pd.concat([existing_dates_series, df_to_upload['date']])).dt.date

            # 计算更新频率=近半年的天数/非空数据点个数
            # 选择六个月前之后的日期
            recent_dates = combined_dates[combined_dates >= six_months_ago]
            # 计算近半年的数据点个数并计算更新频率
            non_null_data_points = recent_dates.count()
            update_freq = 180 / non_null_data_points

            return update_freq

        # 将日期index作为一列
        df_to_upload.index.name = 'date'
        df_to_upload.reset_index(inplace=True)

        update_freq = get_update_freq()

        # 更新metric_static_info 元数据table
        # 如果是通过wind计算得到的数值，那么原来的code没有用，使用自增列internal_id
        unit = map_id_to_unit[code]
        source_code = f'wind_{code}' if len(code) >= 5 else None
        chinese_name = map_id_to_name[code]
        english_name = map_id_to_english[code]

        internal_id = self.insert_metric_static_info(source_code, chinese_name, english_name, unit)

        return (internal_id, update_freq)

    def insert_metric_static_info(self, source_code, chinese_name, english_name, unit, type_identifier):

        """
        Inserts or updates the metadata of a metric in the metric_static_info table.

        Args:
            source_code (str): The source code of the metric.
            chinese_name (str): The Chinese name of the metric.
            english_name (str): The English name of the metric.
            unit (str): The unit of measurement for the metric.
            type_identifier (str): 用于group select，区分不同任务

        Returns:
            int: The internal ID of the inserted or updated record in the metric_static_info table.

        Performs the following steps:
        1. Adjusts the sequence value to ensure unique and consecutive internal IDs.
        2. Inserts or updates the metadata in the metric_static_info table based on the source code and Chinese name.
           - If the source code is provided, the metric is assumed to be not wind_transformed and is inserted or updated based on the source code.
           - If the source code is not provided, the metric is assumed to be wind_transformed. The Chinese name is checked for existence in the table.
             - If the Chinese name already exists, the method returns without performing any insert operation.
             - If the Chinese name is new, a temporary record is inserted and updated with the corresponding source code.
        3. Commits the transaction and returns the internal ID of the inserted or updated record.
        """
        # Ensure the corresponding source_code and chinese_name exist in the metric_static_info table
        self.adjust_seq_val()
        with self.alch_engine.connect() as conn:
            if source_code:
                query = text("""
                            INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name, type_identifier)
                            VALUES (:source_code, :chinese_name, :unit, :english_name, :type_identifier)
                            ON CONFLICT (source_code) DO UPDATE
                            SET english_name = EXCLUDED.english_name,
                                chinese_name = EXCLUDED.chinese_name,
                                unit = EXCLUDED.unit,
                                type_identifier = EXCLUDED.type_identifier
                            RETURNING internal_id;
                            """)
                result = conn.execute(query,
                                      {
                                          'source_code': source_code,
                                          'chinese_name': chinese_name,
                                          'unit': unit,
                                          'english_name': english_name,
                                          'type_identifier': type_identifier,
                                      })
                internal_id = result.fetchone()[0]
            # 对于wind_transformed数据
            else:
                # 首先查询 chinese_name 是否已经存在
                query = text("""
                            SELECT 1
                            FROM metric_static_info
                            WHERE chinese_name = :chinese_name
                            """)
                result = conn.execute(query, {'chinese_name': chinese_name})
                if result.fetchone() is not None:
                    # 如果 chinese_name 已经存在，则直接返回，不执行插入操作
                    return

                # 插入新的记录
                query = text("""
                            INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name)
                            VALUES ('temp_code', :chinese_name, :unit, :english_name)
                            RETURNING internal_id;
                            """)
                result = conn.execute(query,
                                      {
                                          'chinese_name': chinese_name,
                                          'unit': unit,
                                          'english_name': english_name
                                      })
                # Get the internal_id of the corresponding record in the metric_static_info table
                internal_id = result.fetchone()[0]
                source_code = f'wind_transformed_{internal_id}'
                update_query = text("""
                                    UPDATE metric_static_info
                                    SET source_code = :source_code
                                    WHERE internal_id = :internal_id;
                                    """)
                conn.execute(update_query,
                             {
                                 'source_code': source_code,
                                 'internal_id': internal_id
                             })
            conn.commit()
            return internal_id

    def insert_product_static_info(self, row: pd.Series):
        """
        """
        self.adjust_seq_val()
        if row['product_type'] == 'fund':
            with self.alch_engine.connect() as conn:
                query = text("""
                            INSERT INTO product_static_info (code, chinese_name, source, product_type, issueshare, buystartdate, fundfounddate)
                            VALUES (:code, :chinese_name, :source, :product_type, :issueshare, :buystartdate, :fundfounddate)
                            ON CONFLICT (code, product_type) DO UPDATE 
                            SET chinese_name = EXCLUDED.chinese_name,
                                buystartdate = EXCLUDED.buystartdate,
                                fundfounddate = EXCLUDED.fundfounddate,
                                issueshare = EXCLUDED.issueshare
                            RETURNING internal_id;
                            """)
                result = conn.execute(query,
                                      {
                                          'code': row['code'],
                                          'chinese_name': row['chinese_name'],
                                          'source': row['source'],
                                          'product_type': row['product_type'],
                                          'issueshare': None if pd.isnull(row['issueshare']) else row['issueshare'],
                                          'buystartdate': None if pd.isnull(row['buystartdate']) else row[
                                              'buystartdate'],
                                          'fundfounddate': None if pd.isnull(row['fundfounddate']) else row[
                                              'fundfounddate'],
                                      })
                internal_id = result.fetchone()[0]
                conn.commit()
        elif row['product_type'] == 'stock':
            if 'type_identifier' not in row.index:
                query = text("""
                            INSERT INTO product_static_info (code, chinese_name, stk_industry_cs, source, product_type, update_date)
                            VALUES (:code, :chinese_name, :stk_industry_cs, :source, :product_type, :update_date)
                            ON CONFLICT (code, product_type) DO UPDATE 
                            SET chinese_name = EXCLUDED.chinese_name,
                                stk_industry_cs = EXCLUDED.stk_industry_cs,
                                source = EXCLUDED.source,
                                update_date = EXCLUDED.update_date
                            """)
                self.alch_conn.execute(query,
                                       {
                                           'code': row['code'],
                                           'chinese_name': row['chinese_name'],
                                           'source': row['source'],
                                           'product_type': row['product_type'],
                                           'stk_industry_cs': row['stk_industry_cs'],
                                           'update_date': row['update_date'],
                                       })
            else:
                query = text("""
                            INSERT INTO product_static_info (code, chinese_name, stk_industry_cs, source, type_identifier, product_type)
                            VALUES (:code, :chinese_name, :stk_industry_cs, :source, :type_identifier, :product_type)
                            ON CONFLICT (chinese_name, product_type) DO NOTHING
                            RETURNING internal_id;
                            """)
                self.alch_conn.execute(query,
                                       {
                                           'code': row['code'],
                                           'chinese_name': row['chinese_name'],
                                           'source': row['source'],
                                           'product_type': row['product_type'],
                                           'stk_industry_cs': row['stk_industry_cs'],
                                           'type_identifier': row['type_identifier'],
                                       })
            self.alch_conn.commit()
            internal_id = None
        elif row['product_type'] == 'index':
            query = text("""
                        INSERT INTO product_static_info (code, chinese_name, source, type_identifier, product_type)
                        VALUES (:code, :chinese_name, :source, :type_identifier, :product_type)
                        ON CONFLICT (chinese_name, product_type) DO NOTHING
                        RETURNING internal_id;
                        """)
            self.alch_conn.execute(query,
                                   {
                                       'code': row['code'],
                                       'chinese_name': row['chinese_name'],
                                       'source': row['source'],
                                       'product_type': row['product_type'],
                                       'type_identifier': row['type_identifier'],
                                   })
            self.alch_conn.commit()
            internal_id = None
        else:
            raise Exception(f"row['product_type'] = {row['product_type']} not supported.")
        return internal_id

    def upload_product_static_info(self, row, task: str):
        if task == 'fund_name':
            with self.alch_engine.connect() as conn:
                query = text("""
                            UPDATE product_static_info
                            SET fund_fullname = :fund_fullname, english_name = :english_name
                            WHERE code = :code
                            """)
                conn.execute(query,
                             {
                                 'code': row['code'],
                                 'fund_fullname': None if pd.isnull(row['fund_fullname']) else row['fund_fullname'],
                                 'english_name': None if pd.isnull(row['english_name']) else row['english_name'],
                             })
                conn.commit()
        elif task == 'etf_industry_and_type':
            query = text("""
                        UPDATE product_static_info
                        SET stk_industry_cs = :stk_industry_cs, etf_type = :etf_type
                        WHERE code = :code
                        """)
            self.alch_conn.execute(query,
                                   {
                                       'code': row['code'],
                                       'etf_type': row['etf_type'],
                                       'stk_industry_cs': None if pd.isnull(row['stk_industry_cs']) else row[
                                           'stk_industry_cs'],
                                   })
            self.alch_conn.commit()
        else:
            raise Exception(f'task:{task} not supported!')

    def delete_for_renaming(self, names_to_delete):
        with self.alch_engine.connect() as conn:
            # 通过metric_static_info获取所有可能冲突的metric_name
            conflict_metric_names_set = set()
            for name in names_to_delete:
                query = f"""
                SELECT english_name
                FROM metric_static_info
                WHERE chinese_name = '{name}'
                """
                result = conn.execute(text(query))
                for row in result:
                    conflict_metric_names_set.add(row[0])

            # 删除所有可能冲突的 metric_name
            for metric_name in conflict_metric_names_set:
                query = f"""
                DELETE FROM low_freq_long
                WHERE metric_name = '{metric_name}'
                """
                conn.execute(text(query))
                conn.commit()

    def execute_pgsql_function(self, function_name, table_name, view_name, chinese_names):
        query = f"""
        SELECT {function_name}(:table_name, :view_name, ARRAY[:chinese_names])
        """

        self.alch_conn.execute(
            text(query),
            {
                'table_name': table_name,
                'view_name': view_name,
                'chinese_names': chinese_names
            }
        )
        self.alch_conn.commit()

    def get_missing_metrics(self, target_table: str, target_column: str, metrics_at_hand: list):
        """
        Retrieves the missing metrics that are present in the input list but not in the specified target table 的 column.

        Args:
            target_table (str): The name of the target table.
            target_column (str): The name of the target column in the target table.
            metrics_at_hand (list): A list of metrics.

        Returns:
            list: A list of missing metrics.

        Performs the following steps:
        1. Creates a new session using the SQLAlchemy Session object.
        2. Retrieves the metadata of the target table.
        3. Executes a select statement to retrieve the values from the target column in the target table.
        4. Stores the existing values in a set.
        5. Converts the input list of metrics to a set.
        6. Finds the values that are present in the input set but not in the existing values set.
        7. Returns the missing values as a list.

        Returns a list of missing metrics.
        """

        existing_values = self.select_existing_values_in_target_column(target_table, target_column)

        # 将输入的metrics列表转换为集合
        input_set = set(metrics_at_hand)

        # 找出存在于输入列表中，但不存在于目标表中的值
        missing_values = input_set - existing_values

        return sorted(list(missing_values))

    def is_markets_daily_long_updated_today(self, field: str, product_name_key_word: str):
        # TODO: 此函数的存在不太合理 不应该有使用它的情境
        # 获取今天的日期
        today = self.all_dates[-1]

        # 构建原始 SQL 查询
        sql = text(
            f"SELECT MAX(date) FROM markets_daily_long WHERE field = '{field}' AND product_name LIKE '%{product_name_key_word}%' ")

        # 执行查询并获取结果
        result = self.alch_conn.execute(sql).scalar()

        # 比较结果与今天的日期
        if result == today:
            return True
        else:
            return False

    def upload_joined_products_wide_table(self, full_name_keyword: str):
        # 使用SQLAlchemy执行SQL查询
        sql_query = f"""
        SELECT public.markets_daily_long.date AS date, public.markets_daily_long.product_name AS product_name, 
        public.markets_daily_long.field AS field, public.markets_daily_long.value AS value, 
        public.markets_daily_long.date_value AS date_value, 
        product_static_info."code" AS "code", 
        product_static_info."chinese_name" AS "chinese_name", 
        product_static_info."type" AS "type", 
        product_static_info."fundfounddate" AS "fundfounddate", 
        product_static_info."issueshare" AS "issueshare", 
        product_static_info."fund_fullname" AS "fund_fullname"
        FROM "public"."markets_daily_long"
        LEFT JOIN "public"."product_static_info" AS product_static_info 
        ON "public"."markets_daily_long"."product_static_info_id" = product_static_info."internal_id"
        WHERE product_static_info."type" = 'fund' 
        AND product_static_info."fund_fullname" LIKE '%{full_name_keyword}%' 
        AND product_static_info."fund_fullname" NOT LIKE '%债%'
        """
        df = pd.read_sql_query(text(sql_query), self.alch_conn)

        if full_name_keyword == '定期开放':
            # 需要哪些field
            df1 = df[df['field'].isin(['fund_expectedopenday', 'fund_fundscale'])]
            df1.loc[:, 'value_or_date_value'] = df1.apply(
                lambda row: row['date_value'] if pd.isna(row['value']) or row['value'] == 0 else row['value'], axis=1)
            df_wide = df1.pivot(index=['date', 'product_name'], columns='field',
                                values='value_or_date_value').reset_index()
            # 上传到processed_data schema中
            df_wide.to_sql('funds_dk_nobond_wide', self.alch_engine, schema='processed_data', if_exists='replace',
                           index=False)

        elif full_name_keyword == '持有期':
            df1 = df[df['field'].isin(['openrepurchasestartdate', 'fund_fundscale'])]
            df1.loc[:, 'value_or_date_value'] = df1.apply(
                lambda row: row['date_value'] if pd.isna(row['value']) or row['value'] == 0 else row['value'], axis=1)

            df_wide = df1.pivot(index=['product_name'], columns='field', values='value_or_date_value').reset_index()
            # 上传到processed_data schema中
            df_wide.to_sql('funds_cyq_nobond_wide', self.alch_engine, schema='processed_data', if_exists='replace',
                           index=False)

    # def calculate_yoy(self, value_str, yoy_str, cn_value_str, cn_yoy_str, cn_names_to_exhibit):
    #     """
    #     Calculates the year-over-year (YoY) change for specified metrics and inserts the calculated data into the low_freq_long table.
    #
    #     Args:
    #         value_str (str): The string used to identify e.g."*CurrentMonthValue" data in the low_freq_long table.
    #         yoy_str (str): The string used to identify the corresponding YoY change data in the low_freq_long table.
    #         cn_value_str (str): The Chinese translation string for the value_str.
    #         cn_yoy_str (str): The Chinese translation string for the yoy_str.
    #
    #     Performs the following steps:
    #     1. Selects the "*CurrentMonthValue" data from the low_freq_long table, excluding already calculated rows.
    #     2. Matches the value data with the corresponding YoY change data based on metric names and dates.
    #     3. Calculates the YoY change by comparing the current value with the value from the same period last year.
    #     4. Inserts the calculated YoY data into the low_freq_long table, updating the value if a conflict occurs.
    #     5. Updates the source_code in the metric_static_info table for the corresponding YoY metric.
    #     6. Commits the changes to the database.
    #
    #     Note: This method assumes that the necessary data is present in the low_freq_long and metric_static_info tables.
    #     """
    #     # Step 1: Select all "*CurrentMonthValue" data from low_freq_long, bypass already-calculated rows
    #     # Step 1.1: Get the two dataframes
    #     query_value = f"SELECT * FROM low_freq_long WHERE metric_name LIKE '%{value_str}'"
    #     df_value = pd.read_sql_query(text(query_value), self.alch_conn)
    #
    #     query_yoy = f"SELECT * FROM low_freq_long WHERE metric_name LIKE '%{yoy_str}'"
    #     df_yoy = pd.read_sql_query(text(query_yoy), self.alch_conn)
    #
    #     # Step 1.2: Create new columns for matching
    #     df_value['metric_base'] = df_value['metric_name'].str.replace(value_str, '')
    #     df_yoy['metric_base'] = df_yoy['metric_name'].str.replace(yoy_str, '')
    #
    #     # Step 1.3: Find the rows in df_value that have a match in df_yoy
    #     mask = df_value['metric_base'].isin(df_yoy['metric_base']) & df_value['date'].isin(df_yoy['date'])
    #
    #     # Step 1.4: Remove the matching rows from df_value
    #     df = df_value[~mask]
    #
    #     for _, row in df.iterrows():
    #         metric_name_value = row['metric_name']
    #         metric_name_yoy = metric_name_value.replace(value_str, yoy_str)
    #
    #         # Step 2: Find the value from the same period last year
    #         query = f"""
    #         SELECT value
    #         FROM low_freq_long
    #         WHERE metric_name = '{metric_name_value}'
    #         AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CAST('{row['date']}' AS DATE)) - 1
    #         AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CAST('{row['date']}' AS DATE))
    #         """
    #         df_last_year = pd.read_sql_query(text(query), self.alch_conn)
    #
    #         if df_last_year.empty or pd.isnull(df_last_year.loc[0, 'value']):
    #             # print(f"No data for {metric_name_value} '{row['date']}' from the same period last year.")
    #             continue
    #
    #         # Calculate YoY change
    #         current_value = row['value']
    #         last_year_value = df_last_year.loc[0, 'value']
    #         yoy_change = (current_value - last_year_value) / last_year_value * 100
    #
    #         # Step 3: Insert the calculated YoY data into low_freq_long
    #         query = f"""
    #         INSERT INTO low_freq_long (date, metric_name, value)
    #         VALUES ('{row['date']}', '{metric_name_yoy}', {yoy_change})
    #         ON CONFLICT (date, metric_name) DO UPDATE SET value = EXCLUDED.value
    #         """
    #         self.alch_conn.execute(text(query))
    #
    #         # Step 4.1: Get the source_code of the corresponding Value variable
    #         query = f"""
    #         SELECT source_code, chinese_name
    #         FROM metric_static_info
    #         WHERE english_name = '{metric_name_value}'
    #         """
    #         df = pd.read_sql_query(text(query), self.alch_conn)
    #         if df.empty:
    #             # 数据库中存在一些老旧的不需要的数据，它们在metric_static_info没有记录
    #             continue
    #
    #         source_code_value = df.loc[0, 'source_code']
    #         chinese_name_value = df.loc[0, 'chinese_name']
    #         if chinese_name_value not in cn_names_to_exhibit:
    #             # 跳过不需展示的metric
    #             continue
    #
    #         # Step 4.2: Update source_code in metric_static_info
    #         self.adjust_seq_val()
    #         new_source_code = f'calculated from {source_code_value}'
    #         chinese_name_yoy = chinese_name_value.replace(cn_value_str, cn_yoy_str)
    #         query = f"""
    #         INSERT INTO metric_static_info (english_name, source_code, chinese_name, unit)
    #         VALUES ('{metric_name_yoy}', '{new_source_code}', '{chinese_name_yoy}', '%')
    #         ON CONFLICT (english_name, source_code) DO UPDATE
    #         SET source_code = EXCLUDED.source_code, chinese_name = EXCLUDED.chinese_name, unit = EXCLUDED.unit
    #         """
    #
    #         self.alch_conn.execute(text(query))
    #         self.alch_conn.commit()

    # def calculate_custom_metric(self, english_name_a, english_name_b, calculation_function, new_english_name,
    #                             new_chinese_name, new_unit):
    #     """
    #     只能一条一条计算
    #     :param english_name_a:
    #     :param english_name_b:
    #     :param calculation_function:
    #     :param new_english_name:
    #     :param new_chinese_name:
    #     :return:
    #     """
    #     # Check if calculation_function has a name
    #     function_name = getattr(calculation_function, '__name__', None)
    #     if function_name is None:
    #         raise ValueError("calculation_function must have a name, use regular function definition.")
    #
    #     # Step 1: Get data for the specified columns
    #     query_a = f"SELECT date, value as value_a FROM low_freq_long WHERE metric_name = '{english_name_a}'"
    #     query_b = f"SELECT date, value as value_b FROM low_freq_long WHERE metric_name = '{english_name_b}'"
    #     df_a = pd.read_sql_query(text(query_a), self.alch_conn)
    #     df_b = pd.read_sql_query(text(query_b), self.alch_conn)
    #
    #     # Step 2: Merge data on date
    #     df = df_a.merge(df_b, on='date')
    #
    #     # Step 3: Calculate the custom metric using the calculation_function
    #     df['calculated_value'] = df.apply(lambda row: calculation_function(row['value_a'], row['value_b']), axis=1)
    #
    #     # Step 4: Insert calculated values into low_freq_long
    #     for _, row in df.iterrows():
    #         query = f"""
    #         INSERT INTO low_freq_long (date, metric_name, value)
    #         VALUES ('{row['date']}', '{new_english_name}', {row['calculated_value']})
    #         ON CONFLICT (date, metric_name) DO UPDATE SET value = EXCLUDED.value
    #         """
    #         self.alch_conn.execute(text(query))
    #
    #     # Step 5: Update metric_static_info
    #     self.adjust_seq_val()
    #     query = f"""
    #     INSERT INTO metric_static_info (english_name, source_code, chinese_name, unit)
    #     VALUES ('{new_english_name}', 'calculated from {english_name_a} and {english_name_b} using {function_name}', '{new_chinese_name}', '{new_unit}')
    #     ON CONFLICT (english_name, source_code) DO UPDATE
    #     SET source_code = EXCLUDED.source_code, chinese_name = EXCLUDED.chinese_name, unit = EXCLUDED.unit
    #     """
    #     self.alch_conn.execute(text(query))
    #     self.alch_conn.commit()
    #
    # def divide(self, a, b):
    #     return a / b
