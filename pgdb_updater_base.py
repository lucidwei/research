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
from sqlalchemy import text
from sqlalchemy import Table, MetaData
from sqlalchemy.orm import Session
from sqlalchemy import select


class PgDbUpdaterBase(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_dates()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str
        self.months_ends = self.base_config.month_ends
        self.months_ends_str = self.base_config.month_ends_str

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
        dates_missing = self.get_missing_dates(self.tradedays, 'high_freq_long',
                                               english_id=self.conversion_dicts['id_to_english'][code])
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
        code和field根据代码生成器CG获取。
        先获取缺失的日期列表,需要更新的两段日期是：
        - all_dates的最早一天到数据库存在数据的最早一天
        - 数据库存在数据的最后一天到all_dates的最后一天，也就是今天
        wind fetch缺失日期的日度数据，写入DB；
        """
        fields_list = fields.split(',')

        for field in fields_list:
            dates_missing = self.get_missing_dates(self.tradedays, "markets_daily_long",
                                                   english_id=self.conversion_dicts['id_to_english'][code],
                                                   field=field.lower())
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
            with self.alch_engine.connect() as conn:
                query = f"""
                INSERT INTO metric_static_info (source_code, chinese_name)
                VALUES ('{source_code}', '{chinese_name}')
                ON CONFLICT (source_code) DO UPDATE
                SET chinese_name = EXCLUDED.chinese_name;
                """
                conn.execute(text(query))

                # 获取metric_static_info表中对应记录的internal_id
                query = f"""
                SELECT internal_id
                FROM metric_static_info
                WHERE source_code = '{source_code}' AND chinese_name = '{chinese_name}';
                """
                internal_id = conn.execute(text(query)).fetchone()[0]

            # 将新行插入数据库中, df要非空
            if downloaded_df.iloc[0, 0] != 0:
                print('Uploading data to database...')
                downloaded_df['product_name'] = product_name
                downloaded_df['metric_static_info_id'] = internal_id
                downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def update_low_freq_by_edb_id(self, earliest_available_date, code, maps: tuple):
        map_id_to_name, map_id_to_unit, map_id_to_english = maps
        # 更新数据
        existing_dates = self.get_existing_dates_from_db('low_freq_long', map_id_to_english[code])
        dates_missing = self.get_missing_months_ends(self.months_ends, earliest_available_date, 'low_freq_long', map_id_to_english[code])
        if len(dates_missing) == 0:
            print(f'No missing data for low_freq_long {map_id_to_name[code]}, skipping download')
            return

        print(f'Wind downloading for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}')
        downloaded_df = w.edb(code, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
        downloaded_df.columns=[code]

        # 删除与 existing_dates 中日期相同的行
        existing_dates_set = set(existing_dates)
        downloaded_df = downloaded_df.loc[~downloaded_df.index.isin(existing_dates_set)]

        if downloaded_df.empty:
            print(f'No missing data for low_freq_long的{map_id_to_name[code]}, downloaded but will not upload')
            return

        # wind返回的df，日期为一天和多天的格式不一样
        if dates_missing[0] == dates_missing[-1]:
            downloaded_df = downloaded_df.T
            downloaded_df.index = dates_missing

        # 重命名列为数据库中列
        downloaded_df.reset_index(inplace=True)
        downloaded_df.rename(columns={'index': 'date'}, inplace=True)

        # 计算近半年起始日期
        six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
        # 计算近半年的数据点个数并计算更新频率
        existing_dates_series = pd.Series(existing_dates, dtype='datetime64[D]').dt.date
        combined_dates = pd.to_datetime(pd.concat([existing_dates_series, downloaded_df['date']])).dt.date

        # 选择六个月前之后的日期
        recent_dates = combined_dates[combined_dates >= six_months_ago]
        # 计算近半年的数据点个数并计算更新频率
        non_null_data_points = recent_dates.count()
        # 计算更新频率=近半年的天数/非空数据点个数
        update_freq = 180 / non_null_data_points

        # 更新metric_static_info 元数据table
        unit = map_id_to_unit[code]
        source_code = f'wind_{code}'
        chinese_name = map_id_to_name[code]
        english_name = map_id_to_english[code]

        # Ensure the corresponding source_code and chinese_name exist in the metric_static_info table
        with self.alch_engine.connect() as conn:
            query = text("""
                        INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name)
                        VALUES (:source_code, :chinese_name, :unit, :english_name)
                        ON CONFLICT (source_code) DO UPDATE
                        SET english_name = EXCLUDED.english_name,
                            unit = EXCLUDED.unit;
                        """)
            conn.execute(query,
                         {
                             'source_code': source_code,
                             'chinese_name': chinese_name,
                             'unit': unit,
                             'english_name': english_name
                         })
            conn.commit()

            # Get the internal_id of the corresponding record in the metric_static_info table
            query = f"""
                    SELECT internal_id
                    FROM metric_static_info
                    WHERE source_code = :source_code AND chinese_name = :chinese_name;
                    """
            internal_id = conn.execute(text(query), {
                'source_code': source_code,
                'chinese_name': chinese_name,
            }).fetchone()[0]

        # 将新数据插入数据库中
        df_upload = downloaded_df.rename(columns=map_id_to_english)
        df_upload = df_upload.melt(id_vars=['date'], var_name='metric_name', value_name='value')
        df_upload.dropna(subset=['value'], inplace=True)
        # 添加 additional_info:update_freq 和 metric_static_info_id 列
        df_upload['update_freq'] = update_freq
        df_upload['metric_static_info_id'] = internal_id

        df_upload.to_sql('low_freq_long', self.alch_engine, if_exists='append', index=False)


    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, if_rename=False):
        """
        根据excel文件中的metadata更新数据
        """
        def get_start_month_end(s):
            start_date_str = s.split(':')[0]  # 提取开始日期字符串
            start_date = pd.to_datetime(start_date_str, format='%Y-%m')  # 转换成日期格式
            start_month_end = start_date + pd.offsets.MonthEnd(1)  # 获取月末日期
            return start_month_end

        # 剔除 '指标ID' 列字符串长度小于 5 的行，这些是EDB中计算后数值
        self.metadata = self.base_config.process_wind_metadata(excel_file)
        self.metadata = self.metadata[self.metadata['指标ID'].apply(lambda x: len(str(x)) >= 5)]

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
        # renaming中会用到,其实是删除了excel_file对应的所有列，没有进行筛选。但因为很少需要rename，因此不做优化
        names_to_delete = list(map_id_to_chinese.values())

        if if_rename:
            self.delete_for_renaming(names_to_delete)

        maps_tuple = (map_id_to_chinese, map_id_to_unit, map_id_to_english)
        # 更新数据
        for id in col_indicator_id:
            self.update_low_freq_by_edb_id(map_id_to_earlist_date[id], id, maps_tuple)

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
        SELECT {function_name}(:table_name, :view_name, ARRAY{chinese_names})
        """

        self.alch_conn.execute(
            text(query),
            {
                'table_name': table_name,
                'view_name': view_name,
            }
        )
        self.alch_conn.commit()

    def read_from_high_freq_view(self, code_list: List[str]) -> pd.DataFrame:
        """
        从 high_freq_view (wide format) 读取数据，根据 code_list。
        :param code_list: 要查询的代码列表，例如 ['S0059749', 'G0005428']
        :return: 返回一个包含所请求数据的 DataFrame
        """
        # 获取英文列名
        column_names = [self.conversion_dicts['id_to_english'][code] for code in code_list]

        # 构建 SQL 查询
        sql_query = f"SELECT date, {', '.join(column_names)} FROM high_freq_wide"

        # 从数据库中读取数据
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        return result_df.set_index('date').sort_index(ascending=False)

    def read_from_markets_daily_long(self, code: str, fields: str) -> pd.DataFrame:
        """
        从 markets_daily_long (long format) 读取数据，根据 code 和 fields。
        :param code: 要查询的代码，例如 'S0059749'
        :param fields: 要查询的字段，例如 "close,volume,pe_ttm"
        :return: 返回一个包含所请求数据的 DataFrame
        """
        # 将逗号分隔的 fields 转换为列表
        field_list = fields.split(',')

        # 获取 product_name
        product_name = self.conversion_dicts['id_to_english'][code]

        # 构建 SQL 查询
        sql_query = f"""SELECT date, field, value
                       FROM markets_daily_long
                       WHERE product_name = '{product_name}'
                       AND field IN ({', '.join([f"'{field}'" for field in field_list])})"""

        # 从数据库中读取数据
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        # 转换为宽格式
        result_df = result_df.pivot_table(index='date', columns='field', values='value')

        return result_df

    def get_missing_metrics(self, target_table: str, target_column: str, metrics_at_hand: list):
        # 创建session
        session = Session(self.alch_engine)

        # 获取目标表的元数据
        metadata = MetaData()
        target_table = Table(target_table, metadata, autoload_with=self.alch_engine)

        # 查询目标表中的值
        stmt = select(target_table.c[target_column])
        result = session.execute(stmt)

        existing_values = {row[0] for row in result}

        # 将输入的metrics列表转换为集合
        input_set = set(metrics_at_hand)

        # 找出存在于输入列表中，但不存在于目标表中的值
        missing_values = input_set - existing_values

        return list(missing_values)