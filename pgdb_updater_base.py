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

    def update_edb_by_id_to_high_freq(self, code: str):
        """
        TODO: 这种函数不能给上传的数据加description列，但后续描述列是要放在外键的。因此不用担心
        注意：
        1. self.get_missing_dates(self.all_dates导致这个方法只针对日频数据，低频数据需重新定义函数
        2. 如需对原始数据做处理，不应使用此方法
        先获取缺失的日期列表,需要更新的两段日期是：
        - all_dates的最早一天到数据库存在数据的最早一天
        - 数据库存在数据的最后一天到all_dates的最后一天，也就是今天
        """
        dates_missing = self.get_missing_dates(self.tradedays, 'high_freq_long', column_name=self.conversion_dicts['id_to_english'][code])
        if len(dates_missing) == 0:
            return
        old_dates = [date for date in dates_missing if date.year < 2023]
        new_dates = [date for date in dates_missing if date.year >= 2023]
        for dates_update in [old_dates, new_dates]:
            if not dates_update:
                continue
            print(f'Wind downloading {code} for table high_freq_long between {str(dates_update[0])} and {str(dates_update[-1])}')
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
    def update_wsd(self, code: str, fields: str):
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
                                                   column_name=self.conversion_dicts['id_to_english'][code],
                                                   field=field.lower())

            if len(dates_missing) != 0:
                print(
                    f'Wind downloading {code} {field} for markets_daily_long between {str(dates_missing[0])} and {str(dates_missing[-1])}')
                downloaded_df = w.wsd(code, field, str(dates_missing[0]), str(dates_missing[-1]), "", usedf=True)[1]
                # 转换下载的数据框为长格式
                downloaded_df.index.name = 'date'
                downloaded_df.reset_index(inplace=True)
                downloaded_df.columns = [col.lower() for col in downloaded_df.columns]
                downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='field', value_name='value')
                downloaded_df.dropna(subset=['value'], inplace=True)

                # 添加其他所需列
                downloaded_df['product_name'] = self.conversion_dicts['id_to_english'][code]
                downloaded_df['source_code'] = f"wind_{code}"
                downloaded_df['chinese_name'] = downloaded_df['product_name'].map(
                    self.conversion_dicts['english_to_chinese'])

                # 将新行插入数据库中, df要非空
                if downloaded_df.iloc[0, 0] != 0:
                    downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, table_name='low_freq_long'):
        """
        先获取缺失的月末日期列表,需要更新的两段日期是：
        - all_dates的最早一天到数据库存在数据的最早一天
        - 数据库存在数据的最后一天到all_dates的最后一天，也就是今天
        wind fetch缺失日期的EDB数据
        整理好格式传入数据库
        """
        # 剔除 '指标ID' 列字符串长度小于 5 的行，这些是EDB中计算后数值
        self.metadata = self.base_config.process_wind_metadata(excel_file)
        self.metadata = self.metadata[self.metadata['指标ID'].apply(lambda x: len(str(x)) >= 5)]
        # 定义 DataFrame 中列名和表中列名的对应关系
        indicator_id_col = self.metadata.loc[:, '指标ID']
        indicator_name_col = self.metadata.loc[:, '指标名称']

        map_id_to_english = {}
        for ind_id, ind_name in zip(indicator_id_col, indicator_name_col):
            # 使用手动映射的字典来替换列名
            new_string = name_mapping[ind_name]
            # 将指标 ID 与新字符串添加到字典中
            map_id_to_english[ind_id] = new_string
        self.db_col = map_id_to_english.values()

        # 反转 map_id_to_english 字典，将英文列名映射回指标名称
        map_english_to_id = {v: k for k, v in map_id_to_english.items()}
        map_english_to_name = {v: k for k, v in name_mapping.items()}

        # 更新数据
        dates_missing = self.get_missing_months_ends(self.months_ends, table_name)
        ids = self.metadata['指标ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print(f'Wind downloading for {table_name} 数据')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
            # wind返回的df，日期为一天和多天的格式不一样
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing

            # 重命名列为数据库中列
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)
            downloaded_df.rename(columns=map_id_to_english, inplace=True)

            # 计算近半年起始日期
            six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
            # 计算近半年的数据点数并计算更新频率
            update_freq_dict = {}
            for col in downloaded_df.columns:
                if col != 'date':
                    recent_data = downloaded_df.loc[downloaded_df['date'] >= six_months_ago, col]
                    non_null_data_points = recent_data.count()

                    # 计算更新频率
                    date_range = 6 * 30  # 近半年的天数
                    update_freq = date_range / non_null_data_points
                    update_freq_dict[col] = update_freq

            # 将新行插入数据库中
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.dropna(subset=['value'], inplace=True)
            # 添加 chinese_name、update_freq 和 source 列
            downloaded_df['chinese_name'] = downloaded_df['metric_name'].map(map_english_to_name)
            downloaded_df['update_freq'] = downloaded_df['metric_name'].map(update_freq_dict)
            downloaded_df['source'] = downloaded_df['metric_name'].map(map_english_to_id).apply(lambda x: f'wind_{x}')

            downloaded_df.to_sql(table_name, self.alch_engine, if_exists='append', index=False)

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
