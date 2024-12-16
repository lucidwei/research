# coding=gbk
# Time Created: 2022/12/26 17:58
# Author  : Lucid
# FileName: base_config.py
# Software: PyCharm
import platform, os, datetime
from utils import get_tradedays, is_date, get_month_end_dates, split_tradedays_into_weekly_ranges, check_wind
import configparser
import pandas as pd


class BaseConfig:
    """
    :project: str: 每做一个新课题加入新的可选字符串
    :auto_save_fig: bool: 如果代码运行的目的是定期自动生成图片，则设为true，避免运行plt.show()
    'yield_10y', 'firms_cls', 'code_processing', 'export_heavy', 'T_basis'
    """

    def __init__(self, project: str, auto_save_fig: bool = True):
        self.auto_save_fig = auto_save_fig
        self.project = project
        check_wind()
        self.config_paths()
        self.config_dates()
        self.config_db()

    def config_dates(self):
        # 获取区间内交易日
        start_date_dict = {'T_basis': datetime.date(2020, 1, 1),
                           'high_freq': datetime.date(2010, 1, 1),
                           'risk_parity': datetime.date(2012, 1, 1),
                           'low_freq': datetime.date(2004, 1, 1),
                           'equity_liquidity': datetime.date(2019, 1, 2),
                           'equity_sentiment': datetime.date(2016, 1, 2),
                           'quarterly': datetime.date(2000, 1, 2),
                           'boom': datetime.date(2010, 1, 2),
                           'multi-asset': datetime.date(2010, 1, 1),
                           }
        date_start = start_date_dict[self.project]
        date_end = datetime.date.today() # - datetime.timedelta(weeks=100) # - datetime.timedelta(days=1) #开发调试时wind quota受限、节省quota时用
        self.tradedays = get_tradedays(date_start, date_end)
        self.tradedays_str = [str(x) for x in self.tradedays]

        # 获取区间内所有日期
        all_dates = []
        current_date = date_start
        while current_date <= date_end:
            all_dates.append(current_date)
            current_date += datetime.timedelta(days=1)
        self.all_dates = all_dates
        self.all_dates_str = [str(x) for x in self.all_dates]

        # 对于低频数据获取月末日期
        self.month_ends, self.month_ends_str = get_month_end_dates(date_start, date_end)

        self.weekly_date_ranges = split_tradedays_into_weekly_ranges(self.tradedays)

    def config_db(self):
        # 获取 db_config 部分的配置信息
        db_name_dict = {'T_basis': 'wgz_db',
                        'high_freq': 'wgz_db',
                        'risk_parity': 'wgz_db',
                        'low_freq': 'wgz_db',
                        'equity_liquidity': 'wgz_db',
                        'equity_sentiment': 'wgz_db',
                        'quarterly': 'wgz_db',
                        'boom': 'wgz_db',
                        'multi-asset': 'wgz_db',
                        }

        self.db_config = {
            'database': db_name_dict[self.project],
            'user': self.config.get('db_config', 'user'),
            'password': self.config.get('db_config', 'password'),
            'host': self.config.get('db_config', 'host'),
            'port': self.config.get('db_config', 'port')
        }

        self.pg_ctl = self.config.get('db_config', 'pg_ctl')
        self.data_dir = self.config.get('db_config', 'data_dir')

    def config_paths(self):
        # 获取当前脚本的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(current_file_path)
        self.prj_dir = current_dir
        # 从当前目录构建项目根目录或相对目录
        config_file_path = os.path.join(current_dir, 'config.ini')
        # 读取其它必需配置文件
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path, encoding='utf-8')
        # 保存图片
        self.image_folder = self.config.get('image_config', 'folder')
        self.excels_path = self.config.get('excels_path', 'excels_path')

    def process_wind_excel(self, excel_file_name:str, sheet_name=None):
        """
        用wind下载下来的格式化excel整理数据的metadata从而避免手工处理，如指标名称、频率、单位、指标ID、时间区间、来源、更新时间
        """

        # 读取 Excel 文件，其中包含所需要的数据的metadata
        file_path = os.path.join(self.excels_path, excel_file_name)
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, header=None)

        # 定位最后一个 metadata 字段（在第一个日期类型值的位置上一个）
        last_metadata_row = None
        for idx, value in df.index.to_series().items():
            if is_date(value):
                last_metadata_row = idx
                break

        last_metadata_row = df.index.get_loc(last_metadata_row) - 1

        # 提取 metadata
        metadata = df.iloc[:last_metadata_row].copy(deep=True)
        metadata.dropna(inplace=True, how='all')
        metadata = metadata.transpose()

        # 提取数据部分
        data = df.iloc[last_metadata_row + 1:].copy(deep=True)

        # 设置 data 的列索引为 '指标ID'
        indicator_ids = metadata.loc[:, '指标ID']
        data.columns = indicator_ids

        return metadata, data


