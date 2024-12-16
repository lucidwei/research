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
    :project: str: ÿ��һ���¿�������µĿ�ѡ�ַ���
    :auto_save_fig: bool: ����������е�Ŀ���Ƕ����Զ�����ͼƬ������Ϊtrue����������plt.show()
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
        # ��ȡ�����ڽ�����
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
        date_end = datetime.date.today() # - datetime.timedelta(weeks=100) # - datetime.timedelta(days=1) #��������ʱwind quota���ޡ���ʡquotaʱ��
        self.tradedays = get_tradedays(date_start, date_end)
        self.tradedays_str = [str(x) for x in self.tradedays]

        # ��ȡ��������������
        all_dates = []
        current_date = date_start
        while current_date <= date_end:
            all_dates.append(current_date)
            current_date += datetime.timedelta(days=1)
        self.all_dates = all_dates
        self.all_dates_str = [str(x) for x in self.all_dates]

        # ���ڵ�Ƶ���ݻ�ȡ��ĩ����
        self.month_ends, self.month_ends_str = get_month_end_dates(date_start, date_end)

        self.weekly_date_ranges = split_tradedays_into_weekly_ranges(self.tradedays)

    def config_db(self):
        # ��ȡ db_config ���ֵ�������Ϣ
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
        # ��ȡ��ǰ�ű��ľ���·��
        current_file_path = os.path.abspath(__file__)
        # ��ȡ��ǰ�ű����ڵ�Ŀ¼
        current_dir = os.path.dirname(current_file_path)
        self.prj_dir = current_dir
        # �ӵ�ǰĿ¼������Ŀ��Ŀ¼�����Ŀ¼
        config_file_path = os.path.join(current_dir, 'config.ini')
        # ��ȡ�������������ļ�
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path, encoding='utf-8')
        # ����ͼƬ
        self.image_folder = self.config.get('image_config', 'folder')
        self.excels_path = self.config.get('excels_path', 'excels_path')

    def process_wind_excel(self, excel_file_name:str, sheet_name=None):
        """
        ��wind���������ĸ�ʽ��excel�������ݵ�metadata�Ӷ������ֹ�������ָ�����ơ�Ƶ�ʡ���λ��ָ��ID��ʱ�����䡢��Դ������ʱ��
        """

        # ��ȡ Excel �ļ������а�������Ҫ�����ݵ�metadata
        file_path = os.path.join(self.excels_path, excel_file_name)
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, header=None)

        # ��λ���һ�� metadata �ֶΣ��ڵ�һ����������ֵ��λ����һ����
        last_metadata_row = None
        for idx, value in df.index.to_series().items():
            if is_date(value):
                last_metadata_row = idx
                break

        last_metadata_row = df.index.get_loc(last_metadata_row) - 1

        # ��ȡ metadata
        metadata = df.iloc[:last_metadata_row].copy(deep=True)
        metadata.dropna(inplace=True, how='all')
        metadata = metadata.transpose()

        # ��ȡ���ݲ���
        data = df.iloc[last_metadata_row + 1:].copy(deep=True)

        # ���� data ��������Ϊ 'ָ��ID'
        indicator_ids = metadata.loc[:, 'ָ��ID']
        data.columns = indicator_ids

        return metadata, data


