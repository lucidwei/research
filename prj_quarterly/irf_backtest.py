# coding=gbk
# Time Created: 2024/3/8 9:10
# Author  : Lucid

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib, os
import matplotlib.gridspec as gridspec
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text


class DataProcessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.load_from_pgdb()
        self.load_from_excel()
        self.process_into_df()

    def load_from_pgdb(self):
        """
        ��21�°��꿪ʼ��
        pg���ݰ�����
        �������ڡ�ETF�նȾ�����
        """
        pass

    def load_from_excel(self):
        """
        ���û����ܲ�λ����demo.py�еķ����õ�ÿ�ջ���Ĺ�����ҵ������
        """
        pass

    def process_into_df(self):
        pass



class Evaluator:
    def __init__(self, data: DataProcessor):
        self.data = data
        self.calc_weekly_irfs()
        self.evaluate_weekly_spearman()
        self.calc_backtest_nav()

    def calc_weekly_irfs(self):
        """
        ÿ��ĩ���ù�ȥ60������ݲ���IRF��ȷ��������δ�����ݡ�
        �ò�ͬ�ʽ��������������ҵ�ǵ�����ETF�ͻ�������Ȩ�ص�һЩ��ǰ�����ݵ��٣����ߣ��õ���ҵ�ǵ�������
        ����һ��ʵ����ҵ�ǵ���˳��������ء�
        """
        pass

    def evaluate_weekly_spearman(self):
        pass

    def calc_backtest_nav(self):
        pass