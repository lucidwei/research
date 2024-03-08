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
        从21下半年开始的
        pg数据包括：
        北向、两融、ETF日度净流入
        """
        pass

    def load_from_excel(self):
        """
        调用基金总仓位测算demo.py中的方法得到每日基金的估算行业净流入
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
        每周末利用过去60天的数据测算IRF，确保不利用未来数据。
        用不同资金流估算出各个行业涨跌幅，ETF和基金估算的权重低一些（前者数据点少，后者）得到行业涨跌幅排序
        与下一周实际行业涨跌幅顺序做秩相关。
        """
        pass

    def evaluate_weekly_spearman(self):
        pass

    def calc_backtest_nav(self):
        pass