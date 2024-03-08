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
from 基金总仓位测算demo import CalcFundPosition
from sqlalchemy import text


class DataProcessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.base_config = base_config
        self.load_from_pgdb()
        self.load_from_excel()
        self.process_into_df()

    def load_from_pgdb(self):
        """
        从21下半年开始的
        pg数据包括：
        北向、两融、ETF日度净流入
        行业收盘价
        """
        north_inflow_long = self.select_df_from_long_table(table_name='markets_daily_long',
                                                           field='净买入')
        margin_long = self.select_df_from_long_table(table_name='markets_daily_long',
                                                           field='融资净买入额')
        etf_long = self.read_joined_table_as_dataframe(target_table_name='markets_daily_long',
                                                       target_join_column='product_static_info_id',
                                                       join_table_name='product_static_info',
                                                       join_column='internal_id',
                                                       filter_condition="field='净流入额' AND etf_type='行业ETF'")
        north_inflow_long['行业'] = north_inflow_long['product_name'].str.extract(r'CS(\w+)', expand=False)
        df_north_inflow = north_inflow_long.pivot(index='date', columns='行业', values='value') / 1e8
        self.df_north_inflow = df_north_inflow.fillna(0)

        margin_long['行业'] = margin_long['product_name'].str.extract(r'CS(\w+)', expand=False)
        df_margin = margin_long.pivot(index='date', columns='行业', values='value') / 1e8
        self.df_margin = df_margin.fillna(0)

        grouped_etf = etf_long.groupby(['date', 'stk_industry_cs'])['value'].sum().reset_index()
        df_etf = grouped_etf.pivot(index='date', columns='stk_industry_cs', values='value') / 1e8
        self.df_etf = df_etf.fillna(0)

        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='收盘价'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        df_daily_price = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.df_daily_price = df_daily_price
        self.df_daily_return = df_daily_price.pct_change().dropna(how='all')*100


    def load_from_excel(self):
        """
        调用基金总仓位测算demo.py中的方法得到每日基金的估算行业净流入
        """
        obj = CalcFundPosition(self.base_config, '18q4')

        state_estimates_post, return_errors = obj.post_constraint_kf(obj.industry_return, obj.total_return,
                                                                     calibrate=True)
        active_adjustments = obj.calculate_active_adjustment(state_estimates_post, obj.industry_return)
        self.df_fund_estimate = obj.calculate_active_adjustment_amount(active_adjustments)

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


base_config = BaseConfig('quarterly')
data = DataProcessor(base_config)