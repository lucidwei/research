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
from �����ܲ�λ����demo import CalcFundPosition
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
        ��21�°��꿪ʼ��
        pg���ݰ�����
        �������ڡ�ETF�նȾ�����
        ��ҵ���̼�
        """
        north_inflow_long = self.select_df_from_long_table(table_name='markets_daily_long',
                                                           field='������')
        margin_long = self.select_df_from_long_table(table_name='markets_daily_long',
                                                           field='���ʾ������')
        etf_long = self.read_joined_table_as_dataframe(target_table_name='markets_daily_long',
                                                       target_join_column='product_static_info_id',
                                                       join_table_name='product_static_info',
                                                       join_column='internal_id',
                                                       filter_condition="field='�������' AND etf_type='��ҵETF'")
        north_inflow_long['��ҵ'] = north_inflow_long['product_name'].str.extract(r'CS(\w+)', expand=False)
        df_north_inflow = north_inflow_long.pivot(index='date', columns='��ҵ', values='value') / 1e8
        self.df_north_inflow = df_north_inflow.fillna(0)

        margin_long['��ҵ'] = margin_long['product_name'].str.extract(r'CS(\w+)', expand=False)
        df_margin = margin_long.pivot(index='date', columns='��ҵ', values='value') / 1e8
        self.df_margin = df_margin.fillna(0)

        grouped_etf = etf_long.groupby(['date', 'stk_industry_cs'])['value'].sum().reset_index()
        df_etf = grouped_etf.pivot(index='date', columns='stk_industry_cs', values='value') / 1e8
        self.df_etf = df_etf.fillna(0)

        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='���̼�'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        df_daily_price = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.df_daily_price = df_daily_price
        self.df_daily_return = df_daily_price.pct_change().dropna(how='all')*100


    def load_from_excel(self):
        """
        ���û����ܲ�λ����demo.py�еķ����õ�ÿ�ջ���Ĺ�����ҵ������
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
        ÿ��ĩ���ù�ȥ60������ݲ���IRF��ȷ��������δ�����ݡ�
        �ò�ͬ�ʽ��������������ҵ�ǵ�����ETF�ͻ�������Ȩ�ص�һЩ��ǰ�����ݵ��٣����ߣ��õ���ҵ�ǵ�������
        ����һ��ʵ����ҵ�ǵ���˳��������ء�
        """
        pass

    def evaluate_weekly_spearman(self):
        pass

    def calc_backtest_nav(self):
        pass


base_config = BaseConfig('quarterly')
data = DataProcessor(base_config)