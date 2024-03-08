# coding=gbk
# Time Created: 2024/3/8 9:10
# Author  : Lucid

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib, os
import matplotlib.gridspec as gridspec
from typing import Optional
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from �����ܲ�λ����demo import CalcFundPosition
from scipy.stats import spearmanr


class DataProcessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.base_config = base_config
        self.load_from_pgdb()
        self.load_from_excel()
        self.align_dates_industries()
        # self.df_daily_price: Optional[pd.DataFrame] = None
        # self.df_daily_return: Optional[pd.DataFrame] = None
        # self.df_etf: Optional[pd.DataFrame] = None
        # self.df_north_inflow: Optional[pd.DataFrame] = None
        # self.df_margin: Optional[pd.DataFrame] = None
        # self.df_fund_estimate: Optional[pd.DataFrame] = None

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
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(
            columns={'chinese_name': 'industry'})
        df_daily_price = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.df_daily_price = df_daily_price
        self.df_daily_return = df_daily_price.pct_change().dropna(how='all') * 100

    def load_from_excel(self):
        """
        ���û����ܲ�λ����demo.py�еķ����õ�ÿ�ջ���Ĺ�����ҵ������
        """
        obj = CalcFundPosition(self.base_config, '21q2')

        state_estimates_post, return_errors = obj.post_constraint_kf(obj.industry_return, obj.total_return,
                                                                     calibrate=False)
        active_adjustments = obj.calculate_active_adjustment(state_estimates_post, obj.industry_return)
        self.df_fund_estimate = obj.calculate_active_adjustment_amount(active_adjustments)

    def align_dates_industries(self):
        if self.df_fund_estimate is not None:
            industry_columns = self.df_north_inflow.columns
            self.df_fund_estimate = self.df_fund_estimate.reindex(columns=industry_columns, fill_value=0)

            fund_estimate_index = self.df_fund_estimate.index
            if self.df_daily_price is not None:
                self.df_daily_price = self.df_daily_price.reindex(fund_estimate_index, method='ffill')
                self.df_daily_price = self.df_daily_price.reindex(columns=industry_columns, fill_value=0)

            if self.df_daily_return is not None:
                self.df_daily_return = self.df_daily_return.reindex(fund_estimate_index, method='ffill')
                self.df_daily_return = self.df_daily_return.reindex(columns=industry_columns, fill_value=0)

            if self.df_etf is not None:
                self.df_etf = self.df_etf.reindex(fund_estimate_index, method='ffill')
                self.df_etf = self.df_etf.reindex(columns=industry_columns, fill_value=0)

            if self.df_north_inflow is not None:
                self.df_north_inflow = self.df_north_inflow.reindex(fund_estimate_index, method='ffill')

            if self.df_margin is not None:
                self.df_margin = self.df_margin.reindex(fund_estimate_index, method='ffill')
                self.df_margin = self.df_margin.reindex(columns=industry_columns, fill_value=0)


class Evaluator:
    def __init__(self, data: DataProcessor):
        self.data = data
        self.analyze_industries()
        self.calc_weekly_irfs()
        self.evaluate_weekly_spearman()
        self.calc_backtest_nav()

    def calc_weekly_irfs(self):
        """
        ÿ��ĩ���ù�ȥ60������ݲ���IRF��ȷ��������δ�����ݡ�
        �ò�ͬ�ʽ��������������ҵ�ǵ�����ETF�ͻ�������Ȩ�ص�һЩ��ǰ�����ݵ��٣����ߣ��õ���ҵ�ǵ�������
        ����һ��ʵ����ҵ�ǵ���˳��������ء�
        """

        # end_date = pd.to_datetime('today')
        # start_date = end_date - pd.Timedelta(days=60)

        # ��ÿ������֡��ѡ���ȥ60�������
        df_daily_return = self.data.df_daily_return
        df_daily_price = self.data.df_daily_price
        df_north_inflow = self.data.df_north_inflow
        df_margin = self.data.df_margin
        df_etf = self.data.df_etf
        df_fund_estimate = self.data.df_fund_estimate

        # ����ÿ��������
        # next_week_returns = df_daily_price.pct_change(5).shift(-5)  # Ԥ����һ���ܡ���������

        # ��������ת��Ϊ����
        period_starts = pd.date_range(start=df_daily_price.index.min(),
                                      end=df_daily_price.index.max() - timedelta(days=60), freq='5D')

        correlations = []

        for history_start_date in period_starts:
            history_end_date = history_start_date + timedelta(days=60)  # ʹ��60�������
            # next_week_date = history_end_date + timedelta(days=5)  # ��һ���ܡ�����
            next_week_start_date = history_end_date + timedelta(days=1)
            next_week_end_date = next_week_start_date + timedelta(days=4)

            # ���㵽���һ�� ѭ����ֹ
            if next_week_end_date > df_daily_price.index.max():
                break

            # ��ȡ��һ���ڣ����ܡ�����ʵ���ǵ���
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()

            irf_results = {}

            # ��ÿ����ҵ���з���
            for industry in df_daily_return.columns:
                industry_data = {
                    'north_inflow': df_north_inflow[industry].loc[history_start_date:history_end_date],
                    'margin': df_margin[industry].loc[history_start_date:history_end_date],
                    'etf': df_etf[industry].loc[history_start_date:history_end_date],
                    'fund_estimate': df_fund_estimate[industry].loc[history_start_date:history_end_date]
                }
                industry_inflow_history = pd.DataFrame(industry_data)
                merged = pd.merge(history_return[industry], industry_inflow_history,
                              left_index=True, right_index=True)

                if industry_inflow_history.dropna().shape[0] < 25:  # ȷ�����㹻�����ݵ����VAR����
                    continue

                # try:
                # model = sm.tsa.VAR(merged)
                # results = model.fit(maxlags=5, ic='aic')
                # print(f'ѡ���������ͺ���: {results.k_ar}')
                # irf = results.irf(10).irfs#[:, :, -1]  # ȡ�����������ʵ�IRF
                # irf_sum = np.sum(irf, axis=0)
                # irf_results[industry] = np.sum(irf_sum)  # ��͵õ�����Ӱ��
                # except Exception as e:
                #     print(f"Error in VAR model for {industry}: {e}")
                #     continue

            # ����IRF�Ľ��������ҵ
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)

            # ��ȡ��һ����ʵ���ǵ���������
            sorted_actual_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # ����ʵ���ǵ�������
            sorted_real_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # ���������
            if sorted_industries and sorted_real_returns:
                rank_corr, _ = spearmanr(sorted_industries, sorted_real_returns)
                correlations.append(rank_corr)

        # �����������ڵ������ƽ��ֵ
        return np.mean(correlations)

    def analyze_industries(self):
        df_daily_return = self.data.df_daily_return
        df_daily_price = self.data.df_daily_price
        df_north_inflow = self.data.df_north_inflow
        df_margin = self.data.df_margin
        df_etf = self.data.df_etf
        df_fund_estimate = self.data.df_fund_estimate


        # ��������ת��Ϊ����
        period_starts = pd.date_range(start=df_daily_price.index.min(),
                                      end=df_daily_price.index.max() - timedelta(days=60), freq='5D')

        correlations = {}

        for history_start_date in period_starts:
            history_end_date = history_start_date + timedelta(days=60)  # ʹ��60�������
            next_week_start_date = history_end_date + timedelta(days=1)
            next_week_end_date = next_week_start_date + timedelta(days=4)

            # ���㵽���һ�� ѭ����ֹ
            if next_week_end_date > df_daily_price.index.max():
                break

            # ��ȡ��һ���ڣ����ܡ�����ʵ���ǵ���
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()

            irf_results = {}
            # ���費ͬ�ʽ�����ͬ��Ȩ��
            fund_flows_with_weights = {
                'north_inflow': 1.0,  # �����������ʽ�Ȩ��Ϊ1.0
                # 'margin': 0.8,  # ������ȯ�ʽ�Ȩ��Ϊ0.8
                # 'etf': 0.5,  # ETF�ʽ�Ȩ��Ϊ0.5
                # 'fund_estimate': 0.6  # ��������ʽ�Ȩ��Ϊ0.6
            }

            fund_flows_dfs = {
                'north_inflow': df_north_inflow,
                'margin': df_margin,
                'etf': df_etf,
                'fund_estimate': df_fund_estimate
            }

            for industry in df_daily_return.columns:
                weighted_effects = []

                for fund_flow_name, weight in fund_flows_with_weights.items():
                    fund_flow_df = fund_flows_dfs[fund_flow_name]
                    industry_inflow_history = fund_flow_df[industry].loc[history_start_date:history_end_date]
                    effect = self.calculate_irf_for_each_fund_flow(industry, industry_inflow_history, history_return, fund_flow_name)
                    if not np.isnan(effect):
                        weighted_effect = effect * weight  # �����ȨЧӦ
                        weighted_effects.append(weighted_effect)

                if weighted_effects:
                    # ʹ�ü�ȨЧӦ�ĺͳ���Ȩ�صĺͣ��õ���Ȩƽ��ЧӦ
                    irf_results[industry] = sum(weighted_effects) / sum(fund_flows_with_weights.values())

            # ����IRF�Ľ��������ҵ
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)
            # ����ʵ���ǵ�������
            sorted_real_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # ���������
            if sorted_industries and sorted_real_returns:
                rank_corr, _ = spearmanr(sorted_industries, sorted_real_returns)
                correlations[history_end_date] = rank_corr
        return correlations

    def calculate_irf_for_each_fund_flow(self, industry, industry_inflow_history, history_return, fund_flow_name):
        """
        �Ը�������ҵ���ʽ�����ʷ���ݼ���IRF��
        """
        if industry_inflow_history.empty or industry not in history_return:
            return np.nan

        merged = pd.concat([history_return[industry], industry_inflow_history], axis=1).dropna()

        if len(merged) < 25:  # ȷ�����㹻�����ݵ����VAR����
            return np.nan

        try:
            model = sm.tsa.VAR(merged)
            results = model.fit(maxlags=5, ic='aic')
            irf = results.irf(10).irfs
            # ��ȡ����������ʽ����룩��1�ж���Ӧ��������ҵ�����ʣ���0�е��ۻ�Ӱ��
            cumulative_response = np.sum(irf[:10, 1, 0])
            return cumulative_response
        except Exception as e:
            print(f"Error in VAR model for {fund_flow_name} {industry}: {e}")
            return 0

    def evaluate_weekly_spearman(self):
        pass

    def calc_backtest_nav(self):
        pass


base_config = BaseConfig('quarterly')
data = DataProcessor(base_config)
evaluator = Evaluator(data)
