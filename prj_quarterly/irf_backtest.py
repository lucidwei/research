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
from 基金总仓位测算demo import CalcFundPosition
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
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(
            columns={'chinese_name': 'industry'})
        df_daily_price = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.df_daily_price = df_daily_price
        self.df_daily_return = df_daily_price.pct_change().dropna(how='all') * 100

    def load_from_excel(self):
        """
        调用基金总仓位测算demo.py中的方法得到每日基金的估算行业净流入
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
        每周末利用过去60天的数据测算IRF，确保不利用未来数据。
        用不同资金流估算出各个行业涨跌幅，ETF和基金估算的权重低一些（前者数据点少，后者）得到行业涨跌幅排序
        与下一周实际行业涨跌幅顺序做秩相关。
        """

        # end_date = pd.to_datetime('today')
        # start_date = end_date - pd.Timedelta(days=60)

        # 从每个数据帧中选择过去60天的数据
        df_daily_return = self.data.df_daily_return
        df_daily_price = self.data.df_daily_price
        df_north_inflow = self.data.df_north_inflow
        df_margin = self.data.df_margin
        df_etf = self.data.df_etf
        df_fund_estimate = self.data.df_fund_estimate

        # 计算每周收益率
        # next_week_returns = df_daily_price.pct_change(5).shift(-5)  # 预测下一“周”的收益率

        # 日期索引转换为周期
        period_starts = pd.date_range(start=df_daily_price.index.min(),
                                      end=df_daily_price.index.max() - timedelta(days=60), freq='5D')

        correlations = []

        for history_start_date in period_starts:
            history_end_date = history_start_date + timedelta(days=60)  # 使用60天的数据
            # next_week_date = history_end_date + timedelta(days=5)  # 下一“周”日期
            next_week_start_date = history_end_date + timedelta(days=1)
            next_week_end_date = next_week_start_date + timedelta(days=4)

            # 计算到最后一周 循环终止
            if next_week_end_date > df_daily_price.index.max():
                break

            # 获取下一周期（“周”）的实际涨跌幅
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()

            irf_results = {}

            # 对每个行业进行分析
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

                if industry_inflow_history.dropna().shape[0] < 25:  # 确保有足够的数据点进行VAR分析
                    continue

                # try:
                # model = sm.tsa.VAR(merged)
                # results = model.fit(maxlags=5, ic='aic')
                # print(f'选定的最优滞后期: {results.k_ar}')
                # irf = results.irf(10).irfs#[:, :, -1]  # 取对下周收益率的IRF
                # irf_sum = np.sum(irf, axis=0)
                # irf_results[industry] = np.sum(irf_sum)  # 求和得到整体影响
                # except Exception as e:
                #     print(f"Error in VAR model for {industry}: {e}")
                #     continue

            # 根据IRF的结果排序行业
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)

            # 获取下一周期实际涨跌幅的排序
            sorted_actual_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # 计算实际涨跌幅排序
            sorted_real_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # 计算秩相关
            if sorted_industries and sorted_real_returns:
                rank_corr, _ = spearmanr(sorted_industries, sorted_real_returns)
                correlations.append(rank_corr)

        # 返回所有周期的秩相关平均值
        return np.mean(correlations)

    def analyze_industries(self):
        df_daily_return = self.data.df_daily_return
        df_daily_price = self.data.df_daily_price
        df_north_inflow = self.data.df_north_inflow
        df_margin = self.data.df_margin
        df_etf = self.data.df_etf
        df_fund_estimate = self.data.df_fund_estimate


        # 日期索引转换为周期
        period_starts = pd.date_range(start=df_daily_price.index.min(),
                                      end=df_daily_price.index.max() - timedelta(days=60), freq='5D')

        correlations = {}

        for history_start_date in period_starts:
            history_end_date = history_start_date + timedelta(days=60)  # 使用60天的数据
            next_week_start_date = history_end_date + timedelta(days=1)
            next_week_end_date = next_week_start_date + timedelta(days=4)

            # 计算到最后一周 循环终止
            if next_week_end_date > df_daily_price.index.max():
                break

            # 获取下一周期（“周”）的实际涨跌幅
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()

            irf_results = {}
            # 赋予不同资金流不同的权重
            fund_flows_with_weights = {
                'north_inflow': 1.0,  # 举例，北向资金权重为1.0
                # 'margin': 0.8,  # 融资融券资金权重为0.8
                # 'etf': 0.5,  # ETF资金权重为0.5
                # 'fund_estimate': 0.6  # 基金估算资金权重为0.6
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
                        weighted_effect = effect * weight  # 计算加权效应
                        weighted_effects.append(weighted_effect)

                if weighted_effects:
                    # 使用加权效应的和除以权重的和，得到加权平均效应
                    irf_results[industry] = sum(weighted_effects) / sum(fund_flows_with_weights.values())

            # 根据IRF的结果排序行业
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)
            # 计算实际涨跌幅排序
            sorted_real_returns = next_week_return.sort_values(ascending=False).index.tolist()

            # 计算秩相关
            if sorted_industries and sorted_real_returns:
                rank_corr, _ = spearmanr(sorted_industries, sorted_real_returns)
                correlations[history_end_date] = rank_corr
        return correlations

    def calculate_irf_for_each_fund_flow(self, industry, industry_inflow_history, history_return, fund_flow_name):
        """
        对给定的行业和资金流历史数据计算IRF。
        """
        if industry_inflow_history.empty or industry not in history_return:
            return np.nan

        merged = pd.concat([history_return[industry], industry_inflow_history], axis=1).dropna()

        if len(merged) < 25:  # 确保有足够的数据点进行VAR分析
            return np.nan

        try:
            model = sm.tsa.VAR(merged)
            results = model.fit(maxlags=5, ic='aic')
            irf = results.irf(10).irfs
            # 获取冲击变量（资金流入）第1列对响应变量（行业收益率）第0列的累积影响
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
