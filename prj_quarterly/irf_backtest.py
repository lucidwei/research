# coding=gbk
# Time Created: 2024/3/8 9:10
# Author  : Lucid

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from 基金总仓位测算demo import CalcFundPosition
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


class DataProcessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.base_config = base_config
        self.load_from_pgdb()
        self.load_from_excel()
        self.align_dates_industries()

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
        self.calc_weekly_irfs()
        self.predict_next_week_return()
        self.calc_backtest_nav()


    def calc_weekly_irfs(self):
        """
        每周末利用过去60天的数据测算IRF，确保不利用未来数据。
        用不同资金流估算出各个行业涨跌幅，ETF和基金估算的权重低一些（前者数据点少，后者）得到行业涨跌幅排序
        与下一周实际行业涨跌幅顺序做秩相关。
        """
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

            # 获取历史实际涨跌幅
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            history_return_sneak = df_daily_return.loc[history_start_date:next_week_end_date]

            # 赋予不同资金流不同的权重
            fund_flows_with_weights = {
                # 'north_inflow': 1.0,  # 举例，北向资金权重为1.0
                # 'margin': 0.8,  # 融资融券资金权重为0.8
                # 'etf': 0.5,  # ETF资金权重为0.5
                'fund_estimate': 0.6  # 基金估算资金权重为0.6
            }

            fund_flows_dfs = {
                'north_inflow': df_north_inflow,
                'margin': df_margin,
                'etf': df_etf,
                'fund_estimate': df_fund_estimate
            }
            irf_results = {}
            for industry in df_daily_return.columns:
                weighted_effects = []

                for fund_flow_name, weight in fund_flows_with_weights.items():
                    fund_flow_df = fund_flows_dfs[fund_flow_name]
                    industry_inflow_history = fund_flow_df[industry].loc[history_start_date:history_end_date]
                    effect = self.calculate_irf_for_each_fund_flow(industry, industry_inflow_history, history_return, fund_flow_name)
                    # industry_inflow_sneak = fund_flow_df[industry].loc[history_start_date:next_week_end_date]
                    # effect = self.calculate_irf_for_each_fund_flow(industry, industry_inflow_sneak, history_return_sneak, fund_flow_name)
                    if not np.isnan(effect):
                        weighted_effect = effect * weight  # 计算加权效应
                        weighted_effects.append(weighted_effect)

                if weighted_effects:
                    # 使用加权效应的和除以权重的和，得到加权平均效应
                    irf_results[industry] = sum(weighted_effects) / sum(fund_flows_with_weights.values())

            # 根据IRF的结果排序行业
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)
            # 计算实际涨跌幅排序
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()
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

        merged = pd.concat([history_return[industry].rename('return'), industry_inflow_history], axis=1).dropna()

        if len(merged) < 10:  # 确保有足够的数据点进行VAR分析
            return np.nan

        try:
            model = sm.tsa.VAR(merged)
            results = model.fit(maxlags=5, ic='aic')

            # # 第一种方式：irf乘以资金流入
            # irf = results.irf(10).irfs
            # # 获取冲击变量（资金流入）对响应变量（行业收益率）的累积影响
            # cumulative_response = np.sum(irf[:5, merged.columns.get_loc(f'return'),
            #                                  merged.columns.get_loc(industry)])
            # # 获取最近一期的资金流入数据
            # latest_inflow = industry_inflow_history.iloc[-5:].sum()
            # # latest_inflow = industry_inflow_history.iloc[-10:-5].sum()
            # predicted_impact = latest_inflow * cumulative_response

            # 第二种方式：irf自带的forecast
            # 确定需要为forecast方法提供的观测值数量，即模型的最大滞后期
            maxlags = results.k_ar
            print(f'maxlags:{maxlags}')

            # 获取最后几个观测值，其行数应与最大滞后期相匹配
            last_observations = merged.values[-maxlags-1:]
            # last_observations = merged.values[-9-maxlags:-3]

            # 使用forecast方法进行预测，这里假设我们预测未来5期
            forecast_steps = 5
            forecast_result = results.forecast(y=last_observations, steps=forecast_steps)

            # 预测的结果是一个数组，其中每一行对应一个未来时期的预测值
            # 假设您感兴趣的响应变量（如行业收益率）位于第0列
            predicted_return = forecast_result[:, merged.columns.get_loc(f'return')]

            # 预测结果可能需要进一步处理以匹配具体的业务逻辑和需求
            # 例如，您可能只关心预测期间的累积收益率变化
            predicted_impact = predicted_return.sum()

            return predicted_impact
        except Exception as e:
            print(f"Error in VAR model for {fund_flow_name} {industry}: {e}")
            return 0

    def predict_next_week_return(self):
        """
        将本周累积净流入作为冲击，计算下周（取5日、10日、30日的响应累积）具体收益率
        """
        pass

    def calc_backtest_nav(self):
        pass


if __name__ == "__main__":
    base_config = BaseConfig('quarterly')
    data = DataProcessor(base_config)
    evaluator = Evaluator(data)
