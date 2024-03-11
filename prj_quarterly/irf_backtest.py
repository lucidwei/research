# coding=gbk
# Time Created: 2024/3/8 9:10
# Author  : Lucid

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from �����ܲ�λ����demo import CalcFundPosition
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
        self.calc_weekly_irfs()
        self.predict_next_week_return()
        self.calc_backtest_nav()


    def calc_weekly_irfs(self):
        """
        ÿ��ĩ���ù�ȥ60������ݲ���IRF��ȷ��������δ�����ݡ�
        �ò�ͬ�ʽ��������������ҵ�ǵ�����ETF�ͻ�������Ȩ�ص�һЩ��ǰ�����ݵ��٣����ߣ��õ���ҵ�ǵ�������
        ����һ��ʵ����ҵ�ǵ���˳��������ء�
        """
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

            # ��ȡ��ʷʵ���ǵ���
            history_return = df_daily_return.loc[history_start_date:history_end_date]
            history_return_sneak = df_daily_return.loc[history_start_date:next_week_end_date]

            # ���費ͬ�ʽ�����ͬ��Ȩ��
            fund_flows_with_weights = {
                # 'north_inflow': 1.0,  # �����������ʽ�Ȩ��Ϊ1.0
                # 'margin': 0.8,  # ������ȯ�ʽ�Ȩ��Ϊ0.8
                # 'etf': 0.5,  # ETF�ʽ�Ȩ��Ϊ0.5
                'fund_estimate': 0.6  # ��������ʽ�Ȩ��Ϊ0.6
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
                        weighted_effect = effect * weight  # �����ȨЧӦ
                        weighted_effects.append(weighted_effect)

                if weighted_effects:
                    # ʹ�ü�ȨЧӦ�ĺͳ���Ȩ�صĺͣ��õ���Ȩƽ��ЧӦ
                    irf_results[industry] = sum(weighted_effects) / sum(fund_flows_with_weights.values())

            # ����IRF�Ľ��������ҵ
            sorted_industries = sorted(irf_results, key=irf_results.get, reverse=True)
            # ����ʵ���ǵ�������
            next_week_return = df_daily_return.loc[next_week_start_date:next_week_end_date].sum()
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

        merged = pd.concat([history_return[industry].rename('return'), industry_inflow_history], axis=1).dropna()

        if len(merged) < 10:  # ȷ�����㹻�����ݵ����VAR����
            return np.nan

        try:
            model = sm.tsa.VAR(merged)
            results = model.fit(maxlags=5, ic='aic')

            # # ��һ�ַ�ʽ��irf�����ʽ�����
            # irf = results.irf(10).irfs
            # # ��ȡ����������ʽ����룩����Ӧ��������ҵ�����ʣ����ۻ�Ӱ��
            # cumulative_response = np.sum(irf[:5, merged.columns.get_loc(f'return'),
            #                                  merged.columns.get_loc(industry)])
            # # ��ȡ���һ�ڵ��ʽ���������
            # latest_inflow = industry_inflow_history.iloc[-5:].sum()
            # # latest_inflow = industry_inflow_history.iloc[-10:-5].sum()
            # predicted_impact = latest_inflow * cumulative_response

            # �ڶ��ַ�ʽ��irf�Դ���forecast
            # ȷ����ҪΪforecast�����ṩ�Ĺ۲�ֵ��������ģ�͵�����ͺ���
            maxlags = results.k_ar
            print(f'maxlags:{maxlags}')

            # ��ȡ��󼸸��۲�ֵ��������Ӧ������ͺ�����ƥ��
            last_observations = merged.values[-maxlags-1:]
            # last_observations = merged.values[-9-maxlags:-3]

            # ʹ��forecast��������Ԥ�⣬�����������Ԥ��δ��5��
            forecast_steps = 5
            forecast_result = results.forecast(y=last_observations, steps=forecast_steps)

            # Ԥ��Ľ����һ�����飬����ÿһ�ж�Ӧһ��δ��ʱ�ڵ�Ԥ��ֵ
            # ����������Ȥ����Ӧ����������ҵ�����ʣ�λ�ڵ�0��
            predicted_return = forecast_result[:, merged.columns.get_loc(f'return')]

            # Ԥ����������Ҫ��һ��������ƥ������ҵ���߼�������
            # ���磬������ֻ����Ԥ���ڼ���ۻ������ʱ仯
            predicted_impact = predicted_return.sum()

            return predicted_impact
        except Exception as e:
            print(f"Error in VAR model for {fund_flow_name} {industry}: {e}")
            return 0

    def predict_next_week_return(self):
        """
        �������ۻ���������Ϊ������������ܣ�ȡ5�ա�10�ա�30�յ���Ӧ�ۻ�������������
        """
        pass

    def calc_backtest_nav(self):
        pass


if __name__ == "__main__":
    base_config = BaseConfig('quarterly')
    data = DataProcessor(base_config)
    evaluator = Evaluator(data)
