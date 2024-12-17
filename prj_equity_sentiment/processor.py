# coding=gbk
# Time Created: 2023/6/28 16:23
# Author  : Lucid
# FileName: processor.py
# Software: PyCharm
import datetime
import os
import numpy as np
import pandas as pd
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text
import matplotlib

matplotlib.use('TkAgg')
from statsmodels.tsa.api import VAR


def calculate_recent_weight(df: pd.DataFrame, window_short=20, window_long=60):
    df = df.sort_index()
    result_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        recent_20_days = df[column].rolling(window=window_short).sum()
        # 选择正值来计算 recent_60_days，避免分母接近0或为负值的情况，波幅剧烈且难以解释
        # recent_60_days = df[column].rolling(window=window_long).sum()
        # positive_values = df[column].where(df[column] > 0, 0)
        positive_values = df[column].abs()
        recent_60_days = positive_values.rolling(window=window_long).sum()

        weight = recent_20_days / recent_60_days
        result_df[column] = weight
    return result_df.sort_index()


class Processor(PgDbUpdaterBase):
    """
    用来生成各种指标，并上传至数据库
    每个子函数都返回两个，总量（全A）和行业
    """

    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.money_flow = MoneyFlow(base_config)
        self.price_volume = PriceVolume(base_config)
        self.analyst = Analyst(base_config)
        self.market_divergence = MarketDivergence(base_config)
        self.generate_indicators()
        self.calculate_predictions()

    def generate_indicators(self):
        # 生成所有指标
        self.money_flow.calculate()
        self.price_volume.calculate()
        # self.analyst.calculate()
        self.market_divergence.calculate()
        self.calculate_industry_congestion()

    def upload_indicators(self):
        self.upload_indicator(self.money_flow.results)
        self.upload_indicator(self.price_volume.results)
        ## self.upload_indicator(self.analyst.results)
        self.upload_indicator(self.market_divergence.results)
        ## self.upload_indicator(self.industry_congestion.results)

    def upload_indicator(self, results: dict):
        for table_name, df in results.items():
            print(f'uploading {table_name} to database')
            df.dropna().to_sql(name=table_name, con=self.alch_engine, schema='processed_data', if_exists='replace',
                               index=True)

    @property
    def wide_results(self):
        return (self.money_flow.results_wide, self.price_volume.results_wide, self.market_divergence.results_wide)

    def calculate_industry_congestion(self):
        # 将上述指标进行汇总，得到复合拥挤度
        # 需要先对各指标求irf，根据irf结果指定不同指标的权重，不着急做
        pass

    def calculate_predictions(self):
        # 市场行情
        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='收盘价' and product_type='index'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(
            columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        daily_return_df = price_ts.pct_change().dropna(how='all')

        self.predictions_list = []
        for data_dict in self.wide_results:
            dict_predictions = {}
            for indicators_group, df in data_dict.items():
                # 日期对齐 保留交集部分的行
                index_intersection = daily_return_df.index.intersection(df.index)
                daily_return_df = daily_return_df.loc[index_intersection]
                df = df.loc[index_intersection].rename(columns={'总额': '万德全A'})

                if len(df.columns) in (30, 31):
                    # 有行业数据
                    predictions_5_days = {}
                    predictions_25_days = {}
                    for _, industry in enumerate(df.columns):
                        if df[industry].eq(0).all():
                            continue
                        merged = pd.merge(daily_return_df[industry], df[industry],
                                          left_index=True, right_index=True,
                                          suffixes=('_return', '_indicator')).dropna()
                        # 创建VAR模型
                        # 剔除前面不连续的日期和最近一周的影响
                        merged_trimmed = merged[5:-5]
                        model = VAR(merged_trimmed)
                        results = model.fit(maxlags=5)

                        # # 提取单位冲击响应函数
                        # irf = results.irf(30).irfs[1:, 0, 1]
                        # # 计算指标每日变化（作为冲击），除以100使得冲击为%单位的变化是不对的，冲击的单位应该和基准变量一致。
                        # shock = df[industry].diff().dropna()
                        # # 使用 IRF 和指标的过去值来预测未来的股指走势
                        # predictions_5_days[industry] = np.sum([np.multiply(irf[i:i+5], shock[-1-i]) for i in range(5)])
                        # predictions_25_days[industry] = np.sum([np.sum(np.multiply(irf[i:i+25], shock[-1-i])) for i in range(25)])

                        # 直接用VAR预测
                        predictions_5_days[industry] = results.forecast(merged.values, 5)[:, 0].sum()
                        predictions_25_days[industry] = results.forecast(merged.values, 25)[:, 0].sum()

                    predictions_sum = pd.concat(
                        [pd.DataFrame(predictions_5_days, index=['predictions_5_days']).sum(axis=0),
                         pd.DataFrame(predictions_25_days, index=['predictions_25_days']).sum(axis=0)], axis=1).rename(
                        columns={0: 'predictions_5_days', 1: 'predictions_25_days'})
                elif len(df.columns) <= 5:
                    # 无行业数据
                    predictions_5_days = {}
                    predictions_25_days = {}
                    for _, indicator in enumerate(df.columns):
                        merged = pd.merge(daily_return_df['万德全A'], df[indicator],
                                          left_index=True, right_index=True).dropna()
                        # 创建VAR模型
                        # 剔除前面不连续的日期和最近一周的影响
                        merged_trimmed = merged[5:-5]
                        model = VAR(merged_trimmed)
                        results = model.fit(maxlags=5)

                        # 直接用VAR预测
                        predictions_5_days[indicator] = results.forecast(merged.values, 5)[:, 0].sum()
                        predictions_25_days[indicator] = results.forecast(merged.values, 25)[:, 0].sum()

                        # # 提取单位冲击响应函数
                        # irf = results.irf(30).irfs[1:, 0, 1]
                        # # 计算指标每日变化（作为冲击）
                        # shock = df[indicator].diff().dropna()
                        #
                        # # 使用 IRF 和指标的过去值来预测未来的股指走势
                        # predictions_5_days[indicator] = np.sum([np.multiply(irf[i:i+5], shock[-1-i]) for i in range(5)])
                        # predictions_25_days[indicator] = np.sum([np.sum(np.multiply(irf[i:i+25], shock[-1-i])) for i in range(25)])

                    predictions_sum = pd.concat([pd.DataFrame(predictions_5_days, index=['placeholder']).sum(axis=0),
                                                 pd.DataFrame(predictions_25_days, index=['placeholder']).sum(axis=0)],
                                                axis=1).rename(
                        columns={0: 'predictions_5_days', 1: 'predictions_25_days'})
                else:
                    raise Exception(f'indicators df length={len(df.columns)} not supported!')
                dict_predictions[indicators_group] = predictions_sum
            self.predictions_list.append(dict_predictions)
        df_5_days_industry, df_25_days_industry, df_5_days_total, df_25_days_total = self.combine_predictions()
        self.convert_upload_predictions(df_5_days_industry, '5_days_industry_forecast')
        self.convert_upload_predictions(df_5_days_total, '5_days_total_forecast')

    def convert_upload_predictions(self, df_wide, table_name):
        # df_long = df_wide.reset_index().rename(columns={'index': 'industry'}).melt(id_vars=['industry'],
        #                                                                            var_name='indicator',
        #                                                                            value_name='value')
        print(f'uploading {table_name} to database')
        df_wide.to_sql(name=table_name, con=self.alch_engine, schema='processed_data', if_exists='replace',
                       index=True)

    def combine_predictions(self):
        def calculate_combined_mean(df):
            if len(df.index) >= 30:
                # Calculate mean of the first 3 columns (ignoring zero values)
                mean_first_3 = df.iloc[:, 0:3].mean(axis=1)
                # Calculate mean of the last 5 columns (ignoring zero values)
                mean_last_5 = df.iloc[:, -5:].mean(axis=1)
                # Calculate combined mean
                mean_combined = (mean_first_3 + mean_last_5) / 2
                # Add the new column to the DataFrame
                df['加权均值'] = mean_combined
                return df.fillna(0)
            else:
                df['加权均值'] = df.mean(axis=1)
                df = df.rename(columns={'rotation_strength_daily': '日度轮动强度',
                                        'rotation_strength_daily_ma20': '日度轮动强度MA20',
                                        'rotation_strength_5d': '五日轮动强度',
                                        'rotation_strength_5d_ma20': '五日轮动强度MA20'})
                return df.T

        # Initialize empty dataframes for first two tables
        df_5_days_industry = pd.DataFrame()
        df_25_days_industry = pd.DataFrame()

        # Initialize empty dataframes for last two tables
        df_5_days_total = pd.DataFrame()
        df_25_days_total = pd.DataFrame()

        for data in self.predictions_list:
            for key, df in data.items():
                if len(df.index) >= 30:  # industry related data
                    df_5_days_industry[key] = df['predictions_5_days']
                    df_25_days_industry[key] = df['predictions_25_days']
                else:  # total related data
                    for indicator in df.index.tolist():
                        df_5_days_total.loc['万德全A', indicator] = df.loc[indicator, 'predictions_5_days']
                        df_25_days_total.loc['万德全A', indicator] = df.loc[indicator, 'predictions_25_days']

        return calculate_combined_mean(df_5_days_industry), calculate_combined_mean(
            df_25_days_industry), calculate_combined_mean(df_5_days_total), calculate_combined_mean(df_25_days_total)


class MoneyFlow(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_money_flow()

    def calculate(self):
        # 将计算结果保存在self.results中
        print('calculating for MoneyFlow')
        self.calc_finance_net_buy()
        self.calc_north_inflow()
        self.calc_big_order_inflow()
        self.calc_order_inflows()

    @property
    def results(self):
        return {
            'finance_net_buy_percentile_industry': self.fnb_percentile_industry,
            'north_percentile_industry': self.north_percentile_industry,
            'big_order_inflow_percentile': self.combined_order_inflows_percentile,
        }

    @property
    def results_wide(self):
        return {
            'finance_net_buy_percentile_industry': self.fnb_percentile_industry_wide,
            'north_percentile_industry': self.north_percentile_industry_wide,
            'big_order_inflow_percentile': self.big_order_inflow_percentile_wide,
        }

    def read_data_money_flow(self):
        # 还要构建行业的
        # Question 152 各行业两融净买入额(时间序列)
        finance_net_buy_industry_query = text(
            """
            SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '两融净买入额'
            GROUP BY "source"."date", "source"."中信一级行业"
            ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC
            """
        )
        finance_net_buy_industry = self.alch_conn.execute(finance_net_buy_industry_query)
        self.finance_net_buy_industry_df = pd.DataFrame(finance_net_buy_industry,
                                                        columns=['date', '中信行业', '两融净买入额'])

        # Question 156 各行业北向净买入额(时间序列)
        north_inflow_industry_query = text(
            """
            SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "中信一级行业", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '净买入'
            GROUP BY "source"."date", "source"."中信一级行业"
            ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC
            """
        )
        north_inflow_industry = self.alch_conn.execute(north_inflow_industry_query)
        self.north_inflow_industry_df = pd.DataFrame(north_inflow_industry,
                                                     columns=['date', '中信行业', '北向净买入额'])

        # Question 244 主力净流入（全景）
        big_order_inflow_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", SUM("public"."markets_daily_long"."value") AS "sum"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '主力净流入额')
            GROUP BY "public"."markets_daily_long"."date", "Product Static Info"."chinese_name"
            ORDER BY "public"."markets_daily_long"."date" ASC, "Product Static Info"."chinese_name" ASC
            """
        )
        big_order_inflow = self.alch_conn.execute(big_order_inflow_query)
        self.big_order_inflow_df = pd.DataFrame(big_order_inflow,
                                                columns=['date', '中信行业', '主力净流入额'])

    def calc_finance_net_buy(self):
        # 从长格式转换为date-industry的买入额或流入额
        wide_df = self.finance_net_buy_industry_df.pivot(index='date', columns='中信行业', values='两融净买入额')
        wide_df['总额'] = wide_df.sum(axis=1)

        # 最近20个交易日融资买入额占最近60个交易日融资买入额比重作为融资买入情绪
        weights_industry = calculate_recent_weight(df=wide_df).dropna(subset=['交通运输'])

        # 将融资买入情绪进行滚动一年分位处理
        self.fnb_percentile_industry_wide = weights_industry.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.fnb_percentile_industry = self.fnb_percentile_industry_wide.reset_index().melt(id_vars=['date'],
                                                                                            var_name='industry',
                                                                                            value_name='finance_net_buy')

    def calc_north_inflow(self):
        # 注：自建
        # 最近20个交易日北向净流入额占最近60个交易日北向净流入额比重作为北向买入情绪
        wide_df = self.north_inflow_industry_df.pivot(index='date', columns='中信行业', values='北向净买入额')
        wide_df['总额'] = wide_df.sum(axis=1)

        weights_industry = calculate_recent_weight(df=wide_df).dropna(subset=['交通运输'])

        # 将北向买入情绪进行滚动一年分位处理
        self.north_percentile_industry_wide = weights_industry.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.north_percentile_industry = self.north_percentile_industry_wide.reset_index().melt(id_vars=['date'],
                                                                                                var_name='industry',
                                                                                                value_name='north_inflow')

    def calc_big_order_inflow(self):
        # 最近一周主力净流入额作为大单情绪
        wide_df = self.big_order_inflow_df.pivot(index='date', columns='中信行业', values='主力净流入额')
        wide_df_ma = wide_df.rolling(10).mean()
        # 对大单情绪进行滚动一年分位处理
        self.big_order_inflow_percentile_wide = wide_df_ma.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.big_order_inflow_percentile = self.big_order_inflow_percentile_wide.reset_index().melt(id_vars=['date'],
                                                                                                    var_name='industry',
                                                                                                    value_name='big_order_inflow_percentile')

    def calc_order_inflows(self):
        # 定义需要查询的字段
        target_fields = [
            '流入额_机构', '流入额_大户', '流入额_中户', '流入额_散户',
            '净买入额_机构', '净买入额_大户', '净买入额_中户', '净买入额_散户',
            '净主动买入额_机构', '净主动买入额_大户', '净主动买入额_中户', '净主动买入额_散户'
        ]

        # 构建 SQL 查询
        fields_placeholder = ', '.join([f"'{field}'" for field in target_fields])

        order_inflow_query = text(
            f"""
            SELECT
                mdl.date AS date,
                psi.chinese_name AS industry_name,
                mdl.field AS field,
                SUM(mdl.value) AS total_value
            FROM
                markets_daily_long AS mdl
            LEFT JOIN
                product_static_info AS psi
                ON mdl.product_name = psi.code
            WHERE
                psi.product_type = 'index'
                AND mdl.field IN ({fields_placeholder})
            GROUP BY
                mdl.date, psi.chinese_name, mdl.field
            ORDER BY
                mdl.date ASC, psi.chinese_name ASC, mdl.field ASC
            """
        )

        # 执行查询
        result = self.alch_conn.execute(order_inflow_query)
        self.industry_order_inflows_df = pd.DataFrame(result.fetchall(),
                                                      columns=['date', 'industry', 'field', 'total_value'])

        # 处理数据：透视表，以便分别处理不同的字段（order_type）
        # 我们保留 'field' 作为一个维度
        wide_df = self.industry_order_inflows_df.pivot_table(
            index=['date', 'field'],
            columns='industry',
            values='total_value',
            aggfunc='sum'
        )

        # 初始化一个列表存储滚动分位数结果
        rolling_percentile_records = []

        # 设置滚动窗口大小
        rolling_window = 250

        # 遍历每个目标字段
        for field in target_fields:
            # 检查当前field是否存在于wide_df的索引中
            if field not in wide_df.index.get_level_values('field'):
                continue  # 如果不存在，跳过当前field

            # 提取当前field的数据，并按日期排序
            try:
                field_df = wide_df.xs(field, level='field').reset_index().sort_values('date')
            except KeyError:
                # 如果字段不存在，跳过
                continue

            # 遍历每个行业，计算滚动分位数
            for industry in field_df.columns:
                if industry == 'date':
                    continue  # 跳过日期列

                # 提取该行业的时间序列，确保按日期排序
                ts = field_df[['date', industry]].dropna().sort_values('date').set_index('date')[industry]

                if ts.empty:
                    continue

                # 计算滚动平均（例如 10 天）
                rolling_mean = ts.rolling(window=10, min_periods=1).mean()

                # 计算滚动分位数（例如 250 天）
                rolling_percentile = rolling_mean.rolling(window=rolling_window, min_periods=1).apply(
                    lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
                )

                # 创建临时DataFrame存储结果
                temp_df = pd.DataFrame({
                    'date': rolling_percentile.index,
                    'order_inflow_percentile': rolling_percentile.values,
                    'order_type': field,
                    'industry': industry
                })

                # 添加到结果列表中
                rolling_percentile_records.append(temp_df)

        # 合并所有字段和行业的滚动分位数
        if rolling_percentile_records:
            rolling_percentile_df = pd.concat(rolling_percentile_records, ignore_index=True)
        else:
            # 如果没有数据，创建一个空的DataFrame
            rolling_percentile_df = pd.DataFrame(columns=['date', 'order_inflow_percentile', 'order_type', 'industry'])

        # 保留需要的列
        self.industry_order_inflows_percentile = rolling_percentile_df[
            ['date', 'industry', 'order_inflow_percentile', 'order_type']
        ]
        self.combined_order_inflows_percentile = pd.concat([self.industry_order_inflows_percentile, self.big_order_inflow_percentile],
                                ignore_index=True)


class PriceVolume(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_price_volume()

    def calculate(self):
        # 计算各行业量价相关的各指标
        print('calculating for PriceVolume')
        # 将计算结果保存在self.results中
        self.calc_amt_proportion()
        self.calc_amt_quantile()
        self.calc_amt_prop_quantile()
        self.calc_turnover_quantile()
        self.calc_vol_shrink_rate()
        self.calc_momentum()
        self.calc_over_MA_number()
        self.calc_recent_new_highs()

    @property
    def results(self):
        return {'amt_proportion': self.amt_proportion_df,
                'amt_quantile': self.amt_quantile_df,
                'turnover_quantile': self.turnover_quantile_df,
                'shrink_rate': self.shrink_rate_df,
                'amt_prop_quantile': self.amt_prop_quantile_df,
                }

    @property
    def results_wide(self):
        return {'amt_proportion': self.amt_proportion_df_wide,
                'amt_quantile': self.amt_quantile_df_wide,
                'turnover_quantile': self.turnover_quantile_df_wide,
                'shrink_rate': self.shrink_rate_df_wide,
                'amt_prop_quantile': self.amt_prop_quantile_wide,
                }

    def read_data_price_volume(self):
        # 从长格式转换为date-stk_codes的收盘价
        # Question 230 各行业成交额(时间序列)
        amount_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '成交额')
            """
        )
        amount_industry = self.alch_conn.execute(amount_industry_query)
        amount_industry_long_df = pd.DataFrame(amount_industry, columns=['date', '指数代码', 'value', '指数名称'])
        # 从长格式转换为date-industry的成交额
        self.amount_industry_df = amount_industry_long_df.pivot(index='date', columns='指数名称', values='value')

        # Question 231 各行业换手率(时间序列)
        turnover_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '换手率')
            """
        )
        turnover_industry = self.alch_conn.execute(turnover_industry_query)
        turnover_industry_long_df = pd.DataFrame(turnover_industry, columns=['date', '指数代码', 'value', '指数名称'])
        # 从长格式转换为date-industry的换手率
        self.turnover_industry_df = turnover_industry_long_df.pivot(index='date', columns='指数名称', values='value')

        # 各行业-各个股收盘价
        pass

    def calc_amt_proportion(self):
        # 计算各行业的成交占比
        amount_industry_df = self.amount_industry_df.drop(columns=['万德全A'])
        # 计算每一行中各列数值的占比
        row_sums = amount_industry_df.sum(axis=1)  # 计算每行的总和
        self.amt_proportion_df_wide = amount_industry_df.div(row_sums, axis=0)
        self.amt_proportion_df = self.amt_proportion_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                                value_name='amt_proportion')

    def calc_amt_quantile(self):
        # 计算各行业的成交额滚动一年分位
        amount_industry_df = self.amount_industry_df
        self.amt_quantile_df_wide = amount_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.amt_quantile_df = self.amt_quantile_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                            value_name='amt_quantile')

    def calc_amt_prop_quantile(self):
        # 计算各行业的成交额占比的滚动一年分位
        self.amt_prop_quantile_wide = self.amt_proportion_df_wide.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.amt_prop_quantile_df = self.amt_prop_quantile_wide.reset_index().melt(id_vars=['date'],
                                                                                   var_name='industry',
                                                                                   value_name='amt_prop_quantile')

    def calc_turnover_quantile(self):
        # 计算各行业的换手率滚动一年分位
        turnover_industry_df = self.turnover_industry_df
        self.turnover_quantile_df_wide = turnover_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.turnover_quantile_df = self.turnover_quantile_df_wide.reset_index().melt(id_vars=['date'],
                                                                                      var_name='industry',
                                                                                      value_name='turnover_quantile')

    def calc_vol_shrink_rate(self):
        # 计算各行业的缩量率
        amount_industry_df = self.amount_industry_df
        rolling_avg = amount_industry_df.rolling(window=252).mean()
        self.shrink_rate_df_wide = amount_industry_df / rolling_avg - 1
        self.shrink_rate_df = self.shrink_rate_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                          value_name='shrink_rate')

    def calc_momentum(self):
        # 这个暂时不知道怎么做
        pass

    def calc_over_MA_number(self, window=30):
        # 需要个股数据
        # 计算各行业30日均线上数量占比分位
        pass

    def calc_recent_new_highs(self, window=60):
        # 需要个股数据
        # 计算各行业创 60 日新高数量占比分位
        pass


class MarketDivergence(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data()

    def calculate(self):
        # 计算各行业衡量市场分化程度的各指标
        print('calculating for MarketDivergence')
        self.calc_market_breadth()
        self.calc_28_amount_diverge()
        self.calc_rotation_strength()

    @property
    def results(self):
        return {
            'market_breadth_industry': self.mb_industry_combined,
            'rotation_strength': self.rotation_strength,
        }

    @property
    def results_wide(self):
        return {
            'market_breadth_industry_level': self.mb_industry_combined_wide,
            'rotation_strength': self.rotation_strength,
        }

    def read_data(self):
        # 各行业收盘价
        # Question 232 各行业收盘价(时间序列)
        close_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '收盘价')
            ORDER BY "public"."markets_daily_long"."date" ASC
            """
        )
        close_industry = self.alch_conn.execute(close_industry_query)
        close_industry_long_df = pd.DataFrame(close_industry, columns=['date', '指数代码', 'value', '指数名称'])
        # 从长格式转换为date-industry的收盘价
        self.close_industry_df = close_industry_long_df.pivot(index='date', columns='指数名称', values='value')

    def calc_market_breadth(self):
        # 可利用个股或行业计算。前者需要个股数据。
        # 行业层面
        close_industry_df = self.close_industry_df.drop(columns=['万德全A'])
        # 当日收涨比例
        # 计算每个行业的日涨跌幅
        pct_change_df = close_industry_df.pct_change()
        # 判断每天哪些行业收涨
        rising_df = pct_change_df > 0
        # 计算每天收涨的行业的比例
        mb_industry_close = rising_df.mean(axis=1).to_frame()
        mb_industry_close.columns = ['当日收涨比例']
        # mb_industry_close = close_industry_df.apply(lambda x: sum(x > x.shift(1)), axis=1)
        mb_industry_close['当日收涨比例MA20'] = mb_industry_close['当日收涨比例'].rolling(window=20).mean()

        # 回撤幅度中位数减去全A的回撤幅度
        # 计算行业回撤幅度
        industry_drawdown = (close_industry_df / close_industry_df.rolling(window=252).max()) - 1
        industry_drawdown_median = industry_drawdown.median(axis=1)
        # 计算全市场指数回撤幅度
        market_drawdown = (self.close_industry_df['万德全A'] / self.close_industry_df['万德全A'].rolling(
            window=252).max()) - 1
        # 计算市场宽度
        mb_industry_drawdown = (industry_drawdown_median - market_drawdown).to_frame()
        mb_industry_drawdown.columns = ['市场宽度-基于回撤']

        # 行业指数收盘价站上20日均线的比例
        mb_industry_above_ma = (close_industry_df > close_industry_df.rolling(window=20).mean()).mean(axis=1).to_frame()
        # 将最初的19个计算结果改为NaN
        mb_industry_above_ma.iloc[:20] = np.nan
        mb_industry_above_ma.columns = ['市场宽度-基于位置']

        self.mb_industry_combined_wide = pd.concat([mb_industry_close, mb_industry_drawdown, mb_industry_above_ma],
                                                   axis=1)
        self.mb_industry_combined = self.mb_industry_combined_wide.reset_index().melt(id_vars=['date'],
                                                                                      var_name='market_breadth_type',
                                                                                      value_name='value')

    def calc_28_amount_diverge(self):
        # 国盛策略：二八交易分化
        # 需要个股数据
        pass

    def calc_rotation_strength(self):
        # 计算全市场（不同行业之间的）轮动强度
        close_industry_df = self.close_industry_df.drop(columns=['万德全A'])
        daily_returns = close_industry_df.pct_change()  # 计算每日涨跌幅

        rank_changes = daily_returns.rank(axis=1).diff().abs()  # 计算涨跌百分比排名的变化的绝对值
        self.rotation_strength = rank_changes.sum(axis=1).to_frame()  # 求和得到轮动强度
        self.rotation_strength.columns = ['rotation_strength_daily']
        self.rotation_strength['rotation_strength_daily_ma20'] = self.rotation_strength.rolling(window=20).mean()

        # 使用过去5天的涨跌幅进行排序
        rank_changes_5d = daily_returns.rolling(window=5).sum().rank(axis=1).diff().abs()
        self.rotation_strength['rotation_strength_5d'] = rank_changes_5d.sum(axis=1).to_frame()  # 求和得到轮动强度
        self.rotation_strength['rotation_strength_5d_ma20'] = self.rotation_strength['rotation_strength_5d'].rolling(
            window=20).mean()


class Analyst(PgDbUpdaterBase):
    # 无tushare数据权限，暂时不能做
    # TODO：可以看看wind
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        # 初始化MoneyFlow类需要的其他属性

    def calculate(self):
        # 计算各行业的增持或买入评级研报数量分位
        # 将计算结果保存在self.results中
        pass
