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


def calculate_recent_weight(df: pd.DataFrame, window_short=20, window_long=60):
    df = df.sort_index()
    result_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        recent_20_days = df[column].rolling(window=window_short).sum()
        recent_60_days = df[column].rolling(window=window_long).sum()
        weight = recent_20_days / recent_60_days
        result_df[column] = weight
    return result_df


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
        self.upload_indicators()

    def generate_indicators(self):
        # 生成所有指标
        # self.money_flow.calculate()
        self.price_volume.calculate()
        self.analyst.calculate()
        self.market_divergence.calculate()
        self.calculate_industry_congestion()

    def upload_indicators(self):
        self.upload_indicator(self.money_flow.results)
        self.upload_indicator(self.price_volume.results)
        # self.upload_indicator(self.analyst.results)
        self.upload_indicator(self.market_divergence.results)
        # self.upload_indicator(self.industry_congestion.results)
        pass

    def upload_indicator(self, results: dict):
        for table_name, df in results.items():
            print(f'uploading {table_name} to database')
            df.dropna().to_sql(name=table_name, con=self.alch_engine, schema='processed_data', if_exists='replace')

    def calculate_industry_congestion(self):
        # 将上述指标进行汇总，得到复合拥挤度
        # 需要先对各指标求irf，根据irf结果指定不同指标的权重，不着急做
        pass


class MoneyFlow(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_money_flow()

    def calculate(self):
        # 将计算结果保存在self.results中
        self.calc_finance_net_buy()
        self.calc_north_inflow()

    @property
    def results(self):
        return {
            'finance_net_buy_percentile_industry': self.fnb_percentile_industry,
            'finance_net_buy_percentile_total': self.fnb_percentile_total,
            'north_percentile_industry': self.north_percentile_industry,
            'north_percentile_total': self.north_percentile_total,
        }

    def read_data_money_flow(self):
        # Question 153 两融净买入总额(时间序列)
        finance_net_buy_query = text(
            """
            SELECT "source"."date" AS "date", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '两融净买入额'
            GROUP BY "source"."date"
            ORDER BY "source"."date" ASC
            """
        )
        finance_net_buy = self.alch_conn.execute(finance_net_buy_query)
        self.finance_net_buy_df = pd.DataFrame(finance_net_buy, columns=['date', '两融净买入总额']).set_index('date')

        # Question 157 北向资金净买入总额(时间序列)
        north_inflow_query = text(
            """
            SELECT "source"."date" AS "date", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "中信一级行业", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '净买入'
            GROUP BY "source"."date"
            ORDER BY "source"."date" ASC
            """
        )
        north_inflow = self.alch_conn.execute(north_inflow_query)
        self.north_inflow_df = pd.DataFrame(north_inflow, columns=['date', '两融净买入总额']).set_index('date')

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

    def calc_finance_net_buy(self):
        # 从长格式转换为date-industry的买入额或流入额
        wide_df = self.finance_net_buy_industry_df.pivot(index='date', columns='中信行业', values='两融净买入额')

        # 最近20个交易日融资买入额占最近60个交易日融资买入额比重作为融资买入情绪
        weights_industry = calculate_recent_weight(df=wide_df)
        weights_total = calculate_recent_weight(df=self.finance_net_buy_df)

        # 将融资买入情绪进行滚动一年分位处理
        self.fnb_percentile_industry = weights_industry.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.fnb_percentile_total = weights_total.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True)[0])

    def calc_north_inflow(self):
        # 注：自建
        # 最近20个交易日北向净流入额占最近60个交易日北向净流入额比重作为北向买入情绪
        wide_df = self.north_inflow_industry_df.pivot(index='date', columns='中信行业', values='北向净买入额')
        weights_industry = calculate_recent_weight(df=wide_df)
        weights_total = calculate_recent_weight(df=self.north_inflow_df)

        # 将北向买入情绪进行滚动一年分位处理
        self.north_percentile_industry = weights_industry.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.north_percentile_total = weights_total.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True)[0])


class PriceVolume(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_price_volume()

    def calculate(self):
        # 计算各行业量价相关的各指标
        # 将计算结果保存在self.results中
        self.calc_amt_proportion()
        self.calc_amt_quantile()
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
                'shrink_rate': self.shrink_rate_df
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
        amt_proportion_df = amount_industry_df.div(row_sums, axis=0)
        self.amt_proportion_df = amt_proportion_df.reset_index().melt(id_vars=['date'], var_name='industry', value_name='amt_proportion')

    def calc_amt_quantile(self):
        # 计算各行业的成交额滚动一年分位
        amount_industry_df = self.amount_industry_df.drop(columns=['万德全A'])
        amt_quantile_df = amount_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.amt_quantile_df = amt_quantile_df.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                      value_name='amt_quantile')

    def calc_turnover_quantile(self):
        # 计算各行业的换手率滚动一年分位
        turnover_industry_df = self.turnover_industry_df.drop(columns=['万德全A'])
        turnover_quantile_df = turnover_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.turnover_quantile_df = turnover_quantile_df.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                      value_name='turnover_quantile')

    def calc_vol_shrink_rate(self):
        # 计算各行业的缩量率
        amount_industry_df = self.amount_industry_df.drop(columns=['万德全A'])
        rolling_avg = amount_industry_df.rolling(window=252).mean()
        shrink_rate_df = amount_industry_df / rolling_avg - 1
        self.shrink_rate_df = shrink_rate_df.reset_index().melt(id_vars=['date'], var_name='industry',
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


class MarketDivergence(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data()

    def calculate(self):
        # 计算各行业衡量市场分化程度的各指标
        self.calc_market_breadth()
        self.calc_28_amount_diverge()
        self.calc_rotation_strength()

    @property
    def results(self):
        return {
            'market_breadth_industry_close': self.mb_industry_close,
            'market_breadth_industry_drawdown': self.mb_industry_drawdown,
            'market_breadth_industry_above_ma': self.mb_industry_above_ma,
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
        # 当日收涨比例,
        self.mb_industry_close = close_industry_df.apply(lambda x: sum(x > x.shift(1)), axis=1)

        # 回撤幅度中位数减去全A的回撤幅度
        # 计算行业回撤幅度
        industry_drawdown = (close_industry_df / close_industry_df.rolling(window=252).max()) - 1
        industry_drawdown_median = industry_drawdown.median(axis=1)
        # 计算全市场指数回撤幅度
        market_drawdown = (self.close_industry_df['万德全A'] / self.close_industry_df['万德全A'].rolling(
            window=252).max()) - 1
        # 计算市场宽度
        self.mb_industry_drawdown = industry_drawdown_median - market_drawdown

        # 行业指数收盘价站上20日均线的比例
        above_ma_ratio = (close_industry_df > close_industry_df.rolling(window=20).mean()).mean(axis=1)
        # 计算市场宽度
        self.mb_industry_above_ma = above_ma_ratio

    def calc_28_amount_diverge(self):
        # 国盛策略：二八交易分化
        # 需要个股数据
        pass

    def calc_rotation_strength(self):
        # 计算全市场（不同行业之间的）轮动强度
        close_industry_df = self.close_industry_df.drop(columns=['万德全A'])
        daily_returns = close_industry_df.pct_change()  # 计算每日涨跌幅
        rank_changes = daily_returns.rank(axis=1).diff().abs()  # 计算涨跌百分比排名的变化的绝对值
        self.rotation_strength = rank_changes.sum(axis=1)  # 求和得到轮动强度
