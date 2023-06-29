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
        self.money_flow.calculate()
        self.price_volume.calculate()
        self.analyst.calculate()
        self.market_divergence.calculate()
        self.calculate_industry_congestion()

    def upload_indicators(self):
        self.upload_indicator(self.money_flow.results)
        self.upload_indicator(self.price_volume.results)
        self.upload_indicator(self.analyst.results)
        self.upload_indicator(self.market_divergence.results)
        # self.upload_indicator(self.industry_congestion.results)
        pass

    def upload_indicator(self):
        pass

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
        pass

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
        finance_net_buy_df = pd.DataFrame(finance_net_buy, columns=['date', '两融净买入总额'])

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
        north_inflow_df = pd.DataFrame(north_inflow, columns=['date', '两融净买入总额'])

        # TODO: 还要构建行业的，从长格式转换为date-industry的买入额或流入额
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
        finance_net_buy_industry = self.alch_conn.execute(north_inflow_query)
        finance_net_buy_industry_df = pd.DataFrame(finance_net_buy_industry, columns=['date', '中信行业', '两融净买入额'])

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
        north_inflow_industry = self.alch_conn.execute(north_inflow_query)
        north_inflow_industry_df = pd.DataFrame(north_inflow_industry, columns=['date', '中信行业', '北向净买入额'])

    def calc_finance_net_buy(self):
        # 最近20个交易日融资买入额占最近60个交易日融资买入额比重作为融资买入情绪

        # 将融资买入情绪进行滚动一年分位处理
        pass

    def calc_north_inflow(self):
        # 注：自建
        # 最近20个交易日北向净流入额占最近60个交易日北向净流入额比重作为北向买入情绪

        # 将北向买入情绪进行滚动一年分位处理
        pass


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

    def read_data_price_volume(self):
        # 从长格式转换为date-stk_codes的收盘价
        pass

    def calc_amt_proportion(self):
        # 计算各行业的成交占比
        pass

    def calc_amt_quantile(self):
        # 计算各行业的成交额滚动一年分位
        pass

    def calc_turnover_quantile(self):
        # 计算各行业的换手率滚动一年分位
        pass

    def calc_vol_shrink_rate(self):
        # 计算各行业的缩量率
        pass

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

    def calculate(self):
        # 计算各行业衡量市场分化程度的各指标
        # 将计算结果保存在self.results中
        pass

    def calc_market_breadth(self):
        # 可利用个股或行业计算。前者需要个股数据。
        pass

    def calc_28_amount_diverge(self):
        # 国盛策略：二八交易分化
        # 需要个股数据
        pass

    def calc_rotation_strength(self):
        # 计算全市场（不同行业之间的）轮动强度
        pass
