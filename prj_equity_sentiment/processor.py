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
from WindPy import w
from sqlalchemy import text


class Processor(PgDbUpdaterBase):
    """
    用来生成各种指标，并上传至数据库
    """
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)

    def calculate_money_flow(self):
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

        # TODO: 还要构建行业的

    def calculate_price_volume(self):
        pass

    def calculate_market_divergence(self):
        pass

    def calculate_industry_congestion(self):
        pass

    def upload_indicators(self):
        pass