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
    �������ɸ���ָ�꣬���ϴ������ݿ�
    ÿ���Ӻ���������������������ȫA������ҵ
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
        # ��������ָ��
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
        # ������ָ����л��ܣ��õ�����ӵ����
        # ��Ҫ�ȶԸ�ָ����irf������irf���ָ����ָͬ���Ȩ�أ����ż���
        pass


class MoneyFlow(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_money_flow()

    def calculate(self):
        # ��������������self.results��
        self.calc_finance_net_buy()
        self.calc_north_inflow()
        pass

    def read_data_money_flow(self):
        # Question 153 ���ھ������ܶ�(ʱ������)
        finance_net_buy_query = text(
            """
            SELECT "source"."date" AS "date", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '���ھ������'
            GROUP BY "source"."date"
            ORDER BY "source"."date" ASC
            """
        )
        finance_net_buy = self.alch_conn.execute(finance_net_buy_query)
        finance_net_buy_df = pd.DataFrame(finance_net_buy, columns=['date', '���ھ������ܶ�'])

        # Question 157 �����ʽ������ܶ�(ʱ������)
        north_inflow_query = text(
            """
            SELECT "source"."date" AS "date", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "����һ����ҵ", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '������'
            GROUP BY "source"."date"
            ORDER BY "source"."date" ASC
            """
        )
        north_inflow = self.alch_conn.execute(north_inflow_query)
        north_inflow_df = pd.DataFrame(north_inflow, columns=['date', '���ھ������ܶ�'])

        # TODO: ��Ҫ������ҵ�ģ��ӳ���ʽת��Ϊdate-industry�������������
        # Question 152 ����ҵ���ھ������(ʱ������)
        finance_net_buy_industry_query = text(
            """
            SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '���ھ������'
            GROUP BY "source"."date", "source"."����һ����ҵ"
            ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC
            """
        )
        finance_net_buy_industry = self.alch_conn.execute(north_inflow_query)
        finance_net_buy_industry_df = pd.DataFrame(finance_net_buy_industry, columns=['date', '������ҵ', '���ھ������'])

        # Question 156 ����ҵ���������(ʱ������)
        north_inflow_industry_query = text(
            """
            SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum"
            FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "����һ����ҵ", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '������'
            GROUP BY "source"."date", "source"."����һ����ҵ"
            ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC
            """
        )
        north_inflow_industry = self.alch_conn.execute(north_inflow_query)
        north_inflow_industry_df = pd.DataFrame(north_inflow_industry, columns=['date', '������ҵ', '���������'])

    def calc_finance_net_buy(self):
        # ���20�����������������ռ���60����������������������Ϊ������������

        # �����������������й���һ���λ����
        pass

    def calc_north_inflow(self):
        # ע���Խ�
        # ���20�������ձ��������ռ���60�������ձ�������������Ϊ������������

        # �����������������й���һ���λ����
        pass


class PriceVolume(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_price_volume()

    def calculate(self):
        # �������ҵ������صĸ�ָ��
        # ��������������self.results��
        self.calc_amt_proportion()
        self.calc_amt_quantile()
        self.calc_turnover_quantile()
        self.calc_vol_shrink_rate()
        self.calc_momentum()
        self.calc_over_MA_number()
        self.calc_recent_new_highs()

    def read_data_price_volume(self):
        # �ӳ���ʽת��Ϊdate-stk_codes�����̼�
        pass

    def calc_amt_proportion(self):
        # �������ҵ�ĳɽ�ռ��
        pass

    def calc_amt_quantile(self):
        # �������ҵ�ĳɽ������һ���λ
        pass

    def calc_turnover_quantile(self):
        # �������ҵ�Ļ����ʹ���һ���λ
        pass

    def calc_vol_shrink_rate(self):
        # �������ҵ��������
        pass

    def calc_momentum(self):
        # �����ʱ��֪����ô��
        pass

    def calc_over_MA_number(self, window=30):
        # ��Ҫ��������
        # �������ҵ30�վ���������ռ�ȷ�λ
        pass

    def calc_recent_new_highs(self, window=60):
        # ��Ҫ��������
        # �������ҵ�� 60 ���¸�����ռ�ȷ�λ
        pass


class Analyst(PgDbUpdaterBase):
    # ��tushare����Ȩ�ޣ���ʱ������
    # TODO�����Կ���wind
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        # ��ʼ��MoneyFlow����Ҫ����������

    def calculate(self):
        # �������ҵ�����ֻ����������б�������λ
        # ��������������self.results��
        pass


class MarketDivergence(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)

    def calculate(self):
        # �������ҵ�����г��ֻ��̶ȵĸ�ָ��
        # ��������������self.results��
        pass

    def calc_market_breadth(self):
        # �����ø��ɻ���ҵ���㡣ǰ����Ҫ�������ݡ�
        pass

    def calc_28_amount_diverge(self):
        # ��ʢ���ԣ����˽��׷ֻ�
        # ��Ҫ��������
        pass

    def calc_rotation_strength(self):
        # ����ȫ�г�����ͬ��ҵ֮��ģ��ֶ�ǿ��
        pass
