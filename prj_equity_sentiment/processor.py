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
        # self.upload_indicator(self.analyst.results)
        self.upload_indicator(self.market_divergence.results)
        # self.upload_indicator(self.industry_congestion.results)
        pass

    def upload_indicator(self, results: dict):
        for table_name, df in results.items():
            print(f'uploading {table_name} to database')
            df.dropna().to_sql(name=table_name, con=self.alch_engine, schema='processed_data', if_exists='replace')

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

    @property
    def results(self):
        return {
            'finance_net_buy_percentile_industry': self.fnb_percentile_industry,
            'north_percentile_industry': self.north_percentile_industry,
        }

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
        self.finance_net_buy_df = pd.DataFrame(finance_net_buy, columns=['date', '���ھ������ܶ�']).set_index('date')

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
        self.north_inflow_df = pd.DataFrame(north_inflow, columns=['date', '���������ܶ�']).set_index('date')

        # ��Ҫ������ҵ��
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
        finance_net_buy_industry = self.alch_conn.execute(finance_net_buy_industry_query)
        self.finance_net_buy_industry_df = pd.DataFrame(finance_net_buy_industry,
                                                        columns=['date', '������ҵ', '���ھ������'])

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
        north_inflow_industry = self.alch_conn.execute(north_inflow_industry_query)
        self.north_inflow_industry_df = pd.DataFrame(north_inflow_industry,
                                                     columns=['date', '������ҵ', '���������'])

    def calc_finance_net_buy(self):
        # �ӳ���ʽת��Ϊdate-industry�������������
        wide_df = self.finance_net_buy_industry_df.pivot(index='date', columns='������ҵ', values='���ھ������')

        # ���20�����������������ռ���60����������������������Ϊ������������
        weights_industry = calculate_recent_weight(df=wide_df)
        weights_total = calculate_recent_weight(df=self.finance_net_buy_df)

        # �����������������й���һ���λ����
        self.fnb_percentile_industry = weights_industry.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        fnb_percentile_total = weights_total.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True)[0])
        self.fnb_percentile_industry['�ܶ�'] = fnb_percentile_total['���ھ������ܶ�']

    def calc_north_inflow(self):
        # ע���Խ�
        # ���20�������ձ��������ռ���60�������ձ�������������Ϊ������������
        wide_df = self.north_inflow_industry_df.pivot(index='date', columns='������ҵ', values='���������')
        weights_industry = calculate_recent_weight(df=wide_df)
        weights_total = calculate_recent_weight(df=self.north_inflow_df)

        # �����������������й���һ���λ����
        self.north_percentile_industry = weights_industry.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        north_percentile_total = weights_total.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True)[0])
        self.north_percentile_industry['�ܶ�'] = north_percentile_total['���������ܶ�']


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

    @property
    def results(self):
        return {'amt_proportion': self.amt_proportion_df,
                'amt_quantile': self.amt_quantile_df,
                'turnover_quantile': self.turnover_quantile_df,
                'shrink_rate': self.shrink_rate_df
                }

    def read_data_price_volume(self):
        # �ӳ���ʽת��Ϊdate-stk_codes�����̼�
        # Question 230 ����ҵ�ɽ���(ʱ������)
        amount_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '�ɽ���')
            """
        )
        amount_industry = self.alch_conn.execute(amount_industry_query)
        amount_industry_long_df = pd.DataFrame(amount_industry, columns=['date', 'ָ������', 'value', 'ָ������'])
        # �ӳ���ʽת��Ϊdate-industry�ĳɽ���
        self.amount_industry_df = amount_industry_long_df.pivot(index='date', columns='ָ������', values='value')

        # Question 231 ����ҵ������(ʱ������)
        turnover_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '������')
            """
        )
        turnover_industry = self.alch_conn.execute(turnover_industry_query)
        turnover_industry_long_df = pd.DataFrame(turnover_industry, columns=['date', 'ָ������', 'value', 'ָ������'])
        # �ӳ���ʽת��Ϊdate-industry�Ļ�����
        self.turnover_industry_df = turnover_industry_long_df.pivot(index='date', columns='ָ������', values='value')

        # ����ҵ-���������̼�
        pass

    def calc_amt_proportion(self):
        # �������ҵ�ĳɽ�ռ��
        amount_industry_df = self.amount_industry_df.drop(columns=['���ȫA'])
        # ����ÿһ���и�����ֵ��ռ��
        row_sums = amount_industry_df.sum(axis=1)  # ����ÿ�е��ܺ�
        amt_proportion_df = amount_industry_df.div(row_sums, axis=0)
        self.amt_proportion_df = amt_proportion_df.reset_index().melt(id_vars=['date'], var_name='industry', value_name='amt_proportion')

    def calc_amt_quantile(self):
        # �������ҵ�ĳɽ������һ���λ
        amount_industry_df = self.amount_industry_df.drop(columns=['���ȫA'])
        amt_quantile_df = amount_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.amt_quantile_df = amt_quantile_df.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                      value_name='amt_quantile')

    def calc_turnover_quantile(self):
        # �������ҵ�Ļ����ʹ���һ���λ
        turnover_industry_df = self.turnover_industry_df.drop(columns=['���ȫA'])
        turnover_quantile_df = turnover_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[0])
        self.turnover_quantile_df = turnover_quantile_df.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                      value_name='turnover_quantile')

    def calc_vol_shrink_rate(self):
        # �������ҵ��������
        amount_industry_df = self.amount_industry_df.drop(columns=['���ȫA'])
        rolling_avg = amount_industry_df.rolling(window=252).mean()
        shrink_rate_df = amount_industry_df / rolling_avg - 1
        self.shrink_rate_df = shrink_rate_df.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                      value_name='shrink_rate')

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


class MarketDivergence(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data()

    def calculate(self):
        # �������ҵ�����г��ֻ��̶ȵĸ�ָ��
        self.calc_market_breadth()
        self.calc_28_amount_diverge()
        self.calc_rotation_strength()

    @property
    def results(self):
        return {
            'market_breadth_industry': self.mb_industry_combined,
            'rotation_strength': self.rotation_strength,
        }

    def read_data(self):
        # ����ҵ���̼�
        # Question 232 ����ҵ���̼�(ʱ������)
        close_industry_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."value" AS "value", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '���̼�')
            ORDER BY "public"."markets_daily_long"."date" ASC
            """
        )
        close_industry = self.alch_conn.execute(close_industry_query)
        close_industry_long_df = pd.DataFrame(close_industry, columns=['date', 'ָ������', 'value', 'ָ������'])
        # �ӳ���ʽת��Ϊdate-industry�����̼�
        self.close_industry_df = close_industry_long_df.pivot(index='date', columns='ָ������', values='value')

    def calc_market_breadth(self):
        # �����ø��ɻ���ҵ���㡣ǰ����Ҫ�������ݡ�
        # ��ҵ����
        close_industry_df = self.close_industry_df.drop(columns=['���ȫA'])
        # �������Ǳ���
        # ����ÿ����ҵ�����ǵ���
        pct_change_df = close_industry_df.pct_change()
        # �ж�ÿ����Щ��ҵ����
        rising_df = pct_change_df > 0
        # ����ÿ�����ǵ���ҵ�ı���
        mb_industry_close = rising_df.mean(axis=1).to_frame()
        mb_industry_close.columns = ['�������Ǳ���']
        # mb_industry_close = close_industry_df.apply(lambda x: sum(x > x.shift(1)), axis=1)

        # �س�������λ����ȥȫA�Ļس�����
        # ������ҵ�س�����
        industry_drawdown = (close_industry_df / close_industry_df.rolling(window=252).max()) - 1
        industry_drawdown_median = industry_drawdown.median(axis=1)
        # ����ȫ�г�ָ���س�����
        market_drawdown = (self.close_industry_df['���ȫA'] / self.close_industry_df['���ȫA'].rolling(
            window=252).max()) - 1
        # �����г����
        mb_industry_drawdown = (industry_drawdown_median - market_drawdown).to_frame()
        mb_industry_drawdown.columns = ['�г����-���ڻس�']

        # ��ҵָ�����̼�վ��20�վ��ߵı���
        mb_industry_above_ma = (close_industry_df > close_industry_df.rolling(window=20).mean()).mean(axis=1).to_frame()
        mb_industry_above_ma.columns = ['�г����-����λ��']

        mb_industry_combined = pd.concat([mb_industry_close, mb_industry_drawdown, mb_industry_above_ma], axis=1)
        self.mb_industry_combined = mb_industry_combined.reset_index().melt(id_vars=['date'],
                                                                            var_name='market_breadth_type',
                                                                            value_name='value')

    def calc_28_amount_diverge(self):
        # ��ʢ���ԣ����˽��׷ֻ�
        # ��Ҫ��������
        pass

    def calc_rotation_strength(self):
        # ����ȫ�г�����ͬ��ҵ֮��ģ��ֶ�ǿ��
        close_industry_df = self.close_industry_df.drop(columns=['���ȫA'])
        daily_returns = close_industry_df.pct_change()  # ����ÿ���ǵ���
        rank_changes = daily_returns.rank(axis=1).diff().abs()  # �����ǵ��ٷֱ������ı仯�ľ���ֵ
        self.rotation_strength = rank_changes.sum(axis=1)  # ��͵õ��ֶ�ǿ��


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
