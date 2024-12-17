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
        # ѡ����ֵ������ recent_60_days�������ĸ�ӽ�0��Ϊ��ֵ��������������������Խ���
        # recent_60_days = df[column].rolling(window=window_long).sum()
        # positive_values = df[column].where(df[column] > 0, 0)
        positive_values = df[column].abs()
        recent_60_days = positive_values.rolling(window=window_long).sum()

        weight = recent_20_days / recent_60_days
        result_df[column] = weight
    return result_df.sort_index()


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
        self.calculate_predictions()

    def generate_indicators(self):
        # ��������ָ��
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
        # ������ָ����л��ܣ��õ�����ӵ����
        # ��Ҫ�ȶԸ�ָ����irf������irf���ָ����ָͬ���Ȩ�أ����ż���
        pass

    def calculate_predictions(self):
        # �г�����
        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='���̼�' and product_type='index'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(
            columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        daily_return_df = price_ts.pct_change().dropna(how='all')

        self.predictions_list = []
        for data_dict in self.wide_results:
            dict_predictions = {}
            for indicators_group, df in data_dict.items():
                # ���ڶ��� �����������ֵ���
                index_intersection = daily_return_df.index.intersection(df.index)
                daily_return_df = daily_return_df.loc[index_intersection]
                df = df.loc[index_intersection].rename(columns={'�ܶ�': '���ȫA'})

                if len(df.columns) in (30, 31):
                    # ����ҵ����
                    predictions_5_days = {}
                    predictions_25_days = {}
                    for _, industry in enumerate(df.columns):
                        if df[industry].eq(0).all():
                            continue
                        merged = pd.merge(daily_return_df[industry], df[industry],
                                          left_index=True, right_index=True,
                                          suffixes=('_return', '_indicator')).dropna()
                        # ����VARģ��
                        # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
                        merged_trimmed = merged[5:-5]
                        model = VAR(merged_trimmed)
                        results = model.fit(maxlags=5)

                        # # ��ȡ��λ�����Ӧ����
                        # irf = results.irf(30).irfs[1:, 0, 1]
                        # # ����ָ��ÿ�ձ仯����Ϊ�����������100ʹ�ó��Ϊ%��λ�ı仯�ǲ��Եģ�����ĵ�λӦ�úͻ�׼����һ�¡�
                        # shock = df[industry].diff().dropna()
                        # # ʹ�� IRF ��ָ��Ĺ�ȥֵ��Ԥ��δ���Ĺ�ָ����
                        # predictions_5_days[industry] = np.sum([np.multiply(irf[i:i+5], shock[-1-i]) for i in range(5)])
                        # predictions_25_days[industry] = np.sum([np.sum(np.multiply(irf[i:i+25], shock[-1-i])) for i in range(25)])

                        # ֱ����VARԤ��
                        predictions_5_days[industry] = results.forecast(merged.values, 5)[:, 0].sum()
                        predictions_25_days[industry] = results.forecast(merged.values, 25)[:, 0].sum()

                    predictions_sum = pd.concat(
                        [pd.DataFrame(predictions_5_days, index=['predictions_5_days']).sum(axis=0),
                         pd.DataFrame(predictions_25_days, index=['predictions_25_days']).sum(axis=0)], axis=1).rename(
                        columns={0: 'predictions_5_days', 1: 'predictions_25_days'})
                elif len(df.columns) <= 5:
                    # ����ҵ����
                    predictions_5_days = {}
                    predictions_25_days = {}
                    for _, indicator in enumerate(df.columns):
                        merged = pd.merge(daily_return_df['���ȫA'], df[indicator],
                                          left_index=True, right_index=True).dropna()
                        # ����VARģ��
                        # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
                        merged_trimmed = merged[5:-5]
                        model = VAR(merged_trimmed)
                        results = model.fit(maxlags=5)

                        # ֱ����VARԤ��
                        predictions_5_days[indicator] = results.forecast(merged.values, 5)[:, 0].sum()
                        predictions_25_days[indicator] = results.forecast(merged.values, 25)[:, 0].sum()

                        # # ��ȡ��λ�����Ӧ����
                        # irf = results.irf(30).irfs[1:, 0, 1]
                        # # ����ָ��ÿ�ձ仯����Ϊ�����
                        # shock = df[indicator].diff().dropna()
                        #
                        # # ʹ�� IRF ��ָ��Ĺ�ȥֵ��Ԥ��δ���Ĺ�ָ����
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
                df['��Ȩ��ֵ'] = mean_combined
                return df.fillna(0)
            else:
                df['��Ȩ��ֵ'] = df.mean(axis=1)
                df = df.rename(columns={'rotation_strength_daily': '�ն��ֶ�ǿ��',
                                        'rotation_strength_daily_ma20': '�ն��ֶ�ǿ��MA20',
                                        'rotation_strength_5d': '�����ֶ�ǿ��',
                                        'rotation_strength_5d_ma20': '�����ֶ�ǿ��MA20'})
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
                        df_5_days_total.loc['���ȫA', indicator] = df.loc[indicator, 'predictions_5_days']
                        df_25_days_total.loc['���ȫA', indicator] = df.loc[indicator, 'predictions_25_days']

        return calculate_combined_mean(df_5_days_industry), calculate_combined_mean(
            df_25_days_industry), calculate_combined_mean(df_5_days_total), calculate_combined_mean(df_25_days_total)


class MoneyFlow(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data_money_flow()

    def calculate(self):
        # ��������������self.results��
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

        # Question 244 ���������루ȫ����
        big_order_inflow_query = text(
            """
            SELECT "public"."markets_daily_long"."date" AS "date", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", SUM("public"."markets_daily_long"."value") AS "sum"
            FROM "public"."markets_daily_long"
            LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id"
            WHERE ("Product Static Info"."product_type" = 'index')
               AND ("public"."markets_daily_long"."field" = '�����������')
            GROUP BY "public"."markets_daily_long"."date", "Product Static Info"."chinese_name"
            ORDER BY "public"."markets_daily_long"."date" ASC, "Product Static Info"."chinese_name" ASC
            """
        )
        big_order_inflow = self.alch_conn.execute(big_order_inflow_query)
        self.big_order_inflow_df = pd.DataFrame(big_order_inflow,
                                                columns=['date', '������ҵ', '�����������'])

    def calc_finance_net_buy(self):
        # �ӳ���ʽת��Ϊdate-industry�������������
        wide_df = self.finance_net_buy_industry_df.pivot(index='date', columns='������ҵ', values='���ھ������')
        wide_df['�ܶ�'] = wide_df.sum(axis=1)

        # ���20�����������������ռ���60����������������������Ϊ������������
        weights_industry = calculate_recent_weight(df=wide_df).dropna(subset=['��ͨ����'])

        # �����������������й���һ���λ����
        self.fnb_percentile_industry_wide = weights_industry.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.fnb_percentile_industry = self.fnb_percentile_industry_wide.reset_index().melt(id_vars=['date'],
                                                                                            var_name='industry',
                                                                                            value_name='finance_net_buy')

    def calc_north_inflow(self):
        # ע���Խ�
        # ���20�������ձ��������ռ���60�������ձ�������������Ϊ������������
        wide_df = self.north_inflow_industry_df.pivot(index='date', columns='������ҵ', values='���������')
        wide_df['�ܶ�'] = wide_df.sum(axis=1)

        weights_industry = calculate_recent_weight(df=wide_df).dropna(subset=['��ͨ����'])

        # �����������������й���һ���λ����
        self.north_percentile_industry_wide = weights_industry.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.north_percentile_industry = self.north_percentile_industry_wide.reset_index().melt(id_vars=['date'],
                                                                                                var_name='industry',
                                                                                                value_name='north_inflow')

    def calc_big_order_inflow(self):
        # ���һ���������������Ϊ������
        wide_df = self.big_order_inflow_df.pivot(index='date', columns='������ҵ', values='�����������')
        wide_df_ma = wide_df.rolling(10).mean()
        # �Դ��������й���һ���λ����
        self.big_order_inflow_percentile_wide = wide_df_ma.rolling(window=250).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.big_order_inflow_percentile = self.big_order_inflow_percentile_wide.reset_index().melt(id_vars=['date'],
                                                                                                    var_name='industry',
                                                                                                    value_name='big_order_inflow_percentile')

    def calc_order_inflows(self):
        # ������Ҫ��ѯ���ֶ�
        target_fields = [
            '�����_����', '�����_��', '�����_�л�', '�����_ɢ��',
            '�������_����', '�������_��', '�������_�л�', '�������_ɢ��',
            '�����������_����', '�����������_��', '�����������_�л�', '�����������_ɢ��'
        ]

        # ���� SQL ��ѯ
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

        # ִ�в�ѯ
        result = self.alch_conn.execute(order_inflow_query)
        self.industry_order_inflows_df = pd.DataFrame(result.fetchall(),
                                                      columns=['date', 'industry', 'field', 'total_value'])

        # �������ݣ�͸�ӱ��Ա�ֱ���ͬ���ֶΣ�order_type��
        # ���Ǳ��� 'field' ��Ϊһ��ά��
        wide_df = self.industry_order_inflows_df.pivot_table(
            index=['date', 'field'],
            columns='industry',
            values='total_value',
            aggfunc='sum'
        )

        # ��ʼ��һ���б�洢������λ�����
        rolling_percentile_records = []

        # ���ù������ڴ�С
        rolling_window = 250

        # ����ÿ��Ŀ���ֶ�
        for field in target_fields:
            # ��鵱ǰfield�Ƿ������wide_df��������
            if field not in wide_df.index.get_level_values('field'):
                continue  # ��������ڣ�������ǰfield

            # ��ȡ��ǰfield�����ݣ�������������
            try:
                field_df = wide_df.xs(field, level='field').reset_index().sort_values('date')
            except KeyError:
                # ����ֶβ����ڣ�����
                continue

            # ����ÿ����ҵ�����������λ��
            for industry in field_df.columns:
                if industry == 'date':
                    continue  # ����������

                # ��ȡ����ҵ��ʱ�����У�ȷ������������
                ts = field_df[['date', industry]].dropna().sort_values('date').set_index('date')[industry]

                if ts.empty:
                    continue

                # �������ƽ�������� 10 �죩
                rolling_mean = ts.rolling(window=10, min_periods=1).mean()

                # ���������λ�������� 250 �죩
                rolling_percentile = rolling_mean.rolling(window=rolling_window, min_periods=1).apply(
                    lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
                )

                # ������ʱDataFrame�洢���
                temp_df = pd.DataFrame({
                    'date': rolling_percentile.index,
                    'order_inflow_percentile': rolling_percentile.values,
                    'order_type': field,
                    'industry': industry
                })

                # ��ӵ�����б���
                rolling_percentile_records.append(temp_df)

        # �ϲ������ֶκ���ҵ�Ĺ�����λ��
        if rolling_percentile_records:
            rolling_percentile_df = pd.concat(rolling_percentile_records, ignore_index=True)
        else:
            # ���û�����ݣ�����һ���յ�DataFrame
            rolling_percentile_df = pd.DataFrame(columns=['date', 'order_inflow_percentile', 'order_type', 'industry'])

        # ������Ҫ����
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
        # �������ҵ������صĸ�ָ��
        print('calculating for PriceVolume')
        # ��������������self.results��
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
        self.amt_proportion_df_wide = amount_industry_df.div(row_sums, axis=0)
        self.amt_proportion_df = self.amt_proportion_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                                value_name='amt_proportion')

    def calc_amt_quantile(self):
        # �������ҵ�ĳɽ������һ���λ
        amount_industry_df = self.amount_industry_df
        self.amt_quantile_df_wide = amount_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.amt_quantile_df = self.amt_quantile_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
                                                                            value_name='amt_quantile')

    def calc_amt_prop_quantile(self):
        # �������ҵ�ĳɽ���ռ�ȵĹ���һ���λ
        self.amt_prop_quantile_wide = self.amt_proportion_df_wide.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.amt_prop_quantile_df = self.amt_prop_quantile_wide.reset_index().melt(id_vars=['date'],
                                                                                   var_name='industry',
                                                                                   value_name='amt_prop_quantile')

    def calc_turnover_quantile(self):
        # �������ҵ�Ļ����ʹ���һ���λ
        turnover_industry_df = self.turnover_industry_df
        self.turnover_quantile_df_wide = turnover_industry_df.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True)[-1])
        self.turnover_quantile_df = self.turnover_quantile_df_wide.reset_index().melt(id_vars=['date'],
                                                                                      var_name='industry',
                                                                                      value_name='turnover_quantile')

    def calc_vol_shrink_rate(self):
        # �������ҵ��������
        amount_industry_df = self.amount_industry_df
        rolling_avg = amount_industry_df.rolling(window=252).mean()
        self.shrink_rate_df_wide = amount_industry_df / rolling_avg - 1
        self.shrink_rate_df = self.shrink_rate_df_wide.reset_index().melt(id_vars=['date'], var_name='industry',
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
        mb_industry_close['�������Ǳ���MA20'] = mb_industry_close['�������Ǳ���'].rolling(window=20).mean()

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
        # �������19����������ΪNaN
        mb_industry_above_ma.iloc[:20] = np.nan
        mb_industry_above_ma.columns = ['�г����-����λ��']

        self.mb_industry_combined_wide = pd.concat([mb_industry_close, mb_industry_drawdown, mb_industry_above_ma],
                                                   axis=1)
        self.mb_industry_combined = self.mb_industry_combined_wide.reset_index().melt(id_vars=['date'],
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
        self.rotation_strength = rank_changes.sum(axis=1).to_frame()  # ��͵õ��ֶ�ǿ��
        self.rotation_strength.columns = ['rotation_strength_daily']
        self.rotation_strength['rotation_strength_daily_ma20'] = self.rotation_strength.rolling(window=20).mean()

        # ʹ�ù�ȥ5����ǵ�����������
        rank_changes_5d = daily_returns.rolling(window=5).sum().rank(axis=1).diff().abs()
        self.rotation_strength['rotation_strength_5d'] = rank_changes_5d.sum(axis=1).to_frame()  # ��͵õ��ֶ�ǿ��
        self.rotation_strength['rotation_strength_5d_ma20'] = self.rotation_strength['rotation_strength_5d'].rolling(
            window=20).mean()


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
