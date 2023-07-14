# coding=gbk
# Time Created: 2023/6/8 9:10
# Author  : Lucid
# FileName: plotter.py
# Software: PyCharm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib, os
import matplotlib.gridspec as gridspec
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text


class Plotter(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        plt.rcParams['font.sans-serif'] = ['simhei']
        plt.rcParams['axes.unicode_minus'] = False
        self.read_data()
        # self.get_best_order()
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '���ڸ���ҵ��irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '���ڸ���ҵirf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '�������ҵ��irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '�������ҵirf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '�����ܺ͸���ҵ��irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '�����ܺ͸���ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf����ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf����ҵirf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '��ɶ��仯����ҵ��irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '��ɶ��仯����ҵirf', reverse=False)
        self.plot_inflow_windA_irfs(self.daily_return_ts, self.new_fund_ts, '�·�����ȫA��irf', reverse=True)
        self.plot_inflow_windA_irfs(self.daily_return_ts, self.new_fund_ts, '�·�����ȫAirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_extreme_ts, '�������ڸ���ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_extreme_ts, '���˱������ҵirf', reverse=False)

    def calculate_ts_extreme(self, ts_df, upper=0.8, lower=0.2):
        processed_df = ts_df.copy()

        for column in processed_df.columns:
            quantile_80 = processed_df[column].quantile(upper)
            quantile_20 = processed_df[column].quantile(lower)

            processed_df.loc[processed_df[column] > quantile_80, column] = processed_df[column]
            processed_df.loc[processed_df[column] < quantile_20, column] = processed_df[column]
            processed_df.loc[(processed_df[column] >= quantile_20) & (processed_df[column] <= quantile_80), column] = 0

        return processed_df

    def load_metabase_data(self):
        # ������Ҫ��MA10ʱ��Ҫ�õ��������
        df_full = self.get_metabase_full_df()
        # ���̼� ��ӯ�� ��Ϣ�� ��ֵ������MA��ֻ��Ա߼��ʽ���MA
        self.margin_inflow_ts = self.get_metabase_results('margin_inflow', df_full)
        self.north_inflow_ts = self.get_metabase_results('north_inflow', df_full)
        self.aggregate_inflow_ts = self.get_metabase_results('aggregate_inflow', df_full)
        self.etf_inflow_ts = self.get_metabase_results('etf_inflow', df_full)
        self.holder_change_ts = self.get_metabase_results('holder_change', df_full)

    def get_metabase_full_df(self):
        """
        ����metabase question#176 �����ʽ��������(ʱ������)���նȱ���
        ��Ӧ�������ˡ����Ժ�����metabase questionҪ���ü�¼
        """
        metabase_query = text(
            """
            SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."�����ʽ�����") AS "sum", SUM("source"."ETF������") AS "sum_2", SUM("source"."���ھ�����") AS "sum_3", SUM("source"."��������֮��") AS "sum_4", SUM("source"."��ɶ��⾻�ֱֲ䶯���") AS "sum_5"
            FROM (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", "source"."sum" AS "sum", COALESCE("source"."sum", 0) AS "�����ʽ�����", COALESCE("Question 167"."�⾻���ֽ��", 0) * -1 AS "��ɶ��⾻�ֱֲ䶯���", COALESCE("Question 153"."sum", 0) AS "���ھ�����", (COALESCE("Question 153"."sum", 0) + COALESCE("source"."sum", 0) + COALESCE("Question 170"."sum", 0)) - COALESCE("Question 167"."�⾻���ֽ��", 0) AS "��������֮��", COALESCE("Question 170"."sum", 0) AS "ETF������", "Question 153"."date" AS "Question 153__date", "Question 153"."����һ����ҵ" AS "Question 153__����һ����ҵ", "Question 167"."date" AS "Question 167__date", "Question 167"."Product Static Info__stk_industry_cs" AS "Question 167__Product Static Info__stk_industry_cs", "Question 170"."date" AS "Question 170__date", "Question 170"."Product Static Info_2__stk_industry_cs" AS "Question 170__Product Static Info_2__stk_industry_cs", "Question 167"."�⾻���ֽ��" AS "Question 167__�⾻���ֽ��", "Question 153"."sum" AS "Question 153__sum", "Question 170"."sum" AS "Question 170__sum" FROM (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "����һ����ҵ", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '������'
            GROUP BY "source"."date", "source"."����һ����ҵ"
            ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC) AS "source" LEFT JOIN (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long" LEFT JOIN "public"."metric_static_info" AS "Metric Static Info_2" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info_2"."internal_id" WHERE "Metric Static Info_2"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '���ھ������' GROUP BY "source"."date", "source"."����һ����ҵ" ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC) AS "Question 153" ON ("source"."date" = "Question 153"."date")
               AND ("source"."����һ����ҵ" = "Question 153"."����һ����ҵ") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info__stk_industry_cs" AS "Product Static Info__stk_industry_cs", MAX(COALESCE(CASE WHEN "source"."field" = '����ֽ��' THEN "source"."value" END, 0)) - MAX(COALESCE(CASE WHEN "source"."field" = '�����ֽ��' THEN "source"."value" END, 0)) AS "�⾻���ֽ��" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", "public"."markets_daily_long"."product_static_info_id" AS "product_static_info_id", "public"."markets_daily_long"."date_value" AS "date_value", "Product Static Info"."internal_id" AS "Product Static Info__internal_id", "Product Static Info"."code" AS "Product Static Info__code", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", "Product Static Info"."english_name" AS "Product Static Info__english_name", "Product Static Info"."source" AS "Product Static Info__source", "Product Static Info"."type_identifier" AS "Product Static Info__type_identifier", "Product Static Info"."buystartdate" AS "Product Static Info__buystartdate", "Product Static Info"."fundfounddate" AS "Product Static Info__fundfounddate", "Product Static Info"."issueshare" AS "Product Static Info__issueshare", "Product Static Info"."fund_fullname" AS "Product Static Info__fund_fullname", "Product Static Info"."stk_industry_cs" AS "Product Static Info__stk_industry_cs", "Product Static Info"."product_type" AS "Product Static Info__product_type", "Product Static Info"."etf_type" AS "Product Static Info__etf_type" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id" WHERE "Product Static Info"."type_identifier" = 'major_holder') AS "source" WHERE "source"."Product Static Info__type_identifier" = 'major_holder' GROUP BY "source"."date", "source"."Product Static Info__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info__stk_industry_cs" ASC) AS "Question 167" ON ("source"."date" = "Question 167"."date") AND ("source"."����һ����ҵ" = "Question 167"."Product Static Info__stk_industry_cs") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info_2__stk_industry_cs" AS "Product Static Info_2__stk_industry_cs", SUM("source"."value") AS "sum" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "Product Static Info_2"."code" AS "Product Static Info_2__code", "Product Static Info_2"."fund_fullname" AS "Product Static Info_2__fund_fullname", "Product Static Info_2"."stk_industry_cs" AS "Product Static Info_2__stk_industry_cs" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info_2" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info_2"."internal_id" WHERE ("public"."markets_daily_long"."field" = '�������') AND ("Product Static Info_2"."product_type" = 'fund')) AS "source" GROUP BY "source"."date", "source"."Product Static Info_2__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info_2__stk_industry_cs" ASC) AS "Question 170" ON ("source"."date" = "Question 170"."date") AND ("source"."����һ����ҵ" = "Question 170"."Product Static Info_2__stk_industry_cs")) AS "source" GROUP BY "source"."date", "source"."����һ����ҵ" ORDER BY "source"."date" DESC, "source"."����һ����ҵ" ASC
            """
        )
        full_industry_history = self.alch_conn.execute(metabase_query)
        return pd.DataFrame(full_industry_history,
                            columns=['date', '����һ����ҵ', '�����ʽ�����', 'ETF������', '���ھ�����',
                                     '��������֮��', '��ɶ��⾻�ֱֲ䶯���'])

    def get_metabase_new_fund_ts(self):
        """
        metabase question 229, ����-���������й�ģ(�նȱ���ʱ������)
        """
        metabase_query = text(
            """
            SELECT "source"."fundfounddate" AS "fundfounddate", SUM("source"."ETF�����зݶ�") AS "sum", SUM("source"."����������зݶ�") AS "sum_2", SUM("source"."��ծ������зݶ�") AS "sum_3"
            FROM (SELECT "source"."fundfounddate" AS "fundfounddate", "source"."sum" AS "sum", COALESCE("Question 221"."sum", 0) AS "ETF�����зݶ�", COALESCE("source"."sum", 0) AS "��ծ������зݶ�", COALESCE("source"."sum", 0) - COALESCE("Question 221"."sum", 0) AS "����������зݶ�", "Question 221"."fundfounddate" AS "Question 221__fundfounddate", "Question 221"."sum" AS "Question 221__sum", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_2", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_3" FROM (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info"
            WHERE ("public"."product_static_info"."product_type" = 'fund')
               AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%ծ%')
                OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%�浥%') OR ("public"."product_static_info"."fund_fullname" IS NULL))
            GROUP BY "public"."product_static_info"."fundfounddate"
            ORDER BY "public"."product_static_info"."fundfounddate" ASC) AS "source"
            LEFT JOIN (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info" WHERE ("public"."product_static_info"."product_type" = 'fund') AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%ծ%') OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (LOWER("public"."product_static_info"."fund_fullname") LIKE '%�����Ϳ���ʽ%') GROUP BY "public"."product_static_info"."fundfounddate" ORDER BY "public"."product_static_info"."fundfounddate" DESC) AS "Question 221" ON "source"."fundfounddate" = "Question 221"."fundfounddate") AS "source" GROUP BY "source"."fundfounddate" ORDER BY "source"."fundfounddate" DESC
            """
        )
        new_fund = self.alch_conn.execute(metabase_query)
        new_fund_ts = pd.DataFrame(new_fund, columns=['date', 'ETF�����зݶ�', '����������зݶ�', '��ծ�෢�зݶ�']).dropna().set_index('date')
        return new_fund_ts

    def get_metabase_results(self, task, df_full):
        print(f'Calculating for task:{task}')
        match task:
            case 'aggregate_inflow':
                df_inflow_sum = df_full.groupby(['date', '����һ����ҵ'])['��������֮��'].apply(
                    lambda x: pd.Series(x.values)).unstack('����һ����ҵ').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_inflow_sum

            case 'margin_inflow':
                df_margin_buy = df_full.groupby(['date', '����һ����ҵ'])['���ھ�����'].apply(
                    lambda x: pd.Series(x.values)).unstack('����һ����ҵ').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_margin_buy

            case 'north_inflow':
                df_net_buy = df_full.groupby(['date', '����һ����ҵ'])['�����ʽ�����'].apply(
                    lambda x: pd.Series(x.values)).unstack('����һ����ҵ').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_net_buy

            case 'etf_inflow':
                df_etf_flow = df_full.groupby(['date', '����һ����ҵ'])['ETF������'].apply(
                    lambda x: pd.Series(x.values)).unstack('����һ����ҵ').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_etf_flow
            case 'holder_change':
                df_holder_change = df_full.groupby(['date', '����һ����ҵ'])['��ɶ��⾻�ֱֲ䶯���'].apply(
                    lambda x: pd.Series(x.values)).unstack('����һ����ҵ').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_holder_change
            case _:
                raise Exception(f'task={task} not supported!')

    def read_data(self):
        self.load_metabase_data()
        self.margin_inflow_ts = self.margin_inflow_ts/1e8
        # ����ȫA��Ӱ��ͳһ��Ϊÿ10�ڵ�Ӱ��
        self.margin_inflow_ts['���ȫA'] = self.margin_inflow_ts.sum(axis=1)/10
        # self.margin_inflow_extreme_ts = self.calculate_ts_extreme(self.margin_inflow_ts, 0.9, 0.1)

        self.north_inflow_ts = self.north_inflow_ts/1e8
        self.north_inflow_ts['���ȫA'] = self.north_inflow_ts.sum(axis=1)/10
        # self.north_inflow_extreme_ts = self.calculate_ts_extreme(self.north_inflow_ts, 0.9, 0.1)

        self.aggregate_inflow_ts = self.aggregate_inflow_ts/1e8
        self.aggregate_inflow_ts['���ȫA'] = self.aggregate_inflow_ts.sum(axis=1)/10
        # self.aggregate_inflow_extreme_ts = self.calculate_ts_extreme(self.aggregate_inflow_ts, 0.9, 0.1)

        self.etf_inflow_ts = self.etf_inflow_ts/1e8
        self.etf_inflow_ts['���ȫA'] = self.etf_inflow_ts.sum(axis=1)/10
        # self.etf_inflow_extreme_ts = self.calculate_ts_extreme(self.etf_inflow_ts, 0.9, 0.1)

        self.holder_change_ts = self.holder_change_ts/1e8
        self.holder_change_ts['���ȫA'] = self.holder_change_ts.sum(axis=1)/10
        # self.holder_change_extreme_ts = self.calculate_ts_extreme(self.holder_change_ts, 0.9, 0.1)

        self.new_fund_ts = self.get_metabase_new_fund_ts()
        # self.new_fund_ts['���ȫA'] = self.new_fund_ts['��ծ�෢�зݶ�']
        # self.new_fund_extreme_ts = self.calculate_ts_extreme(self.new_fund_ts, 0.9, 0.1)

        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='���̼�'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.daily_return_ts = price_ts.pct_change().dropna(how='all')*100

    def plot_inflow_industry_irfs(self, daily_return_df, inflow_df, fig_name, reverse):
        # ���ڶ��� �����������ֵ���
        # ��5������ͣ�������ֱ�������ܶȸ������
        inflow_5d = inflow_df.rolling(5).sum().dropna()
        index_intersection = daily_return_df.index.intersection(inflow_5d.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_5d.loc[index_intersection]

        # ����Figure��Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 7, figure=fig, hspace=0, wspace=0)  # �����hspace�������м�࣬����Ը�����Ҫ��������ֵ
        plt.rcParams['axes.titlesize'] = 5

        for index, industry in enumerate(daily_return_df.columns):
            if inflow_df[industry].eq(0).all():
                continue

            merged = pd.merge(daily_return_df[industry], inflow_df[industry],
                              left_index=True, right_index=True, suffixes=('_return', '_inflow'))
            # ����VARģ��
            # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
            merged = merged[15:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # ����VARģ��
            # 5��bic����ֵ������ƽʱҲ���ῼ�ǹ�ȥ���ܵ�Ӱ�죬���������ǵ�Ӧ�ó�����10�Ļ���̫zigzag�ˣ����ѽ���
            results = model.fit(maxlags=5)

            # ��ȡ��λ�����Ӧ����
            irf = results.irf(periods=30)  # �趨�����Ӧ����������

            # ����ָ���������Ӧ���ۻ������Ӧ����
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'{industry}_return'),
                                             merged.columns.get_loc(f'{industry}_inflow')])
            else:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'{industry}_inflow'),
                                             merged.columns.get_loc(f'{industry}_return')])

            # ���ƶ�̬��Ӧ��������ʱ�ļ�
            filename = f"temp_plot_industry_{industry}.png"
            if reverse:
                irf.plot(impulse=f'{industry}_return', response=f'{industry}_inflow', signif=0.2)

            else:
                irf.plot(impulse=f'{industry}_inflow', response=f'{industry}_return', signif=0.2)
                if industry == '���ȫA':
                    plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # ��Figure����ʾ��ͼ��
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # Ϊÿ����ͼ��ӱ��⣬���ڱ����а����ۻ������Ӧ������ֵ
            if reverse:
                if industry == '���ȫA':
                    ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}ʮ��Ԫ")
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}��Ԫ")
            else:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}%")
            # ɾ����ʱ�ļ�
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=2000)

    def plot_inflow_windA_irfs(self, daily_return_df, inflow_df, fig_name, reverse):
        inflow_5d = inflow_df.rolling(5).sum().dropna()
        index_intersection = daily_return_df.index.intersection(inflow_5d.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_5d.loc[index_intersection]

        # ����Figure��Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0, wspace=0)
        plt.rcParams['axes.titlesize'] = 5

        for index, metric in enumerate(inflow_df.columns):
            if inflow_df[metric].eq(0).all():
                continue

            merged = pd.merge(daily_return_df['���ȫA'], inflow_df[metric],
                              left_index=True, right_index=True, suffixes=('_return', '_inflow'))
            # ����VARģ��
            # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # ����VARģ��
            # 5��bic����ֵ������ƽʱҲ���ῼ�ǹ�ȥ���ܵ�Ӱ�죬���������ǵ�Ӧ�ó�����10�Ļ���̫zigzag�ˣ����ѽ���
            results = model.fit(maxlags=5)

            # ��ȡ��λ�����Ӧ����
            irf = results.irf(periods=30)  # �趨�����Ӧ����������

            # ����ָ���������Ӧ���ۻ������Ӧ����
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'���ȫA'),
                                             merged.columns.get_loc(metric)])
            else:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(metric),
                                             merged.columns.get_loc(f'���ȫA')])

            # ���ƶ�̬��Ӧ��������ʱ�ļ�
            filename = f"temp_plot_industry_{metric}.png"
            if reverse:
                irf.plot(impulse='���ȫA', response=metric, signif=0.2)

            else:
                irf.plot(impulse=metric, response='���ȫA', signif=0.2)
            plt.savefig(filename, dpi=100)
            plt.close()

            # ��Figure����ʾ��ͼ��
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # Ϊÿ����ͼ��ӱ��⣬���ڱ����а����ۻ������Ӧ������ֵ
            if reverse:
                ax.set_title(f"{metric}\nCumulative: {cumulative_response:.2f}��Ԫ")
            else:
                ax.set_title(f"{metric}\nCumulative: {cumulative_response:.2f}%")
            # ɾ����ʱ�ļ�
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=400, bbox_inches='tight')

    def get_best_order(self):
        # ���ý�����Χ
        max_order = 10

        # ��ʼ����СAIC�����Ž���
        min_aic = float('inf')
        best_order = 0

        # ������ͬ����
        for order in range(1, max_order + 1):
            # ����VARģ��
            model = sm.tsa.VAR(self.merged_agagregate_both[['���ȫA', '��������֮��_MA10']])

            # ����VARģ��
            results = model.fit(order)

            # ����AIC
            aic = results.aic

            # �ж��Ƿ�Ϊ��СAIC
            if aic < min_aic:
                min_aic = aic
                best_order = order

        # ������Ž���
        print("���Ž���:", best_order)

