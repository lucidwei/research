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
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业逆irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业irf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业逆irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业irf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '四项总和各行业逆irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '四项总和各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf各行业irf', reverse=False)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '大股东变化各行业逆irf', reverse=True)
        self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '大股东变化各行业irf', reverse=False)
        self.plot_inflow_windA_irfs(self.daily_return_ts, self.new_fund_ts, '新发基金全A逆irf', reverse=True)
        self.plot_inflow_windA_irfs(self.daily_return_ts, self.new_fund_ts, '新发基金全Airf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_extreme_ts, '极端两融各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_extreme_ts, '极端北向各行业irf', reverse=False)

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
        # 复盘需要看MA10时需要用到这个函数
        df_full = self.get_metabase_full_df()
        # 收盘价 市盈率 股息率 市值不用求MA，只需对边际资金求MA
        self.margin_inflow_ts = self.get_metabase_results('margin_inflow', df_full)
        self.north_inflow_ts = self.get_metabase_results('north_inflow', df_full)
        self.aggregate_inflow_ts = self.get_metabase_results('aggregate_inflow', df_full)
        self.etf_inflow_ts = self.get_metabase_results('etf_inflow', df_full)
        self.holder_change_ts = self.get_metabase_results('holder_change', df_full)

    def get_metabase_full_df(self):
        """
        不是metabase question#176 五项资金流动情况(时间序列)的日度变体
        对应不起来了……以后引用metabase question要做好记录
        """
        metabase_query = text(
            """
            SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."北向资金净买入") AS "sum", SUM("source"."ETF净流入") AS "sum_2", SUM("source"."两融净买入") AS "sum_3", SUM("source"."四项流入之和") AS "sum_4", SUM("source"."大股东拟净持仓变动金额") AS "sum_5"
            FROM (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", "source"."sum" AS "sum", COALESCE("source"."sum", 0) AS "北向资金净买入", COALESCE("Question 167"."拟净减持金额", 0) * -1 AS "大股东拟净持仓变动金额", COALESCE("Question 153"."sum", 0) AS "两融净买入", (COALESCE("Question 153"."sum", 0) + COALESCE("source"."sum", 0) + COALESCE("Question 170"."sum", 0)) - COALESCE("Question 167"."拟净减持金额", 0) AS "四项流入之和", COALESCE("Question 170"."sum", 0) AS "ETF净流入", "Question 153"."date" AS "Question 153__date", "Question 153"."中信一级行业" AS "Question 153__中信一级行业", "Question 167"."date" AS "Question 167__date", "Question 167"."Product Static Info__stk_industry_cs" AS "Question 167__Product Static Info__stk_industry_cs", "Question 170"."date" AS "Question 170__date", "Question 170"."Product Static Info_2__stk_industry_cs" AS "Question 170__Product Static Info_2__stk_industry_cs", "Question 167"."拟净减持金额" AS "Question 167__拟净减持金额", "Question 153"."sum" AS "Question 153__sum", "Question 170"."sum" AS "Question 170__sum" FROM (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "中信一级行业", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '净买入'
            GROUP BY "source"."date", "source"."中信一级行业"
            ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC) AS "source" LEFT JOIN (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long" LEFT JOIN "public"."metric_static_info" AS "Metric Static Info_2" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info_2"."internal_id" WHERE "Metric Static Info_2"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '两融净买入额' GROUP BY "source"."date", "source"."中信一级行业" ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC) AS "Question 153" ON ("source"."date" = "Question 153"."date")
               AND ("source"."中信一级行业" = "Question 153"."中信一级行业") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info__stk_industry_cs" AS "Product Static Info__stk_industry_cs", MAX(COALESCE(CASE WHEN "source"."field" = '拟减持金额' THEN "source"."value" END, 0)) - MAX(COALESCE(CASE WHEN "source"."field" = '拟增持金额' THEN "source"."value" END, 0)) AS "拟净减持金额" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", "public"."markets_daily_long"."product_static_info_id" AS "product_static_info_id", "public"."markets_daily_long"."date_value" AS "date_value", "Product Static Info"."internal_id" AS "Product Static Info__internal_id", "Product Static Info"."code" AS "Product Static Info__code", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", "Product Static Info"."english_name" AS "Product Static Info__english_name", "Product Static Info"."source" AS "Product Static Info__source", "Product Static Info"."type_identifier" AS "Product Static Info__type_identifier", "Product Static Info"."buystartdate" AS "Product Static Info__buystartdate", "Product Static Info"."fundfounddate" AS "Product Static Info__fundfounddate", "Product Static Info"."issueshare" AS "Product Static Info__issueshare", "Product Static Info"."fund_fullname" AS "Product Static Info__fund_fullname", "Product Static Info"."stk_industry_cs" AS "Product Static Info__stk_industry_cs", "Product Static Info"."product_type" AS "Product Static Info__product_type", "Product Static Info"."etf_type" AS "Product Static Info__etf_type" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id" WHERE "Product Static Info"."type_identifier" = 'major_holder') AS "source" WHERE "source"."Product Static Info__type_identifier" = 'major_holder' GROUP BY "source"."date", "source"."Product Static Info__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info__stk_industry_cs" ASC) AS "Question 167" ON ("source"."date" = "Question 167"."date") AND ("source"."中信一级行业" = "Question 167"."Product Static Info__stk_industry_cs") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info_2__stk_industry_cs" AS "Product Static Info_2__stk_industry_cs", SUM("source"."value") AS "sum" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "Product Static Info_2"."code" AS "Product Static Info_2__code", "Product Static Info_2"."fund_fullname" AS "Product Static Info_2__fund_fullname", "Product Static Info_2"."stk_industry_cs" AS "Product Static Info_2__stk_industry_cs" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info_2" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info_2"."internal_id" WHERE ("public"."markets_daily_long"."field" = '净流入额') AND ("Product Static Info_2"."product_type" = 'fund')) AS "source" GROUP BY "source"."date", "source"."Product Static Info_2__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info_2__stk_industry_cs" ASC) AS "Question 170" ON ("source"."date" = "Question 170"."date") AND ("source"."中信一级行业" = "Question 170"."Product Static Info_2__stk_industry_cs")) AS "source" GROUP BY "source"."date", "source"."中信一级行业" ORDER BY "source"."date" DESC, "source"."中信一级行业" ASC
            """
        )
        full_industry_history = self.alch_conn.execute(metabase_query)
        return pd.DataFrame(full_industry_history,
                            columns=['date', '中信一级行业', '北向资金净买入', 'ETF净流入', '两融净买入',
                                     '四项流入之和', '大股东拟净持仓变动金额'])

    def get_metabase_new_fund_ts(self):
        """
        metabase question 229, 基金-主被动发行规模(日度被引时间序列)
        """
        metabase_query = text(
            """
            SELECT "source"."fundfounddate" AS "fundfounddate", SUM("source"."ETF基金发行份额") AS "sum", SUM("source"."主动类基金发行份额") AS "sum_2", SUM("source"."非债类基金发行份额") AS "sum_3"
            FROM (SELECT "source"."fundfounddate" AS "fundfounddate", "source"."sum" AS "sum", COALESCE("Question 221"."sum", 0) AS "ETF基金发行份额", COALESCE("source"."sum", 0) AS "非债类基金发行份额", COALESCE("source"."sum", 0) - COALESCE("Question 221"."sum", 0) AS "主动类基金发行份额", "Question 221"."fundfounddate" AS "Question 221__fundfounddate", "Question 221"."sum" AS "Question 221__sum", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_2", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_3" FROM (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info"
            WHERE ("public"."product_static_info"."product_type" = 'fund')
               AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%债%')
                OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%存单%') OR ("public"."product_static_info"."fund_fullname" IS NULL))
            GROUP BY "public"."product_static_info"."fundfounddate"
            ORDER BY "public"."product_static_info"."fundfounddate" ASC) AS "source"
            LEFT JOIN (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info" WHERE ("public"."product_static_info"."product_type" = 'fund') AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%债%') OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (LOWER("public"."product_static_info"."fund_fullname") LIKE '%交易型开放式%') GROUP BY "public"."product_static_info"."fundfounddate" ORDER BY "public"."product_static_info"."fundfounddate" DESC) AS "Question 221" ON "source"."fundfounddate" = "Question 221"."fundfounddate") AS "source" GROUP BY "source"."fundfounddate" ORDER BY "source"."fundfounddate" DESC
            """
        )
        new_fund = self.alch_conn.execute(metabase_query)
        new_fund_ts = pd.DataFrame(new_fund, columns=['date', 'ETF基金发行份额', '主动类基金发行份额', '非债类发行份额']).dropna().set_index('date')
        return new_fund_ts

    def get_metabase_results(self, task, df_full):
        print(f'Calculating for task:{task}')
        match task:
            case 'aggregate_inflow':
                df_inflow_sum = df_full.groupby(['date', '中信一级行业'])['四项流入之和'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_inflow_sum

            case 'margin_inflow':
                df_margin_buy = df_full.groupby(['date', '中信一级行业'])['两融净买入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_margin_buy

            case 'north_inflow':
                df_net_buy = df_full.groupby(['date', '中信一级行业'])['北向资金净买入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_net_buy

            case 'etf_inflow':
                df_etf_flow = df_full.groupby(['date', '中信一级行业'])['ETF净流入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_etf_flow
            case 'holder_change':
                df_holder_change = df_full.groupby(['date', '中信一级行业'])['大股东拟净持仓变动金额'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_holder_change
            case _:
                raise Exception(f'task={task} not supported!')

    def read_data(self):
        self.load_metabase_data()
        self.margin_inflow_ts = self.margin_inflow_ts/1e8
        # 对于全A的影响统一改为每10亿的影响
        self.margin_inflow_ts['万德全A'] = self.margin_inflow_ts.sum(axis=1)/10
        # self.margin_inflow_extreme_ts = self.calculate_ts_extreme(self.margin_inflow_ts, 0.9, 0.1)

        self.north_inflow_ts = self.north_inflow_ts/1e8
        self.north_inflow_ts['万德全A'] = self.north_inflow_ts.sum(axis=1)/10
        # self.north_inflow_extreme_ts = self.calculate_ts_extreme(self.north_inflow_ts, 0.9, 0.1)

        self.aggregate_inflow_ts = self.aggregate_inflow_ts/1e8
        self.aggregate_inflow_ts['万德全A'] = self.aggregate_inflow_ts.sum(axis=1)/10
        # self.aggregate_inflow_extreme_ts = self.calculate_ts_extreme(self.aggregate_inflow_ts, 0.9, 0.1)

        self.etf_inflow_ts = self.etf_inflow_ts/1e8
        self.etf_inflow_ts['万德全A'] = self.etf_inflow_ts.sum(axis=1)/10
        # self.etf_inflow_extreme_ts = self.calculate_ts_extreme(self.etf_inflow_ts, 0.9, 0.1)

        self.holder_change_ts = self.holder_change_ts/1e8
        self.holder_change_ts['万德全A'] = self.holder_change_ts.sum(axis=1)/10
        # self.holder_change_extreme_ts = self.calculate_ts_extreme(self.holder_change_ts, 0.9, 0.1)

        self.new_fund_ts = self.get_metabase_new_fund_ts()
        # self.new_fund_ts['万德全A'] = self.new_fund_ts['非债类发行份额']
        # self.new_fund_extreme_ts = self.calculate_ts_extreme(self.new_fund_ts, 0.9, 0.1)

        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='收盘价'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.daily_return_ts = price_ts.pct_change().dropna(how='all')*100

    def plot_inflow_industry_irfs(self, daily_return_df, inflow_df, fig_name, reverse):
        # 日期对齐 保留交集部分的行
        # 求5日流入和，测算结果直接能与周度跟踪相乘
        inflow_5d = inflow_df.rolling(5).sum().dropna()
        index_intersection = daily_return_df.index.intersection(inflow_5d.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_5d.loc[index_intersection]

        # 创建Figure和Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 7, figure=fig, hspace=0, wspace=0)  # 这里的hspace设置了行间距，你可以根据需要调整它的值
        plt.rcParams['axes.titlesize'] = 5

        for index, industry in enumerate(daily_return_df.columns):
            if inflow_df[industry].eq(0).all():
                continue

            merged = pd.merge(daily_return_df[industry], inflow_df[industry],
                              left_index=True, right_index=True, suffixes=('_return', '_inflow'))
            # 创建VAR模型
            # 剔除前面不连续的日期和最近一周的影响
            merged = merged[15:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # 估计VAR模型
            # 5是bic给的值，而且平时也不会考虑过去两周的影响，更符合我们的应用场景。10的话就太zigzag了，很难解释
            results = model.fit(maxlags=5)

            # 提取单位冲击响应函数
            irf = results.irf(periods=30)  # 设定冲击响应函数的期数

            # 计算指定冲击和响应的累积冲击响应函数
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'{industry}_return'),
                                             merged.columns.get_loc(f'{industry}_inflow')])
            else:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'{industry}_inflow'),
                                             merged.columns.get_loc(f'{industry}_return')])

            # 绘制动态响应函数到临时文件
            filename = f"temp_plot_industry_{industry}.png"
            if reverse:
                irf.plot(impulse=f'{industry}_return', response=f'{industry}_inflow', signif=0.2)

            else:
                irf.plot(impulse=f'{industry}_inflow', response=f'{industry}_return', signif=0.2)
                if industry == '万德全A':
                    plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # 在Figure上显示该图像
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # 为每个子图添加标题，并在标题中包含累积冲击响应函数的值
            if reverse:
                if industry == '万德全A':
                    ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}十亿元")
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}亿元")
            else:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}%")
            # 删除临时文件
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=2000)

    def plot_inflow_windA_irfs(self, daily_return_df, inflow_df, fig_name, reverse):
        inflow_5d = inflow_df.rolling(5).sum().dropna()
        index_intersection = daily_return_df.index.intersection(inflow_5d.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_5d.loc[index_intersection]

        # 创建Figure和Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0, wspace=0)
        plt.rcParams['axes.titlesize'] = 5

        for index, metric in enumerate(inflow_df.columns):
            if inflow_df[metric].eq(0).all():
                continue

            merged = pd.merge(daily_return_df['万德全A'], inflow_df[metric],
                              left_index=True, right_index=True, suffixes=('_return', '_inflow'))
            # 创建VAR模型
            # 剔除前面不连续的日期和最近一周的影响
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # 估计VAR模型
            # 5是bic给的值，而且平时也不会考虑过去两周的影响，更符合我们的应用场景。10的话就太zigzag了，很难解释
            results = model.fit(maxlags=5)

            # 提取单位冲击响应函数
            irf = results.irf(periods=30)  # 设定冲击响应函数的期数

            # 计算指定冲击和响应的累积冲击响应函数
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(f'万德全A'),
                                             merged.columns.get_loc(metric)])
            else:
                cumulative_response = np.sum(irf.irfs[:30, merged.columns.get_loc(metric),
                                             merged.columns.get_loc(f'万德全A')])

            # 绘制动态响应函数到临时文件
            filename = f"temp_plot_industry_{metric}.png"
            if reverse:
                irf.plot(impulse='万德全A', response=metric, signif=0.2)

            else:
                irf.plot(impulse=metric, response='万德全A', signif=0.2)
            plt.savefig(filename, dpi=100)
            plt.close()

            # 在Figure上显示该图像
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # 为每个子图添加标题，并在标题中包含累积冲击响应函数的值
            if reverse:
                ax.set_title(f"{metric}\nCumulative: {cumulative_response:.2f}亿元")
            else:
                ax.set_title(f"{metric}\nCumulative: {cumulative_response:.2f}%")
            # 删除临时文件
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=400, bbox_inches='tight')

    def get_best_order(self):
        # 设置阶数范围
        max_order = 10

        # 初始化最小AIC和最优阶数
        min_aic = float('inf')
        best_order = 0

        # 遍历不同阶数
        for order in range(1, max_order + 1):
            # 创建VAR模型
            model = sm.tsa.VAR(self.merged_agagregate_both[['万德全A', '四项流入之和_MA10']])

            # 估计VAR模型
            results = model.fit(order)

            # 计算AIC
            aic = results.aic

            # 判断是否为最小AIC
            if aic < min_aic:
                min_aic = aic
                best_order = order

        # 输出最优阶数
        print("最优阶数:", best_order)

