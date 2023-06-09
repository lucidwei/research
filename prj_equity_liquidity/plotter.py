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


class Plotter(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        plt.rcParams['font.sans-serif'] = ['simhei']
        plt.rcParams['axes.unicode_minus']=False
        self.read_data()
        # self.get_best_order()
        # self.plot_irf()
        # self.plot_inflow_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业逆irf', reverse=True)
        # self.plot_inflow_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业逆irf', reverse=True)
        # self.plot_inflow_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业irf', reverse=False)
        # self.plot_inflow_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业irf', reverse=False)
        self.plot_inflow_irfs(self.daily_return_ts, self.margin_inflow_extreme_ts, '极端两融各行业irf', reverse=False)
        self.plot_inflow_irfs(self.daily_return_ts, self.north_inflow_extreme_ts, '极端北向各行业irf', reverse=False)

    def calculate_ts_extreme(self, ts_df, upper=0.8, lower=0.2):
        processed_df = ts_df.copy()

        for column in processed_df.columns:
            quantile_80 = processed_df[column].quantile(upper)
            quantile_20 = processed_df[column].quantile(lower)

            processed_df.loc[processed_df[column] > quantile_80, column] = processed_df[column]
            processed_df.loc[processed_df[column] < quantile_20, column] = processed_df[column]
            processed_df.loc[(processed_df[column] >= quantile_20) & (processed_df[column] <= quantile_80), column] = 0

        return processed_df

    def read_data(self):
        MA10_margin_inflow_long = self.read_table_from_schema('processed_data', 'MA10_margin_inflow')
        self.margin_inflow_ts = MA10_margin_inflow_long.pivot(index='date', columns='中信一级行业', values='两融净买入')/1e8
        self.margin_inflow_ts['万德全A'] = self.margin_inflow_ts.sum(axis=1)
        self.margin_inflow_extreme_ts = self.calculate_ts_extreme(self.margin_inflow_ts, 0.9, 0.1)
        self.margin_inflow_ma10_ts = MA10_margin_inflow_long.pivot(index='date', columns='中信一级行业', values='两融净买入_MA10')/1e8

        MA10_north_inflow_long = self.read_table_from_schema('processed_data', 'MA10_north_inflow')
        self.north_inflow_ts = MA10_north_inflow_long.pivot(index='date', columns='中信一级行业', values='北向净买入')/1e8
        self.north_inflow_ts['万德全A'] = self.north_inflow_ts.sum(axis=1)
        self.north_inflow_extreme_ts = self.calculate_ts_extreme(self.north_inflow_ts, 0.9, 0.1)
        self.north_inflow_ma10_ts = MA10_north_inflow_long.pivot(index='date', columns='中信一级行业', values='北向净买入_MA10')/1e8

        MA10_aggregate_inflow_long = self.read_table_from_schema('processed_data', 'MA10_aggregate_inflow')
        MA10_aggregate_inflow_ts = MA10_aggregate_inflow_long.pivot(index='date', columns='var_name', values='value')/1e8

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

        # 合并涨跌幅数据和净流入数据
        # self.merged_agagregate_both = pd.merge(self.daily_return_ts, MA10_aggregate_inflow_ts, left_index=True, right_index=True)
        # self.merged_margin = pd.merge(self.daily_return_ts, margin_inflow_ts, left_index=True, right_index=True)
        # self.merged_north = pd.merge(self.daily_return_ts, north_inflow_ts, left_index=True, right_index=True)

    # def plot_irf(self):
    #     # 创建VAR模型
    #     model = sm.tsa.VAR(self.merged_agagregate_both[['万德全A', '四项流入之和_MA10']])
    #
    #     # 创建Figure和Subplot
    #     fig = plt.figure(constrained_layout=True)
    #     gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.05)  # 这里的hspace设置了行间距，你可以根据需要调整它的值
    #
    #     for order in range(1, 11):
    #         # 估计VAR模型
    #         results = model.fit(order)
    #
    #         # 提取单位冲击响应函数
    #         irf = results.irf(30)  # 设定冲击响应函数的期数
    #
    #         # 绘制动态响应函数到临时文件
    #         filename = f"temp_plot_order_{order}.png"
    #         irf.plot(impulse='万德全A', response='四项流入之和_MA10')
    #         plt.savefig(filename)
    #         plt.close()
    #
    #         # 在Figure上显示该图像
    #         ax = fig.add_subplot(gs[order - 1])
    #         img = mpimg.imread(filename)
    #         ax.imshow(img)
    #         ax.axis('off')
    #
    #         # 为每个子图添加标题
    #         ax.set_title(f"order={order}")
    #
    #         # 删除临时文件
    #         os.remove(filename)
    #
    #     plt.show()

    def plot_inflow_irfs(self, daily_return_df, inflow_df, fig_name, reverse):
        # 创建Figure和Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 7, figure=fig, hspace=0, wspace=0)  # 这里的hspace设置了行间距，你可以根据需要调整它的值
        plt.rcParams['axes.titlesize'] = 5

        for index, industry in enumerate(daily_return_df.columns):
            merged = pd.merge(daily_return_df[industry], inflow_df[industry],
                              left_index=True, right_index=True, suffixes=('_return', '_inflow'))
            # 创建VAR模型
            model = sm.tsa.VAR(merged)

            # 估计VAR模型
            results = model.fit(6)

            # 提取单位冲击响应函数
            irf = results.irf(30)  # 设定冲击响应函数的期数

            # 计算累积冲击响应函数
            # cumulative_irf = np.cumsum(irf.irfs, axis=0)

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
                irf.plot(impulse=f'{industry}_return', response=f'{industry}_inflow')
            else:
                irf.plot(impulse=f'{industry}_inflow', response=f'{industry}_return')
            plt.savefig(filename, dpi=60)
            plt.close()

            # 在Figure上显示该图像
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # 为每个子图添加标题，并在标题中包含累积冲击响应函数的值
            if reverse:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}亿元")
            else:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}%")
            # 删除临时文件
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=2000)


    # def plot_irf(self):
    #     # 创建VAR模型
    #     model = sm.tsa.VAR(self.merged_data[['万德全A', '四项流入之和_MA10']])
    #
    #     # 估计VAR模型
    #     order = 5  # VAR模型阶数
    #     results = model.fit(order)
    #
    #     # 提取单位冲击响应函数
    #     irf = results.irf(30)  # 设定冲击响应函数的期数
    #
    #     # 绘制动态响应函数
    #     # irf.plot(impulse='四项流入之和_MA10', response='万德全A')
    #     irf.plot(impulse='万德全A', response='四项流入之和_MA10')
    #     plt.show()

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

