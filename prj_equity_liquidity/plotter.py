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
from prj_equity_liquidity.db_updater import DatabaseUpdater


class Plotter(DatabaseUpdater):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        plt.rcParams['font.sans-serif'] = ['simhei']
        plt.rcParams['axes.unicode_minus'] = False
        self.read_data()
        # self.get_best_order()
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '两融各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '北向各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '四项总和各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '四项总和各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf各行业irf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '大股东变化各行业逆irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '大股东变化各行业irf', reverse=False)
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

    def read_data(self):
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
        index_intersection = daily_return_df.index.intersection(inflow_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_df.loc[index_intersection]
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
        # 日期对齐 保留交集部分的行
        index_intersection = daily_return_df.index.intersection(inflow_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_df.loc[index_intersection]
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

