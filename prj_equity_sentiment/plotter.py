# coding=gbk
# Time Created: 2023/7/6 15:29
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
from prj_equity_sentiment.processor import Processor
from pgdb_manager import PgDbManager


class Plotter(PgDbManager):
    def __init__(self, base_config: BaseConfig, processed_data: Processor):
        super().__init__(base_config)
        self.processed_data = processed_data
        plt.rcParams['font.sans-serif'] = ['simhei']
        plt.rcParams['axes.unicode_minus'] = False
        self.read_data()
        self.make_plots()

    def make_plots(self):
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.finance_net_buy_percentile_industry, '情绪-两融各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.finance_net_buy_percentile_industry, '情绪-两融各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.north_percentile_industry, '情绪-北向各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.north_percentile_industry, '情绪-北向各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.big_order_inflow_percentile, '情绪-主力资金各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.big_order_inflow_percentile, '情绪-主力资金各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_quantile, '情绪-成交额各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_quantile, '情绪-成交额各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.turnover_quantile, '情绪-换手率各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.turnover_quantile, '情绪-换手率各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.shrink_rate, '情绪-缩量率各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.shrink_rate, '情绪-缩量率各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_proportion, '情绪-成交占比各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_proportion, '情绪-成交占比各行业逆irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_prop_quantile, '情绪-成交占比分位各行业irf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_prop_quantile, '情绪-成交占比分位各行业逆irf', reverse=True)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.market_breadth, '情绪-市场宽度全Airf', reverse=False)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.market_breadth, '情绪-市场宽度全A逆irf', reverse=True)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.rotation_strength, '情绪-轮动强度全Airf', reverse=False)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.rotation_strength, '情绪-轮动强度全A逆irf', reverse=True)

    def read_data(self):
        money_flow_dict, price_volume_dict, market_diverg_dict = self.processed_data

        # 仅总量
        self.market_breadth = market_diverg_dict['market_breadth_industry_level']*100
        self.rotation_strength = market_diverg_dict['rotation_strength']

        # 总量行业一起
        self.finance_net_buy_percentile_industry = money_flow_dict['finance_net_buy_percentile_industry'].rename(columns={'总额': '万德全A'})*100
        self.north_percentile_industry = money_flow_dict['north_percentile_industry'].rename(columns={'总额': '万德全A'})*100
        self.big_order_inflow_percentile = money_flow_dict['big_order_inflow_percentile']*100

        self.amt_quantile = price_volume_dict['amt_quantile']*100
        self.turnover_quantile = price_volume_dict['turnover_quantile']*100
        self.shrink_rate = price_volume_dict['shrink_rate']*100

        # 仅行业
        self.amt_proportion = price_volume_dict['amt_proportion']*100
        self.amt_prop_quantile = price_volume_dict['amt_prop_quantile']*100

        # 市场行情
        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='收盘价' and product_type='index'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.daily_return_ts = price_ts.pct_change().dropna(how='all')*100

    def plot_indicator_industry_irfs(self, daily_return_df, indicator_df, fig_name, reverse):
        # 日期对齐 保留交集部分的行
        index_intersection = daily_return_df.index.intersection(indicator_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        indicator_df = indicator_df.loc[index_intersection]
        # 创建Figure和Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 7, figure=fig, hspace=0, wspace=0)  # 这里的hspace设置了行间距，你可以根据需要调整它的值
        plt.rcParams['axes.titlesize'] = 5

        for index, industry in enumerate(indicator_df.columns):
            if indicator_df[industry].eq(0).all():
                continue

            merged = pd.merge(daily_return_df[industry], indicator_df[industry],
                              left_index=True, right_index=True, suffixes=('_return', '_indicator')).dropna()
            # 创建VAR模型
            # 剔除前面不连续的日期和最近一周的影响
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # 估计VAR模型
            # 5是bic给的值，而且平时也不会考虑过去两周的影响，更符合我们的应用场景。10的话就太zigzag了，很难解释
            results = model.fit(maxlags=5)

            # 提取单位冲击响应函数
            irf = results.irf(periods=25)  # 设定冲击响应函数的期数

            # 计算指定冲击和响应的累积冲击响应函数
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{industry}_return'),
                                             merged.columns.get_loc(f'{industry}_indicator')])
            else:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{industry}_indicator'),
                                             merged.columns.get_loc(f'{industry}_return')])

            # 绘制动态响应函数到临时文件
            filename = f"temp_plot_industry_{industry}.png"
            if reverse:
                irf.plot(impulse=f'{industry}_return', response=f'{industry}_indicator', signif=0.2)

            else:
                irf.plot(impulse=f'{industry}_indicator', response=f'{industry}_return', signif=0.2)
                # if industry == '万德全A':
                #     plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # 在Figure上显示该图像
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # 为每个子图添加标题，并在标题中包含累积冲击响应函数的值
            if reverse:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}逆向%")
            else:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}%")
            # 删除临时文件
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=2000)

    def plot_indicator_overall_irfs(self, daily_return_df, indicator_df, fig_name, reverse):
        # 日期对齐 保留交集部分的行
        index_intersection = daily_return_df.index.intersection(indicator_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        indicator_df = indicator_df.loc[index_intersection]
        # 创建Figure和Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0, wspace=0)  # 这里的hspace设置了行间距，你可以根据需要调整它的值
        plt.rcParams['axes.titlesize'] = 5

        for index, indicator in enumerate(indicator_df.columns):
            if indicator_df[indicator].eq(0).all():
                continue

            merged = pd.merge(daily_return_df['万德全A'], indicator_df[indicator],
                              left_index=True, right_index=True).dropna()
            # 创建VAR模型
            # 剔除前面不连续的日期和最近一周的影响
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # 估计VAR模型
            # 5是bic给的值，而且平时也不会考虑过去两周的影响，更符合我们的应用场景。10的话就太zigzag了，很难解释
            results = model.fit(maxlags=5)

            # 提取单位冲击响应函数
            irf = results.irf(periods=25)  # 设定冲击响应函数的期数

            # 计算指定冲击和响应的累积冲击响应函数
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'万德全A'),
                                             merged.columns.get_loc(f'{indicator}')])
            else:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{indicator}'),
                                             merged.columns.get_loc(f'万德全A')])

            # 绘制动态响应函数到临时文件
            filename = f"temp_plot_industry_{indicator}.png"
            if reverse:
                irf.plot(impulse=f'万德全A', response=f'{indicator}', signif=0.2)

            else:
                irf.plot(impulse=f'{indicator}', response=f'万德全A', signif=0.2)
                # if industry == '万德全A':
                #     plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # 在Figure上显示该图像
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # 为每个子图添加标题，并在标题中包含累积冲击响应函数的值
            if reverse:
                ax.set_title(f"{indicator}\nCumulative: {cumulative_response:.2f}逆向%")
            else:
                ax.set_title(f"{indicator}\nCumulative: {cumulative_response:.2f}%")
            # 删除临时文件
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=400)
