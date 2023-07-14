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
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.finance_net_buy_percentile_industry, '����-���ڸ���ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.finance_net_buy_percentile_industry, '����-���ڸ���ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.north_percentile_industry, '����-�������ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.north_percentile_industry, '����-�������ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.big_order_inflow_percentile, '����-�����ʽ����ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.big_order_inflow_percentile, '����-�����ʽ����ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_quantile, '����-�ɽ������ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_quantile, '����-�ɽ������ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.turnover_quantile, '����-�����ʸ���ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.turnover_quantile, '����-�����ʸ���ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.shrink_rate, '����-�����ʸ���ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.shrink_rate, '����-�����ʸ���ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_proportion, '����-�ɽ�ռ�ȸ���ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_proportion, '����-�ɽ�ռ�ȸ���ҵ��irf', reverse=True)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_prop_quantile, '����-�ɽ�ռ�ȷ�λ����ҵirf', reverse=False)
        # self.plot_indicator_industry_irfs(self.daily_return_ts, self.amt_prop_quantile, '����-�ɽ�ռ�ȷ�λ����ҵ��irf', reverse=True)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.market_breadth, '����-�г����ȫAirf', reverse=False)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.market_breadth, '����-�г����ȫA��irf', reverse=True)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.rotation_strength, '����-�ֶ�ǿ��ȫAirf', reverse=False)
        self.plot_indicator_overall_irfs(self.daily_return_ts, self.rotation_strength, '����-�ֶ�ǿ��ȫA��irf', reverse=True)

    def read_data(self):
        money_flow_dict, price_volume_dict, market_diverg_dict = self.processed_data

        # ������
        self.market_breadth = market_diverg_dict['market_breadth_industry_level']*100
        self.rotation_strength = market_diverg_dict['rotation_strength']

        # ������ҵһ��
        self.finance_net_buy_percentile_industry = money_flow_dict['finance_net_buy_percentile_industry'].rename(columns={'�ܶ�': '���ȫA'})*100
        self.north_percentile_industry = money_flow_dict['north_percentile_industry'].rename(columns={'�ܶ�': '���ȫA'})*100
        self.big_order_inflow_percentile = money_flow_dict['big_order_inflow_percentile']*100

        self.amt_quantile = price_volume_dict['amt_quantile']*100
        self.turnover_quantile = price_volume_dict['turnover_quantile']*100
        self.shrink_rate = price_volume_dict['shrink_rate']*100

        # ����ҵ
        self.amt_proportion = price_volume_dict['amt_proportion']*100
        self.amt_prop_quantile = price_volume_dict['amt_prop_quantile']*100

        # �г�����
        df_price_joined = self.read_joined_table_as_dataframe(
            target_table_name='markets_daily_long',
            target_join_column='product_static_info_id',
            join_table_name='product_static_info',
            join_column='internal_id',
            filter_condition="field='���̼�' and product_type='index'"
        )
        df_price_joined = df_price_joined[['date', 'chinese_name', 'field', 'value']].rename(columns={'chinese_name': 'industry'})
        price_ts = df_price_joined.pivot(index='date', columns='industry', values='value')
        self.daily_return_ts = price_ts.pct_change().dropna(how='all')*100

    def plot_indicator_industry_irfs(self, daily_return_df, indicator_df, fig_name, reverse):
        # ���ڶ��� �����������ֵ���
        index_intersection = daily_return_df.index.intersection(indicator_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        indicator_df = indicator_df.loc[index_intersection]
        # ����Figure��Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 7, figure=fig, hspace=0, wspace=0)  # �����hspace�������м�࣬����Ը�����Ҫ��������ֵ
        plt.rcParams['axes.titlesize'] = 5

        for index, industry in enumerate(indicator_df.columns):
            if indicator_df[industry].eq(0).all():
                continue

            merged = pd.merge(daily_return_df[industry], indicator_df[industry],
                              left_index=True, right_index=True, suffixes=('_return', '_indicator')).dropna()
            # ����VARģ��
            # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # ����VARģ��
            # 5��bic����ֵ������ƽʱҲ���ῼ�ǹ�ȥ���ܵ�Ӱ�죬���������ǵ�Ӧ�ó�����10�Ļ���̫zigzag�ˣ����ѽ���
            results = model.fit(maxlags=5)

            # ��ȡ��λ�����Ӧ����
            irf = results.irf(periods=25)  # �趨�����Ӧ����������

            # ����ָ���������Ӧ���ۻ������Ӧ����
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{industry}_return'),
                                             merged.columns.get_loc(f'{industry}_indicator')])
            else:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{industry}_indicator'),
                                             merged.columns.get_loc(f'{industry}_return')])

            # ���ƶ�̬��Ӧ��������ʱ�ļ�
            filename = f"temp_plot_industry_{industry}.png"
            if reverse:
                irf.plot(impulse=f'{industry}_return', response=f'{industry}_indicator', signif=0.2)

            else:
                irf.plot(impulse=f'{industry}_indicator', response=f'{industry}_return', signif=0.2)
                # if industry == '���ȫA':
                #     plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # ��Figure����ʾ��ͼ��
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # Ϊÿ����ͼ��ӱ��⣬���ڱ����а����ۻ������Ӧ������ֵ
            if reverse:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}����%")
            else:
                ax.set_title(f"{industry}\nCumulative: {cumulative_response:.2f}%")
            # ɾ����ʱ�ļ�
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=2000)

    def plot_indicator_overall_irfs(self, daily_return_df, indicator_df, fig_name, reverse):
        # ���ڶ��� �����������ֵ���
        index_intersection = daily_return_df.index.intersection(indicator_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        indicator_df = indicator_df.loc[index_intersection]
        # ����Figure��Subplot
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0, wspace=0)  # �����hspace�������м�࣬����Ը�����Ҫ��������ֵ
        plt.rcParams['axes.titlesize'] = 5

        for index, indicator in enumerate(indicator_df.columns):
            if indicator_df[indicator].eq(0).all():
                continue

            merged = pd.merge(daily_return_df['���ȫA'], indicator_df[indicator],
                              left_index=True, right_index=True).dropna()
            # ����VARģ��
            # �޳�ǰ�治���������ں����һ�ܵ�Ӱ��
            merged = merged[5:-5]
            model = sm.tsa.VAR(merged)
            # a = model.select_order()

            # ����VARģ��
            # 5��bic����ֵ������ƽʱҲ���ῼ�ǹ�ȥ���ܵ�Ӱ�죬���������ǵ�Ӧ�ó�����10�Ļ���̫zigzag�ˣ����ѽ���
            results = model.fit(maxlags=5)

            # ��ȡ��λ�����Ӧ����
            irf = results.irf(periods=25)  # �趨�����Ӧ����������

            # ����ָ���������Ӧ���ۻ������Ӧ����
            if not reverse:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'���ȫA'),
                                             merged.columns.get_loc(f'{indicator}')])
            else:
                cumulative_response = np.sum(irf.irfs[:25, merged.columns.get_loc(f'{indicator}'),
                                             merged.columns.get_loc(f'���ȫA')])

            # ���ƶ�̬��Ӧ��������ʱ�ļ�
            filename = f"temp_plot_industry_{indicator}.png"
            if reverse:
                irf.plot(impulse=f'���ȫA', response=f'{indicator}', signif=0.2)

            else:
                irf.plot(impulse=f'{indicator}', response=f'���ȫA', signif=0.2)
                # if industry == '���ȫA':
                #     plt.ylim(-0.1, 0.1)
            plt.savefig(filename, dpi=60)
            plt.close()

            # ��Figure����ʾ��ͼ��
            ax = fig.add_subplot(gs[index])
            img = mpimg.imread(filename)
            ax.imshow(img, interpolation='none')
            ax.axis('off')

            # Ϊÿ����ͼ��ӱ��⣬���ڱ����а����ۻ������Ӧ������ֵ
            if reverse:
                ax.set_title(f"{indicator}\nCumulative: {cumulative_response:.2f}����%")
            else:
                ax.set_title(f"{indicator}\nCumulative: {cumulative_response:.2f}%")
            # ɾ����ʱ�ļ�
            os.remove(filename)
        print(f'{fig_name} saved!')
        plt.savefig(self.base_config.image_folder+fig_name, dpi=400)
