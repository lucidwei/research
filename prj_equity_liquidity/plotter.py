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
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '���ڸ���ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.margin_inflow_ts, '���ڸ���ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '�������ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.north_inflow_ts, '�������ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '�����ܺ͸���ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.aggregate_inflow_ts, '�����ܺ͸���ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf����ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.etf_inflow_ts, 'etf����ҵirf', reverse=False)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '��ɶ��仯����ҵ��irf', reverse=True)
        # self.plot_inflow_industry_irfs(self.daily_return_ts, self.holder_change_ts, '��ɶ��仯����ҵirf', reverse=False)
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

    def read_data(self):
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
        index_intersection = daily_return_df.index.intersection(inflow_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_df.loc[index_intersection]
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
        # ���ڶ��� �����������ֵ���
        index_intersection = daily_return_df.index.intersection(inflow_df.index)
        daily_return_df = daily_return_df.loc[index_intersection]
        inflow_df = inflow_df.loc[index_intersection]
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

