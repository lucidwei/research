# coding=gbk
# Time Created: 2024/4/8 17:28
# Author  : Lucid
# FileName: modeler.py
# Software: PyCharm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from preprocess import DataPreprocessor
from datetime import datetime

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


class DynamicFactorModeler:
    def __init__(self, preprocessor: DataPreprocessor, k_factors: int, factor_orders: int, financial: str):
        """
        DynamicFactorMQ 建模和评估类的初始化方法
        :param data: 预处理后的数据,DataFrame 格式
        :param k_factors: 因子数量
        :param financial: 金融指标序列,用于评估模型效果,Series 格式
        """
        self.preprocessor = preprocessor
        self.data = preprocessor.data
        self.financial = preprocessor.df_finalcials[financial]
        self.k_factors = k_factors
        self.factor_orders = factor_orders

    def apply_dynamic_factor_model(self):
        """
        应用 DynamicFactorMQ 模型进行建模和计算
        """
        em_kwargs = {
            'tolerance': 1e-7,  # 设置收敛阈值
        }
        model = DynamicFactorMQ(self.data, factors=self.k_factors, factor_orders=self.factor_orders, idiosyncratic_ar1=False)

        self.results = model.fit_em(maxiter=1000)
        print(self.results.summary())
        # fitted_data用来观察补全后的空值（但对原始数据变化很大）
        self.fitted_data = self.results.predict()

    def evaluate_model(self):
        """
        评估模型效果,计算提取的因子与金融指标的相关系数
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        self.extracted_factor = extracted_factor

        # 将 self.financial 的索引转换为月频
        financial_monthly = self.financial.resample('M').last()

        # 对齐两个时间序列的索引
        combined_data = pd.merge(extracted_factor, financial_monthly, left_index=True, right_index=True, how='inner')
        combined_data = combined_data.dropna()
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze().astype(float)

        corr = np.corrcoef(extracted_factor_filtered[15:], factor_filtered[15:])[0, 1]
        print(f"后期Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered[:15], factor_filtered[:15])[0, 1]
        print(f"早期Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")
        self.corr = corr

    def analyze_factor_contribution(self, start_date, end_date):
        """
        分析给定时间段内各变量对共同因子变化的贡献
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        # 获取平滑后的状态分解
        decomposition = self.results.get_smoothed_decomposition(decomposition_of='smoothed_state')
        data_contributions = decomposition[0].loc[pd.IndexSlice['0', :], :]

        # 将日期转换为 DataFrame 的行索引
        data_contributions.index = data_contributions.index.droplevel(0)

        # 自动计算默认的起止日期为最后三个月
        if start_date is None or end_date is None:
            end_date = data_contributions.index[-1]
            start_date = data_contributions.index[-3]
        print(f"Variable contributions to factor change from {start_date} to {end_date}:")

        # 提取给定时间段内的贡献
        factor_contributions = data_contributions.loc[start_date:end_date].T
        if self.corr < 0:
            factor_contributions *= -1

        df = factor_contributions.copy(deep=True)

        output = ""  # 存储所有的输出信息
        for column in df.columns:
            print(f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:")
            output += f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:\n"

            # 对当前列进行降序排序并去除nan值
            sorted_column = df[column].sort_values(ascending=False).dropna()

            # 获取前三个值及其Index
            head_values = sorted_column.head(3)
            print("Top 3 正贡献:")
            output += "Top 3 正贡献:\n"
            for index, value in head_values.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}\n"

            # 获取后三个值及其Index
            tail_values = sorted_column.tail(3)
            print("Bottom 3 负贡献:")
            output += "Bottom 3 负贡献:\n"
            for index, value in tail_values.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}\n"

            print("\n")
            output += "\n"

        # 增量分析：计算最后一次数据变化（从4月到5月）的贡献
        if len(df.columns) >= 2:
            prev_month = df.columns[-2]
            curr_month = df.columns[-1]

            print(
                f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}")
            output += f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}\n"

            # 计算每个变量的贡献变化
            contrib_change = df[curr_month] - df[prev_month]

            # 对贡献变化进行排序
            sorted_contrib_change = contrib_change.sort_values(ascending=False).dropna()

            print(f"对于{curr_month.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数变化:")
            output += f"对于{curr_month.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数变化:\n"

            # 获取前三个正贡献变化值及其Index
            top_positive_changes = sorted_contrib_change.head(3)
            print("Top 3 正贡献变化:")
            output += "Top 3 正贡献变化:\n"
            for index, value in top_positive_changes.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}\n"
            # 获取后三个负贡献变化值及其Index
            bottom_negative_changes = sorted_contrib_change.tail(3)
            print("Bottom 3 负贡献变化:")
            output += "Bottom 3 负贡献变化:\n"
            for index, value in bottom_negative_changes.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}\n"

            print("\n")
            output += "\n"

        return output

    def plot_factors(self, save_or_show='show'):
        """
        绘制提取的因子和原始因子的图像
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        if self.corr < 0:
            extracted_factor *= -1

        factor = self.financial.dropna().astype(float)

        # 对齐两个时间序列的索引
        combined_data = pd.merge(extracted_factor, factor, left_index=True,
                                 right_index=True, how='outer')
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # 绘制提取的因子和原始因子的图像
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(extracted_factor_filtered, label='景气综合指标')

        # 获取最新的两个时间点
        latest_dates = extracted_factor_filtered.index[-2:]
        # 在图中标注最新一期的日期范围
        start_date = latest_dates[0].strftime('%Y-%m-%d')
        end_date = latest_dates[1].strftime('%Y-%m-%d')
        latest_period_label = f"Latest Period: {start_date} to {end_date}"
        # 绘制最新一期数据变化的红线
        ax1.plot(latest_dates, extracted_factor_filtered[latest_dates], color='red', linewidth=2, label=latest_period_label)

        # 创建第二个 y 轴
        ax2 = ax1.twinx()
        ax2.scatter(factor_filtered.index, factor_filtered.values, label=factor.name, color='red')

        # 绘制每年的纵向栅格
        years = sorted(set(dt.year for dt in extracted_factor_filtered.index))
        for year in years:
            ax1.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', linewidth=0.8)

        # 设置第一个 y 轴标签
        ax1.set_ylabel('景气综合指标')

        # 设置第二个 y 轴标签
        ax2.set_ylabel(factor.name)

        # 合并两个 y 轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(rf'{self.preprocessor.industry}')
        if save_or_show == 'show':
            plt.show()
        elif save_or_show == 'save':
            # 保存图像到文件
            current_time = datetime.now().strftime("%Y%m%d_%H%M")
            stationary_flag = 'stationary' if self.preprocessor.stationary else 'fast'
            image_path = rf'{self.preprocessor.base_config.excels_path}/景气/pics/{self.preprocessor.industry}_{stationary_flag}_factor_plot_{current_time}.png'
            plt.savefig(image_path)
            plt.close(fig)

            return image_path

    def run(self):
        """
        运行 DynamicFactorMQ 建模和评估的完整流程
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
        # 分析给定时间段内各变量对共同因子变化的贡献，默认为最后三个月
        self.analyze_factor_contribution(None, None)
        self.plot_factors(save_or_show='show')
