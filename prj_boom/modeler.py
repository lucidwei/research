# coding=gbk
# Time Created: 2024/4/8 17:28
# Author  : Lucid
# FileName: modeler.py
# Software: PyCharm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from preprocess import DataPreprocessor
from datetime import datetime

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class DynamicFactorModeler:
    def __init__(self, preprocessor: DataPreprocessor, k_factors: int, factor_orders: int, compare_to: str):
        """
        DynamicFactorMQ 建模和评估类的初始化方法
        :param data: 预处理后的数据,DataFrame 格式
        :param k_factors: 因子数量
        :param compare_to: 金融指标序列,用于评估模型效果,Series 格式
        """
        self.preprocessor = preprocessor
        self.data = preprocessor.data
        self.k_factors = k_factors
        self.factor_orders = factor_orders

        # 在 df_finalcials 和 df_indicators 中寻找 compare_to 字符串
        if compare_to in preprocessor.df_finalcials:
            self.series_compared_to = preprocessor.df_finalcials[compare_to]
        elif compare_to in preprocessor.df_indicators:
            self.series_compared_to = preprocessor.df_indicators[compare_to]
        else:
            raise ValueError(f"'{compare_to}' not found in df_finalcials or df_indicators")

        self.find_statistically_significant_leading_indicators()

    def find_statistically_significant_leading_indicators(self, max_lag=5, alpha=0.05, resample_freq='M'):
        """
        找到 df_indicators 中在统计学上显著领先于 compare_to 的时间序列
        :param max_lag: 最大滞后阶数
        :param alpha: 显著性水平
        :param resample_freq: 重新采样频率，例如 'M' 表示按月
        :return: 领先的时间序列名称及其显著滞后期数的字典
        """
        leading_indicators = {}

        # 重新采样 self.series_compared_to
        series_compared_to_resampled = self.series_compared_to.resample(resample_freq).mean().interpolate()

        # 遍历 df_indicators 中的每个时间序列
        for column in self.preprocessor.df_indicators.columns:
            # 重新采样每个指标时间序列
            indicator_series_resampled = self.preprocessor.df_indicators[column].resample(resample_freq).mean().interpolate()

            combined_data = pd.concat([series_compared_to_resampled, indicator_series_resampled], axis=1).dropna()

            # 检查数据长度是否足够
            if combined_data.shape[0] <= max_lag:
                print(f"Skipping {column} due to insufficient data length.")
                continue

            try:
                test_result = grangercausalitytests(combined_data, max_lag, verbose=False)

                # 检查每个滞后阶数下的 F-检验 p 值
                for lag in range(1, max_lag + 1):
                    p_value = test_result[lag][0]['ssr_ftest'][1]
                    if p_value < alpha:
                        leading_indicators[column] = lag
                        break  # 如果在任何滞后阶数下显著，则添加该列并跳出循环
            except ValueError as e:
                print(f"Error processing {column}: {e}")
                continue

        return leading_indicators

    def apply_dynamic_factor_model(self):
        """
        应用 DynamicFactorMQ 模型进行建模和计算
        """
        em_kwargs = {
            'tolerance': 1e-7,  # 设置收敛阈值
        }
        model = DynamicFactorMQ(self.data, factors=self.k_factors, factor_orders=self.factor_orders,
                                idiosyncratic_ar1=False)

        self.results = model.fit_em(maxiter=1000)
        print(self.results.summary())

        # 提取因子载荷（factor loadings），即每个观察变量的权重
        num_loadings = len([param for param in self.results.params.index if 'loading' in param])
        self.factor_loadings = self.results.params[:num_loadings]
        self.factor_loadings.index = self.factor_loadings.index.str.replace('loading.0->', '', regex=False)
        # 提取状态转移矩阵，历史数据对当前因子的影响通过状态方程和滞后项来实现，而不是通过因子载荷。
        # self.transition_matrix = self.results.transition

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
        financial_monthly = self.series_compared_to.resample('M').last()

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

    def analyze_factor_contribution(self, start_date=None, end_date=None):
        """
        分析给定时间段内各变量对共同因子变化的贡献
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        # 获取平滑后的状态分解
        decomposition = self.results.get_smoothed_decomposition(decomposition_of='smoothed_state')
        data_contributions = decomposition[0].loc[pd.IndexSlice['0', :], :]
        # decomposition = self.results.filtered_state
        # dates = self.results.data.dates  # 获取时间序列的日期
        # variables = self.results.data.param_names  # 获取变量名称
        # data_contributions = pd.DataFrame(data=decomposition.T, index=dates, columns=variables)


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
            self.factor_loadings *= -1

        factor_contributions_adjusted = factor_contributions.copy(deep=True)

        # 新增：存储各个变量的权重
        output = f"各指标权重：{self.factor_loadings}\n"  # 存储所有的输出信息
        print(output)
        for column in factor_contributions_adjusted.columns:
            print(f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:")
            output += f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:\n"

            # 对当前列进行降序排序并去除nan值
            sorted_column = factor_contributions_adjusted[column].sort_values(ascending=False).dropna()

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
        if len(factor_contributions_adjusted.columns) >= 2:
            prev_month = factor_contributions_adjusted.columns[-2]
            curr_month = factor_contributions_adjusted.columns[-1]

            print(
                f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}")
            output += f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}\n"

            # 计算每个变量的贡献变化
            contrib_change = factor_contributions_adjusted[curr_month] - factor_contributions_adjusted[prev_month]

            # 不应再对contrib_change取负，因为factor_contributions_adjusted已经调整了符号
            # if self.corr < 0:
            #     contrib_change *= -1

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

            # 计算所有正贡献变化的总和
            positive_sum = sorted_contrib_change[sorted_contrib_change > 0].sum()
            print(f"正贡献变化总和: {positive_sum:.3f}")
            output += f"正贡献变化总和: {positive_sum:.3f}\n"

            # 计算所有负贡献变化的总和
            negative_sum = sorted_contrib_change[sorted_contrib_change < 0].sum()
            print(f"负贡献变化总和: {negative_sum:.3f}")
            output += f"负贡献变化总和: {negative_sum:.3f}\n"

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

        factor = self.series_compared_to.dropna().astype(float)

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
        ax1.plot(latest_dates, extracted_factor_filtered[latest_dates], color='red', linewidth=2,
                 label=latest_period_label)

        # 创建第二个 y 轴
        ax2 = ax1.twinx()
        # 判断 NaN 的数量是否多于一半
        nan_count = factor_filtered.isna().sum()
        total_count = len(factor_filtered)

        if nan_count > total_count / 2 or self.preprocessor.industry == '社零综指':
            ax2.scatter(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red')
        else:
            ax2.plot(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red')

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
