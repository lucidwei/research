# coding=gbk
# Time Created: 2024/4/8 17:28
# Author  : Lucid
# FileName: modeler.py
# Software: PyCharm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
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
    def __init__(self, preprocessor: DataPreprocessor, k_factors: int, factor_orders: int, compare_to: str,
                 leading_prediction=False, single_line=False, plot_y0=False):
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
        self.leading_prediction = leading_prediction
        self.single_line = single_line
        self.plot_y0 = plot_y0

        # 在 df_finalcials 和 df_indicators 中寻找 compare_to 字符串
        if compare_to in preprocessor.df_finalcials:
            self.series_compared_to = preprocessor.df_finalcials[compare_to]
        elif compare_to in preprocessor.df_indicators:
            self.series_compared_to = preprocessor.df_indicators[compare_to]
        else:
            raise ValueError(f"'{compare_to}' not found in df_finalcials or df_indicators")

    def run(self):
        """
        运行 DynamicFactorMQ 建模和评估的完整流程
        """
        if self.single_line:
            mannual_indicators_group = {
                'PPI': {
                    '中国:M1:同比': 9,
                    '中国:实体经济部门杠杆率:同比增加': 9,
                },
                '出口': {'上海出口集装箱运价指数:综合指数:同比': 4,
                         '美国:销售总额:季调:同比-美国:库存总额:季调:同比': 2,
                         '美国:库存销售比:季调': 15,
                         '期货收盘价(连续):COMEX铜:同比:月:平均值': 3}
            }
            # 先计算同步，再计算leading，再解决画图问题
            results_concurrent = self.apply_dynamic_factor_model()
            results_leading = self.apply_dynamic_factor_model(mannual_indicators_group[self.preprocessor.industry])
            self.plot_factors_single_line(results_concurrent, results_leading)
            return

        if self.leading_prediction:
            leading_indicators, synchronous_indicators, _ = self.find_statistically_significant_indicators(
                # lag1_as_sync=True) #仅作展示各指标的领先期数用（获取字符串）
                lag1_as_sync=False)
            self.apply_dynamic_factor_model(leading_indicators)
        else:
            self.find_statistically_significant_indicators()
            self.apply_dynamic_factor_model()
        self.evaluate_model()
        # 分析给定时间段内各变量对共同因子变化的贡献，默认为最后三个月
        self.analyze_factor_contribution(None, None)
        self.plot_factors(save_or_show='show')

    def find_statistically_significant_indicators(self, max_lag=15, alpha=0.05, resample_freq='M', lag1_as_sync=True):
        """
        找到 df_indicators 中在统计学上显著领先于 compare_to 的时间序列, 并筛选出同步指标
        :param max_lag: 最大滞后阶数
        :param alpha: 显著性水平
        :param resample_freq: 重新采样频率，例如 'M' 表示按月
        :return: 领先的时间序列及其显著滞后期数的字典, 同步时间序列及其显著相关性的字典, 被遗弃的时间序列列表
        """
        leading_indicators = {}
        synchronous_indicators = {}
        discarded_indicators = []

        # 重新采样 self.series_compared_to
        series_compared_to_resampled = self.series_compared_to.resample(resample_freq).mean().interpolate()

        # 遍历 df_indicators 中的每个时间序列
        for column in self.preprocessor.df_indicators.columns:
            # 重新采样每个指标时间序列
            indicator_series_resampled = self.preprocessor.df_indicators[column].resample(
                resample_freq).mean().interpolate()

            combined_data = pd.concat([series_compared_to_resampled, indicator_series_resampled], axis=1).dropna()

            # 检查数据长度是否足够
            if combined_data.shape[0] <= max_lag:
                print(f"Skipping {column} due to insufficient data length.")
                discarded_indicators.append(column)
                self.preprocessor.data.drop(columns=[column], inplace=True)
                continue

            try:
                # 同步性检验
                correlation_sync, pear_p_value = pearsonr(combined_data.iloc[:, 0], combined_data.iloc[:, 1])
                if pear_p_value < alpha:
                    synchronous_indicators[column] = correlation_sync

                best_lag = 0
                highest_correlation = correlation_sync

                # 遍历不同的滞后期数，计算滞后后的相关性
                for lag in range(0, max_lag + 1):
                    # 滞后处理
                    lagged_data = combined_data.copy()
                    lagged_data.iloc[:, 1] = lagged_data.iloc[:, 1].shift(lag)
                    lagged_data = lagged_data.dropna()

                    # 计算滞后后的相关性
                    correlation, p_val = pearsonr(lagged_data.iloc[:, 0], lagged_data.iloc[:, 1])

                    # 选择相关性最高的滞后期数，不关注显著性
                    if correlation > highest_correlation:
                        highest_correlation = correlation
                        best_lag = lag

                if best_lag == 0 and pear_p_value <= alpha:
                    print(f'{column}仅为同步指标')
                elif best_lag == 1 and lag1_as_sync:
                    print(f'{column}为领先1期，被视为同步指标')
                elif best_lag >= 1:
                    leading_indicators[column] = best_lag
                elif best_lag == 0 and pear_p_value >= alpha:
                    discarded_indicators.append(column)
                    self.preprocessor.data.drop(columns=[column], inplace=True)
                else:
                    raise ValueError('Unexpected condition encountered')

                # # 使用最佳滞后期数绘制时间序列图
                # if best_lag is not None:
                #     lagged_data = combined_data.copy()
                #     lagged_data.iloc[:, 1] = lagged_data.iloc[:, 1].shift(best_lag)
                #     lagged_data = lagged_data.dropna()
                #
                #     plt.figure(figsize=(12, 6))
                #     plt.plot(lagged_data.iloc[:, 0], label='Target')
                #     plt.plot(lagged_data.iloc[:, 1], label=f'Indicator (lag={best_lag})')
                #     plt.xlabel('Time')
                #     plt.ylabel('Value')
                #     plt.title(f'{column}Time Series Comparison')
                #     plt.legend()
                #     plt.show()
                # else:
                #     print("No valid lag found")

                # # 进行格兰杰因果检验
                # test_result = grangercausalitytests(combined_data, max_lag, verbose=False)

                # def plot_granger_causality_pvalues(combined_data, max_lag):
                #     p_values = []
                #     for lag in range(1, max_lag + 1):
                #         test_result = grangercausalitytests(combined_data, lag, verbose=False)
                #         p_value = test_result[lag][0]['ssr_ftest'][1]
                #         p_values.append(p_value)
                #
                #     plt.plot(range(1, max_lag + 1), p_values, marker='o')
                #     plt.xlabel('Lag')
                #     plt.ylabel('p-value')
                #     plt.title('Granger Causality Test p-values')
                #     plt.show()
                #
                # # 绘制不同滞后期数下p值的变化图
                # plot_granger_causality_pvalues(combined_data, max_lag=12)

                # # 检查每个滞后阶数下的 F-检验 p 值，选取显著性最高的期数
                # best_lag = None
                # best_p_value = float('inf')
                #
                # for lag in range(1, max_lag + 1):
                #     p_value = test_result[lag][0]['ssr_ftest'][1]
                #     if p_value < alpha and p_value < best_p_value:
                #         best_p_value = p_value
                #         best_lag = lag
                # if best_lag == 1 and lag1_as_sync:
                #     continue
                # elif best_lag is not None:
                #     leading_indicators[column] = best_lag
                # elif best_lag is None and pear_p_value >= alpha:
                #     discarded_indicators.append(column)
                # else:
                #     print(f'{column}仅为同步指标')
            except ValueError as e:
                print(f"Error processing {column}: {e}")
                discarded_indicators.append(column)
                continue

        print(f'领先指标(领先期数)：{leading_indicators}')
        print(
            f"同步指标(相关性)：{{{', '.join([f'{key}: {value:.2f}' for key, value in synchronous_indicators.items()])}}}")
        print(f'遗弃的指标：{discarded_indicators}')
        return leading_indicators, synchronous_indicators, discarded_indicators

    def apply_dynamic_factor_model(self, indicators_group=None):
        """
        应用 DynamicFactorMQ 模型进行建模和计算
        """
        if indicators_group is not None:
            # 从指标组中提取列名和对应的领先期数
            selected_columns = indicators_group.keys()
            leading_periods = indicators_group.values()
            periods_to_extend = max(leading_periods)

            # 创建一个新的 DataFrame 来存储推移后的数据
            future_dates = pd.date_range(start=self.data.index[-1], periods=periods_to_extend + 1, freq='M')[1:]
            extended_index = self.data.index.append(future_dates)
            extended_data = self.data.reindex(extended_index)

            shifted_data = pd.DataFrame(index=extended_index)

            for column, period in zip(selected_columns, leading_periods):
                # 尝试直接使用 column 进行匹配
                if column in extended_data.columns:
                    target_column = column
                else:
                    # 如果匹配不上，则尝试在前面加上 '(月度化)'
                    modified_column = f"(月度化){column}"
                    if modified_column in extended_data.columns:
                        target_column = modified_column
                    else:
                        raise ValueError(f"Column '{column}' or '{modified_column}' not found in extended_data")

                # 将列数据推移到未来
                # TODO 手动-1对齐高低点，具体原因未明
                if self.leading_prediction:
                    shifted_series = extended_data[target_column].shift(period - 1)
                else:
                    shifted_series = extended_data[target_column].shift(period)

                # 将推移后的系列添加到新的 DataFrame 中
                shifted_data[column] = shifted_series

            # 对齐新的日期索引，处理可能不存在的月末日期
            filtered_data = shifted_data.resample('M').asfreq().interpolate(method='time')
        else:
            # 如果没有指定指标组，使用全部数据
            filtered_data = self.data

        model = DynamicFactorMQ(filtered_data, factors=self.k_factors, factor_orders=self.factor_orders,
                                idiosyncratic_ar1=False)

        results = model.fit_em(maxiter=1000)
        print(results.summary())
        self.results = results

        # 提取因子载荷（factor loadings），即每个观察变量的权重
        num_loadings = len([param for param in self.results.params.index if 'loading' in param])
        self.factor_loadings = self.results.params[:num_loadings]
        self.factor_loadings.index = self.factor_loadings.index.str.replace('loading.0->', '', regex=False)
        # 提取状态转移矩阵，历史数据对当前因子的影响通过状态方程和滞后项来实现，而不是通过因子载荷。
        # self.transition_matrix = self.results.transition

        # fitted_data用来观察补全后的空值（但对原始数据变化很大）
        self.fitted_data = self.results.predict()
        return results

    def evaluate_model(self):
        """
        评估模型效果,计算提取的因子与金融指标的相关系数
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        # 将 self.financial 的索引转换为月频
        financial_monthly = self.series_compared_to.resample('M').last()

        # 对齐两个时间序列的索引
        extracted_factor_filtered, factor_filtered = self.align_index_scale_corr(extracted_factor, financial_monthly,
                                                                                 'inner')

        corr = np.corrcoef(extracted_factor_filtered[15:], factor_filtered[15:])[0, 1]
        print(f"后期Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered[:15], factor_filtered[:15])[0, 1]
        print(f"早期Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")

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
        # start_date = '2024-03-31'
        # end_date = '2024-04-30'
        print(f"Variable contributions to factor change from {start_date} to {end_date}:")

        # 提取给定时间段内的贡献
        factor_contributions = data_contributions.loc[start_date:end_date].T

        # 新增：存储各个变量的权重
        output = f"各指标权重：{self.factor_loadings}\n"  # 存储所有的输出信息
        print(output)
        for column in factor_contributions.columns:
            print(f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:")
            output += f"对于{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} 景气度指数:\n"

            # 对当前列进行降序排序并去除nan值
            sorted_column = factor_contributions[column].sort_values(ascending=False).dropna()

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

        # 增量分析：计算最后一次(或adjust后的月份)数据变化的贡献
        adjust = 0
        if len(factor_contributions.columns) >= 2:
            prev_month = factor_contributions.columns[-2 - adjust]
            curr_month = factor_contributions.columns[-1 - adjust]

            print(
                f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}")
            output += f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}\n"

            # 计算每个变量的贡献变化
            contrib_change = factor_contributions[curr_month] - factor_contributions[prev_month]

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
        factor = self.series_compared_to.dropna().astype(float)

        extracted_factor_filtered, factor_filtered = self.align_index_scale_corr(extracted_factor, factor, 'outer')

        # 获取 factor_filtered 实际存在的真实数据中的最新日期
        latest_date_existing = factor_filtered.dropna().index.max()
        # 找到今天之后的所有日期
        predicted_dates = extracted_factor_filtered.index[extracted_factor_filtered.index > latest_date_existing]
        if len(predicted_dates) == 0:
            extracted_factor_filtered_without_predicted = extracted_factor_filtered
        else:
            extracted_factor_filtered_without_predicted = extracted_factor_filtered[
                extracted_factor_filtered.index < predicted_dates[0]]
        # predicted_dates添加一个历史日期，保证画预测虚线时的连贯
        prev_date = extracted_factor_filtered_without_predicted.index.max()
        predicted_dates = extracted_factor_filtered.index[extracted_factor_filtered.index >= prev_date]

        # 绘制提取的因子和原始因子的图像
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(extracted_factor_filtered_without_predicted, label='景气综合指标')

        # 添加一条纵坐标为0的虚线
        if self.plot_y0:
            ax1.axhline(y=0, color='gray', linestyle='--')

        # 在图中标注预测期的日期范围
        start_date = predicted_dates[0].strftime('%Y-%m-%d') if len(predicted_dates) == 1 else predicted_dates[
            1].strftime('%Y-%m-%d')
        end_date = predicted_dates[-1].strftime('%Y-%m-%d')
        latest_period_label = f"预测期: {start_date} to {end_date}" if start_date != end_date else f"预测期: {start_date}"
        # 绘制最新一期数据变化的红线
        ax1.plot(predicted_dates, extracted_factor_filtered[predicted_dates], color='purple', linewidth=3,
                 linestyle=':' if self.leading_prediction else '-', label=latest_period_label)

        # 创建第二个 y 轴
        ax2 = ax1.twinx()
        # 判断 NaN 的数量是否多于一半
        nan_count = factor_filtered.isna().sum()
        total_count = len(factor_filtered)

        if nan_count > total_count / 2 or self.preprocessor.industry == '社零综指':
            ax2.scatter(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red')
        else:
            ax2.plot(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red', alpha=0.6)

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

    def plot_factors_single_line(self, results_concurrent, results_leading):
        """
        绘制提取的因子和原始因子的图像
        """

        factor = self.series_compared_to.dropna().astype(float)
        extracted_factor_concurrent = results_concurrent.factors.filtered['0']
        extracted_factor_leading = results_leading.factors.filtered['0']

        extracted_factor_aligned_concurrent, factor_aligned_concurrent = self.align_index_scale_corr(
            extracted_factor_concurrent, factor, 'outer')
        extracted_factor_aligned_leading, factor_aligned_leading = self.align_index_scale_corr(extracted_factor_leading,
                                                                                               factor, 'outer')

        # 获取 factor_filtered 实际存在的真实数据中的最新日期
        latest_date_existing = factor.dropna().index.max()
        # 找到今天之后的所有日期
        predicted_dates_concurrent = extracted_factor_aligned_concurrent.index[
            extracted_factor_aligned_concurrent.index > latest_date_existing]
        predicted_dates_leading = extracted_factor_aligned_leading.index[
            extracted_factor_aligned_leading.index > latest_date_existing]

        # 针对同步指标可能不存在当月预测值的情况
        if len(predicted_dates_concurrent) == 0:
            extracted_factor_concurrent_without_predicted = extracted_factor_aligned_concurrent
        else:
            extracted_factor_concurrent_without_predicted = extracted_factor_aligned_concurrent[
                extracted_factor_aligned_concurrent.index < predicted_dates_concurrent[0]]
        # predicted_dates添加一个历史日期，保证画预测虚线时的连贯
        prev_date = extracted_factor_concurrent_without_predicted.index.max()
        predicted_dates = extracted_factor_aligned_concurrent.index[
            extracted_factor_aligned_concurrent.index >= prev_date]

        # 绘制提取的因子预测和原始因子的图像
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(factor, label=f'{self.preprocessor.industry}(历史)')

        # 在图中标注预测期的日期范围
        start_date = predicted_dates[0].strftime('%Y-%m-%d') if len(predicted_dates) == 1 else predicted_dates[
            1].strftime('%Y-%m-%d')
        end_date = predicted_dates[-1].strftime('%Y-%m-%d')
        latest_period_label = f"当期预测: {start_date} to {end_date}" if start_date != end_date else f"当期预测: {start_date}"
        # 调整数值保证连贯
        extracted_factor_aligned_concurrent -= extracted_factor_aligned_concurrent.loc[predicted_dates[0]] - factor.loc[
            predicted_dates[0]]
        # 绘制最新一期数据变化的红线
        ax1.plot(predicted_dates, extracted_factor_aligned_concurrent[predicted_dates], color='purple', linewidth=2,
                 linestyle='-', label=latest_period_label)

        start_date = predicted_dates_leading[0].strftime('%Y-%m-%d')
        end_date = predicted_dates_leading[-1].strftime('%Y-%m-%d')
        latest_period_label = f"远期预期: {start_date} to {end_date}" if start_date != end_date else f"远期预期: {start_date}"
        # 调整数值保证连贯
        extracted_factor_aligned_leading -= extracted_factor_aligned_leading.loc[predicted_dates_leading[0]] - \
                                            extracted_factor_aligned_concurrent.loc[predicted_dates_leading[0]]
        # 绘制最新一期数据变化的红线
        ax1.plot(predicted_dates_leading, extracted_factor_aligned_leading[predicted_dates_leading], color='purple',
                 linewidth=3,
                 linestyle=':', label=latest_period_label)

        # ax1.plot(factor_aligned_concurrent, label='原始值')

        # 绘制每年的纵向栅格
        years = sorted(set(dt.year for dt in extracted_factor_aligned_leading.index))
        for year in years:
            ax1.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', linewidth=0.8)

        ax1.legend()
        # ax1.set_ylabel('因子值')
        ax1.set_title(rf'{self.preprocessor.industry} 综合指标')
        plt.show()

    def align_index_scale_corr(self, extracted_factor, factor, merge_how):
        """
        对齐两个时间序列的索引
        对齐两个时间序列的scale
        调整正负号
        """
        # 计算相关性的正负号
        factor_monthly = factor.resample('M').last()
        # 对齐两个时间序列的索引
        combined_data = pd.merge(extracted_factor, factor_monthly, left_index=True,
                                 right_index=True, how=merge_how)
        combined_data = combined_data.dropna()

        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # 方向调整
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        if corr < 0:
            extracted_factor *= -1
            extracted_factor_filtered *= -1

        # 保留原index
        if merge_how == 'outer':
            # 重新对齐两个时间序列的索引，不进行dropna
            combined_data = pd.merge(extracted_factor, factor, left_index=True,
                                     right_index=True, how=merge_how)

            extracted_factor_filtered = combined_data['0']
            factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # 对齐两个时间序列的scale
        # 使用 MinMaxScaler 对 extracted_factor 进行缩放，缩放范围为 factor 的最小值和最大值
        scaler = MinMaxScaler(feature_range=(factor_filtered.min(), factor_filtered.max()))
        # 对 extracted_factor_filtered 进行缩放
        extracted_factor_scaled = scaler.fit_transform(extracted_factor_filtered.values.reshape(-1, 1))
        # 将缩放后的值转换回 Series，并保持原始索引
        extracted_factor_scaled = pd.Series(extracted_factor_scaled.flatten(),
                                            index=extracted_factor_filtered.index)

        return extracted_factor_scaled, factor_filtered
