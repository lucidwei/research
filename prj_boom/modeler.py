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

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False  # �������ͼ���Ǹ���'-'��ʾΪ���������


class DynamicFactorModeler:
    def __init__(self, preprocessor: DataPreprocessor, k_factors: int, factor_orders: int, compare_to: str,
                 leading_prediction=False, single_line=False, plot_y0=False):
        """
        DynamicFactorMQ ��ģ��������ĳ�ʼ������
        :param data: Ԥ����������,DataFrame ��ʽ
        :param k_factors: ��������
        :param compare_to: ����ָ������,��������ģ��Ч��,Series ��ʽ
        """
        self.preprocessor = preprocessor
        self.data = preprocessor.data
        self.k_factors = k_factors
        self.factor_orders = factor_orders
        self.leading_prediction = leading_prediction
        self.single_line = single_line
        self.plot_y0 = plot_y0

        # �� df_finalcials �� df_indicators ��Ѱ�� compare_to �ַ���
        if compare_to in preprocessor.df_finalcials:
            self.series_compared_to = preprocessor.df_finalcials[compare_to]
        elif compare_to in preprocessor.df_indicators:
            self.series_compared_to = preprocessor.df_indicators[compare_to]
        else:
            raise ValueError(f"'{compare_to}' not found in df_finalcials or df_indicators")

    def run(self):
        """
        ���� DynamicFactorMQ ��ģ����������������
        """
        if self.single_line:
            mannual_indicators_group = {
                'PPI': {
                    '�й�:M1:ͬ��': 9,
                    '�й�:ʵ�徭�ò��Ÿܸ���:ͬ������': 9,
                },
                '����': {'�Ϻ����ڼ�װ���˼�ָ��:�ۺ�ָ��:ͬ��': 4,
                         '����:�����ܶ�:����:ͬ��-����:����ܶ�:����:ͬ��': 2,
                         '����:������۱�:����': 15,
                         '�ڻ����̼�(����):COMEXͭ:ͬ��:��:ƽ��ֵ': 3}
            }
            # �ȼ���ͬ�����ټ���leading���ٽ����ͼ����
            results_concurrent = self.apply_dynamic_factor_model()
            results_leading = self.apply_dynamic_factor_model(mannual_indicators_group[self.preprocessor.industry])
            self.plot_factors_single_line(results_concurrent, results_leading)
            return

        if self.leading_prediction:
            leading_indicators, synchronous_indicators, _ = self.find_statistically_significant_indicators(
                # lag1_as_sync=True) #����չʾ��ָ������������ã���ȡ�ַ�����
                lag1_as_sync=False)
            self.apply_dynamic_factor_model(leading_indicators)
        else:
            self.find_statistically_significant_indicators()
            self.apply_dynamic_factor_model()
        self.evaluate_model()
        # ��������ʱ����ڸ������Թ�ͬ���ӱ仯�Ĺ��ף�Ĭ��Ϊ���������
        self.analyze_factor_contribution(None, None)
        self.plot_factors(save_or_show='show')

    def find_statistically_significant_indicators(self, max_lag=15, alpha=0.05, resample_freq='M', lag1_as_sync=True):
        """
        �ҵ� df_indicators ����ͳ��ѧ������������ compare_to ��ʱ������, ��ɸѡ��ͬ��ָ��
        :param max_lag: ����ͺ����
        :param alpha: ������ˮƽ
        :param resample_freq: ���²���Ƶ�ʣ����� 'M' ��ʾ����
        :return: ���ȵ�ʱ�����м��������ͺ��������ֵ�, ͬ��ʱ�����м�����������Ե��ֵ�, ��������ʱ�������б�
        """
        leading_indicators = {}
        synchronous_indicators = {}
        discarded_indicators = []

        # ���²��� self.series_compared_to
        series_compared_to_resampled = self.series_compared_to.resample(resample_freq).mean().interpolate()

        # ���� df_indicators �е�ÿ��ʱ������
        for column in self.preprocessor.df_indicators.columns:
            # ���²���ÿ��ָ��ʱ������
            indicator_series_resampled = self.preprocessor.df_indicators[column].resample(
                resample_freq).mean().interpolate()

            combined_data = pd.concat([series_compared_to_resampled, indicator_series_resampled], axis=1).dropna()

            # ������ݳ����Ƿ��㹻
            if combined_data.shape[0] <= max_lag:
                print(f"Skipping {column} due to insufficient data length.")
                discarded_indicators.append(column)
                self.preprocessor.data.drop(columns=[column], inplace=True)
                continue

            try:
                # ͬ���Լ���
                correlation_sync, pear_p_value = pearsonr(combined_data.iloc[:, 0], combined_data.iloc[:, 1])
                if pear_p_value < alpha:
                    synchronous_indicators[column] = correlation_sync

                best_lag = 0
                highest_correlation = correlation_sync

                # ������ͬ���ͺ������������ͺ��������
                for lag in range(0, max_lag + 1):
                    # �ͺ���
                    lagged_data = combined_data.copy()
                    lagged_data.iloc[:, 1] = lagged_data.iloc[:, 1].shift(lag)
                    lagged_data = lagged_data.dropna()

                    # �����ͺ��������
                    correlation, p_val = pearsonr(lagged_data.iloc[:, 0], lagged_data.iloc[:, 1])

                    # ѡ���������ߵ��ͺ�����������ע������
                    if correlation > highest_correlation:
                        highest_correlation = correlation
                        best_lag = lag

                if best_lag == 0 and pear_p_value <= alpha:
                    print(f'{column}��Ϊͬ��ָ��')
                elif best_lag == 1 and lag1_as_sync:
                    print(f'{column}Ϊ����1�ڣ�����Ϊͬ��ָ��')
                elif best_lag >= 1:
                    leading_indicators[column] = best_lag
                elif best_lag == 0 and pear_p_value >= alpha:
                    discarded_indicators.append(column)
                    self.preprocessor.data.drop(columns=[column], inplace=True)
                else:
                    raise ValueError('Unexpected condition encountered')

                # # ʹ������ͺ���������ʱ������ͼ
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

                # # ���и������������
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
                # # ���Ʋ�ͬ�ͺ�������pֵ�ı仯ͼ
                # plot_granger_causality_pvalues(combined_data, max_lag=12)

                # # ���ÿ���ͺ�����µ� F-���� p ֵ��ѡȡ��������ߵ�����
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
                #     print(f'{column}��Ϊͬ��ָ��')
            except ValueError as e:
                print(f"Error processing {column}: {e}")
                discarded_indicators.append(column)
                continue

        print(f'����ָ��(��������)��{leading_indicators}')
        print(
            f"ͬ��ָ��(�����)��{{{', '.join([f'{key}: {value:.2f}' for key, value in synchronous_indicators.items()])}}}")
        print(f'������ָ�꣺{discarded_indicators}')
        return leading_indicators, synchronous_indicators, discarded_indicators

    def apply_dynamic_factor_model(self, indicators_group=None):
        """
        Ӧ�� DynamicFactorMQ ģ�ͽ��н�ģ�ͼ���
        """
        if indicators_group is not None:
            # ��ָ��������ȡ�����Ͷ�Ӧ����������
            selected_columns = indicators_group.keys()
            leading_periods = indicators_group.values()
            periods_to_extend = max(leading_periods)

            # ����һ���µ� DataFrame ���洢���ƺ������
            future_dates = pd.date_range(start=self.data.index[-1], periods=periods_to_extend + 1, freq='M')[1:]
            extended_index = self.data.index.append(future_dates)
            extended_data = self.data.reindex(extended_index)

            shifted_data = pd.DataFrame(index=extended_index)

            for column, period in zip(selected_columns, leading_periods):
                # ����ֱ��ʹ�� column ����ƥ��
                if column in extended_data.columns:
                    target_column = column
                else:
                    # ���ƥ�䲻�ϣ�������ǰ����� '(�¶Ȼ�)'
                    modified_column = f"(�¶Ȼ�){column}"
                    if modified_column in extended_data.columns:
                        target_column = modified_column
                    else:
                        raise ValueError(f"Column '{column}' or '{modified_column}' not found in extended_data")

                # �����������Ƶ�δ��
                # TODO �ֶ�-1����ߵ͵㣬����ԭ��δ��
                if self.leading_prediction:
                    shifted_series = extended_data[target_column].shift(period - 1)
                else:
                    shifted_series = extended_data[target_column].shift(period)

                # �����ƺ��ϵ����ӵ��µ� DataFrame ��
                shifted_data[column] = shifted_series

            # �����µ�����������������ܲ����ڵ���ĩ����
            filtered_data = shifted_data.resample('M').asfreq().interpolate(method='time')
        else:
            # ���û��ָ��ָ���飬ʹ��ȫ������
            filtered_data = self.data

        model = DynamicFactorMQ(filtered_data, factors=self.k_factors, factor_orders=self.factor_orders,
                                idiosyncratic_ar1=False)

        results = model.fit_em(maxiter=1000)
        print(results.summary())
        self.results = results

        # ��ȡ�����غɣ�factor loadings������ÿ���۲������Ȩ��
        num_loadings = len([param for param in self.results.params.index if 'loading' in param])
        self.factor_loadings = self.results.params[:num_loadings]
        self.factor_loadings.index = self.factor_loadings.index.str.replace('loading.0->', '', regex=False)
        # ��ȡ״̬ת�ƾ�����ʷ���ݶԵ�ǰ���ӵ�Ӱ��ͨ��״̬���̺��ͺ�����ʵ�֣�������ͨ�������غɡ�
        # self.transition_matrix = self.results.transition

        # fitted_data�����۲첹ȫ��Ŀ�ֵ������ԭʼ���ݱ仯�ܴ�
        self.fitted_data = self.results.predict()
        return results

    def evaluate_model(self):
        """
        ����ģ��Ч��,������ȡ�����������ָ������ϵ��
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        # �� self.financial ������ת��Ϊ��Ƶ
        financial_monthly = self.series_compared_to.resample('M').last()

        # ��������ʱ�����е�����
        extracted_factor_filtered, factor_filtered = self.align_index_scale_corr(extracted_factor, financial_monthly,
                                                                                 'inner')

        corr = np.corrcoef(extracted_factor_filtered[15:], factor_filtered[15:])[0, 1]
        print(f"����Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered[:15], factor_filtered[:15])[0, 1]
        print(f"����Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")

    def analyze_factor_contribution(self, start_date=None, end_date=None):
        """
        ��������ʱ����ڸ������Թ�ͬ���ӱ仯�Ĺ���
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        # ��ȡƽ�����״̬�ֽ�
        decomposition = self.results.get_smoothed_decomposition(decomposition_of='smoothed_state')
        data_contributions = decomposition[0].loc[pd.IndexSlice['0', :], :]
        # decomposition = self.results.filtered_state
        # dates = self.results.data.dates  # ��ȡʱ�����е�����
        # variables = self.results.data.param_names  # ��ȡ��������
        # data_contributions = pd.DataFrame(data=decomposition.T, index=dates, columns=variables)

        # ������ת��Ϊ DataFrame ��������
        data_contributions.index = data_contributions.index.droplevel(0)

        # �Զ�����Ĭ�ϵ���ֹ����Ϊ���������
        if start_date is None or end_date is None:
            end_date = data_contributions.index[-1]
            start_date = data_contributions.index[-3]
        # start_date = '2024-03-31'
        # end_date = '2024-04-30'
        print(f"Variable contributions to factor change from {start_date} to {end_date}:")

        # ��ȡ����ʱ����ڵĹ���
        factor_contributions = data_contributions.loc[start_date:end_date].T

        # �������洢����������Ȩ��
        output = f"��ָ��Ȩ�أ�{self.factor_loadings}\n"  # �洢���е������Ϣ
        print(output)
        for column in factor_contributions.columns:
            print(f"����{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ��:")
            output += f"����{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ��:\n"

            # �Ե�ǰ�н��н�������ȥ��nanֵ
            sorted_column = factor_contributions[column].sort_values(ascending=False).dropna()

            # ��ȡǰ����ֵ����Index
            head_values = sorted_column.head(3)
            print("Top 3 ������:")
            output += "Top 3 ������:\n"
            for index, value in head_values.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}\n"

            # ��ȡ������ֵ����Index
            tail_values = sorted_column.tail(3)
            print("Bottom 3 ������:")
            output += "Bottom 3 ������:\n"
            for index, value in tail_values.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact {value:.3f}\n"

            print("\n")
            output += "\n"

        # �����������������һ��(��adjust����·�)���ݱ仯�Ĺ���
        adjust = 0
        if len(factor_contributions.columns) >= 2:
            prev_month = factor_contributions.columns[-2 - adjust]
            curr_month = factor_contributions.columns[-1 - adjust]

            print(
                f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}")
            output += f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}\n"

            # ����ÿ�������Ĺ��ױ仯
            contrib_change = factor_contributions[curr_month] - factor_contributions[prev_month]

            # �Թ��ױ仯��������
            sorted_contrib_change = contrib_change.sort_values(ascending=False).dropna()

            print(f"����{curr_month.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ���仯:")
            output += f"����{curr_month.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ���仯:\n"

            # ��ȡǰ���������ױ仯ֵ����Index
            top_positive_changes = sorted_contrib_change.head(3)
            print("Top 3 �����ױ仯:")
            output += "Top 3 �����ױ仯:\n"
            for index, value in top_positive_changes.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}\n"
            # ��ȡ�����������ױ仯ֵ����Index
            bottom_negative_changes = sorted_contrib_change.tail(3)
            print("Bottom 3 �����ױ仯:")
            output += "Bottom 3 �����ױ仯:\n"
            for index, value in bottom_negative_changes.items():
                print(f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}")
                output += f"'{index[0]}' at '{index[1].strftime('%Y-%m-%d')}', impact change {value:.3f}\n"

            # �������������ױ仯���ܺ�
            positive_sum = sorted_contrib_change[sorted_contrib_change > 0].sum()
            print(f"�����ױ仯�ܺ�: {positive_sum:.3f}")
            output += f"�����ױ仯�ܺ�: {positive_sum:.3f}\n"

            # �������и����ױ仯���ܺ�
            negative_sum = sorted_contrib_change[sorted_contrib_change < 0].sum()
            print(f"�����ױ仯�ܺ�: {negative_sum:.3f}")
            output += f"�����ױ仯�ܺ�: {negative_sum:.3f}\n"

            print("\n")
            output += "\n"

        return output

    def plot_factors(self, save_or_show='show'):
        """
        ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        factor = self.series_compared_to.dropna().astype(float)

        extracted_factor_filtered, factor_filtered = self.align_index_scale_corr(extracted_factor, factor, 'outer')

        # ��ȡ factor_filtered ʵ�ʴ��ڵ���ʵ�����е���������
        latest_date_existing = factor_filtered.dropna().index.max()
        # �ҵ�����֮�����������
        predicted_dates = extracted_factor_filtered.index[extracted_factor_filtered.index > latest_date_existing]
        if len(predicted_dates) == 0:
            extracted_factor_filtered_without_predicted = extracted_factor_filtered
        else:
            extracted_factor_filtered_without_predicted = extracted_factor_filtered[
                extracted_factor_filtered.index < predicted_dates[0]]
        # predicted_dates���һ����ʷ���ڣ���֤��Ԥ������ʱ������
        prev_date = extracted_factor_filtered_without_predicted.index.max()
        predicted_dates = extracted_factor_filtered.index[extracted_factor_filtered.index >= prev_date]

        # ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(extracted_factor_filtered_without_predicted, label='�����ۺ�ָ��')

        # ���һ��������Ϊ0������
        if self.plot_y0:
            ax1.axhline(y=0, color='gray', linestyle='--')

        # ��ͼ�б�עԤ���ڵ����ڷ�Χ
        start_date = predicted_dates[0].strftime('%Y-%m-%d') if len(predicted_dates) == 1 else predicted_dates[
            1].strftime('%Y-%m-%d')
        end_date = predicted_dates[-1].strftime('%Y-%m-%d')
        latest_period_label = f"Ԥ����: {start_date} to {end_date}" if start_date != end_date else f"Ԥ����: {start_date}"
        # ��������һ�����ݱ仯�ĺ���
        ax1.plot(predicted_dates, extracted_factor_filtered[predicted_dates], color='purple', linewidth=3,
                 linestyle=':' if self.leading_prediction else '-', label=latest_period_label)

        # �����ڶ��� y ��
        ax2 = ax1.twinx()
        # �ж� NaN �������Ƿ����һ��
        nan_count = factor_filtered.isna().sum()
        total_count = len(factor_filtered)

        if nan_count > total_count / 2 or self.preprocessor.industry == '������ָ':
            ax2.scatter(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red')
        else:
            ax2.plot(factor_filtered.index, factor_filtered.values, label=factor_filtered.name, color='red', alpha=0.6)

        # ����ÿ�������դ��
        years = sorted(set(dt.year for dt in extracted_factor_filtered.index))
        for year in years:
            ax1.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', linewidth=0.8)

        # ���õ�һ�� y ���ǩ
        ax1.set_ylabel('�����ۺ�ָ��')

        # ���õڶ��� y ���ǩ
        ax2.set_ylabel(factor.name)

        # �ϲ����� y ���ͼ��
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(rf'{self.preprocessor.industry}')
        if save_or_show == 'show':
            plt.show()
        elif save_or_show == 'save':
            # ����ͼ���ļ�
            current_time = datetime.now().strftime("%Y%m%d_%H%M")
            stationary_flag = 'stationary' if self.preprocessor.stationary else 'fast'
            image_path = rf'{self.preprocessor.base_config.excels_path}/����/pics/{self.preprocessor.industry}_{stationary_flag}_factor_plot_{current_time}.png'
            plt.savefig(image_path)
            plt.close(fig)

            return image_path

    def plot_factors_single_line(self, results_concurrent, results_leading):
        """
        ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        """

        factor = self.series_compared_to.dropna().astype(float)
        extracted_factor_concurrent = results_concurrent.factors.filtered['0']
        extracted_factor_leading = results_leading.factors.filtered['0']

        extracted_factor_aligned_concurrent, factor_aligned_concurrent = self.align_index_scale_corr(
            extracted_factor_concurrent, factor, 'outer')
        extracted_factor_aligned_leading, factor_aligned_leading = self.align_index_scale_corr(extracted_factor_leading,
                                                                                               factor, 'outer')

        # ��ȡ factor_filtered ʵ�ʴ��ڵ���ʵ�����е���������
        latest_date_existing = factor.dropna().index.max()
        # �ҵ�����֮�����������
        predicted_dates_concurrent = extracted_factor_aligned_concurrent.index[
            extracted_factor_aligned_concurrent.index > latest_date_existing]
        predicted_dates_leading = extracted_factor_aligned_leading.index[
            extracted_factor_aligned_leading.index > latest_date_existing]

        # ���ͬ��ָ����ܲ����ڵ���Ԥ��ֵ�����
        if len(predicted_dates_concurrent) == 0:
            extracted_factor_concurrent_without_predicted = extracted_factor_aligned_concurrent
        else:
            extracted_factor_concurrent_without_predicted = extracted_factor_aligned_concurrent[
                extracted_factor_aligned_concurrent.index < predicted_dates_concurrent[0]]
        # predicted_dates���һ����ʷ���ڣ���֤��Ԥ������ʱ������
        prev_date = extracted_factor_concurrent_without_predicted.index.max()
        predicted_dates = extracted_factor_aligned_concurrent.index[
            extracted_factor_aligned_concurrent.index >= prev_date]

        # ������ȡ������Ԥ���ԭʼ���ӵ�ͼ��
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(factor, label=f'{self.preprocessor.industry}(��ʷ)')

        # ��ͼ�б�עԤ���ڵ����ڷ�Χ
        start_date = predicted_dates[0].strftime('%Y-%m-%d') if len(predicted_dates) == 1 else predicted_dates[
            1].strftime('%Y-%m-%d')
        end_date = predicted_dates[-1].strftime('%Y-%m-%d')
        latest_period_label = f"����Ԥ��: {start_date} to {end_date}" if start_date != end_date else f"����Ԥ��: {start_date}"
        # ������ֵ��֤����
        extracted_factor_aligned_concurrent -= extracted_factor_aligned_concurrent.loc[predicted_dates[0]] - factor.loc[
            predicted_dates[0]]
        # ��������һ�����ݱ仯�ĺ���
        ax1.plot(predicted_dates, extracted_factor_aligned_concurrent[predicted_dates], color='purple', linewidth=2,
                 linestyle='-', label=latest_period_label)

        start_date = predicted_dates_leading[0].strftime('%Y-%m-%d')
        end_date = predicted_dates_leading[-1].strftime('%Y-%m-%d')
        latest_period_label = f"Զ��Ԥ��: {start_date} to {end_date}" if start_date != end_date else f"Զ��Ԥ��: {start_date}"
        # ������ֵ��֤����
        extracted_factor_aligned_leading -= extracted_factor_aligned_leading.loc[predicted_dates_leading[0]] - \
                                            extracted_factor_aligned_concurrent.loc[predicted_dates_leading[0]]
        # ��������һ�����ݱ仯�ĺ���
        ax1.plot(predicted_dates_leading, extracted_factor_aligned_leading[predicted_dates_leading], color='purple',
                 linewidth=3,
                 linestyle=':', label=latest_period_label)

        # ax1.plot(factor_aligned_concurrent, label='ԭʼֵ')

        # ����ÿ�������դ��
        years = sorted(set(dt.year for dt in extracted_factor_aligned_leading.index))
        for year in years:
            ax1.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', linewidth=0.8)

        ax1.legend()
        # ax1.set_ylabel('����ֵ')
        ax1.set_title(rf'{self.preprocessor.industry} �ۺ�ָ��')
        plt.show()

    def align_index_scale_corr(self, extracted_factor, factor, merge_how):
        """
        ��������ʱ�����е�����
        ��������ʱ�����е�scale
        ����������
        """
        # ��������Ե�������
        factor_monthly = factor.resample('M').last()
        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor, factor_monthly, left_index=True,
                                 right_index=True, how=merge_how)
        combined_data = combined_data.dropna()

        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # �������
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        if corr < 0:
            extracted_factor *= -1
            extracted_factor_filtered *= -1

        # ����ԭindex
        if merge_how == 'outer':
            # ���¶�������ʱ�����е�������������dropna
            combined_data = pd.merge(extracted_factor, factor, left_index=True,
                                     right_index=True, how=merge_how)

            extracted_factor_filtered = combined_data['0']
            factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # ��������ʱ�����е�scale
        # ʹ�� MinMaxScaler �� extracted_factor �������ţ����ŷ�ΧΪ factor ����Сֵ�����ֵ
        scaler = MinMaxScaler(feature_range=(factor_filtered.min(), factor_filtered.max()))
        # �� extracted_factor_filtered ��������
        extracted_factor_scaled = scaler.fit_transform(extracted_factor_filtered.values.reshape(-1, 1))
        # �����ź��ֵת���� Series��������ԭʼ����
        extracted_factor_scaled = pd.Series(extracted_factor_scaled.flatten(),
                                            index=extracted_factor_filtered.index)

        return extracted_factor_scaled, factor_filtered
