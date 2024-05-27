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
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class DynamicFactorModeler:
    def __init__(self, preprocessor: DataPreprocessor, k_factors: int, factor_orders: int, financial: str):
        """
        DynamicFactorMQ ��ģ��������ĳ�ʼ������
        :param data: Ԥ����������,DataFrame ��ʽ
        :param k_factors: ��������
        :param financial: ����ָ������,��������ģ��Ч��,Series ��ʽ
        """
        self.preprocessor = preprocessor
        self.data = preprocessor.data
        self.financial = preprocessor.df_finalcials[financial]
        self.k_factors = k_factors
        self.factor_orders = factor_orders

    def apply_dynamic_factor_model(self):
        """
        Ӧ�� DynamicFactorMQ ģ�ͽ��н�ģ�ͼ���
        """
        em_kwargs = {
            'tolerance': 1e-7,  # ����������ֵ
        }
        model = DynamicFactorMQ(self.data, factors=self.k_factors, factor_orders=self.factor_orders, idiosyncratic_ar1=False)

        self.results = model.fit_em(maxiter=1000)
        print(self.results.summary())
        # fitted_data�����۲첹ȫ��Ŀ�ֵ������ԭʼ���ݱ仯�ܴ�
        self.fitted_data = self.results.predict()

    def evaluate_model(self):
        """
        ����ģ��Ч��,������ȡ�����������ָ������ϵ��
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']
        self.extracted_factor = extracted_factor

        # �� self.financial ������ת��Ϊ��Ƶ
        financial_monthly = self.financial.resample('M').last()

        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor, financial_monthly, left_index=True, right_index=True, how='inner')
        combined_data = combined_data.dropna()
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze().astype(float)

        corr = np.corrcoef(extracted_factor_filtered[15:], factor_filtered[15:])[0, 1]
        print(f"����Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered[:15], factor_filtered[:15])[0, 1]
        print(f"����Correlation: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")
        self.corr = corr

    def analyze_factor_contribution(self, start_date, end_date):
        """
        ��������ʱ����ڸ������Թ�ͬ���ӱ仯�Ĺ���
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        # ��ȡƽ�����״̬�ֽ�
        decomposition = self.results.get_smoothed_decomposition(decomposition_of='smoothed_state')
        data_contributions = decomposition[0].loc[pd.IndexSlice['0', :], :]

        # ������ת��Ϊ DataFrame ��������
        data_contributions.index = data_contributions.index.droplevel(0)

        # �Զ�����Ĭ�ϵ���ֹ����Ϊ���������
        if start_date is None or end_date is None:
            end_date = data_contributions.index[-1]
            start_date = data_contributions.index[-3]
        print(f"Variable contributions to factor change from {start_date} to {end_date}:")

        # ��ȡ����ʱ����ڵĹ���
        factor_contributions = data_contributions.loc[start_date:end_date].T
        if self.corr < 0:
            factor_contributions *= -1

        df = factor_contributions.copy(deep=True)

        output = ""  # �洢���е������Ϣ
        for column in df.columns:
            print(f"����{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ��:")
            output += f"����{column.strftime('%Y-%m-%d')} {self.preprocessor.industry} ������ָ��:\n"

            # �Ե�ǰ�н��н�������ȥ��nanֵ
            sorted_column = df[column].sort_values(ascending=False).dropna()

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

        # �����������������һ�����ݱ仯����4�µ�5�£��Ĺ���
        if len(df.columns) >= 2:
            prev_month = df.columns[-2]
            curr_month = df.columns[-1]

            print(
                f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}")
            output += f"Comparing contributions from {prev_month.strftime('%Y-%m-%d')} to {curr_month.strftime('%Y-%m-%d')}\n"

            # ����ÿ�������Ĺ��ױ仯
            contrib_change = df[curr_month] - df[prev_month]

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
        if self.corr < 0:
            extracted_factor *= -1

        factor = self.financial.dropna().astype(float)

        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor, factor, left_index=True,
                                 right_index=True, how='outer')
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(extracted_factor_filtered, label='�����ۺ�ָ��')

        # ��ȡ���µ�����ʱ���
        latest_dates = extracted_factor_filtered.index[-2:]
        # ��ͼ�б�ע����һ�ڵ����ڷ�Χ
        start_date = latest_dates[0].strftime('%Y-%m-%d')
        end_date = latest_dates[1].strftime('%Y-%m-%d')
        latest_period_label = f"Latest Period: {start_date} to {end_date}"
        # ��������һ�����ݱ仯�ĺ���
        ax1.plot(latest_dates, extracted_factor_filtered[latest_dates], color='red', linewidth=2, label=latest_period_label)

        # �����ڶ��� y ��
        ax2 = ax1.twinx()
        ax2.scatter(factor_filtered.index, factor_filtered.values, label=factor.name, color='red')

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

    def run(self):
        """
        ���� DynamicFactorMQ ��ģ����������������
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
        # ��������ʱ����ڸ������Թ�ͬ���ӱ仯�Ĺ��ף�Ĭ��Ϊ���������
        self.analyze_factor_contribution(None, None)
        self.plot_factors(save_or_show='show')
