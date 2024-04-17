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

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class DynamicFactorModeler:
    def __init__(self, data: pd.DataFrame, k_factors: int, financial: pd.Series):
        """
        DynamicFactorMQ ��ģ��������ĳ�ʼ������
        :param data: Ԥ����������,DataFrame ��ʽ
        :param k_factors: ��������
        :param financial: ����ָ������,��������ģ��Ч��,Series ��ʽ
        """
        self.data = data
        self.k_factors = k_factors
        self.financial = financial
        self.results = None

    def apply_dynamic_factor_model(self):
        """
        Ӧ�� DynamicFactorMQ ģ�ͽ��н�ģ�ͼ���
        """
        em_kwargs = {
            'tolerance': 1e-7,  # ����������ֵ
        }
        model = DynamicFactorMQ(self.data, factors=self.k_factors, factor_orders=2, idiosyncratic_ar1=False)
        print(model.summary())
        self.results = model.fit_em(maxiter=1000)

    def evaluate_model(self):
        """
        ����ģ��Ч��,������ȡ�����������ָ������ϵ��
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        # fitted_data�����۲첹ȫ��Ŀ�ֵ������ԭʼ���ݱ仯�ܴ�
        fitted_data = self.results.predict()
        extracted_factor = self.results.factors.filtered['0']

        # �� self.financial ������ת��Ϊ��Ƶ
        financial_monthly = self.financial.resample('M').last()

        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor, financial_monthly, left_index=True, right_index=True, how='inner')
        combined_data = combined_data.dropna()
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze().astype(float)

        corr = np.corrcoef(extracted_factor_filtered[15:], factor_filtered[15:])[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered[:15], factor_filtered[:15])[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")
        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")

    def plot_factors(self):
        """
        ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']

        # �� extracted_factor ת��Ϊ��Ƶ����
        extracted_factor_quarterly = extracted_factor.resample('Q').last()

        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor_quarterly, self.financial.dropna().astype(float), left_index=True,
                                 right_index=True, how='outer')
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # ������ȡ�����Ӻ�ԭʼ���ӵ�ͼ��
        plt.figure(figsize=(12, 6))
        plt.plot(extracted_factor_filtered, label='Extracted Factor')
        plt.scatter(factor_filtered.index, factor_filtered.values, label='roe_ttm', color='red')
        # ����ÿ�������դ��
        years = sorted(set(dt.year for dt in extracted_factor_filtered.index))
        for year in years:
            plt.axvline(pd.to_datetime(f'{year}-01-01'), color='gray', linestyle='--', linewidth=0.8)

        plt.xlabel('Date')
        plt.ylabel('Factor Value')
        plt.legend()
        plt.title('Extracted Factor vs. roe_ttm')
        plt.show()

    def run(self):
        """
        ���� DynamicFactorMQ ��ģ����������������
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
        self.plot_factors()
