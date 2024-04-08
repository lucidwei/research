# coding=gbk
# Time Created: 2024/4/8 17:28
# Author  : Lucid
# FileName: modeler.py
# Software: PyCharm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


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

        fitted_data = self.results.predict()
        extracted_factor = self.results.factors.filtered['0']

        # ȷ���������еĳ��Ⱥ�ʱ���ƥ��
        extracted_factor_series = pd.Series(extracted_factor.values, index=self.financial.index, name='0')

        # ��������ʱ�����е�����
        combined_data = pd.merge(extracted_factor_series, self.financial, left_index=True, right_index=True,
                                 how='inner')
        combined_data.dropna(inplace=True)
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")

    def run(self):
        """
        ���� DynamicFactorMQ ��ģ����������������
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
