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
        DynamicFactorMQ 建模和评估类的初始化方法
        :param data: 预处理后的数据,DataFrame 格式
        :param k_factors: 因子数量
        :param financial: 金融指标序列,用于评估模型效果,Series 格式
        """
        self.data = data
        self.k_factors = k_factors
        self.financial = financial
        self.results = None

    def apply_dynamic_factor_model(self):
        """
        应用 DynamicFactorMQ 模型进行建模和计算
        """
        em_kwargs = {
            'tolerance': 1e-7,  # 设置收敛阈值
        }
        model = DynamicFactorMQ(self.data, factors=self.k_factors, factor_orders=2, idiosyncratic_ar1=False)
        print(model.summary())
        self.results = model.fit_em(maxiter=1000)

    def evaluate_model(self):
        """
        评估模型效果,计算提取的因子与金融指标的相关系数
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        fitted_data = self.results.predict()
        extracted_factor = self.results.factors.filtered['0']

        # 确保两个序列的长度和时间点匹配
        extracted_factor_series = pd.Series(extracted_factor.values, index=self.financial.index, name='0')

        # 对齐两个时间序列的索引
        combined_data = pd.merge(extracted_factor_series, self.financial, left_index=True, right_index=True,
                                 how='inner')
        combined_data.dropna(inplace=True)
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
        print(f"Correlation between extracted factor and original factor: {corr:.4f}")

    def run(self):
        """
        运行 DynamicFactorMQ 建模和评估的完整流程
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
