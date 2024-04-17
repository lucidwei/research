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
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


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

        # fitted_data用来观察补全后的空值（但对原始数据变化很大）
        fitted_data = self.results.predict()
        extracted_factor = self.results.factors.filtered['0']

        # 将 self.financial 的索引转换为月频
        financial_monthly = self.financial.resample('M').last()

        # 对齐两个时间序列的索引
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
        绘制提取的因子和原始因子的图像
        """
        if self.results is None:
            raise ValueError("Please run apply_dynamic_factor_model() first.")

        extracted_factor = self.results.factors.filtered['0']

        # 将 extracted_factor 转换为季频数据
        extracted_factor_quarterly = extracted_factor.resample('Q').last()

        # 对齐两个时间序列的索引
        combined_data = pd.merge(extracted_factor_quarterly, self.financial.dropna().astype(float), left_index=True,
                                 right_index=True, how='outer')
        extracted_factor_filtered = combined_data['0']
        factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

        # 绘制提取的因子和原始因子的图像
        plt.figure(figsize=(12, 6))
        plt.plot(extracted_factor_filtered, label='Extracted Factor')
        plt.scatter(factor_filtered.index, factor_filtered.values, label='roe_ttm', color='red')
        # 绘制每年的纵向栅格
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
        运行 DynamicFactorMQ 建模和评估的完整流程
        """
        self.apply_dynamic_factor_model()
        self.evaluate_model()
        self.plot_factors()
