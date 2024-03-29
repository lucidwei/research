# coding=gbk
# Time Created: 2024/3/26 14:45
# Author  : Lucid
# FileName: 景气.py
# Software: PyCharm
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 设置随机数种子以获得可重现的结果
np.random.seed(42)

# 1. 生成示例数据
# def generate_data(factor, freq_list, noise_std):
#     data = {}
#     start_date = pd.to_datetime('2010-01-01')
#     end_date = start_date + pd.Timedelta(days=len(factor)-1)
#     for freq in freq_list:
#         index = pd.date_range(start=start_date, end=end_date, freq=freq)
#         data[freq] = pd.Series(factor[:len(index)] + np.random.normal(0, noise_std, size=len(index)), index=index)
#     return pd.DataFrame(data)
#
# periods = 500
# factor = np.zeros(periods)
# phi = 0.8
# for i in range(1, periods):
#     factor[i] = phi * factor[i-1] + np.random.normal(0, 0.1)
# freq_list = ['D', 'W', 'M']
# noise_std = 0.005
# data = generate_data(factor, freq_list, noise_std)
#
# # 2. 模拟缺失值
# data.loc[::3, 'M'] = np.nan

# 设置参数
file_path = rf"D:\Downloads"
k_factors = 1

# 读取数据
data = pd.read_excel(rf'{file_path}/期货结算价(连续)_WTI原油.xlsx', header=1)
data_cleaned = data.drop(index=[0, 1]).reset_index(drop=True)
# Convert the index to datetime format
data_cleaned['指标名称'] = pd.to_datetime(data_cleaned['指标名称'])
# Set the new index
data_cleaned.set_index('指标名称', inplace=True)
data_cleaned = data_cleaned[data_cleaned.index >= pd.Timestamp('2010-01-01')]
data_cleaned.sort_index(ascending=True, inplace=True)

financials = pd.read_excel(rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\行业财务数据.xlsx", header=3, sheet_name='油气开采q')
financials.set_index('Date', inplace=True)
financials = financials[financials.index >= pd.Timestamp('2010-01-01')]
financials.sort_index(ascending=True, inplace=True)


combined_data = pd.merge(data_cleaned, financials, left_index=True, right_index=True, how='outer')

df_indicators = combined_data.loc[:, combined_data.columns != 'roe_ttm2']
df_finalcials = combined_data.loc[:, 'roe_ttm2']

data = df_indicators.replace(0, np.nan)
factor = df_finalcials

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')



# 3. 应用动态因子模型
def apply_dynamic_factor_model(data, k_factors):
    em_kwargs = {
        'tolerance': 1e-7,  # 设置收敛阈值
    }
    model = DynamicFactorMQ(data, factors=k_factors, factor_orders=2, idiosyncratic_ar1=False)
    print(model.summary())
    results = model.fit_em(maxiter=1000)
    return results

results = apply_dynamic_factor_model(data, k_factors)

# 4. 评估模型效果
def evaluate_model(results, factor):
    fitted_data = results.predict()
    extracted_factor = results.factors.filtered['0']
    # # 确保两个序列的长度和时间点匹配
    extracted_factor_series = pd.Series(extracted_factor.values, index=factor.index, name='0')

    # 对齐两个时间序列的索引
    combined_data = pd.merge(extracted_factor_series, factor, left_index=True, right_index=True, how='inner')
    combined_data.dropna(inplace=True)
    extracted_factor_filtered = combined_data['0']
    factor_filtered = combined_data.loc[:, combined_data.columns != '0'].squeeze()

    corr = np.corrcoef(extracted_factor_filtered, factor_filtered)[0, 1]
    # mse = mean_squared_error(extracted_factor, factor)
    print(f"Correlation between extracted factor and original factor: {corr:.4f}")
    # print(f"MSE between extracted factor and original factor: {mse:.4f}")

    # for freq in data.columns:
    #     original_data = data[freq]
    #     fitted_data = results.predict(start=original_data.index[0], end=original_data.index[-1], dynamic=False).loc[:, freq]
    #     mse = mean_squared_error(original_data[~original_data.isnull()], fitted_data[~original_data.isnull()])
    #     print(f"MSE for {freq} frequency data: {mse:.4f}")
    return extracted_factor_series, corr

extracted_factor_series, corr = evaluate_model(results, factor)

print('q')


def plot_factors_mixed_freq(results, factor, corr):
    extracted_factor = results.factors.filtered['0']
    # 根据相关性的符号调整共同因子的符号
    if corr < 0:
        extracted_factor = -extracted_factor

    extracted_factor_series = pd.Series(extracted_factor.values, index=factor.index, name='0')
    # 对原始因子进行插值
    factor_interpolated = factor.interpolate(method='time')

    # 绘制原始因子和提取因子的时间序列图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(extracted_factor_series, label='景气指标(日频)')
    ax.plot(factor_interpolated, '-')
    ax.plot(factor, 'o', label='ROE_TTM(季频)')

    for x in factor.dropna().index:
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.legend()
    ax.set_title('景气指标(日频) vs ROE_TTM(季频)')
    ax.set_ylabel('Factor Value')

    # 设置x轴日期标签的格式和间隔
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.show()

plot_factors_mixed_freq(results, factor, corr)

# 6. 结果解释
print("\nFactor loadings:")
# print(results.factors)

# 5. 调整优化
# k_factors_list = [1, 2, 3]
# for k in k_factors_list:
#     print(f"\nResults for {k} factor(s):")
#     results = apply_dynamic_factor_model(data, k)
#     evaluate_model(results, factor, data)