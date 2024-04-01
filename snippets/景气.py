# coding=gbk
# Time Created: 2024/3/26 14:45
# Author  : Lucid
# FileName: ����.py
# Software: PyCharm
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# ��������������Ի�ÿ����ֵĽ��
np.random.seed(42)

# 1. ����ʾ������
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
# # 2. ģ��ȱʧֵ
# data.loc[::3, 'M'] = np.nan

# 1. ���ò���
file_path = rf"D:\WPS����\WPS����\����-���\�о�trial"
k_factors = 1

# 2. ��ȡ����
data = pd.read_excel(rf'{file_path}/�к��indicators.xlsx', header=1)
data_cleaned = data.drop(index=[0, 1]).reset_index(drop=True)
# Convert the index to datetime format
data_cleaned['ָ������'] = pd.to_datetime(data_cleaned['ָ������'])
# Set the new index
data_cleaned.set_index('ָ������', inplace=True)
data_cleaned = data_cleaned[data_cleaned.index >= pd.Timestamp('2010-01-01')]
data_cleaned.sort_index(ascending=True, inplace=True)

financials = pd.read_excel(rf"{file_path}/��ҵ��������.xlsx", header=3, sheet_name='��������q')
financials.set_index('Date', inplace=True)
financials = financials[financials.index >= pd.Timestamp('2010-01-01')]
financials.sort_index(ascending=True, inplace=True)


combined_data = pd.merge(data_cleaned, financials, left_index=True, right_index=True, how='outer')

# ����һ���б�,�洢Ҫ�޳�������
financials_cols = ['roe_ttm2', 'yoyprofit']
indicators_cols = [col for col in combined_data.columns if col not in financials_cols]

df_indicators = combined_data[indicators_cols]
df_finalcials = combined_data[financials_cols]

# �޳�Ӱ������0ֵ
data = df_indicators.replace(0, np.nan)
finalcials = df_finalcials

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')



# 3. Ӧ�ö�̬����ģ��
def apply_dynamic_factor_model(data, k_factors):
    em_kwargs = {
        'tolerance': 1e-7,  # ����������ֵ
    }
    model = DynamicFactorMQ(data, factors=k_factors, factor_orders=2, idiosyncratic_ar1=False)
    print(model.summary())
    results = model.fit_em(maxiter=1000)
    return results

results = apply_dynamic_factor_model(data, k_factors)

# 4. ����ģ��Ч��
def evaluate_model(results, financial: pd.Series):
    fitted_data = results.predict()
    extracted_factor = results.factors.filtered['0']
    # # ȷ���������еĳ��Ⱥ�ʱ���ƥ��
    extracted_factor_series = pd.Series(extracted_factor.values, index=financial.index, name='0')

    # ��������ʱ�����е�����
    combined_data = pd.merge(extracted_factor_series, financial, left_index=True, right_index=True, how='inner')
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

# extracted_factor_series, corr = evaluate_model(results, finalcials['yoyprofit'])
extracted_factor_series, corr = evaluate_model(results, finalcials['roe_ttm2'])


# 5. ��ͼ��Ч��
def plot_factors_mixed_freq(results, financial, corr):
    extracted_factor = results.factors.filtered['0']
    # ��������Եķ��ŵ�����ͬ���ӵķ���
    if corr < 0:
        extracted_factor = -extracted_factor

    extracted_factor_series = pd.Series(extracted_factor.values, index=financial.index, name='0')
    # ��ԭʼ���ӽ��в�ֵ
    factor_interpolated = financial.interpolate(method='time')

    # ����ԭʼ���Ӻ���ȡ���ӵ�ʱ������ͼ
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(extracted_factor_series, label='����ָ��(��Ƶ)')
    ax.plot(factor_interpolated, '-')
    match financial.name:
        case 'roe_ttm2':
            ax.plot(financial, 'o', label='ROE_TTM(��Ƶ)')
            ax.set_title('����ָ��(��Ƶ) vs ROE_TTM(��Ƶ)')
        case 'yoyprofit':
            ax.plot(financial, 'o', label='������ͬ��(��Ƶ)')
    ax.set_title('����ָ��(��Ƶ) vs ������ͬ��(��Ƶ)')

    # ��ĩ��������
    for x in financial.dropna().index:
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.legend()
    ax.set_ylabel('Factor Value')

    # ����x�����ڱ�ǩ�ĸ�ʽ�ͼ��
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.show()

plot_factors_mixed_freq(results, finalcials['roe_ttm2'], corr)
plot_factors_mixed_freq(results, finalcials['yoyprofit'] / 100, corr)

# 6. �������
print("\nFactor loadings:")
# print(results.factors)

# 5. �����Ż�
# k_factors_list = [1, 2, 3]
# for k in k_factors_list:
#     print(f"\nResults for {k} factor(s):")
#     results = apply_dynamic_factor_model(data, k)
#     evaluate_model(results, factor, data)