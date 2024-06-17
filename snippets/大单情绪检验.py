# coding=gbk
# Time Created: 2024/6/17 10:57
# Author  : Lucid
# FileName: ����������.py
# Software: PyCharm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������
# �����ض��ľ���
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")


# ��ȡExcel�ļ�
file_path = r'D:\WPS����\WPS����\����-���\���ڻ㱨\�ձ�ģ������\�����ʽ����͸�ӱ�.xlsx'
sheet_name = '������'

data = pd.read_excel(file_path, sheet_name=sheet_name)

# ����Ԥ����
# ����һ����Ϊ��������ɾ��ǰ����
data.columns = data.iloc[1]
data = data.drop([0, 1]).reset_index(drop=True)

# ��������ת��Ϊ��������
data['����'] = pd.to_datetime(data['����'])
data.set_index('����', inplace=True)
# �������ת��Ϊ��ֵ����
data['���ȫA'] = pd.to_numeric(data['���ȫA'], errors='coerce')
data['������'] = pd.to_numeric(data['������'], errors='coerce')
data['������_MA5'] = pd.to_numeric(data['������_MA5'], errors='coerce')

# ɾ������NaN����
data = data.dropna(subset=['���ȫA', '������', '������_MA5'])

# ��������Ƿ�ʱ��˳������
if not data.index.is_monotonic_increasing:
    data = data.sort_index()

# USE_COLUMN = '������_MA5'
USE_COLUMN = '������'
# USE_COLUMN = 'Signal'




# # �趨�������ڴ�С
# n = 10
#
# # ������������ڵ����ֵ
# data['Max_Sentiment_Rolling'] = data['������'].rolling(window=n).max()
# data['Min_Sentiment_Rolling'] = data['������'].rolling(window=n).min()
#
# # �������ź���
# data['Signal'] = np.where(data[USE_COLUMN] < (data['Max_Sentiment_Rolling'] - 0.2), -1,
#                         np.where(data[USE_COLUMN] > (data['Min_Sentiment_Rolling'] + 0.2), 1, 0))
#
#
# df = pd.DataFrame(data)
#
# # ��ͼ
# fig, ax1 = plt.subplots(figsize=(14, 7))
#
# # �����ȫA�۸�
# ax1.plot(df.index, df['���ȫA'], color='blue', label='���ȫA �۸�')
# ax1.set_xlabel('����')
# ax1.set_ylabel('���ȫA �۸�', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # ʹ�ò�ͬ����ɫ��͸���������Ӱ��ʾ Signal
# ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == 1), color='red', alpha=0.3, label='Signal = 1')
# ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == -1), color='green', alpha=0.3, label='Signal = -1')
# # ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == 0), color='blue', alpha=0.1, label='Signal = 0')
#
# fig.tight_layout()
# fig.suptitle('���ȫA �۸��� Signal', y=1.02)
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
# plt.show()




USE_COLUMN = '������_MA5'
# USE_COLUMN = '������'
# USE_COLUMN = 'Signal'


# ���������
correlation = data['���ȫA'].corr(data[USE_COLUMN])
print(f"Correlation between ���ȫA and ������: {correlation:.4f}")

# ����1��5�׵�Granger�������
max_lag = 25
granger_results = grangercausalitytests(data[['���ȫA', USE_COLUMN]], max_lag)
# ��ӡÿ���ͺ������µ�p-value
for lag in range(1, max_lag+1):
    p_value = granger_results[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_value}")

# ����VARģ��
model = VAR(data[['���ȫA', USE_COLUMN]])
results = model.fit(maxlags=25, ic='aic')

# ����������
lead_lag_summary = results.test_causality('���ȫA', USE_COLUMN, kind='f')
print(f"Lead-lag relationship (causality test) result:\n {lead_lag_summary}")

# # ����������Ӧ������IRF��
irf = results.irf(10)  # ����10�ڵ�IRF
irf.plot(orth=True)
plt.show()

# Ԥ������ֽ�
fevd = results.fevd(10)
fevd.plot()

lag_order = results.k_ar
print(f"Optimal lag order: {lag_order}")

# ��ӡIRFֵ
# irf_values = irf.irfs
# print("Impulse Response Function values:\n", irf_values)

# ����data�ǰ��������ȫA���͡�����������DataFrame
model11 = VAR(data[['���ȫA', USE_COLUMN]])
results = model11.fit(11)  # ʹ������ͺ�����11

forecast = results.forecast(data[['���ȫA', USE_COLUMN]].values[-11:], steps=10)
forecast_df = pd.DataFrame(forecast, columns=['���ȫA', USE_COLUMN])
# print(forecast_df)


# ��ʼ���ź���
data['signal'] = 0

# �زⴰ��
train_size = 200  # ѵ������С
test_size = 100  # ���Լ���С���زⲿ�֣�

for t in range(train_size, train_size + test_size):
    # ѵ��������
    train_data = data.iloc[t - train_size:t]

    # ���VARģ��
    model = VAR(train_data[['���ȫA', USE_COLUMN]])
    results = model.fit(lag_order)

    # ����һ��Ԥ��
    forecast = results.forecast(train_data[['���ȫA', USE_COLUMN]].values[-lag_order:], steps=1)

    # ��ȡԤ����
    predicted_value = forecast[0, 0]  # Ԥ��� '���ȫA' ֵ

    # ���������ź�
    if predicted_value > train_data['���ȫA'].iloc[-1]:
        data.iloc[t, data.columns.get_loc('signal')] = 1  # �����ź�
    else:
        data.iloc[t, data.columns.get_loc('signal')] = -1  # �����ź�

# �����������
data['returns'] = data['���ȫA'].pct_change()
data['strategy_returns'] = data['signal'].shift(1) * data['returns']
cumulative_returns = (1 + data['strategy_returns']).cumprod() - 1

print(f"Cumulative returns from strategy: {cumulative_returns.iloc[-1]:.2%}")


# ����10����������
window = 10
data['rolling_corr'] = data['���ȫA'].rolling(window).corr(data[USE_COLUMN])

# �������ȫA�ļ۸�����
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['���ȫA'], label='���ȫA')

# �����Ӱ����
for start_date, end_date in zip(data.index[:-window], data.index[window:]):
    corr_value = data.loc[start_date:end_date, 'rolling_corr'].iloc[-1]
    if corr_value > 0.8:
        plt.axvspan(start_date, end_date, color='red', alpha=0.3)
    elif corr_value < -0.5:
        plt.axvspan(start_date, end_date, color='blue', alpha=0.3)

# ����ͼ���ͱ���
plt.legend()
plt.title(f'���ȫA�۸���{USE_COLUMN}���������')
plt.xlabel('Date')
plt.ylabel('���ȫA')
plt.show()