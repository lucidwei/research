# coding=gbk
# Time Created: 2024/7/29 10:57
# Author  : Lucid
# FileName: demo.py
# Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# ��ȡExcel�ļ�
file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\FMP\�ʲ�������������.xlsx"  # ���Excel�ļ�·��
asset_data = pd.read_excel(file_path, sheet_name='�ʲ�����', header=1)
macro_data = pd.read_excel(file_path, sheet_name='�������')

# ����Ԥ����
# �ʲ���������
asset_data.columns = asset_data.iloc[0]
asset_data = asset_data.drop([0, 1]).reset_index(drop=True)
asset_data['����'] = pd.to_datetime(asset_data['����'])
asset_data.set_index('����', inplace=True)
asset_data = asset_data.apply(pd.to_numeric, errors='coerce')

# ���ʲ��۸�����ת��Ϊͬ�ȱ仯��
asset_data = asset_data.resample('M').last()
asset_data_yoy = asset_data.pct_change(periods=12) * 100
# TODO:��ʱ�Դ��� ɸѡ������2016��֮������ݣ�ɾ��֮ǰ������
asset_data_yoy = asset_data_yoy.loc['2016-01-01':]

# ��drop������Ϊnan���У�Ȼ��drop��nanֵ���У�������Щ������ӡ����
# ɾ��������Ϊ NaN ����
asset_data_yoy = asset_data_yoy.dropna(how='all')
# �ҵ��� NaN ֵ����
cols_with_nan = asset_data_yoy.columns[asset_data_yoy.isna().any()].tolist()
# ��ȡɾ�������������Ӧ������������ݵ������
earliest_dates = {}
for col in cols_with_nan:
    earliest_date = asset_data_yoy[col].first_valid_index()
    earliest_dates[col] = earliest_date
print("asset_data_yoyɾ�����м�������������ݵ������:")
for col, date in earliest_dates.items():
    print(f"{col}: {date}")
# ɾ���� NaN ֵ����
asset_data_yoy = asset_data_yoy.dropna(axis=1)



# �������
macro_data.columns = macro_data.iloc[0]
macro_data = macro_data.drop(range(0, 6)).reset_index(drop=True)
macro_data = macro_data.rename(columns={'ָ������': '����'})
macro_data['����'] = pd.to_datetime(macro_data['����'])
macro_data.set_index('����', inplace=True)
macro_data = macro_data[['�й�:����ҵPMI:12���ƶ�ƽ��:����ƽ��']]
macro_data = macro_data.apply(pd.to_numeric, errors='coerce')


# �ϲ�����
data = asset_data_yoy.join(macro_data, how='inner').dropna()
X = data.iloc[:, :-1]  # �ʲ���������
y = data.iloc[:, -1]   # PMI����

# LassoCV���н�����֤ѡ����ѳͷ�ϵ��
lasso_cv = LassoCV(cv=10, max_iter=10000).fit(X, y)  # ��������������
alpha = lasso_cv.alpha_

# ʹ����ѳͷ�ϵ������Lasso�ع�
lasso = Lasso(alpha=alpha, max_iter=10000).fit(X, y)  # ��������������
selected_features = X.columns[(lasso.coef_ != 0)]
selected_weights = lasso.coef_[lasso.coef_ != 0]
intercept = lasso.intercept_

# ��ӡѡ����ʲ�����Ȩ��
print("Selected features for FMP and their weights:")
for feature, weight in zip(selected_features, selected_weights):
    print(f"{feature}: {weight:.3f}")

# FMP����
fmp = np.dot(data[selected_features], selected_weights) + intercept

# ��������FMP���ݵ���DataFrame
fmp_series = pd.Series(fmp, index=data.index, name='FMP')

# �ϲ�FMP��ԭʼPMI���ݣ�ֻ�������߾������ݵ����ڲ���
combined_data = pd.concat([macro_data['�й�:����ҵPMI:12���ƶ�ƽ��:����ƽ��'], fmp_series], axis=1).dropna()

# ��ͼ�Ա�
plt.figure(figsize=(12, 6))
plt.plot(combined_data.index, combined_data['�й�:����ҵPMI:12���ƶ�ƽ��:����ƽ��'], label='ԭʼPMI', color='b')
plt.plot(combined_data.index, combined_data['FMP'], label='FMP', color='r')
plt.xlabel('����')
plt.ylabel('ֵ')
plt.title('FMP��ԭʼPMI�Ա�')
plt.legend()
plt.grid(True)
plt.show()

# # ��ͼ�Ա�
# plt.figure(figsize=(12, 6))
# plt.plot(macro_data.index, macro_data['�й�:����ҵPMI:12���ƶ�ƽ��:����ƽ��'], label='ԭʼPMI', color='b')
# plt.plot(data.index, fmp, label='FMP', color='r')
# plt.xlabel('����')
# plt.ylabel('ֵ')
# plt.title('FMP��ԭʼPMI�Ա�')
# plt.legend()
# plt.grid(True)
# plt.show()