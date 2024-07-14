# coding=gbk
# Time Created: 2024/7/1 17:15
# Author  : Lucid
# FileName: ȫA�ײ���������.py
# Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# ��ȡExcel�ļ�
file_path = rf"D:\WPS����\WPS����\����-���\�о�trial\���ȫA.xlsx"  # �滻Ϊ���Excel�ļ�·��
df = pd.read_excel(file_path, header=3)

# ȷ��Date����datetime����
df['Date'] = pd.to_datetime(df['Date'])

# ��Date������Ϊ����
df.set_index('Date', inplace=True)

# ����100�չ�����λ���ĵ�5%��λ
# df['100d_pct'] = df['close'].rolling(window=100).quantile(0.5)
# df['100d_pct'] = df['close'].rolling(window=400).min() * 1.05
df['100d_pct'] = df['close'].rolling(window=100).min() * 1.05
df['400d_pct'] = df['close'].rolling(window=400).min() * 1.10

# ����20���ƶ�ƽ���ɽ���
df['20d_avg_amt'] = df['amt'].rolling(window=10).mean()

# �������̼��Ƿ�
df['pct_change'] = df['close'].pct_change() * 100

# ���������
df['cond_close_100d'] = df['close'].shift(1) <= df['100d_pct']
df['cond_close_400d'] = df['close'].shift(1) <= df['400d_pct']
df['cond_amt'] = df['amt'] > df['20d_avg_amt'] * 1.1
df['cond_pct_change'] = df['pct_change'] > 2

# ɸѡ����
condition = (
    df['cond_close_100d'] &
    df['cond_close_400d'] &
    df['cond_amt'] &
    df['cond_pct_change']
)


# ɸѡ��������������
filtered_df = df[condition]

# # ���ĳһ�����ϸ��Ϣ
# def check_date_details(date):
#     if date in df.index:
#         details = df.loc[date]
#         print(f"Details for {date}:")
#         print(details)
#         print("\nConditions:")
#         print(f"Close <= 100d_pct: {details['close']} <= {details['100d_pct']} -> {details['close'] <= details['100d_pct']}")
#         print(f"Amt > 20d_avg_amt * 1.2: {details['amt']} > {details['20d_avg_amt'] * 1.2} -> {details['amt'] > details['20d_avg_amt'] * 1.2}")
#         print(f"Pct_change > 1: {details['pct_change']} > 1 -> {details['pct_change'] > 1}")
#     else:
#         print(f"No data available for {date}")
#
# # ���磬���1995-01-06
# check_date_details(pd.Timestamp('2008-11-10'))

# ������
print(filtered_df)




# ��ȡ����7��1�յ�����
july_1_data = df[(df.index.month == 7) & (df.index.day == 1)]

# �����ǵ���
# july_1_data['pct_change'] = july_1_data['close'].pct_change() * 100

# ��ӡ���߱�����
print(july_1_data[['close', 'pct_change']])




# �������̼�����
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='���̼�', color='blue')

# ��עɸѡ����������
plt.scatter(filtered_df.index, filtered_df['close'], color='red', label='ɸѡ����', marker='o')

# ��ӱ���ͱ�ǩ
plt.title('���̼����Ƽ�ɸѡ����')
plt.xlabel('����')
plt.ylabel('���̼�')
plt.legend()

# ��ʾ����
plt.grid(True)

# ��ʾͼ��
plt.show()

# �����Ҫ����������Excel��Ҳ����ʹ�����´��룺
# filtered_df.to_excel('ɸѡ���.xlsx')