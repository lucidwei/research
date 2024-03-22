# coding=gbk
# Time Created: 2024/3/20 22:34
# Author  : Lucid
# FileName: ��Ѷ����ع�.py
# Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# ����Excel�ļ�
file_path = "D:\Downloads\��Ѷ�ع� 0700.HK_��Ʊ�ع�.xlsx"
df = pd.read_excel(file_path)

# ������ϴ��ȥ��ǰ���в��趨��ȷ������
df_cleaned = df.drop([0, 1]).reset_index(drop=True)
df_cleaned.columns = ['����', '�ع�����', '�ع����', '��߼�', '��ͼ�', '�������ع���']

# ת����������
df_cleaned['����'] = pd.to_datetime(df_cleaned['����'])
df_cleaned['�ع����'] = df_cleaned['�ع����'].astype(float)

# ��ȡ��ݣ�������ݻ��ܻع����
df_cleaned['���'] = df_cleaned['����'].dt.year
annual_repurchase_amount = df_cleaned.groupby('���')['�ع����'].sum().reset_index()

# ����ֱ��ͼ
plt.figure(figsize=(8, 5))
plt.bar(annual_repurchase_amount['���'], annual_repurchase_amount['�ع����'], color='skyblue')
plt.xlabel('���')
plt.ylabel('�ع���� (��)')
plt.title('��Ѷ�ع������Ʊ�ع����')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
