# coding=gbk
# Time Created: 2024/4/10 15:12
# Author  : Lucid
# FileName: data_visualizer.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_index_col_wind
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class DataVisualizer:
    def __init__(self, file_path, sheet_name):
        """
        ��ʼ�� DataVisualizer �ࡣ

        ����:
        - file_path: Excel �ļ���·����
        - sheet_name: Ҫ��ȡ�Ĺ��������ơ�
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.df_dict = self._split_dataframe()

    def _split_dataframe(self):
        """
        ���ݿ��н� DataFrame �ָ�ɶ���� DataFrame,����������������������

        ����:
        - df_dict: �����ָ����� DataFrame ���ֵ�,��Ϊ '����'��'����' �� '������'��
        """
        df_dict = {}
        df_names = ['����', '����', '������']
        col_locators = ['����', '����', 'ָ������']

        start_idx = 0
        for idx, col in enumerate(self.df.columns):
            if pd.isna(self.df[col]).all() or idx == len(self.df.columns) - 1:
                if start_idx <= idx:
                    sub_df = self.df.iloc[:, start_idx:idx + 1]
                    df_name = df_names.pop(0)
                    col_locator = col_locators.pop(0)
                    sub_df = set_index_col_wind(sub_df, col_locator)
                    # ɾ��ȫ��ΪNaN����
                    sub_df = sub_df.dropna(axis=1, how='all')

                    if df_name == '������':
                        sub_df = sub_df.replace(0, np.nan)

                    df_dict[df_name] = sub_df.astype(float)
                start_idx = idx + 1

        return df_dict

    def plot_data(self, df1_key, df2_key, df1_col, df2_col, start_date=None, end_date=None, marker1=None, marker2=None):
        """
        �������� DataFrame ��ָ���е����ݡ�

        ����:
        - df1_key: ��һ�� DataFrame �ļ���
        - df2_key: �ڶ��� DataFrame �ļ���
        - df1_col: ��һ�� DataFrame ��Ҫ���Ƶ�������
        - df2_col: �ڶ��� DataFrame ��Ҫ���Ƶ�������
        - start_date: ��ѡ����ʼ����,���ڹ������ݡ�
        - end_date: ��ѡ�Ľ�������,���ڹ������ݡ�
        """
        df1 = self.df_dict[df1_key]
        df2 = self.df_dict[df2_key]

        # ���ָ���������Ƿ�����ڶ�Ӧ�� DataFrame ��
        if df1_col not in df1.columns:
            raise ValueError(f"Column '{df1_col}' does not exist in DataFrame '{df1_key}'.")
        if df2_col not in df2.columns:
            raise ValueError(f"Column '{df2_col}' does not exist in DataFrame '{df2_key}'.")

        # ������ DataFrame �����������ϲ�,�����ȱʧֵ
        merged_df = pd.merge(df1[[df1_col]], df2[[df2_col]], left_index=True, right_index=True, how='outer').dropna(how='all')
        # merged_df.fillna(method='ffill', inplace=True)
        # merged_df.fillna(method='bfill', inplace=True)

        if start_date:
            merged_df = merged_df.loc[start_date:]
        if end_date:
            merged_df = merged_df.loc[:end_date]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(merged_df[df1_col], label=df1_col, linestyle='-', marker=marker1)
        ax2.plot(merged_df[df2_col], label=df2_col, linestyle='-', color='red', marker=marker2)

        ax1.set_xlabel('Date')
        ax1.set_ylabel(df1_col)
        ax2.set_ylabel(df2_col)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()


# ʹ��ʾ��
visualizer = DataVisualizer(rf"D:\WPS����\WPS����\����-���\ר���о�\�����о�\��ҵ�������ݿ���չʾ.xlsx", 'ʯ��ʯ��')
# visualizer.plot_data('����', '������', 'Ӫҵ����(ͬ��������)', '�й�:��ģ���Ϲ�ҵ����ֵ:ʯ�ͺ���Ȼ������ҵ:����ͬ��')#, start_date='2022-01-01', end_date='2023-12-31')
# visualizer.plot_data('����', '����', 'Ӫҵ����(ͬ��������)', '���̼�')#, start_date='2022-01-01', end_date='2023-12-31')
visualizer.plot_data('����', '����', 'Ӫҵ����(ͬ��������)', '���̼�', marker1='o', marker2=None)
