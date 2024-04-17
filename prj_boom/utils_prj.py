# coding=gbk
# Time Created: 2024/4/10 15:41
# Author  : Lucid
# FileName: utils_prj.py
# Software: PyCharm

import pandas as pd
import numpy as np


def set_index_col_wind(data, col_locator: str) -> pd.DataFrame:
    """
    ���� DataFrame ������������������

    ����:
    - data: Ҫ����� DataFrame��
    - col_locator: ���ڶ�λ�����е��ַ�����

    ����:
    - ������ DataFrame��
    """
    # �ҵ���һ���������ڵ���
    date_row = data.iloc[:, 0].apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().idxmax()
    # �ҵ��������ڵ���
    col_name_row = data.iloc[:date_row, 0].str.contains(col_locator).fillna(False).idxmax()
    # ��������
    data.columns = data.iloc[col_name_row]
    # ɾ�������м����Ϸ���������
    data = data.iloc[date_row + 1:]
    # ���õ�һ��Ϊ index,������Ϊ "date"
    data = data.set_index(data.columns[0])
    data.index.name = 'date'
    return data


def cap_outliers(data, threshold: float = 3.0):
    """
    ���쳣ֵ�趨Ϊ������׼��λ��
    :param threshold: �쳣ֵ�ж���ֵ,Ĭ��Ϊ3.0(������3����׼��)
    """
    for col in data.columns:
        series = data[col]
        mean = series.mean()
        std = series.std()
        upper_bound = mean + threshold * std
        lower_bound = mean - threshold * std

        # ���쳣ֵ�趨Ϊ������׼��λ��
        data.loc[series > upper_bound, col] = upper_bound
        data.loc[series < lower_bound, col] = lower_bound
    return data


def split_dataframe(whole_df):
    """
    ���ݿ��н� DataFrame �ָ�ɶ���� DataFrame,����������������������

    ����:
    - df_dict: �����ָ����� DataFrame ���ֵ�,��Ϊ '����'��'����' �� '������'��
    """
    df_dict = {}
    df_names = ['����', '����', '������']
    col_locators = ['����', '����', 'ָ������']

    start_idx = 0
    for idx, col in enumerate(whole_df.columns):
        if pd.isna(whole_df[col]).all() or idx == len(whole_df.columns) - 1:
            if start_idx <= idx:
                sub_df = whole_df.iloc[:, start_idx:idx + 1]
                df_name = df_names.pop(0)
                col_locator = col_locators.pop(0)
                sub_df = set_index_col_wind(sub_df, col_locator)
                # ɾ��ȫ��ΪNaN����
                sub_df = sub_df.dropna(axis=1, how='all')

                if df_name == '������':
                    sub_df = sub_df.replace(0, np.nan)
                if df_name == '����':
                    # ��ÿ��������MA4
                    sub_df = sub_df.sort_index(ascending=True).rolling(window=4).mean().sort_index(ascending=False)
                    # ɾ�������е�"������."�ַ���
                    sub_df.columns = sub_df.columns.str.replace('������.', '')

                df_dict[df_name] = sub_df.astype(float)
            start_idx = idx + 1

    return df_dict
