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
    # �����������ڵ�����
    data = data.iloc[date_row:]
    # ���õ�һ��Ϊ index,������Ϊ "date"
    data = data.set_index(data.columns[0])
    data.index.name = 'date'
    return data


def cap_outliers(data, threshold: float = 3.0):
    """
    ���쳣ֵ�趨Ϊ������׼��λ��
    :param threshold: �쳣ֵ�ж���ֵ,Ĭ��Ϊ3.0(������3����׼��)
    """
    outliers = pd.DataFrame()

    for col in data.columns:
        series = data[col]
        mean = series.mean()
        std = series.std()
        upper_bound = mean + threshold * std
        lower_bound = mean - threshold * std

        # �ҳ��쳣ֵ
        col_outliers = data[(series > upper_bound) | (series < lower_bound)].copy()

        if not col_outliers.empty:
            # ֻ�����쳣ֵ���ڵ���
            col_outliers[col + '_original'] = col_outliers[col]

            # ���һ�б�ʾ���Ͻ绹���½��쳣
            col_outliers['bound'] = np.where(col_outliers[col] > upper_bound, 'upper', 'lower')

            # ���쳣ֵ��ӵ�outliers DataFrame
            outliers = pd.concat([outliers, col_outliers[[col + '_original', 'bound']]], axis=1)

        # ���쳣ֵ�趨Ϊ������׼��λ��
        data.loc[series > upper_bound, col] = upper_bound
        data.loc[series < lower_bound, col] = lower_bound

    # ����������outliers����ӡ
    if not outliers.empty:
        outliers = outliers.sort_index()
        print("���쳣ֵ��ɾ��:")
        # print(outliers)
    else:
        print("û�з����쳣ֵ��")

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
                elif df_name == '����':
                    # ��ÿ��������MA4
                    sub_df = sub_df.sort_index(ascending=True).rolling(window=4, min_periods=1).mean().sort_index(ascending=False)
                    # ɾ�������е�"������."�ַ���
                    sub_df.columns = sub_df.columns.str.replace('������.', '', regex=True)
                elif df_name == '����':
                    # �ԡ����顱����û�����⴦������ʽ˵��
                    pass

                # �������ո��ַ�����cell������ת��Ϊ NaN����������ת���ᱨ��
                sub_df = sub_df.replace(r'^\s*$', np.nan, regex=True)
                df_dict[df_name] = sub_df.astype(float)
            start_idx = idx + 1

    return df_dict
