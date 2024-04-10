# coding=gbk
# Time Created: 2024/4/10 15:41
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm

import pandas as pd

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