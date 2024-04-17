# coding=gbk
# Time Created: 2024/4/10 15:41
# Author  : Lucid
# FileName: utils_prj.py
# Software: PyCharm

import pandas as pd
import numpy as np


def set_index_col_wind(data, col_locator: str) -> pd.DataFrame:
    """
    设置 DataFrame 的日期索引和列名。

    参数:
    - data: 要处理的 DataFrame。
    - col_locator: 用于定位列名行的字符串。

    返回:
    - 处理后的 DataFrame。
    """
    # 找到第一个日期所在的行
    date_row = data.iloc[:, 0].apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().idxmax()
    # 找到列名所在的行
    col_name_row = data.iloc[:date_row, 0].str.contains(col_locator).fillna(False).idxmax()
    # 设置列名
    data.columns = data.iloc[col_name_row]
    # 删除列名行及其上方的所有行
    data = data.iloc[date_row + 1:]
    # 设置第一列为 index,并命名为 "date"
    data = data.set_index(data.columns[0])
    data.index.name = 'date'
    return data


def cap_outliers(data, threshold: float = 3.0):
    """
    将异常值设定为三个标准差位置
    :param threshold: 异常值判断阈值,默认为3.0(即超过3个标准差)
    """
    for col in data.columns:
        series = data[col]
        mean = series.mean()
        std = series.std()
        upper_bound = mean + threshold * std
        lower_bound = mean - threshold * std

        # 将异常值设定为三个标准差位置
        data.loc[series > upper_bound, col] = upper_bound
        data.loc[series < lower_bound, col] = lower_bound
    return data


def split_dataframe(whole_df):
    """
    根据空列将 DataFrame 分割成多个子 DataFrame,并设置日期索引和列名。

    返回:
    - df_dict: 包含分割后的子 DataFrame 的字典,键为 '财务'、'行情' 和 '基本面'。
    """
    df_dict = {}
    df_names = ['财务', '行情', '基本面']
    col_locators = ['日期', '日期', '指标名称']

    start_idx = 0
    for idx, col in enumerate(whole_df.columns):
        if pd.isna(whole_df[col]).all() or idx == len(whole_df.columns) - 1:
            if start_idx <= idx:
                sub_df = whole_df.iloc[:, start_idx:idx + 1]
                df_name = df_names.pop(0)
                col_locator = col_locators.pop(0)
                sub_df = set_index_col_wind(sub_df, col_locator)
                # 删除全部为NaN的列
                sub_df = sub_df.dropna(axis=1, how='all')

                if df_name == '基本面':
                    sub_df = sub_df.replace(0, np.nan)
                if df_name == '财务':
                    # 对每列数据求MA4
                    sub_df = sub_df.sort_index(ascending=True).rolling(window=4).mean().sort_index(ascending=False)
                    # 删除列名中的"单季度."字符串
                    sub_df.columns = sub_df.columns.str.replace('单季度.', '')

                df_dict[df_name] = sub_df.astype(float)
            start_idx = idx + 1

    return df_dict
