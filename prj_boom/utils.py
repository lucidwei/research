# coding=gbk
# Time Created: 2024/4/10 15:41
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm

import pandas as pd

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