# coding=gbk
# Time Created: 2022/12/26 17:37
# Author  : Lucid
# FileName: funcs.py
# Software: PyCharm
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def get_bull_or_bear(series, order=100):
    """获取时间区间所处的牛熊市状态
    传入: series如close, order代表划分前后追朔的数据量,数据量越大,精度越小
    返回: 交易日的牛熊市的分类，series"""

    # 利用scipy在前后order个交易日内寻找极值点
    x = series.values
    high = argrelextrema(x, np.greater, order=order)[0]
    low = argrelextrema(x, np.less, order=order)[0]

    high_s = pd.Series('high', series.iloc[high].index)
    low_s = pd.Series('low', series.iloc[low].index)

    data1 = pd.concat([high_s, low_s]).sort_index()
    other = []
    for i in range(len(data1) - 1):  # 去除重复值划分
        if data1.iloc[i] == data1.iloc[i + 1]:
            other.append(data1.index[i])
    data1.drop(other, inplace=True)

    data1[series.index[-1]] = data1.iloc[-2]  # 加上开头与结束的归类
    data1[series.index[0]] = data1.iloc[1]
    data1.sort_index(inplace=True)  # 获得牛熊分界点

    bull_data = pd.Series(False, series.index, name='is_bull')  # 获得每一交易日属于牛市期还是熊市期
    if data1[0] == 'high':
        is_bull = False
    else:
        is_bull = True
    for i in range(len(data1) - 1):
        if is_bull:
            bull_data[data1.index[i]:data1.index[i + 1]] = True
            is_bull = False
        else:
            is_bull = True
    return bull_data


def get_stats(bull_dates, ori_series, if_bull: bool):
    """
    从牛熊划分与原始行情序列得到行情统计量。
    :param bull_dates: 牛熊划分结果序列，用于计算"start","end","length"
    :param ori_series: 原始行情序列，用于计算"magnitude"
    :param if_bull: 统计上涨(or下跌)序列
    :return: dataframe，四列 "start","end","length","magnitude"
    """
    stats_df = pd.DataFrame()
    counting = False
    for index, value in bull_dates.items():
        if value == True and not counting:
            # start = index
            counting = True
            stats_df = stats_df.append({'start': index}, ignore_index=True)
        elif value == False and counting:
            counting = False
            stats_df = stats_df.append({'end': last_index}, ignore_index=True)
        last_index = index
    stats_df['end'] = stats_df['end'].shift(-1)
    stats_df = stats_df.dropna().reset_index(drop=True)
    stats_df['length'] = (stats_df['end'] - stats_df['start']).dt.days
    stats_df['magnitude'] = 100*(ori_series[stats_df['end']].reset_index(drop=True) - ori_series[stats_df['start']].reset_index(drop=True))
    return stats_df




