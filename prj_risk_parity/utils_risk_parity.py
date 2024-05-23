# coding=gbk
# Time Created: 2024/5/23 13:40
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm
import pandas as pd


def align_signals(*signals):
    """
    对齐多个 pandas Series，根据它们的公共索引，并打印出每个 Series 中多余的日期。

    参数:
    *signals: 多个 pandas Series 需要对齐的信号

    返回:
    对齐后的多个 pandas Series 的元组
    """
    if not signals:
        raise ValueError("至少需要一个 Series 作为输入。")

    # 找出共有的索引
    common_index = signals[0].index
    for signal in signals[1:]:
        common_index = common_index.intersection(signal.index)

    # 打印提示信息
    for i, signal in enumerate(signals):
        extra_dates = signal.index.difference(common_index)
        if not extra_dates.empty:
            print(f"signal {i} 多了以下日期: {extra_dates.tolist()}")

    # 选取共有的索引
    aligned_signals = tuple(signal[common_index] for signal in signals)

    return aligned_signals