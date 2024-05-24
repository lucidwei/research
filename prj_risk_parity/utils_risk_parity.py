# coding=gbk
# Time Created: 2024/5/23 13:40
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm
import pandas as pd


def align_signals(**signals):
    """
    对齐多个 pandas Series，根据它们的公共索引，并打印出每个 Series 中多余的日期。

    参数:
    **signals: 多个 pandas Series 需要对齐的信号，使用关键字参数形式传递

    返回:
    对齐后的多个 pandas Series 的字典
    """
    if not signals:
        raise ValueError("至少需要一个 Series 作为输入。")

    # 打印正在对齐的变量名
    signal_names = ', '.join(signals.keys())
    print(f"正在对齐以下变量: {signal_names}")

    # 找出共有的索引
    common_index = None
    for signal in signals.values():
        if common_index is None:
            common_index = signal.index
        else:
            common_index = common_index.intersection(signal.index)

    # 打印提示信息
    for name, signal in signals.items():
        extra_dates = signal.index.difference(common_index)
        if not extra_dates.empty:
            print(f"{name} 多了以下日期: {extra_dates.tolist()}")

    # 选取共有的索引
    aligned_signals = {name: signal[common_index] for name, signal in signals.items()}

    return aligned_signals

# 示例用法
# 假设您有三个 pandas Series: erp_signal, us_tips_signal, volume_ma_signal
# aligned_signals = align_signals(erp_signal=erp_signal, us_tips_signal=us_tips_signal, volume_ma_signal=volume_ma_signal)
# erp_signal = aligned_signals['erp_signal']
# us_tips_signal = aligned_signals['us_tips_signal']
# volume_ma_signal = aligned_signals['volume_ma_signal']