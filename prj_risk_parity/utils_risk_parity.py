# coding=gbk
# Time Created: 2024/5/23 13:40
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm
import pandas as pd


def align_signals(*signals):
    """
    ������ pandas Series���������ǵĹ�������������ӡ��ÿ�� Series �ж�������ڡ�

    ����:
    *signals: ��� pandas Series ��Ҫ������ź�

    ����:
    �����Ķ�� pandas Series ��Ԫ��
    """
    if not signals:
        raise ValueError("������Ҫһ�� Series ��Ϊ���롣")

    # �ҳ����е�����
    common_index = signals[0].index
    for signal in signals[1:]:
        common_index = common_index.intersection(signal.index)

    # ��ӡ��ʾ��Ϣ
    for i, signal in enumerate(signals):
        extra_dates = signal.index.difference(common_index)
        if not extra_dates.empty:
            print(f"signal {i} ������������: {extra_dates.tolist()}")

    # ѡȡ���е�����
    aligned_signals = tuple(signal[common_index] for signal in signals)

    return aligned_signals