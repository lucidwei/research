# coding=gbk
# Time Created: 2024/5/23 13:40
# Author  : Lucid
# FileName: utils.py
# Software: PyCharm
import pandas as pd


def align_signals(**signals):
    """
    ������ pandas Series���������ǵĹ�������������ӡ��ÿ�� Series �ж�������ڡ�

    ����:
    **signals: ��� pandas Series ��Ҫ������źţ�ʹ�ùؼ��ֲ�����ʽ����

    ����:
    �����Ķ�� pandas Series ���ֵ�
    """
    if not signals:
        raise ValueError("������Ҫһ�� Series ��Ϊ���롣")

    # ��ӡ���ڶ���ı�����
    signal_names = ', '.join(signals.keys())
    print(f"���ڶ������±���: {signal_names}")

    # �ҳ����е�����
    common_index = None
    for signal in signals.values():
        if common_index is None:
            common_index = signal.index
        else:
            common_index = common_index.intersection(signal.index)

    # ��ӡ��ʾ��Ϣ
    for name, signal in signals.items():
        extra_dates = signal.index.difference(common_index)
        if not extra_dates.empty:
            print(f"{name} ������������: {extra_dates.tolist()}")

    # ѡȡ���е�����
    aligned_signals = {name: signal[common_index] for name, signal in signals.items()}

    return aligned_signals

# ʾ���÷�
# ������������ pandas Series: erp_signal, us_tips_signal, volume_ma_signal
# aligned_signals = align_signals(erp_signal=erp_signal, us_tips_signal=us_tips_signal, volume_ma_signal=volume_ma_signal)
# erp_signal = aligned_signals['erp_signal']
# us_tips_signal = aligned_signals['us_tips_signal']
# volume_ma_signal = aligned_signals['volume_ma_signal']