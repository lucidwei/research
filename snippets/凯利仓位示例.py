# coding=gbk
# Time Created: 2025/1/16 16:24
# Author  : Lucid
# FileName: ������λʾ��.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def cal_zero_cost_times(f, b, c, total_nums):
    A = 1
    count = 0
    df = pd.DataFrame({})
    win_or_lose = ['-'] + [True] * total_nums
    Assets = [A] * (total_nums + 1)
    df.insert(0, 'win_or_lose', win_or_lose)
    df.insert(1, 'Assets', Assets)
    # ����ʵ�����
    while count < total_nums:
        win_or_loss = np.random.uniform(0, 1)
        count += 1
        # ���� or ����
        if win_or_loss >= 0.5:
            df.loc[count, 'Assets'] = df.loc[count - 1, 'Assets'] * (1 + f * b)
            df.loc[count, 'win_or_lose'] = True
        else:
            df.loc[count, 'Assets'] = df.loc[count - 1, 'Assets'] * (1 - f * c)
            df.loc[count, 'win_or_lose'] = False
    return df


b = 2
c = 1
f = 0.5
total_nums = 1000
df = cal_zero_cost_times(f, b, c, total_nums)
print('ʵ�ʻ�ʤ����:', df['win_or_lose'][1:].sum() / total_nums)
print('ƽ��ÿ��ӯ��:', pow(df['Assets'][total_nums], 1 / total_nums) - 1)
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
print(df)
# ��������
plt.plot(df.index[:500], df['Assets'][:500])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Assets Curve')
plt.show()