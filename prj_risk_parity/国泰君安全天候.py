# coding=gbk
# Time Created: 2024/5/14 14:02
# Author  : Lucid
# FileName: ��̩����ȫ���.py
# Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime

# �����ʲ����
assets = {
    '000300.SS': '����300 (China Equity)',
    'TLT': 'iShares 20+ Year Treasury Bond ETF (Bonds)',
    'GLD': 'SPDR Gold Trust (Gold)',
    'DBC': 'Invesco DB Commodity Index Tracking Fund (Commodities)',
    'SPY': 'S&P 500 ETF (US Equity)'
}

# ��ȡ��ʷ����
start_date = '2011-01-01'
end_date = '2021-12-31'
df = pd.DataFrame()

for asset in assets:
    df[asset] = pdr.get_data_yahoo(asset, start=start_date, end=end_date)['Adj Close']

# ����ÿ��������
returns = df.pct_change().dropna()

# �������ƽ��ģ��
def risk_parity_weights(returns):
    vol = returns.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    return weights

# �����ʲ�Ȩ��
weights = risk_parity_weights(returns)
print("�ʲ�Ȩ��:\n", weights)

# �������ÿ��������
portfolio_returns = (returns * weights).sum(axis=1)

# ������ϵľ�ֵ����
portfolio_value = (1 + portfolio_returns).cumprod()

# ���ƾ�ֵ����
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='ȫ�����ʲ����')
plt.title('ȫ�����ʲ���Ͼ�ֵ����')
plt.xlabel('����')
plt.ylabel('��ֵ')
plt.legend()
plt.show()

# �������ձ��ʺͿ������
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns.mean() - risk_free_rate / 252
    return excess_returns / returns.std() * np.sqrt(252)

def calculate_calmar_ratio(portfolio_value):
    drawdown = portfolio_value / portfolio_value.cummax() - 1
    max_drawdown = drawdown.min()
    annual_return = (portfolio_value[-1] / portfolio_value[0]) ** (252 / len(portfolio_value)) - 1
    return annual_return / abs(max_drawdown)

sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
calmar_ratio = calculate_calmar_ratio(portfolio_value)

print(f"���ձ���: {sharpe_ratio:.2f}")
print(f"�������: {calmar_ratio:.2f}")