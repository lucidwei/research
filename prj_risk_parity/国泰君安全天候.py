# coding=gbk
# Time Created: 2024/5/14 14:02
# Author  : Lucid
# FileName: 国泰君安全天候.py
# Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime

# 定义资产类别
assets = {
    '000300.SS': '沪深300 (China Equity)',
    'TLT': 'iShares 20+ Year Treasury Bond ETF (Bonds)',
    'GLD': 'SPDR Gold Trust (Gold)',
    'DBC': 'Invesco DB Commodity Index Tracking Fund (Commodities)',
    'SPY': 'S&P 500 ETF (US Equity)'
}

# 获取历史数据
start_date = '2011-01-01'
end_date = '2021-12-31'
df = pd.DataFrame()

for asset in assets:
    df[asset] = pdr.get_data_yahoo(asset, start=start_date, end=end_date)['Adj Close']

# 计算每日收益率
returns = df.pct_change().dropna()

# 定义风险平价模型
def risk_parity_weights(returns):
    vol = returns.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    return weights

# 计算资产权重
weights = risk_parity_weights(returns)
print("资产权重:\n", weights)

# 计算组合每日收益率
portfolio_returns = (returns * weights).sum(axis=1)

# 计算组合的净值曲线
portfolio_value = (1 + portfolio_returns).cumprod()

# 绘制净值曲线
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='全天候多资产组合')
plt.title('全天候多资产组合净值曲线')
plt.xlabel('日期')
plt.ylabel('净值')
plt.legend()
plt.show()

# 计算夏普比率和卡玛比率
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

print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"卡玛比率: {calmar_ratio:.2f}")