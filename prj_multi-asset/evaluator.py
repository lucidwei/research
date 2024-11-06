# coding=gbk
# Time Created: 2024/10/30 14:58
# Author  : Lucid
# FileName: evaluator.py
# Software: PyCharm
import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, strategy_name, price_data, weights_history):
        self.strategy_name = strategy_name
        self.price_data = price_data
        self.weights_history = weights_history
        self.net_value = self.calculate_net_value()
        self.performance = self.calculate_performance()

    def calculate_net_value(self):
        # 计算资产的日收益率
        returns = self.price_data.pct_change().dropna()

        # 权重向前填充，对齐到收益率数据
        aligned_weights = self.weights_history.reindex(returns.index).fillna(method='ffill')

        # 计算组合的日收益率
        portfolio_returns = (aligned_weights * returns).sum(axis=1)
        net_value = (1 + portfolio_returns).cumprod()

        # 在序列开始前插入初始净值1
        start_date = net_value.index[0] - pd.Timedelta(days=1)
        net_value = pd.concat([pd.Series([1], index=[start_date]), net_value])

        return net_value

    def calculate_performance(self):
        net_value = self.net_value
        returns = net_value.pct_change().dropna()
        cumulative_return = net_value.iloc[-1] / net_value.iloc[0] - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(net_value)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        max_drawdown = ((net_value / net_value.cummax()) - 1).min()
        # 持有一年正收益比率和收益中位数
        holding_period = 252
        if len(net_value) > holding_period:
            holding_returns = net_value.pct_change(periods=holding_period).dropna()
            positive_return_ratio = (holding_returns > 0).mean()
            median_return = holding_returns.median()
        else:
            positive_return_ratio = np.nan
            median_return = np.nan
        performance = {
            '累计收益率': cumulative_return,
            '年化收益率': annualized_return,
            '年化波动率': annualized_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '持有一年正收益比率': positive_return_ratio,
            '持有一年收益中位数': median_return
        }
        return performance