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
        # �����ʲ�����������
        returns = self.price_data.pct_change().dropna()

        # Ȩ����ǰ��䣬���뵽����������
        aligned_weights = self.weights_history.reindex(returns.index).fillna(method='ffill')

        # ������ϵ���������
        portfolio_returns = (aligned_weights * returns).sum(axis=1)
        net_value = (1 + portfolio_returns).cumprod()

        # �����п�ʼǰ�����ʼ��ֵ1
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
        # ����һ����������ʺ�������λ��
        holding_period = 252
        if len(net_value) > holding_period:
            holding_returns = net_value.pct_change(periods=holding_period).dropna()
            positive_return_ratio = (holding_returns > 0).mean()
            median_return = holding_returns.median()
        else:
            positive_return_ratio = np.nan
            median_return = np.nan
        performance = {
            '�ۼ�������': cumulative_return,
            '�껯������': annualized_return,
            '�껯������': annualized_volatility,
            '���ձ���': sharpe_ratio,
            '���س�': max_drawdown,
            '����һ�����������': positive_return_ratio,
            '����һ��������λ��': median_return
        }
        return performance