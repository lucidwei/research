# coding=gbk
# Time Created: 2023/4/27 22:23
# Author  : Lucid
# FileName: evaluator.py
# Software: PyCharm
import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, asset_allocator):
        self.asset_allocator = asset_allocator
        self.data = asset_allocator.data
        self.prepare_monthly_returns()
        self.calculate_portfolio_value()
        self.get_all_performance_metrics()

    def calculate_portfolio_value(self):
        # 获取月度仓位
        monthly_weights = self.asset_allocator.adjusted_monthly_weights

        # 计算投资组合收益率
        portfolio_returns = self.calculate_port_returns(monthly_weights,
                                                        self.stock_monthly_returns_aligned,
                                                        self.bond_monthly_returns_aligned,
                                                        self.gold_monthly_returns_aligned)

        # 计算基准组合收益率
        benchmark_weights = pd.DataFrame(
            [self.asset_allocator.parameters['benchmark_weight']] * len(monthly_weights.index),
            columns=['stock', 'bond', 'gold'],
            index=monthly_weights.index)

        benchmark_returns = self.calculate_port_returns(benchmark_weights,
                                                        self.stock_monthly_returns_aligned,
                                                        self.bond_monthly_returns_aligned,
                                                        self.gold_monthly_returns_aligned)

        # 计算组合净值
        # Shift the index to next month end，从而表示月末之前这个月的收益和月末的净值
        portfolio_returns.index = portfolio_returns.index.to_series().apply(lambda date: pd.to_datetime(date + pd.offsets.MonthEnd()))
        benchmark_returns.index = benchmark_returns.index.to_series().apply(lambda date: pd.to_datetime(date + pd.offsets.MonthEnd()))
        self.portfolio_value = (1 + portfolio_returns).cumprod()
        self.benchmark_value = (1 + benchmark_returns).cumprod()

    def prepare_monthly_returns(self):
        # 获取股票、债券和黄金的收盘价数据
        stock_prices = self.data.data_easy_dict['stock_prices']
        bond_prices = self.data.data_dict['cba']['close']
        gold_prices = self.data.data_easy_dict['gold_prices']

        # 对日度价格数据进行月末对齐
        stock_prices.index = pd.to_datetime(stock_prices.index)
        stock_prices_month_end = stock_prices.resample('M').last()

        bond_prices.index = pd.to_datetime(bond_prices.index)
        bond_prices_month_end = bond_prices.resample('M').last()

        gold_prices.index = pd.to_datetime(gold_prices.index)
        gold_prices_month_end = gold_prices.resample('M').last()

        # 计算月度收益率
        stock_monthly_returns = stock_prices_month_end.pct_change().shift(-1)
        bond_monthly_returns = bond_prices_month_end.pct_change().shift(-1)
        gold_monthly_returns = gold_prices_month_end.pct_change().shift(-1)

        # 对齐索引
        monthly_weights = self.asset_allocator.adjusted_monthly_weights
        self.stock_monthly_returns_aligned = stock_monthly_returns.reindex(monthly_weights.index)
        self.bond_monthly_returns_aligned = bond_monthly_returns.reindex(monthly_weights.index)
        self.gold_monthly_returns_aligned = gold_monthly_returns.reindex(monthly_weights.index)

    def calculate_port_returns(self, weights, stock_returns, bond_returns, gold_returns):
        return (weights.iloc[:, 0] * stock_returns +
                weights.iloc[:, 1] * bond_returns +
                weights.iloc[:, 2] * gold_returns)

    def calculate_sharpe_ratio(self, values, risk_free_rate=0):
        returns = values.pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self, values):
        value_max = values.cummax()
        drawdown = (values - value_max) / value_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def get_performance_metrics(self, values):
        sharpe_ratio = self.calculate_sharpe_ratio(values)
        max_drawdown = self.calculate_max_drawdown(values)
        performance_metrics = {'Sharpe Ratio': sharpe_ratio, 'Max Drawdown': max_drawdown}
        return performance_metrics

    def get_all_performance_metrics(self):
        portfolio_metrics = self.get_performance_metrics(self.portfolio_value)
        benchmark_metrics = self.get_performance_metrics(self.benchmark_value)
        self.all_performance_metrics = {'Portfolio': portfolio_metrics, 'Benchmark': benchmark_metrics}

