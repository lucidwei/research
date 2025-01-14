# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy import stats


class SignalGenerator:
    def __init__(self, df):
        """
        Initializes the SignalGenerator.

        Parameters:
            df (pd.DataFrame): Preprocessed data from DataHandler.
        """
        self.df = df.copy()
        self.strategies_results = {}
        self.strategy_names = {}  # 字典：策略编号 -> 策略名称

    def generate_signals_for_all_strategies(self, strategies_params=None, strategy_names=None):
        """
        Generates signals for all strategies with optional parameter optimization.

        Parameters:
            strategies_params (dict): Dictionary containing strategy parameters for optimization.
            strategy_names (dict): Dictionary mapping strategy numbers to strategy names.
        """
        if strategy_names:
            self.strategy_names = strategy_names
        else:
            # 默认策略名称为 strategy1, strategy2, ...
            self.strategy_names = {num: f'strategy{num}' for num in range(1, 6)}

        if strategies_params:
            for strategy_num, params in strategies_params.items():
                signals = self.generate_strategy_signals(strategy_num, **params)
                strategy_name = self.strategy_names.get(strategy_num, f'strategy{strategy_num}')
                self.df[f'{strategy_name}_signal'] = signals
        else:
            # Default strategies without parameter optimization
            for strategy_num in range(1, 6):
                signals = self.generate_strategy_signals(strategy_num)
                strategy_name = self.strategy_names.get(strategy_num, f'strategy{strategy_num}')
                self.df[f'{strategy_name}_signal'] = signals

            # Strategy 6
            strategy6_signals = self.generate_strategy6_signals()
            self.df['strategy6_signal'] = strategy6_signals

        return self.df

    def generate_strategy_signals(self, strategy_num, **kwargs):
        """
        Generates buy/sell signals for a specific strategy.

        Parameters:
            strategy_num (int): Strategy number (1-5).
            **kwargs: Additional parameters for the strategy.

        Returns:
            pd.Series: Signal series.
        """
        df = self.df
        signals = pd.Series(index=df.index, data=0)

        if strategy_num == 1:
            # Strategy 1: 中长期贷款同比MA2
            # df['中长期贷款同比MA2'] = calculate_moving_average(df, '中长期贷款同比MA2', window=2)
            df['中长期贷款同比MA2_prev'] = df['中长期贷款同比MA2'].shift(1)
            signals = np.where(df['中长期贷款同比MA2'] > df['中长期贷款同比MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index)

            # 将信号向前移动一个周期，表示次月末建仓
            signals = signals.shift(1)

        elif strategy_num == 2:
            # Strategy 2: M1同比MA3 and M1-PPI同比MA3
            df['M1同比MA3_prev'] = df['M1同比MA3'].shift(1)
            df['M1-PPI同比MA3_prev'] = df['M1-PPI同比MA3'].shift(1)
            condition1 = df['M1同比MA3'] > df['M1同比MA3_prev']
            condition2 = df['M1-PPI同比MA3'] > df['M1-PPI同比MA3_prev']
            signals = np.where(condition1 | condition2, 1, 0)
            signals = pd.Series(signals, index=df.index)

            # 将信号向前移动一个周期，表示次月末建仓
            signals = signals.shift(1)

        elif strategy_num == 3:
            # Strategy 3: 美元指数MA2
            df['美元指数MA2_prev'] = df['美元指数MA2'].shift(1)
            signals = np.where(df['美元指数MA2'] < df['美元指数MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 4:
            # Strategy 4: Parameters like 'percentile_threshold', 'volume_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '市净率:上证指数:月:最后一条', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] < percentile_threshold
            condition_yoy = df['上证综合指数:成交金额:月:合计值:同比'] > 0
            condition_mom = df['上证综合指数:成交金额:月:合计值:环比'] > 0
            condition_volume = condition_yoy & condition_mom

            condition_index_yoy = df['上证综合指数:月:最后一条:同比'] > 0
            condition_index_mom = df['上证综合指数:月:最后一条:环比'] > 0
            condition_price = condition_index_yoy & condition_index_mom

            signals = np.where(condition_per & condition_volume & condition_price, 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 5:
            # Strategy 5: Parameters like 'percentile_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '市净率:上证指数:月:最后一条', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] > percentile_threshold

            condition_mom_volume = (df['上证综合指数:成交金额:月:合计值:环比'] > 0) & (
                    df['上证综合指数:月:最后一条:环比'] < 0)
            condition_yoy_price = (df['上证综合指数:月:最后一条:同比'] > 0) & (
                    df['上证综合指数:成交金额:月:合计值:同比'] < 0)

            condition_sell = condition_mom_volume | condition_yoy_price

            signals = np.where(condition_per & condition_sell, -1, 0)
            signals = pd.Series(signals, index=df.index)

        return signals

    def generate_strategy6_signals(self):
        """
        Generates signals for Strategy 6 based on Strategies 1-5.

        Returns:
            pd.Series: Strategy 6 signals.
        """
        # 假设 strategy_names 已包含所有策略的名称
        df = self.df
        # 获取策略1-5的信号列名
        strategy_signals = [f"{self.strategy_names.get(num, f'strategy{num}')}_signal" for num in range(1, 6)]

        # 基本面改善信号：策略1、策略2、策略3中至少两个为1
        basic_improved = df[[f'strategy{num}_signal' for num in range(1, 4)]].sum(axis=1) >= 2

        # 技术面卖出信号：策略5信号为-1
        technical_sell = df[f"{self.strategy_names.get(5, 'strategy5')}_signal"] == -1

        # 技术面买入信号：策略4信号为1
        technical_buy = df[f"{self.strategy_names.get(4, 'strategy4')}_signal"] == 1

        # **模型规则**：
        # 条件1：基本面改善且无技术面卖出信号
        condition1 = basic_improved & (~technical_sell)

        # 条件2：技术面买入信号
        condition2 = technical_buy

        # 最终信号：
        # 满足条件1或条件2则买入（1）
        # 满足技术面卖出信号则卖出（-1）
        # 否则保持持仓（0）
        signals = np.where(
            condition1 | condition2, 1,
            np.where(technical_sell, -1, 0)
        )

        return pd.Series(signals, index=df.index)

    def calculate_rolling_percentile_rank(self, df, column, window):
        """
        Calculates the rolling percentile rank for a specified column.

        Parameters:
            df (pd.DataFrame): Dataframe.
            column (str): Column name.
            window (int): Rolling window size.

        Returns:
            pd.Series: Rolling percentile rank.
        """
        return df[column].rolling(window=window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100,
            raw=False
        )

    def optimize_parameters(self, strategy_num, param_grid):
        """
        Optimizes parameters for a given strategy by evaluating performance over a grid.

        Parameters:
            strategy_num (int): Strategy number to optimize.
            param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter settings.

        Returns:
            dict: Best parameters and corresponding performance metric.
        """
        from itertools import product

        best_metric = -np.inf
        best_params = None

        # Generate all combinations of parameters
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        for params in param_combinations:
            # Generate signals with current parameters
            signals = self.generate_strategy_signals(strategy_num, **params)

            # Temporarily store signals in dataframe
            temp_df = self.df.copy()
            temp_df[f'strategy{strategy_num}_signal'] = signals

            # Assume we have a PerformanceEvaluator instance to evaluate performance
            # Here, we'll simulate performance metric calculation (e.g., Sharpe Ratio)
            # In practice, you should integrate with PerformanceEvaluator

            # Placeholder: Calculate a mock performance metric
            # Replace this with actual backtesting and performance calculation
            mock_metric = np.random.random()  # Replace with actual metric

            if mock_metric > best_metric:
                best_metric = mock_metric
                best_params = params

        return {'best_params': best_params, 'best_metric': best_metric}


