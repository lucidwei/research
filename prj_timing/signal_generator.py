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

    def generate_signals_for_all_strategies(self, strategies_params=None):
        """
        Generates signals for all strategies with optional parameter optimization.

        Parameters:
            strategies_params (dict): Dictionary containing strategy parameters for optimization.

        """
        if strategies_params:
            for strategy_num, params in strategies_params.items():
                signals = self.generate_strategy_signals(strategy_num, **params)
                self.df[f'strategy{strategy_num}_signal'] = signals
        else:
            # Default strategies without parameter optimization
            for strategy_num in range(1, 6):
                signals = self.generate_strategy_signals(strategy_num)
                self.df[f'strategy{strategy_num}_signal'] = signals

            # Strategy 6
            strategy6_signals = self.generate_strategy6_signals()
            self.df['strategy6_signal'] = strategy6_signals

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
            # Strategy 1: Example with parameter 'ma_window'
            ma_window = kwargs.get('ma_window', 2)
            df['中长期贷款同比MA2_prev'] = df['中长期贷款同比MA2'].shift(1)
            signals = np.where(df['中长期贷款同比MA2'] > df['中长期贷款同比MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index).shift(1)

        elif strategy_num == 2:
            # Strategy 2: Example with parameters 'ma_window1', 'ma_window2'
            ma_window1 = kwargs.get('ma_window1', 3)
            ma_window2 = kwargs.get('ma_window2', 3)
            df[f'M1同比MA{ma_window1}_prev'] = df['M1同比MA3'].shift(1)
            df[f'M1-PPI同比MA{ma_window2}_prev'] = df['M1-PPI同比MA3'].shift(1)
            condition1 = df['M1同比MA3'] > df[f'M1同比MA{ma_window1}_prev']
            condition2 = df['M1-PPI同比MA3'] > df[f'M1-PPI同比MA{ma_window2}_prev']
            signals = np.where(condition1 | condition2, 1, 0)
            signals = pd.Series(signals, index=df.index).shift(1)

        elif strategy_num == 3:
            # Strategy 3: Example with parameter 'ma_window'
            ma_window = kwargs.get('ma_window', 2)
            df['美元指数MA2_prev'] = df['美元指数MA2'].shift(1)
            signals = np.where(df['美元指数MA2'] < df['美元指数MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index).shift(1)

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
        df = self.df
        basic_improved = (
                                 (df['strategy1_signal']) +
                                 (df['strategy2_signal']) +
                                 (df['strategy3_signal'])
                         ) >= 2

        technical_sell = df['strategy5_signal'] == -1
        technical_buy = df['strategy4_signal'] == 1

        condition1 = basic_improved & (~technical_sell)
        condition2 = technical_buy

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


