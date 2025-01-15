# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy import stats


class SignalGenerator:
    def __init__(self, indices_data, macro_data):
        """
        Initializes the SignalGenerator.

        Parameters:
            df (pd.DataFrame): Preprocessed data from DataHandler.
        """
        self.indices_data = indices_data.copy()
        self.macro_data = macro_data.copy()
        self.strategies_results = {}
        self.strategy_names = {}  # 字典：策略编号 -> 策略名称

    def generate_signals_for_all_strategies(self, strategies_params=None, strategy_names=None, selected_indices=None):
        """
        Generates signals for all strategies with optional parameter optimization.

        Parameters:
            strategies_params (dict): Dictionary containing strategy parameters for optimization.
            strategy_names (dict): Dictionary mapping strategy numbers to strategy names.
        """
        if strategy_names:
            self.strategy_names = strategy_names
        else:
            # Default strategy names (for strategies 1-6 per index)
            self.strategy_names = {}
            for index_name in self.indices_data.keys():
                for num in range(1, 7):
                    strategy_id = f"{index_name}_strategy{num}"
                    self.strategy_names[strategy_id] = strategy_id

        if selected_indices is None:
            # Default to all indices
            selected_indices = list(self.indices_data.keys())

        for index_name in selected_indices:
            for strategy_num in range(1, 6):
                strategy_id = f"{index_name}_strategy{strategy_num}"
                strategy_name = self.strategy_names.get(strategy_id, strategy_id)

                if strategies_params and strategy_id in strategies_params:
                    params = strategies_params[strategy_id]
                    signals = self.generate_strategy_signals(index_name, strategy_num, **params)
                else:
                    signals = self.generate_strategy_signals(index_name, strategy_num)

                self.indices_data[index_name][f'{strategy_name}_signal'] = signals

            # Generate strategy6 (aggregated strategy)
            strategy6_id = f"{index_name}_strategy6"
            strategy6_name = self.strategy_names.get(strategy6_id, strategy6_id)
            strategy6_signals = self.generate_strategy6_signals(index_name)
            self.indices_data[index_name][f'{strategy6_name}_signal'] = strategy6_signals

        return self.indices_data

    def generate_strategy_signals(self, index_name, strategy_num, **kwargs):
        """
        Generates buy/sell signals for a specific strategy.

        Parameters:
            strategy_num (int): Strategy number (1-5).
            **kwargs: Additional parameters for the strategy.

        Returns:
            pd.Series: Signal series.
        """
        df = self.indices_data[index_name]
        macro_data = self.macro_data.copy()
        signals = pd.Series(index=df.index, data=0)

        if strategy_num == 1:
            # Strategy 1: 中长期贷款同比MA2
            macro_data['中长期贷款同比MA2_prev'] = macro_data['中长期贷款同比MA2'].shift(1)
            signals = np.where(macro_data['中长期贷款同比MA2'] > macro_data['中长期贷款同比MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

            # 将信号向前移动一个周期，表示次月末建仓
            signals = signals.shift(1)

        elif strategy_num == 2:
            # Strategy 2: M1同比MA3 and M1-PPI同比MA3
            macro_data['M1同比MA3_prev'] = macro_data['M1同比MA3'].shift(1)
            macro_data['M1-PPI同比MA3_prev'] = macro_data['M1-PPI同比MA3'].shift(1)
            condition1 = macro_data['M1同比MA3'] > macro_data['M1同比MA3_prev']
            condition2 = macro_data['M1-PPI同比MA3'] > macro_data['M1-PPI同比MA3_prev']
            signals = np.where(condition1 | condition2, 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

            # 将信号向前移动一个周期，表示次月末建仓
            signals = signals.shift(1)

        elif strategy_num == 3:
            # Strategy 3: 美元指数MA2
            macro_data['美元指数MA2_prev'] = macro_data['美元指数MA2'].shift(1)
            signals = np.where(macro_data['美元指数MA2'] < macro_data['美元指数MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

        elif strategy_num == 4:
            # Strategy 4: Parameters like 'percentile_threshold', 'volume_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '市净率:指数', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] < percentile_threshold
            condition_yoy = df['指数:成交金额:合计值:同比'] > 0
            condition_mom = df['指数:成交金额:合计值:环比'] > 0
            condition_volume = condition_yoy & condition_mom

            condition_index_yoy = df['指数:最后一条:同比'] > 0
            condition_index_mom = df['指数:最后一条:环比'] > 0
            condition_price = condition_index_yoy & condition_index_mom

            signals = np.where(condition_per & condition_volume & condition_price, 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 5:
            # Strategy 5: Parameters like 'percentile_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '市净率:指数', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] > percentile_threshold

            condition_mom_volume = (df['指数:成交金额:合计值:环比'] > 0) & (
                    df['指数:最后一条:环比'] < 0)
            condition_yoy_price = (df['指数:最后一条:同比'] > 0) & (
                    df['指数:成交金额:合计值:同比'] < 0)

            condition_sell = condition_mom_volume | condition_yoy_price

            signals = np.where(condition_per & condition_sell, -1, 0)
            signals = pd.Series(signals, index=df.index)

        return signals

    def generate_strategy6_signals(self, index_name):
        """
        Generates signals for Strategy 6 based on Strategies 1-5.

        Returns:
            pd.Series: Strategy 6 signals.
        """
        # 假设 strategy_names 已包含所有策略的名称
        df = self.indices_data[index_name]

        # 基本面改善信号：策略1、策略2、策略3中至少两个为1
        basic_improved = df[[f'{index_name}_strategy{num}_signal' for num in range(1, 4)]].sum(axis=1) >= 2

        # 技术面卖出信号：策略5信号为-1
        technical_sell = df[f"{index_name}_strategy5_signal"] == -1

        # 技术面买入信号：策略4信号为1
        technical_buy = df[f"{index_name}_strategy4_signal"] == 1

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

    def generate_turnover_strategy_signals(self, index_name, holding_days=10, percentile_window_years=4, next_day_open=True):
        """
        Generates signals for the turnover-based strategy.

        Parameters:
            index_name (str): Name of the index.
            holding_days (int): Number of days to hold the position.
            percentile_window_years (int): Number of years to calculate historical percentiles.
            next_day_open (bool): Whether to open positions the next day after signal.

        Returns:
            pd.Series: Signal series with 1 for buy, -1 for sell, 0 for hold.
        """
        df = self.indices_data[index_name].copy()
        strategy_name = f"{index_name}_strategy6"

        # Initialize signal column
        df[strategy_name + '_signal'] = 0

        # Record holding end dates
        holding_end_date = None

        for current_date in df.index:
            if holding_end_date and current_date >= holding_end_date:
                # Sell signal to close position
                df.at[current_date, strategy_name + '_signal'] = -1
                holding_end_date = None

            if holding_end_date:
                # Currently holding, skip further signals
                continue

            # Calculate recent 60 days turnover trend
            recent_turnover = df.loc[:current_date].tail(60)['指数:成交金额:合计值']
            if len(recent_turnover) < 60:
                continue  # Not enough data

            current_turnover = recent_turnover.iloc[-1]
            avg_turnover_60 = recent_turnover.mean()
            turnover_trend = current_turnover / avg_turnover_60

            # Calculate historical turnover trends over the percentile window
            start_date = current_date - pd.DateOffset(years=percentile_window_years)
            historical_df = df.loc[start_date:current_date].tail(60 * percentile_window_years)
            if len(historical_df) < 60 * percentile_window_years:
                continue  # Not enough historical data

            historical_df = historical_df.copy()
            historical_df['turnover_trend'] = historical_df['指数:成交金额:合计值'] / historical_df['指数:成交金额:合计值'].rolling(window=60).mean()
            turnover_trends = historical_df['turnover_trend'].dropna()

            if turnover_trends.empty:
                continue  # No valid turnover trends

            # Calculate percentiles
            percentile_95 = np.percentile(turnover_trends, 95)
            percentile_5 = np.percentile(turnover_trends, 5)

            # Generate buy or sell signal based on trend percentiles
            signal = 0
            if turnover_trend >= percentile_95:
                signal = 1  # Buy signal
            elif turnover_trend <= percentile_5:
                signal = -1  # Sell signal

            if signal != 0:
                if next_day_open:
                    # Place signal for the next trading day if possible
                    try:
                        next_day = df.index[df.index.get_loc(current_date) + 1]
                        df.at[next_day, strategy_name + '_signal'] = signal
                        # Set holding end date
                        holding_end_date = next_day + pd.Timedelta(days=holding_days)
                        # Find the nearest trading day after holding_end_date
                        if holding_end_date not in df.index:
                            holding_end_date = df.index[df.index <= holding_end_date].max()
                    except IndexError:
                        # If current_date is the last trading day, cannot place next day signal
                        pass
                else:
                    # Place signal on the same day
                    df.at[current_date, strategy_name + '_signal'] = signal
                    # Set holding end date
                    holding_end_date = current_date + pd.Timedelta(days=holding_days)
                    # Find the nearest trading day after holding_end_date
                    if holding_end_date not in df.index:
                        holding_end_date = df.index[df.index <= holding_end_date].max()

        self.indices_data[index_name] = df
        return df[strategy_name + '_signal']

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
            temp_df = self.indices_data.copy()
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


