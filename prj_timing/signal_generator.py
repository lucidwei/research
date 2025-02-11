# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
"""
signal_generator.py

本模块负责根据宏观数据和指数数据生成各类策略信号。
支持两类策略：
    1. 基于宏观数据的“招商”系列策略（strategy_zhaoshang），
       内部将各原始宏观信号与组合信号分离，提供足够的自由度供后续扩展。
    2. 成交额策略（turnover strategy）。

生成的信号初始为月频数据，模块内提供 convert_monthly_signals_to_daily 方法，
将月频信号转换为日频信号，保证后续组合回测时能统一使用日频数据。

注意：
    - 使用 generate_signals_for_all_strategies 时，必须传入 strategy_names 列表，
      各名称格式建议为 "上证指数_strategy_1"、"上证指数_strategy_turnover" 等，
      各名称中的“_strategy_”后面部分对应策略类型标识（数字或字符串）。
    - 若 selected_indices 为 None，则默认采用 indices_data 中所有的指数数据。
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import percentileofscore
from data_handler import DataHandler


class SignalGenerator:
    def __init__(self, data_handler: DataHandler):
        """
        初始化 SignalGenerator 对象。

        参数:
            data_handler (DataHandler): 数据处理实例，其中包含日频和月频的指数数据
        """
        self.data_handler = data_handler
        # 为了兼容原来代码，将月频数据作为策略计算的基础
        self.indices_data_monthly = self.data_handler.monthly_indices_data
        self.indices_data_daily = self.data_handler.daily_indices_data
        self.macro_data = self.data_handler.macro_data

        # 保存策略名称列表（例如："上证指数_macro_loan" 或 "上证指数_daily_turnover"）
        self.strategy_names = []
        self.monthly_signals_dict = {}

    def generate_signals_for_all_strategies(self, strategies_params=None, strategy_names=None, selected_indices=None):
        """
        生成所有策略信号。

        注意：
          - 本方法根据策略名称判断频率：若名称中包含 "daily"，则为日频信号；否则默认为月频信号。
          - 月频信号直接与月频价格数据合并，日频信号直接与日频价格数据合并。
          - PerformanceEvaluator 内部根据 signal 列名（例如包含 "_monthly"）判断策略类型，
            并在日频回测前完成月→日仓位映射。

        参数：
            strategies_params (dict, optional): 策略参数字典，例如 {策略全名: {参数名: 参数值, ...}}。
            strategy_names (list): 策略名称列表，例如：
               ["上证指数_macro_loan", "上证指数_macro_m1ppi", "上证指数_macro_usd",
                "上证指数_daily_turnover", "上证指数_tech_long", "上证指数_tech_sell",
                "上证指数_composite_basic_tech", "上证指数_composite_basic"]
            selected_indices (list, optional): 指数名称列表；若为 None，则采用所有指数数据。

        返回:
            dict: 包含两个键：
                "M" -> 月频信号合并结果字典（键为指数名，值为合并的月频 DataFrame）；
                "D" -> 日频信号合并结果字典。
        """
        if not strategy_names or not isinstance(strategy_names, list):
            raise ValueError("必须提供策略名称列表（strategy_names）。")
        self.strategy_names = strategy_names

        if selected_indices is None:
            # 默认选取月频和日频数据中的所有指数（注意：两个数据源有可能不完全相同）
            selected_indices = list(set(list(self.indices_data_monthly.keys()) + list(self.indices_data_daily.keys())))

        # 定义月频策略类型映射
        strategy_mapping_monthly = {
            "macro_loan_monthly": self._macro_signal_loan_monthly,
            "macro_m1ppi_monthly": self._macro_signal_m1ppi_monthly,
            "macro_usd_monthly": self._macro_signal_usd_monthly,
            "tech_long_monthly": self._macro_signal_tech_long_monthly,
            "tech_sell_monthly": self._macro_signal_tech_sell_monthly,
            "composite_basic_tech_monthly": self.generate_composite_basic_tech_monthly,
            "composite_basic_monthly": self.generate_composite_basic_monthly,
        }
        # 定义日频策略映射，此处仅演示成交额策略的日频版本，
        # 其他日频策略可按需扩展。例如，如果宏观数据不适合日频，则不用提供。
        strategy_mapping_daily = {
            "turnover": self.generate_turnover_strategy_signals
            # 可以添加更多日频策略映射
        }

        # 分别存储月频和日频信号，按指数存放
        monthly_signals = {}
        daily_signals = {}

        # 重新排序：这里不做 composite 分组，直接遍历所有策略名称
        for full_name in self.strategy_names:
            try:
                index_name, strat_type = self._parse_strategy_name(full_name)
            except ValueError as e:
                print(e)
                continue

            if index_name not in selected_indices:
                print(f"指数 '{index_name}' 不在 selected_indices 中，跳过策略 '{full_name}'。")
                continue

            params = strategies_params.get(full_name, {}) if strategies_params else {}

            # 判断频率：如果策略名称中含有 "daily"，则频率为日频；否则默认为月频。
            if "daily" in strat_type:
                freq = "D"
            else:
                freq = "M"

            # 根据频率选择生成函数和数据源
            if freq == "M":
                if strat_type not in strategy_mapping_monthly:
                    print(f"月频策略类型 '{strat_type}' 未定义，跳过策略 '{full_name}'。")
                    continue
                signal = strategy_mapping_monthly[strat_type](index_name, **params)
                # 保存时建议将信号列名称中含有标识“_monthly”，以便后续识别
                col_name = f"{full_name}_signal" if "_monthly" in full_name else f"{full_name}_monthly_signal"
                if index_name not in monthly_signals:
                    monthly_signals[index_name] = pd.DataFrame(index=signal.index)
                monthly_signals[index_name][col_name] = signal
            else:
                # 日频信号
                if strat_type not in strategy_mapping_daily:
                    print(f"日频策略类型 '{strat_type}' 未定义，跳过策略 '{full_name}'。")
                    continue
                signal = strategy_mapping_daily[strat_type](index_name, **params)
                col_name = f"{full_name}_signal" if "_daily" in full_name else f"{full_name}_daily_signal"
                if index_name not in daily_signals:
                    daily_signals[index_name] = pd.DataFrame(index=signal.index)
                daily_signals[index_name][col_name] = signal

        # 合并信号与价格数据分别处理月频和日频部分
        merged_monthly = {}
        for index_name, df_signals in monthly_signals.items():
            if index_name not in self.indices_data_monthly:
                print(f"月频价格数据中未找到指数 {index_name}，跳过。")
                continue
            monthly_prices = self.indices_data_monthly[index_name].copy()
            merged_df = monthly_prices.join(df_signals, how='left')
            for col in df_signals.columns:
                merged_df[col] = merged_df[col].fillna(0)
            merged_monthly[index_name] = merged_df

        merged_daily = {}
        for index_name, df_signals in daily_signals.items():
            if index_name not in self.indices_data_daily:
                print(f"日频价格数据中未找到指数 {index_name}，跳过。")
                continue
            daily_prices = self.indices_data_daily[index_name].copy()
            merged_df = daily_prices.join(df_signals, how='left')
            for col in df_signals.columns:
                merged_df[col] = merged_df[col].fillna(0)
            merged_daily[index_name] = merged_df

        return {"M": merged_monthly, "D": merged_daily}

    def _macro_signal_loan_monthly(self, index_name, shift=True):
        """
        生成基于中长期贷款同比MA2的信号。

        若当前值大于前一期，信号为1；否则0。
        参数：
            shift (bool)：是否将信号向后平移一次（默认 True，对应次月末建仓）。
        """
        df_macro = self.macro_data.copy()
        signal = pd.Series(
            np.where(df_macro['中长期贷款同比MA2'] > df_macro['中长期贷款同比MA2'].shift(1), 1, 0),
            index=df_macro.index
        )
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_m1ppi_monthly(self, index_name, shift=True):
        """
        生成基于 M1同比MA3 与 M1-PPI同比MA3 的信号。

        任一指标当前值大于前一期时信号为1；否则0。
        """
        df_macro = self.macro_data.copy()
        condition1 = df_macro['M1同比MA3'] > df_macro['M1同比MA3'].shift(1)
        condition2 = df_macro['M1-PPI同比MA3'] > df_macro['M1-PPI同比MA3'].shift(1)
        signal = pd.Series(np.where(condition1 | condition2, 1, 0), index=df_macro.index)
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_usd_monthly(self, index_name, shift=False):
        """
        生成基于美元指数MA2下降的信号。

        当美元指数MA2下降时，信号为1；否则0。默认不平移。
        """
        df_macro = self.macro_data.copy()
        signal = pd.Series(
            np.where(df_macro['美元指数MA2'] < df_macro['美元指数MA2'].shift(1), 1, 0),
            index=df_macro.index
        )
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_tech_long_monthly(self, index_name, window=60, percentile_threshold=0.5):
        """
        生成技术指标买入信号（买入时机）。

        当市净率百分位较低，且成交金额同比和环比、指数同比和环比均为正时，信号为1；否则0。
        """
        df_index = self.indices_data_monthly[index_name].copy()
        df_index['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(df_index, '市净率:指数', window)
        condition_per = df_index['PER_5Y_Pct_Rank'] < percentile_threshold
        condition_volume = (df_index['指数:成交金额:合计值:同比'] > 0) & (df_index['指数:成交金额:合计值:环比'] > 0)
        condition_price = (df_index['指数:最后一条:同比'] > 0) & (df_index['指数:最后一条:环比'] > 0)
        signal = pd.Series(np.where(condition_per & condition_volume & condition_price, 1, 0), index=df_index.index)
        self.monthly_signals_dict['tech_buy'] = signal
        return signal

    def _macro_signal_tech_sell_monthly(self, index_name, window=60, percentile_threshold=0.5):
        """
        生成技术指标卖出信号（卖出时机）。

        当市净率百分位较高且满足成交金额与指数价格的条件时，信号为-1；否则0。
        """
        df_index = self.indices_data_monthly[index_name].copy()
        df_index['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(df_index, '市净率:指数', window)
        condition_per = df_index['PER_5Y_Pct_Rank'] > percentile_threshold
        condition_sell = ((df_index['指数:成交金额:合计值:环比'] > 0) & (df_index['指数:最后一条:环比'] < 0)) | \
                         ((df_index['指数:最后一条:同比'] > 0) & (df_index['指数:成交金额:合计值:同比'] < 0))
        signal = pd.Series(np.where(condition_per & condition_sell, -1, 0), index=df_index.index)
        self.monthly_signals_dict['tech_sell'] = signal
        return signal

    def _parse_strategy_name(self, strategy_full_name):
        """
        解析完整策略名称，将其拆分为指数名称和策略类型。

        要求策略名称格式为 "<指数名称>_<策略类型>"，
        例如 "上证指数_macro_loan"，其中策略类型为诸如 "macro_loan", "tech_long" 等描述性名称。

        参数：
            strategy_full_name (str): 完整策略名称，例如 "上证指数_macro_loan"

        返回：
            tuple: (index_name, strategy_type)，其中 strategy_type 为小写形式。

        若格式不符合要求，则抛出 ValueError 异常。
        """
        # 使用 rsplit 分割字符串，确保当指数名称中包含下划线时不会误分割
        parts = strategy_full_name.split("_", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"策略名称 '{strategy_full_name}' 格式不正确，应为 '<指数名称>_<策略类型>' 格式。")
        return parts[0], parts[1].lower()

    def generate_composite_basic_tech_monthly(self, index_name, **kwargs):
        """
        生成组合信号：综合基本面改善信号与技术信号。

        基本面改善信号由 macro_loan、macro_m1ppi、macro_usd 三个信号构成，
        当其至少2个为1时视为改善；再结合技术指标买入（tech_long）和卖出（tech_sell），
        组合逻辑：只要基本面改善或技术指标买入则信号为1；若技术指标卖出则信号为-1。

        返回：
            pd.Series：组合信号序列。
        """
        df = self.indices_data_monthly[index_name]
        basic_signals = []
        for sig_type in ["macro_loan", "macro_m1ppi", "macro_usd"]:
            col = f"{index_name}_{sig_type}_signal"
            if col in df.columns:
                basic_signals.append(df[col])
            else:
                if sig_type == "macro_loan":
                    basic_signals.append(self._macro_signal_loan_monthly(index_name))
                elif sig_type == "macro_m1ppi":
                    basic_signals.append(self._macro_signal_m1ppi_monthly(index_name))
                elif sig_type == "macro_usd":
                    basic_signals.append(self._macro_signal_usd_monthly(index_name))
        if basic_signals:
            basic_improved = (pd.concat(basic_signals, axis=1).sum(axis=1) >= 2).astype(int)
        else:
            basic_improved = pd.Series(0, index=df.index)
        # 对 tech_buy 和 tech_sell 进行 index 对齐
        tech_buy = self.monthly_signals_dict['tech_buy'].reindex(df.index, method='ffill').fillna(0)
        tech_sell = self.monthly_signals_dict['tech_sell'].reindex(df.index, method='ffill').fillna(0)

        # 根据基本面改善和技术信号确定最终组合信号
        combined = pd.Series(0, index=df.index)
        # 如果基本面改善或技术买入均有效（假设有效信号为1），则记为买入（1）
        combined[basic_improved | (tech_buy == 1)] = 1
        # 如果技术卖出有效（即为 -1），且当天没有技术买入信号，则记为卖出（-1）
        combined[(tech_sell == -1) & ~(tech_buy == 1)] = -1
        return combined

    def generate_composite_basic_monthly(self, index_name, **kwargs):
        """
        生成仅基于基本面改善的组合信号。

        基本面改善信号由 macro_loan、macro_m1ppi、macro_usd 三个信号构成，
        当至少2个信号为1时，组合信号为1；否则为0。

        返回：
            pd.Series：组合信号序列（仅买入信号）。
        """
        df = self.indices_data_monthly[index_name]
        basic_signals = []
        for sig_type in ["macro_loan", "macro_m1ppi", "macro_usd"]:
            col = f"{index_name}_{sig_type}_signal"
            if col in df.columns:
                basic_signals.append(df[col])
            else:
                if sig_type == "macro_loan":
                    basic_signals.append(self._macro_signal_loan_monthly(index_name))
                elif sig_type == "macro_m1ppi":
                    basic_signals.append(self._macro_signal_m1ppi_monthly(index_name))
                elif sig_type == "macro_usd":
                    basic_signals.append(self._macro_signal_usd_monthly(index_name))
        if basic_signals:
            basic_improved = (pd.concat(basic_signals, axis=1).sum(axis=1) >= 2).astype(int)
            return basic_improved
        else:
            return pd.Series(0, index=df.index)

    def calculate_rolling_percentile_rank(self, df, column, window):
        """
        计算给定列的滚动百分位排名。

        参数：
            df (pd.DataFrame): 数据集。
            column (str): 需要计算百分位的列名称。
            window (int): 滑动窗口大小。

        返回：
            pd.Series: 百分位排名（0～1）。
        """
        return df[column].rolling(window=window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100,
            raw=False
        )

    def generate_turnover_strategy_signals(self, index_name, holding_days=10, percentile_window_years=4,
                                           next_day_open=True):
        """
        生成成交额策略信号。

        参数：
            index_name (str)
            holding_days (int): 持仓天数设置。
            percentile_window_years (int): 用于计算历史百分位的回溯年数。
            next_day_open (bool): 是否在信号发生次日开仓。

        返回：
            pd.Series：信号序列，1 表示买入、-1 表示卖出，0 表示空仓。
        """
        df = self.indices_data_monthly[index_name].copy()
        strategy_name = f"{index_name}_strategy_turnover"

        # 初始化信号相关列
        df[f'{strategy_name}_signal'] = 0
        df['turnover_trend'] = df['指数:成交金额:合计值'] / df['指数:成交金额:合计值'].rolling(window=60,
                                                                                               min_periods=1).mean()

        # 计算每个交易日对应的历史百分位
        total_lookback_days = 252 * percentile_window_years  # 以252交易日计一年的标准
        trend_percentiles = []
        turnover_trend_values = df['turnover_trend'].values
        for i in range(len(df)):
            if i < 1:
                trend_percentiles.append(np.nan)
                continue
            start_idx = max(0, i - total_lookback_days)
            window_vals = turnover_trend_values[start_idx:i]
            current_value = turnover_trend_values[i]
            if len(window_vals) > 0:
                percentile = percentileofscore(window_vals, current_value, kind='rank') / 100
                trend_percentiles.append(percentile)
            else:
                trend_percentiles.append(np.nan)
        df['trend_percentile'] = trend_percentiles

        signal_series = [0] * len(df)
        holding_end_idx = -1
        current_signal = 0
        for i in range(len(df)):
            current_percentile = df.iloc[i]['trend_percentile']
            if pd.isna(current_percentile):
                signal_series[i] = current_signal
                continue
            if i > holding_end_idx:
                if current_percentile >= 0.95:
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif current_percentile <= 0.05:
                    current_signal = -1
                    holding_end_idx = i + holding_days
                else:
                    current_signal = 0
            else:
                # 持仓期间仍可刷新持仓信号
                if current_percentile >= 0.95:
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif current_percentile <= 0.05:
                    current_signal = -1
                    holding_end_idx = i + holding_days
            signal_series[i] = current_signal

        absolute_signal = [abs(val) for val in signal_series]
        df[f'{strategy_name}_signal'] = absolute_signal

        if not next_day_open:
            df[f'{strategy_name}_signal'] = df[f'{strategy_name}_signal'].shift(-1).fillna(0)

        # 保存回原数据
        self.indices_data_monthly[index_name] = df
        return df[f'{strategy_name}_signal']

    def generate_turnover_strategy_signals(self, index_name, holding_days=10, percentile_window_years=4,
                                           next_day_open=True):
        """
        Generates signals for the turnover-based strategy.

        Parameters:
            index_name (str): Name of the index.
            holding_days (int): Number of trading days to hold the position.
            percentile_window_years (int): Number of years to calculate historical percentiles.
            next_day_open (bool): Whether to open positions the next day after signal.
        """
        df = self.indices_data_monthly[index_name].copy()
        strategy_name = f"{index_name}_strategy_turnover"

        # 初始化新列
        df[f'{strategy_name}_signal'] = 0
        df['turnover_trend'] = np.nan
        df['trend_percentile'] = np.nan

        # 近四年的交易日数量
        trading_days_per_year = 252
        total_lookback_days = trading_days_per_year * percentile_window_years

        # 1. 计算成交额趋势指标
        df['turnover_trend'] = df['指数:成交金额:合计值'] / df['指数:成交金额:合计值'].rolling(window=60, min_periods=1).mean()

        # 2. 计算成交额趋势指标的百分位
        trend_percentiles = []

        # 遍历每个交易日计算百分位
        turnover_trend_values = df['turnover_trend'].values
        for i in range(len(df)):
            if i < 1:
                trend_percentiles.append(np.nan)
                continue
            # 定义过去的窗口
            start_idx = max(0, i - total_lookback_days)
            window = turnover_trend_values[start_idx:i]
            current_value = turnover_trend_values[i]
            if len(window) > 0:
                percentile = percentileofscore(window, current_value, kind='rank') / 100
                trend_percentiles.append(percentile)
            else:
                trend_percentiles.append(np.nan)

        df['trend_percentile'] = trend_percentiles

        # 3. 生成持仓信号
        signal_series = [0] * len(df)
        holding_end_idx = -1  # 持仓结束的索引
        current_signal = 0  # 当前持仓信号

        for i in range(len(df)):
            percentile = df.iloc[i]['trend_percentile']

            if pd.isna(percentile):
                # 如果百分位为空，则不生成信号
                signal_series[i] = current_signal
                continue

            if i > holding_end_idx:
                # 当前不在持仓中
                if percentile >= 0.95:
                    # 异常放量信号
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif percentile <= 0.05:
                    # 异常缩量信号
                    current_signal = -1
                    holding_end_idx = i + holding_days
                else:
                    current_signal = 0
            else:
                # 当前在持仓中
                if percentile >= 0.95:
                    # 重新触发异常放量信号，延长持仓期
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif percentile <= 0.05:
                    # 重新触发异常缩量信号，延长持仓期
                    current_signal = -1
                    holding_end_idx = i + holding_days
                # 否则保持当前持仓状态

            signal_series[i] = current_signal

        # 将信号写入DataFrame
        absolute_signal = [abs(item) for item in signal_series]
        df[f'{strategy_name}_signal'] = absolute_signal

        # 如果需要在信号发生的次日开仓，则在PerformanceEvaluator.backtest_strategy中已实现，不需额外操作。否则，将信号提前一天。
        if next_day_open:
            pass
        else:
            df[f'{strategy_name}_signal'] = df[f'{strategy_name}_signal'].shift(-1).fillna(0)

        # 更新 self.indices_data
        self.indices_data_monthly[index_name] = df

        return df[f'{strategy_name}_signal']

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
            signals = self.generate_strategy_zhaoshang_signals(strategy_num, **params)

            # Temporarily store signals in dataframe
            temp_df = self.indices_data_monthly.copy()
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


