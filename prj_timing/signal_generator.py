# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import percentileofscore


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
        self.strategy_names = {}  # �ֵ䣺���Ա�� -> ��������

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

        for strategy_id, strategy_name in self.strategy_names.items():
            # ���� strategy_id������ '��ָ֤��_strategy_turnover'
            if '_strategy_' not in strategy_id:
                print(f"���Ա�ʶ�� '{strategy_id}' ��ʽ����ȷ��Ӧ���� '_strategy_'�������˲��ԡ�")
                continue  # ������ʽ����ȷ�Ĳ���

            index_name, strategy_type = strategy_id.split('_strategy_', 1)

            if index_name not in selected_indices:
                print(f"ָ�� '{index_name}' δ�� selected_indices �С��������� '{strategy_id}'��")
                continue  # ����δѡ���ָ��

            if 'turnover' in strategy_type.lower():
                # ����������Ͱ��� 'turnover'������ generate_turnover_strategy_signals
                params = strategies_params.get(strategy_id, {}) if strategies_params else {}
                signals = self.generate_turnover_strategy_signals(
                    index_name=index_name,
                    holding_days=params.get('holding_days', 10),
                    percentile_window_years=params.get('percentile_window_years', 4),
                    next_day_open=params.get('next_day_open', True)
                )

            elif strategy_type.isdigit():
                signals = self.generate_strategy_signals(index_name, int(strategy_type))

            else:
                print(f"�������ͱ�ʶ�� '{strategy_type}' ��֧�֡������˲���({strategy_id})��")
                continue  # ������ʽ����ȷ�Ĳ���

            self.indices_data[index_name][f'{strategy_name}_signal'] = signals

        # for index_name in selected_indices:
        #     for strategy_num in range(1, 6):
        #         strategy_id = f"{index_name}_strategy{strategy_num}"
        #         strategy_name = self.strategy_names.get(strategy_id, strategy_id)
        #
        #         if strategies_params and strategy_id in strategies_params:
        #             params = strategies_params[strategy_id]
        #             signals = self.generate_strategy_signals(index_name, strategy_num, **params)
        #         else:
        #             signals = self.generate_strategy_signals(index_name, strategy_num)
        #
        #         self.indices_data[index_name][f'{strategy_name}_signal'] = signals
        #
        #     # Generate strategy6 (aggregated strategy)
        #     strategy6_id = f"{index_name}_strategy6"
        #     strategy6_name = self.strategy_names.get(strategy6_id, strategy6_id)
        #     strategy6_signals = self.generate_strategy6_signals(index_name)
        #     self.indices_data[index_name][f'{strategy6_name}_signal'] = strategy6_signals

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
            # Strategy 1: �г��ڴ���ͬ��MA2
            macro_data['�г��ڴ���ͬ��MA2_prev'] = macro_data['�г��ڴ���ͬ��MA2'].shift(1)
            signals = np.where(macro_data['�г��ڴ���ͬ��MA2'] > macro_data['�г��ڴ���ͬ��MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

            # ���ź���ǰ�ƶ�һ�����ڣ���ʾ����ĩ����
            signals = signals.shift(1)

        elif strategy_num == 2:
            # Strategy 2: M1ͬ��MA3 and M1-PPIͬ��MA3
            macro_data['M1ͬ��MA3_prev'] = macro_data['M1ͬ��MA3'].shift(1)
            macro_data['M1-PPIͬ��MA3_prev'] = macro_data['M1-PPIͬ��MA3'].shift(1)
            condition1 = macro_data['M1ͬ��MA3'] > macro_data['M1ͬ��MA3_prev']
            condition2 = macro_data['M1-PPIͬ��MA3'] > macro_data['M1-PPIͬ��MA3_prev']
            signals = np.where(condition1 | condition2, 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

            # ���ź���ǰ�ƶ�һ�����ڣ���ʾ����ĩ����
            signals = signals.shift(1)

        elif strategy_num == 3:
            # Strategy 3: ��Ԫָ��MA2
            macro_data['��Ԫָ��MA2_prev'] = macro_data['��Ԫָ��MA2'].shift(1)
            signals = np.where(macro_data['��Ԫָ��MA2'] < macro_data['��Ԫָ��MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=macro_data.index)

        elif strategy_num == 4:
            # Strategy 4: Parameters like 'percentile_threshold', 'volume_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '�о���:ָ��', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] < percentile_threshold
            condition_yoy = df['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] > 0
            condition_mom = df['ָ��:�ɽ����:�ϼ�ֵ:����'] > 0
            condition_volume = condition_yoy & condition_mom

            condition_index_yoy = df['ָ��:���һ��:ͬ��'] > 0
            condition_index_mom = df['ָ��:���һ��:����'] > 0
            condition_price = condition_index_yoy & condition_index_mom

            signals = np.where(condition_per & condition_volume & condition_price, 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 5:
            # Strategy 5: Parameters like 'percentile_threshold'
            ma_window = kwargs.get('window', 60)
            percentile_threshold = kwargs.get('percentile_threshold', 0.5)

            df['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(
                df, '�о���:ָ��', window=ma_window
            )

            condition_per = df['PER_5Y_Pct_Rank'] > percentile_threshold

            condition_mom_volume = (df['ָ��:�ɽ����:�ϼ�ֵ:����'] > 0) & (
                    df['ָ��:���һ��:����'] < 0)
            condition_yoy_price = (df['ָ��:���һ��:ͬ��'] > 0) & (
                    df['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] < 0)

            condition_sell = condition_mom_volume | condition_yoy_price

            signals = np.where(condition_per & condition_sell, -1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 6:
            signals = self.generate_strategy6_signals(index_name)

        return signals

    def generate_strategy6_signals(self, index_name):
        """
        Generates signals for Strategy 6 based on Strategies 1-5.

        Returns:
            pd.Series: Strategy 6 signals.
        """
        # ���� strategy_names �Ѱ������в��Ե�����
        df = self.indices_data[index_name]

        # ����������źţ�����1������2������3����������Ϊ1
        basic_improved = df[[f'{index_name}_strategy{num}_signal' for num in range(1, 4)]].sum(axis=1) >= 2

        # �����������źţ�����5�ź�Ϊ-1
        technical_sell = df[f"{index_name}_strategy5_signal"] == -1

        # �����������źţ�����4�ź�Ϊ1
        technical_buy = df[f"{index_name}_strategy4_signal"] == 1

        # **ģ�͹���**��
        # ����1��������������޼����������ź�
        condition1 = basic_improved & (~technical_sell)

        # ����2�������������ź�
        condition2 = technical_buy

        # �����źţ�
        # ��������1������2�����루1��
        # ���㼼���������ź���������-1��
        # ���򱣳ֲֳ֣�0��
        signals = np.where(
            condition1 | condition2, 1,
            np.where(technical_sell, -1, 0)
        )

        return pd.Series(signals, index=df.index)


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
        df = self.indices_data[index_name].copy()
        strategy_name = f"{index_name}_strategy_turnover"

        # ��ʼ������
        df[f'{strategy_name}_signal'] = 0
        df['turnover_trend'] = np.nan
        df['trend_percentile'] = np.nan

        # ������Ľ���������
        trading_days_per_year = 252
        total_lookback_days = trading_days_per_year * percentile_window_years

        # 1. ����ɽ�������ָ��
        df['turnover_trend'] = df['ָ��:�ɽ����:�ϼ�ֵ'] / df['ָ��:�ɽ����:�ϼ�ֵ'].rolling(window=60, min_periods=1).mean()

        # 2. ����ɽ�������ָ��İٷ�λ
        trend_percentiles = []

        # ����ÿ�������ռ���ٷ�λ
        turnover_trend_values = df['turnover_trend'].values
        for i in range(len(df)):
            if i < 1:
                trend_percentiles.append(np.nan)
                continue
            # �����ȥ�Ĵ���
            start_idx = max(0, i - total_lookback_days)
            window = turnover_trend_values[start_idx:i]
            current_value = turnover_trend_values[i]
            if len(window) > 0:
                percentile = percentileofscore(window, current_value, kind='rank') / 100
                trend_percentiles.append(percentile)
            else:
                trend_percentiles.append(np.nan)

        df['trend_percentile'] = trend_percentiles

        # 3. ���ɳֲ��ź�
        signal_series = [0] * len(df)
        holding_end_idx = -1  # �ֲֽ���������
        current_signal = 0  # ��ǰ�ֲ��ź�

        for i in range(len(df)):
            percentile = df.iloc[i]['trend_percentile']

            if pd.isna(percentile):
                # ����ٷ�λΪ�գ��������ź�
                signal_series[i] = current_signal
                continue

            if i > holding_end_idx:
                # ��ǰ���ڳֲ���
                if percentile >= 0.95:
                    # �쳣�����ź�
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif percentile <= 0.05:
                    # �쳣�����ź�
                    current_signal = -1
                    holding_end_idx = i + holding_days
                else:
                    current_signal = 0
            else:
                # ��ǰ�ڳֲ���
                if percentile >= 0.95:
                    # ���´����쳣�����źţ��ӳ��ֲ���
                    current_signal = 1
                    holding_end_idx = i + holding_days
                elif percentile <= 0.05:
                    # ���´����쳣�����źţ��ӳ��ֲ���
                    current_signal = -1
                    holding_end_idx = i + holding_days
                # ���򱣳ֵ�ǰ�ֲ�״̬

            signal_series[i] = current_signal

        # ���ź�д��DataFrame
        absolute_signal = [abs(item) for item in signal_series]
        df[f'{strategy_name}_signal'] = absolute_signal

        # �����Ҫ���źŷ����Ĵ��տ��֣����ź���ǰ�ƶ�һ��
        if next_day_open:
            df[f'{strategy_name}_signal'] = df[f'{strategy_name}_signal'].shift(1).fillna(0)

        # ���� self.indices_data
        self.indices_data[index_name] = df

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


