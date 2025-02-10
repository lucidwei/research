# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
"""
signal_generator.py

��ģ�鸺����ݺ�����ݺ�ָ���������ɸ�������źš�
֧��������ԣ�
    1. ���ں�����ݵġ����̡�ϵ�в��ԣ�strategy_zhaoshang����
       �ڲ�����ԭʼ����ź�������źŷ��룬�ṩ�㹻�����ɶȹ�������չ��
    2. �ɽ�����ԣ�turnover strategy����

���ɵ��źų�ʼΪ��Ƶ���ݣ�ģ�����ṩ convert_monthly_signals_to_daily ������
����Ƶ�ź�ת��Ϊ��Ƶ�źţ���֤������ϻز�ʱ��ͳһʹ����Ƶ���ݡ�

ע�⣺
    - ʹ�� generate_signals_for_all_strategies ʱ�����봫�� strategy_names �б�
      �����Ƹ�ʽ����Ϊ "��ָ֤��_strategy_1"��"��ָ֤��_strategy_turnover" �ȣ�
      �������еġ�_strategy_�����沿�ֶ�Ӧ�������ͱ�ʶ�����ֻ��ַ�������
    - �� selected_indices Ϊ None����Ĭ�ϲ��� indices_data �����е�ָ�����ݡ�
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import percentileofscore
from data_handler import DataHandler


class SignalGenerator:
    def __init__(self, data_handler: DataHandler):
        """
        ��ʼ�� SignalGenerator ����

        ����:
            data_handler (DataHandler): ���ݴ���ʵ�������а�����Ƶ����Ƶ��ָ������
        """
        self.data_handler = data_handler
        # Ϊ�˼���ԭ�����룬����Ƶ������Ϊ���Լ���Ļ���
        self.indices_data_monthly = self.data_handler.monthly_indices_data
        self.indices_data_daily = self.data_handler.daily_indices_data
        self.macro_data = self.data_handler.macro_data

        # ������������б����磺"��ָ֤��_strategy_1"��
        self.strategy_names = []
        self.monthly_signals_dict = {}

    def generate_signals_for_all_strategies(self, strategies_params=None, strategy_names=None, selected_indices=None):
        """
        �������в����źš�

        ע�⣺
          - �������������ɵĲ����źţ�ͨ��Ϊ��Ƶ��������-->��ת����
            ���޸� self.indices_data �е�ԭʼ��Ƶ�۸����ݡ�
          - ����������Ҫʹ����Ƶ��ָ�����ݣ����� DataHandler �����ṩ��
            ���������ص���ת����� ��Ƶ�ź� DataFrame��

        ������
            strategies_params (dict, optional): ���Բ����ֵ䣬���� {����ȫ��: {������: ����ֵ, ...}}��
            strategy_names (list): ���������б����磺
               ["��ָ֤��_macro_loan", "��ָ֤��_macro_m1ppi", "��ָ֤��_macro_usd",
                "��ָ֤��_tech_long", "��ָ֤��_tech_sell", "��ָ֤��_composite_basic_tech",
                "��ָ֤��_composite_basic", "��ָ֤��_turnover"]
            selected_indices (list, optional): ָ�������б���Ϊ None������� self.indices_data ������ָ�����ݡ�

        ����:
            pd.DataFrame: ��������һ��ָ��ת��Ϊ��Ƶ����ź����ݣ����ź��У���
        """
        if not strategy_names or not isinstance(strategy_names, list):
            raise ValueError("�����ṩ���������б�strategy_names����")
        self.strategy_names = strategy_names

        if selected_indices is None:
            selected_indices = list(self.indices_data_monthly.keys())

        # �����������ӳ�䣺�������͵����ɺ�����ӳ���ֵ�
        strategy_mapping = {
            "turnover": self.generate_turnover_strategy_signals,
            "macro_loan": self._macro_signal_loan,
            "macro_m1ppi": self._macro_signal_m1ppi,
            "macro_usd": self._macro_signal_usd,
            "tech_long": self._macro_signal_tech_long,
            "tech_sell": self._macro_signal_tech_sell,
            "composite_basic_tech": self.generate_composite_basic_tech,
            "composite_basic": self.generate_composite_basic
        }

        # �� composite ���Է�Ϊ���飺�� composite �� composite ����
        non_composite_strategies = []
        composite_strategies = []
        for full_name in self.strategy_names:
            # �жϲ����������Ƿ���� "composite"��Ҳ�ɸ���ʵ�ʹ�������жϷ�ʽ��
            if "composite" in full_name:
                composite_strategies.append(full_name)
            else:
                non_composite_strategies.append(full_name)

        # ���ź�Ĳ�������˳���ȷ� composite ���ԣ��� composite ����
        ordered_strategy_names = non_composite_strategies + composite_strategies

        # ���ڴ洢���ɵ���Ƶ�źţ�������۸����ݷ���
        signals_dict = {}
        for full_name in ordered_strategy_names:
            try:
                index_name, strat_type = self._parse_strategy_name(full_name)
            except ValueError as e:
                print(e)
                continue

            if index_name not in selected_indices:
                print(f"ָ�� '{index_name}' ���� selected_indices �У��������� '{full_name}'��")
                continue

            strat_key = strat_type.lower()
            if strat_key not in strategy_mapping:
                print(f"�������� '{strat_key}' δ���壬�������� '{full_name}'��")
                continue

            params = strategies_params.get(full_name, {}) if strategies_params else {}
            # ������Ƶ�ź�
            signals = strategy_mapping[strat_key](index_name, **params)
            # �����ɵ��źŴ��� signals_dict����ָ�����֣�
            if index_name not in signals_dict:
                signals_dict[index_name] = pd.DataFrame(index=signals.index)
            signals_dict[index_name][f'{full_name}_signal'] = signals

        # �� signals_dict �е���Ƶ�ź�ת��Ϊ��Ƶ�źţ����ź�����ת����
        signals_daily = {}
        for index_name, df_signals in signals_dict.items():
            signals_daily[index_name] = self.convert_monthly_signals_to_daily_signals(df_signals)

        # �ϲ�ԭʼ��Ƶָ�������������ź�����
        merged_dfs = {}
        for index_name, daily_signals in signals_daily.items():
            # ԭʼ��Ƶ�۸����ݣ����ֲ��䣨�� DataHandler �ṩ��
            original_df = self.indices_data_daily[index_name].copy()
            # �������ϲ����ź����ݿ���ֻ�����ض���������
            merged_df = original_df.join(daily_signals, how='left')
            # ���ź��е�ȱʧֵ���Ϊ 0
            for col in daily_signals.columns:
                merged_df[col] = merged_df[col].fillna(0)
            merged_dfs[index_name] = merged_df

        # ��������һ��ָ��ĺϲ����������ֻ��һ��ָ���ָ��������һ����
        return list(merged_dfs.values())[-1]

    def _macro_signal_loan(self, index_name, shift=True):
        """
        ���ɻ����г��ڴ���ͬ��MA2���źš�

        ����ǰֵ����ǰһ�ڣ��ź�Ϊ1������0��
        ������
            shift (bool)���Ƿ��ź����ƽ��һ�Σ�Ĭ�� True����Ӧ����ĩ���֣���
        """
        df_macro = self.macro_data.copy()
        signal = pd.Series(
            np.where(df_macro['�г��ڴ���ͬ��MA2'] > df_macro['�г��ڴ���ͬ��MA2'].shift(1), 1, 0),
            index=df_macro.index
        )
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_m1ppi(self, index_name, shift=True):
        """
        ���ɻ��� M1ͬ��MA3 �� M1-PPIͬ��MA3 ���źš�

        ��һָ�굱ǰֵ����ǰһ��ʱ�ź�Ϊ1������0��
        """
        df_macro = self.macro_data.copy()
        condition1 = df_macro['M1ͬ��MA3'] > df_macro['M1ͬ��MA3'].shift(1)
        condition2 = df_macro['M1-PPIͬ��MA3'] > df_macro['M1-PPIͬ��MA3'].shift(1)
        signal = pd.Series(np.where(condition1 | condition2, 1, 0), index=df_macro.index)
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_usd(self, index_name, shift=False):
        """
        ���ɻ�����Ԫָ��MA2�½����źš�

        ����Ԫָ��MA2�½�ʱ���ź�Ϊ1������0��Ĭ�ϲ�ƽ�ơ�
        """
        df_macro = self.macro_data.copy()
        signal = pd.Series(
            np.where(df_macro['��Ԫָ��MA2'] < df_macro['��Ԫָ��MA2'].shift(1), 1, 0),
            index=df_macro.index
        )
        if shift:
            signal = signal.shift(1)
        return signal

    def _macro_signal_tech_long(self, index_name, window=60, percentile_threshold=0.5):
        """
        ���ɼ���ָ�������źţ�����ʱ������

        ���о��ʰٷ�λ�ϵͣ��ҳɽ����ͬ�Ⱥͻ��ȡ�ָ��ͬ�Ⱥͻ��Ⱦ�Ϊ��ʱ���ź�Ϊ1������0��
        """
        df_index = self.indices_data_monthly[index_name].copy()
        df_index['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(df_index, '�о���:ָ��', window)
        condition_per = df_index['PER_5Y_Pct_Rank'] < percentile_threshold
        condition_volume = (df_index['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] > 0) & (df_index['ָ��:�ɽ����:�ϼ�ֵ:����'] > 0)
        condition_price = (df_index['ָ��:���һ��:ͬ��'] > 0) & (df_index['ָ��:���һ��:����'] > 0)
        signal = pd.Series(np.where(condition_per & condition_volume & condition_price, 1, 0), index=df_index.index)
        self.monthly_signals_dict['tech_buy'] = signal
        return signal

    def _macro_signal_tech_sell(self, index_name, window=60, percentile_threshold=0.5):
        """
        ���ɼ���ָ�������źţ�����ʱ������

        ���о��ʰٷ�λ�ϸ�������ɽ������ָ���۸������ʱ���ź�Ϊ-1������0��
        """
        df_index = self.indices_data_monthly[index_name].copy()
        df_index['PER_5Y_Pct_Rank'] = self.calculate_rolling_percentile_rank(df_index, '�о���:ָ��', window)
        condition_per = df_index['PER_5Y_Pct_Rank'] > percentile_threshold
        condition_sell = ((df_index['ָ��:�ɽ����:�ϼ�ֵ:����'] > 0) & (df_index['ָ��:���һ��:����'] < 0)) | \
                         ((df_index['ָ��:���һ��:ͬ��'] > 0) & (df_index['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] < 0))
        signal = pd.Series(np.where(condition_per & condition_sell, -1, 0), index=df_index.index)
        self.monthly_signals_dict['tech_sell'] = signal
        return signal

    def _parse_strategy_name(self, strategy_full_name):
        """
        ���������������ƣ�������Ϊָ�����ƺͲ������͡�

        Ҫ��������Ƹ�ʽΪ "<ָ������>_<��������>"��
        ���� "��ָ֤��_macro_loan"�����в�������Ϊ���� "macro_loan", "tech_long" �����������ơ�

        ������
            strategy_full_name (str): �����������ƣ����� "��ָ֤��_macro_loan"

        ���أ�
            tuple: (index_name, strategy_type)������ strategy_type ΪСд��ʽ��

        ����ʽ������Ҫ�����׳� ValueError �쳣��
        """
        # ʹ�� rsplit �ָ��ַ�����ȷ����ָ�������а����»���ʱ������ָ�
        parts = strategy_full_name.split("_", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"�������� '{strategy_full_name}' ��ʽ����ȷ��ӦΪ '<ָ������>_<��������>' ��ʽ��")
        return parts[0], parts[1].lower()

    def generate_composite_basic_tech(self, index_name, **kwargs):
        """
        ��������źţ��ۺϻ���������ź��뼼���źš�

        ����������ź��� macro_loan��macro_m1ppi��macro_usd �����źŹ��ɣ�
        ��������2��Ϊ1ʱ��Ϊ���ƣ��ٽ�ϼ���ָ�����루tech_long����������tech_sell����
        ����߼���ֻҪ��������ƻ���ָ���������ź�Ϊ1��������ָ���������ź�Ϊ-1��

        ���أ�
            pd.Series������ź����С�
        """
        df = self.indices_data_monthly[index_name]
        basic_signals = []
        for sig_type in ["macro_loan", "macro_m1ppi", "macro_usd"]:
            col = f"{index_name}_{sig_type}_signal"
            if col in df.columns:
                basic_signals.append(df[col])
            else:
                if sig_type == "macro_loan":
                    basic_signals.append(self._macro_signal_loan(index_name))
                elif sig_type == "macro_m1ppi":
                    basic_signals.append(self._macro_signal_m1ppi(index_name))
                elif sig_type == "macro_usd":
                    basic_signals.append(self._macro_signal_usd(index_name))
        if basic_signals:
            basic_improved = (pd.concat(basic_signals, axis=1).sum(axis=1) >= 2).astype(int)
        else:
            basic_improved = pd.Series(0, index=df.index)
        # �� tech_buy �� tech_sell ���� index ����
        tech_buy = self.monthly_signals_dict['tech_buy'].reindex(df.index, method='ffill').fillna(0)
        tech_sell = self.monthly_signals_dict['tech_sell'].reindex(df.index, method='ffill').fillna(0)

        # ���ݻ�������ƺͼ����ź�ȷ����������ź�
        combined = pd.Series(0, index=df.index)
        # �����������ƻ����������Ч��������Ч�ź�Ϊ1�������Ϊ���루1��
        combined[basic_improved | (tech_buy == 1)] = 1
        # �������������Ч����Ϊ -1�����ҵ���û�м��������źţ����Ϊ������-1��
        combined[(tech_sell == -1) & ~(tech_buy == 1)] = -1
        return combined

    def generate_composite_basic(self, index_name, **kwargs):
        """
        ���ɽ����ڻ�������Ƶ�����źš�

        ����������ź��� macro_loan��macro_m1ppi��macro_usd �����źŹ��ɣ�
        ������2���ź�Ϊ1ʱ������ź�Ϊ1������Ϊ0��

        ���أ�
            pd.Series������ź����У��������źţ���
        """
        df = self.indices_data_monthly[index_name]
        basic_signals = []
        for sig_type in ["macro_loan", "macro_m1ppi", "macro_usd"]:
            col = f"{index_name}_{sig_type}_signal"
            if col in df.columns:
                basic_signals.append(df[col])
            else:
                if sig_type == "macro_loan":
                    basic_signals.append(self._macro_signal_loan(index_name))
                elif sig_type == "macro_m1ppi":
                    basic_signals.append(self._macro_signal_m1ppi(index_name))
                elif sig_type == "macro_usd":
                    basic_signals.append(self._macro_signal_usd(index_name))
        if basic_signals:
            basic_improved = (pd.concat(basic_signals, axis=1).sum(axis=1) >= 2).astype(int)
            return basic_improved
        else:
            return pd.Series(0, index=df.index)

    def calculate_rolling_percentile_rank(self, df, column, window):
        """
        ��������еĹ����ٷ�λ������

        ������
            df (pd.DataFrame): ���ݼ���
            column (str): ��Ҫ����ٷ�λ�������ơ�
            window (int): �������ڴ�С��

        ���أ�
            pd.Series: �ٷ�λ������0��1����
        """
        return df[column].rolling(window=window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100,
            raw=False
        )

    def generate_turnover_strategy_signals(self, index_name, holding_days=10, percentile_window_years=4,
                                           next_day_open=True):
        """
        ���ɳɽ�������źš�

        ������
            index_name (str)
            holding_days (int): �ֲ��������á�
            percentile_window_years (int): ���ڼ�����ʷ�ٷ�λ�Ļ���������
            next_day_open (bool): �Ƿ����źŷ������տ��֡�

        ���أ�
            pd.Series���ź����У�1 ��ʾ���롢-1 ��ʾ������0 ��ʾ�ղ֡�
        """
        df = self.indices_data_monthly[index_name].copy()
        strategy_name = f"{index_name}_strategy_turnover"

        # ��ʼ���ź������
        df[f'{strategy_name}_signal'] = 0
        df['turnover_trend'] = df['ָ��:�ɽ����:�ϼ�ֵ'] / df['ָ��:�ɽ����:�ϼ�ֵ'].rolling(window=60,
                                                                                               min_periods=1).mean()

        # ����ÿ�������ն�Ӧ����ʷ�ٷ�λ
        total_lookback_days = 252 * percentile_window_years  # ��252�����ռ�һ��ı�׼
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
                # �ֲ��ڼ��Կ�ˢ�³ֲ��ź�
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

        # �����ԭ����
        self.indices_data_monthly[index_name] = df
        return df[f'{strategy_name}_signal']

    def convert_monthly_signals_to_daily_signals(self, df_signals):
        """
        ����Ƶ�����ź�����ת��Ϊ��Ƶ�ź����ݡ�

        ���� df_signals �е�ÿ���ź��У�����ÿ��ԭʼ�ź����ڣ�
        �����ź�ֵ��չ������Ч�·ݡ��ڣ���ԭ���ڼ�1���������й����գ���

        ������
            df_signals (pd.DataFrame): ������Ƶ�����źŵ� DataFrame��������Ϊ��Ƶ���ڡ�

        ����:
             pd.DataFrame: ��Ƶ�ź� DataFrame��������Ϊ�����ա�
        """
        # ȷ����Ƶ�������䣺�������ź����ڼ�1���µ��׸������յ������ź����ڼ�1���µ���ĩ
        start_date = (df_signals.index.min() + pd.offsets.MonthBegin(1)).normalize()
        end_date = (df_signals.index.max() + pd.offsets.MonthEnd(1)).normalize()
        daily_index = pd.date_range(start=start_date, end=end_date, freq='B')
        new_df = pd.DataFrame(index=daily_index)

        for col in df_signals.columns:
            new_signal = pd.Series(0, index=daily_index)
            for date, value in df_signals[col].iteritems():
                effective_date = (date + pd.offsets.MonthBegin(1)).normalize()  # ��Ч�·���ʼ
                effective_month_end = (effective_date + pd.offsets.MonthEnd(0)).normalize()  # ��Ч�·�ĩ
                effective_range = pd.date_range(start=effective_date, end=effective_month_end, freq='B')
                new_signal.loc[effective_range] = value
            new_df[col] = new_signal

        return new_df

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

        # �����Ҫ���źŷ����Ĵ��տ��֣�����PerformanceEvaluator.backtest_strategy����ʵ�֣����������������򣬽��ź���ǰһ�졣
        if next_day_open:
            pass
        else:
            df[f'{strategy_name}_signal'] = df[f'{strategy_name}_signal'].shift(-1).fillna(0)

        # ���� self.indices_data
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


