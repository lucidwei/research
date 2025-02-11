# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: performance_evaluator.py
# Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from data_handler import DataHandler

mpl.rcParams['font.sans-serif'] = ['STZhongsong']
mpl.rcParams['axes.unicode_minus'] = False


class PerformanceEvaluator:
    """
    PerformanceEvaluator���ڶԶ��ֲ����źŽ��лز�ͼ�Ч������
    ֧��������Ƶ����Ƶ���ԣ�
      1. ���ʶ������Ƶ���ԣ������¶Ȳ�����㾻ֵ��ָ�꣬�Ա�֤ʤ�ʡ����ʡ�Kelly��ͳ�Ƶ���ȷ�ԣ�
      2. ���¶��ź�����ӳ��Ϊ�նȲ�λ���Ա���������������ͬ�ն�����ϵ�½��о�ֵ���ߺϲ�����ͼ��
    """
    def __init__(self, data_handler: DataHandler, signals_dict: dict, signals_columns):
        """
        ����:
            data_handler: ������DataHandler�������ڻ�ȡdaily_indices_data��monthly_indices_data��
            df_signals: ���� generate_signals_for_all_strategies���ṹ���磺
                 {"M": merged_monthly, "D": merged_daily}��
            signals_columns (list): ��ʾ��Щ���ǲ����źš���Ϊ None����Ĭ��df_signals��ȫ���С�
        """
        self.data_handler = data_handler
        # �ֱ𱣴���Ƶ����Ƶ�ĺϲ��źţ���Ϊ�ֵ䣺��Ϊָ�����ƣ�ֵΪ��Ӧ�� DataFrame��
        self.df_signals_monthly = signals_dict.get("M", {})
        self.df_signals_daily = signals_dict.get("D", {})

        # ���������ź��к�Ƶ��ӳ��
        self.signals_columns = signals_columns
        self.is_monthly_signal = {}
        # ������Ƶ�źţ�������ͳһ���� "_monthly_signal"
        for index_name, df in self.df_signals_monthly.items():
            for col in df.columns:
                self.is_monthly_signal[col] = True
        # ������Ƶ�źţ�������ͳһ���� "_daily_signal"
        for index_name, df in self.df_signals_daily.items():
            for col in df.columns:
                self.is_monthly_signal[col] = False

        # ������ʼ�����ñ��ֲ���
        # Ĭ���껯���������Ƶ���ԣ���Ƶ�����ڼ��㼨Чָ��ʱʹ�� 12
        self.annual_factor_default = 252
        self.time_delta = 'Y'
        self.strategies_results = {}
        self.metrics_df = None
        self.stats_by_each_year = {}
        self.detailed_data = {}

        # �� data_handler ��ȡdict��ʽ���ն�/�¶�����
        self.daily_data_dict = self.data_handler.daily_indices_data  # {index_name: df_daily}
        self.monthly_data_dict = self.data_handler.monthly_indices_data  # {index_name: df_monthly}

        # ׼��һ����������ŵ�ǰָ�����ն�����
        self.index_df_daily = None

    def prepare_data(self, index_name):
        """
        �� daily_data_dict ȡ�ն����ݣ����浽 self.index_df_daily��
        �ز�ʱ�������DataFrame����� 'Position', 'Strategy_Return' ���С�
        """
        if index_name not in self.daily_data_dict:
            raise ValueError(f"�� data_handler ��δ�ҵ��ն�����: {index_name}")
        self.index_df_daily = self.daily_data_dict[index_name].copy()

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        �ֱ����Ƶ����Ƶ�źŽ��лز⡣
        """
        # ������Ƶ�źŻز�
        for index_name, df_signals in self.df_signals_monthly.items():
            for signal_col in self.signals_columns:
                print(f"\n��ʼ�ز���Ƶ����: {signal_col} (ָ��: {index_name})...")
                self.prepare_data(index_name)  # �� daily_data_dict �л�ȡ�ն����ݣ��ز�ʱ��Ҫ��Ƶ������ͼ��
                # ֱ�Ӵ���Ƶ�ź� DataFrame ����ȡ��Ӧ�ź�
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date, signal_series=current_signal_series)
                base_name = signal_col.replace("_monthly_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"���� {base_name} ���վ�ֵ: {final_strategy:.2f}")
                # self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

        # ������Ƶ�źŻز�
        for index_name, df_signals in self.df_signals_daily.items():
            for signal_col in self.signals_columns:
                print(f"\n��ʼ�ز���Ƶ����: {signal_col} (ָ��: {index_name})...")
                self.prepare_data(index_name)
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date, signal_series=current_signal_series)
                base_name = signal_col.replace("_daily_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"���� {base_name} ���վ�ֵ: {final_strategy:.2f}")
                self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

    def backtest_single_strategy(self, index_name, signal_col, start_date='2001-12', signal_series=None):
        """
        ���� signal_series ������ֱ��ʹ�ô�����ź����ݣ����� df_signals_monthly �� df_signals_daily����
        ����Ƶ���ԣ�
            1. ���¶������ϼ�����Ƶ��ֵ�����棻
            2. ���� convert_monthly_signals_to_daily_positions ����Ƶ�ź�ӳ��Ϊ��Ƶ��λ���ټ�����Ƶ��ֵ��
        ����Ƶ���ԣ�
            ֱ�����ն������ϼ�����Ծ�ֵ��
        �޸ĵ㣺���ӷ��ؽ���е� 'Position' �ֶΣ����ڱ���ÿ�ղ�λ��Ϣ��������Ϣ����
        """
        df_daily = self.daily_data_dict[index_name].copy()
        df_monthly = self.monthly_data_dict[index_name].copy()

        df_daily = df_daily[df_daily.index >= pd.to_datetime(start_date)].copy()
        df_monthly = df_monthly[df_monthly.index >= pd.to_datetime(start_date)].copy()

        if self.is_monthly_signal.get(signal_col, False):
            # ��Ƶ���Դ���
            monthly_net_value, monthly_strategy_returns = self._backtest_monthly(df_monthly, signal_series)
            daily_positions = self.convert_monthly_signals_to_daily_positions(df_daily, signal_series)
            # ע�⣺��λ���ú�Ҫ�����һ�գ���֤�ź��ͺ���Ч
            df_daily['Position'] = daily_positions.shift(1).fillna(0)
            df_daily['Index_Return'] = df_daily['ָ��:���һ��'].pct_change()
            df_daily['Strategy_Return'] = df_daily['Position'] * df_daily['Index_Return']
            df_daily['Strategy_Return'].fillna(0, inplace=True)
            df_daily['Cumulative_Strategy'] = (1 + df_daily['Strategy_Return']).cumprod()
            df_daily['Cumulative_Index'] = (1 + df_daily['Index_Return']).cumprod()
            return {
                'Daily_Strategy_Return': df_daily['Strategy_Return'],
                'Daily_Cumulative_Strategy': df_daily['Cumulative_Strategy'],
                'Daily_Cumulative_Index': df_daily['Cumulative_Index'],
                'Monthly_Strategy_Return': monthly_strategy_returns,
                'Monthly_Cumulative_Strategy': monthly_net_value,
                'is_monthly': True,
                'Position': df_daily['Position']  # ����������Ϣ����λ��
            }
        else:
            # ��Ƶ���Դ���ֱ��ʹ����Ƶ�źŽ��лز�
            df_daily['Position'] = signal_series.shift(1).reindex(df_daily.index).fillna(0)
            df_daily['Index_Return'] = df_daily['ָ��:���һ��'].pct_change()
            df_daily['Strategy_Return'] = df_daily['Position'] * df_daily['Index_Return']
            df_daily['Strategy_Return'].fillna(0, inplace=True)
            df_daily['Cumulative_Strategy'] = (1 + df_daily['Strategy_Return']).cumprod()
            df_daily['Cumulative_Index'] = (1 + df_daily['Index_Return']).cumprod()
            return {
                'Daily_Strategy_Return': df_daily['Strategy_Return'],
                'Daily_Cumulative_Strategy': df_daily['Cumulative_Strategy'],
                'Daily_Cumulative_Index': df_daily['Cumulative_Index'],
                'is_monthly': False,
                'Position': df_daily['Position']  # ����������Ϣ����λ��
            }


    def _backtest_monthly(self, df_monthly, monthly_signal_series):
        """
        �����¶Ȳ�����лز⣬������¶Ȳ���Ĳ��Ծ�ֵ�Ͳ������档
        ����ͳ��ʤ�ʡ����ʡ�Kelly��; �����նȳֲ֡�
        """
        df_m_temp = df_monthly.copy()
        df_m_temp['Signal'] = monthly_signal_series.reindex(df_m_temp.index).fillna(0)
        df_m_temp['MonthlyReturn'] = df_m_temp['ָ��:���һ��'].pct_change()

        # �������� = Signal.shift(1) * MonthlyReturn
        df_m_temp['MonthlyStrategyReturn'] = df_m_temp['Signal'].shift(1).fillna(0) * df_m_temp['MonthlyReturn']
        df_m_temp['MonthlyStrategyReturn'].fillna(0, inplace=True)
        df_m_temp['MonthlyCumStrategy'] = (1 + df_m_temp['MonthlyStrategyReturn']).cumprod()

        monthly_net_value = df_m_temp['MonthlyCumStrategy']
        monthly_strategy_returns = df_m_temp['MonthlyStrategyReturn']
        return monthly_net_value, monthly_strategy_returns

    def plot_results(self, cumulative_strategy, cumulative_index, strategy_id):
        """
        Plots cumulative returns of the strategy against the index.

        Parameters:
            cumulative_strategy (pd.Series): Cumulative strategy returns.
            cumulative_index (pd.Series): Cumulative index returns.
            strategy_id (int): Strategy id.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_index, label='��׼ָ��')
        plt.plot(cumulative_strategy, label=f'{strategy_id} ��ֵ')
        plt.title(f'{strategy_id} �ز���')
        plt.xlabel('ʱ��')
        plt.ylabel('�ۼ�����')
        plt.legend()
        plt.grid(True)
        plt.show()

    def convert_monthly_signals_to_daily_positions(self, df_daily, monthly_signal_series,
                                                   next_day_open=True):
        """
        ���¶��ź�ӳ��Ϊ�նȲ�λ��ʾ��д����
          next_day_open=True��ʾ��Ŀ����Ч�յ���һ�����տ��̽��֡�
        """
        daily_positions = pd.Series(0, index=df_daily.index, dtype=float)

        # �����¶��źţ����� next_day_open �������¸������ջ��ǵ���ĩ�ս���
        monthly_signal_series = monthly_signal_series.dropna()
        # �¶����ݵ�index��ÿ�������һ��(����csv��ζ�)��������Ҫ�ӵ���/���µ��ն�����ɸѡ

        for month_date, sig_val in monthly_signal_series.items():
            if sig_val == 0:
                continue

            # ��ĩ (month_date), �¸��µ�һ��
            from pandas.tseries.offsets import MonthBegin
            month_first_day = (month_date + MonthBegin(0)).replace(day=1)
            next_month_first_day = (month_date + MonthBegin(1))

            if next_day_open:
                # ��һ��������(����������)�Ž���
                start_date_for_position = next_month_first_day
            else:
                # ����ĩ�ͽ���
                start_date_for_position = month_date

            # ��signalͨ����Ч���¸���ĩ��
            end_date_for_position = (next_month_first_day + MonthBegin(1)) - pd.Timedelta(days=1)

            # ��ֵ���ն�����
            mask = (df_daily.index >= start_date_for_position) & (df_daily.index <= end_date_for_position)
            daily_positions.loc[mask] = sig_val

        return daily_positions

    def load_kelly_fractions(self):
        """
        �� self.metrics_df �ж�ȡ������ Kelly ��λ��Ϣ
        ����һ���ֵ䣬��Ϊ���Ա�ʶ��ֵΪ��Ӧ�� Kelly ��λ��
        """
        if self.metrics_df is None:
            return {}
        kelly_dict = self.metrics_df['Kelly��λ'].to_dict()
        return kelly_dict

    def compose_strategies_by_kelly(self, method='sum_to_one'):
        """
        ���ݸ����Ե� Kelly ��λ�Լ������Ƿ�ֲ֣�ͨ�� Daily_Strategy_Return �жϣ�
        ��ϳ�һ���ܲ��ԣ�
          - �������ò��Ե� Daily_Strategy_Return Ϊ 0������Ϊ�޳ֲ֣��� Kelly ��λ�����뵱�ռ�Ȩ��
          - �����гֲֵĲ��ԣ��乱������Ϊ Daily_Strategy_Return ���� Kelly ��λ��
            ������Ч Kelly ��λΪ Kelly ��λ�������������Գֲ�ָʾ�������� -> 1������ 0����
          - ���ۺ�����Ϊ���в��Թ��������ܺͳ��Ե������л�Ծ���� Kelly ��λ֮�ͣ�
            �������޻�Ծ���ԣ����ۺ������Ϊ 0��

        �ر�˵����
          ���ڲ��� '��ָ֤��_tech_sell' �� '��ָ֤��_composite_basic_tech'�����ղ��ԣ���
          ����Ч��λ��ȡ��ֵ���Ӷ����ܲ�λ�С���ȥ�����ǵĲ�λ���ܲ�λ����Ϊ�����������գ���

        ����:
            method:
                'sum_to_one'��ʹ�õ������л�Ծ���Ե� Kelly ��λ��͹�һ�������ۺ����棻
                'avg_to_one'����չ������Ŀǰʾ������ sum_to_one �÷�һ�£���

        ����:
            pd.DataFrame: �����ۺϲ��������桢�ۼƾ�ֵ�������Ե��չ������漰����Ч Kelly ��λ��
                          �Լ�ÿ���ۺϲ�λ��Composite_Position����
        """
        if not self.strategies_results:
            print("�޲��Իز�����")
            return None

        # �Ե�һ�����Ե� Daily_Strategy_Return ������Ϊ������������
        base_index = list(self.strategies_results.values())[0]['Daily_Strategy_Return'].index
        combined_df = pd.DataFrame(index=base_index)
        kelly_dict = self.load_kelly_fractions()

        composite_raw = pd.Series(0.0, index=base_index)  # �����ۼƸ����Թ��׵�������
        active_kelly_sum = pd.Series(0.0, index=base_index)  # ���ջ�Ծ���Ե� Kelly ��λ����

        # ����ÿ�����ԣ��жϵ����Ƿ�ֲ֣�ͨ�� Daily_Strategy_Return �Ƿ�Ϊ 0 �жϣ�
        for strategy_id, results in self.strategies_results.items():
            # ���ݲ�������ȥ��ǰ׺������ '��ָ֤��_tech_sell' �õ� "tech_sell"
            strategy_id_ = "_".join(strategy_id.split("_")[1:])
            if strategy_id_ not in kelly_dict:
                print(f"���� {strategy_id_} ���� Kelly �����У�������")
                continue
            fraction = kelly_dict[strategy_id_]
            # ��ȡ�ò��Ե�����������
            daily_return = results['Daily_Strategy_Return'].reindex(base_index).fillna(0)

            effective_fraction = fraction * results['Position'].reindex(base_index).fillna(0)

            active_kelly_sum += effective_fraction
            composite_raw += daily_return * effective_fraction

            # ��������ԵĹ��������ÿ����Ч��λ����� DataFrame ��
            combined_df[f'{strategy_id_}_ret'] = daily_return * effective_fraction
            combined_df[f'{strategy_id_}_position'] = effective_fraction

        # �Ե�һ�����ԵĻ�׼ָ���ۼƾ�ֵ��Ϊ�ԱȻ�׼
        first_result = list(self.strategies_results.values())[0]
        combined_df['Daily_Cumulative_Index'] = first_result['Daily_Cumulative_Index'].reindex(base_index).fillna(
            method='ffill')

        # �������޻�Ծ���ԣ�active_kelly_sum Ϊ 0�������ۺ������Ϊ 0
        if method == 'sum_to_one':
            composite_daily_return = composite_raw.divide(active_kelly_sum.max()).fillna(0)
            active_kelly_sum = active_kelly_sum.divide(active_kelly_sum.max()).fillna(0)
        elif method == 'avg_to_one':
            composite_daily_return = composite_raw.divide(active_kelly_sum.mean()).fillna(0)
        else:
            composite_daily_return = composite_raw.divide(active_kelly_sum.replace(0, np.nan)).fillna(0)

        composite_cum = (1 + composite_daily_return).cumprod()
        combined_df['Composite_Return'] = composite_daily_return
        combined_df['Composite_Cum'] = composite_cum

        # ���浱�����л�Ծ���� Kelly ��λ֮��Ϊ�ۺϲ�λ
        combined_df['Composite_Position'] = active_kelly_sum

        # ��������ָ����㺯����������Щ������������ʵ�֣������ۺϲ���ҵ��ָ�꣬�껯���Ӷ���Ƶ����ȡ 252
        annual_return = self.calculate_annualized_return(composite_daily_return, annual_factor=252)
        annual_vol = self.calculate_annualized_volatility(composite_daily_return, annual_factor=252)
        sharpe = self.calculate_sharpe_ratio(composite_daily_return, risk_free_rate=0, annual_factor=252)
        sortino = self.calculate_sortino_ratio(composite_daily_return, target=0, annual_factor=252)
        max_dd = self.calculate_max_drawdown(composite_cum)
        win_rate = self.calculate_win_rate(composite_daily_return)
        odds = self.calculate_odds_ratio(composite_daily_return)

        metrics = {
            '�껯������': annual_return,
            '�껯������': annual_vol,
            '���ձ���': sharpe,
            '����ŵ����': sortino,
            '���س�': max_dd,
            'ʤ��': win_rate,
            '����': odds,
            'Composite_Position_Today': active_kelly_sum.iloc[-1]
        }
        print("�ۺϲ���ҵ��ָ��:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return combined_df

    def calculate_metrics_all_strategies(self):
        """
        �������в��Եļ�Чָ�꣨���ڲ������棩��
        �����¶Ȳ��ԣ�ʹ���¶Ȳ��������;�ֵ����֤ʤ�ʡ����ʡ�Kelly��ָ�����ȷ�ԣ�
        �����նȲ��ԣ�ʹ���ն��������ָ�ꡣ
        """
        metrics = {
            '��������': [],
            '�껯������': [],
            '�껯������': [],
            '���ձ���': [],
            '���س�': [],
            '����ŵ����': [],
            'ʤ��': [],
            '����': [],
            'Kelly��λ': [],
            '����źŴ���': []
        }

        for strategy_id, results in self.strategies_results.items():
            print(f'���ڼ������ {strategy_id} �ļ�Чָ��...')
            if results.get('is_monthly', False):
                ret_series = results['Monthly_Strategy_Return']
                cumulative_strategy = results['Monthly_Cumulative_Strategy']
                factor = 12
            else:
                ret_series = results['Daily_Strategy_Return']
                cumulative_strategy = results['Daily_Cumulative_Strategy']
                factor = 252

            annualized_return = self.calculate_annualized_return(ret_series, annual_factor=factor)
            annualized_volatility = self.calculate_annualized_volatility(ret_series, annual_factor=factor)
            sharpe_ratio = self.calculate_sharpe_ratio(ret_series, risk_free_rate=0, annual_factor=factor)
            max_drawdown = self.calculate_max_drawdown(cumulative_strategy)
            sortino_ratio = self.calculate_sortino_ratio(ret_series, target=0, annual_factor=factor)
            win_rate = self.calculate_win_rate(ret_series)
            odds_ratio = self.calculate_odds_ratio(ret_series)
            kelly_fraction = self.calculate_kelly_fraction(win_rate, odds_ratio)
            average_signals = self.calculate_average_signal_count(ret_series)

            display_name = strategy_id[strategy_id.find('_')+1:] if '_' in strategy_id else strategy_id
            metrics['��������'].append(display_name)
            metrics['�껯������'].append(annualized_return)
            metrics['�껯������'].append(annualized_volatility)
            metrics['���ձ���'].append(sharpe_ratio)
            metrics['���س�'].append(max_drawdown)
            metrics['����ŵ����'].append(sortino_ratio)
            metrics['ʤ��'].append(win_rate)
            metrics['����'].append(odds_ratio)
            metrics['Kelly��λ'].append(kelly_fraction)
            metrics['����źŴ���'].append(average_signals)

        self.metrics_df = pd.DataFrame(metrics)
        self.metrics_df.set_index('��������', inplace=True)

    def calculate_annual_metrics_for(self, strategy_names):
        """
        Calculates annual metrics for specific strategies and saves detailed data.
        Parameters:
            strategy_names (list): List of strategy names (e.g., ['strategy6', 'strategy7']).
        """
        if not isinstance(strategy_names, list):
            raise TypeError("strategy_names Ӧ����һ���б�")

        for strategy_name in strategy_names:
            if strategy_name not in self.strategies_results:
                raise ValueError(f"�������� '{strategy_name}' �������ڻز����С�")

            # ע�⣺�˴���Ҫ��֤ self.index_df_with_signal �Ѿ��ʵ�����
            strategy_returns = self.strategies_results[strategy_name].get('Daily_Strategy_Return')
            index_returns = self.index_df_with_signal['Index_Return']

            annual_strategy_returns = (1 + strategy_returns).resample('Y').prod() - 1
            annual_index_returns = (1 + index_returns).resample('Y').prod() - 1
            annual_excess_returns = annual_strategy_returns - annual_index_returns

            trade_counts = self.calculate_trade_counts(self.index_df_with_signal[f'{strategy_name}_signal'])

            self.stats_by_each_year[strategy_name] = pd.DataFrame({
                '�����������': annual_strategy_returns,
                'ָ���������': annual_index_returns,
                '��������': annual_excess_returns,
                '���ж൥����': trade_counts['Annual_Long_Trades'],
                '���пյ�����': trade_counts['Annual_Short_Trades']
            })
            self.stats_by_each_year[strategy_name].index = self.stats_by_each_year[strategy_name].index.year  # ����������Ϊ���

            # ��ȡ�û�ָ������
            signal_column = f'{strategy_name}_signal'
            if signal_column not in self.index_df_with_signal.columns:
                raise ValueError(f"�ź��� '{signal_column}' �������� index_df_with_signal �С�")

            detailed_df = self.index_df_with_signal[
                [signal_column, 'Position', 'Strategy_Return', 'Cumulative_Strategy', 'Cumulative_Index']].copy()
            detailed_df.rename(columns={
                signal_column: '������Signal'
            }, inplace=True)

            # ��ȡ�û�ָ�����м�������¼
            signal_columns = [col for col in self.index_df_with_signal.columns if col.endswith('_signal')]
            for signal_column in signal_columns:
                # ��ȡ��һ��'_'֮����ַ�����Ϊ����
                new_col_name = signal_column.split('_', 1)[1]
                detailed_df[f'{new_col_name}'] = self.index_df_with_signal[signal_column]

            self.detailed_data[strategy_name] = detailed_df

    def generate_excel_reports(self, output_file, annual_metrics_strategy_names):
        """
        ���ɲ��������ͳ�ƺ���ϸ���ݵ�ͬһ��Excel�ļ��Ķ���������С�
        Parameters:
            output_file (str): ���Excel�ļ���·����
            annual_metrics_strategy_names (list): List of strategy names to generate annual metrics for.
        """
        with pd.ExcelWriter(output_file) as writer:
            for strategy_name in annual_metrics_strategy_names:
                if strategy_name in self.stats_by_each_year:
                    self.stats_by_each_year[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_���ͳ��')
                else:
                    print(f"���� {strategy_name} �����ͳ�����ݲ����ڣ�������")

                if strategy_name in self.detailed_data:
                    self.detailed_data[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_��ϸ����')
                else:
                    print(f"���� {strategy_name} ����ϸ���ݲ����ڣ�������")

            if self.metrics_df is not None:
                self.metrics_df.to_excel(writer, sheet_name='���Լ�Чָ��')
            else:
                print("���Լ�Чָ�����ݲ����ڣ�������")

    def calculate_average_signal_count(self, strategy_returns):
        """
        Calculates the average number of signals per year.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
        Returns:
            float: Average number of signals per year.
        """
        signals = strategy_returns != 0
        annual_signals = signals.resample(self.time_delta).sum()
        average_signals = annual_signals.mean()
        return average_signals

    def calculate_annualized_return(self, strategy_returns, annual_factor=None):
        """
        Calculates the annualized return.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            annual_factor (int): Annualization factor. ��Ϊ None����ʹ��Ĭ��ֵ��
        Returns:
            float: Annualized return.
        """
        cumulative_return = (1 + strategy_returns).prod()
        n_periods = strategy_returns.count()
        if n_periods == 0:
            return np.nan
        annualized_return = cumulative_return ** (annual_factor / n_periods) - 1
        return annualized_return

    def calculate_annualized_volatility(self, strategy_returns, annual_factor=None):
        """
        Calculates the annualized volatility.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            annual_factor (int): Annualization factor.
        Returns:
            float: Annualized volatility.
        """
        return strategy_returns.std() * np.sqrt(annual_factor)

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0, annual_factor=None):
        """
        Calculates the Sharpe Ratio.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            risk_free_rate (float): Risk-free rate.
            annual_factor (int): Annualization factor.
        Returns:
            float: Sharpe Ratio.
        """
        excess_returns = strategy_returns.mean() - (risk_free_rate / annual_factor)
        volatility = strategy_returns.std()
        if volatility == 0:
            return np.nan
        return (excess_returns / volatility) * np.sqrt(annual_factor)

    def calculate_sortino_ratio(self, strategy_returns, target=0, annual_factor=None):
        """
        Calculates the Sortino Ratio.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            target (float): Target return.
            annual_factor (int): Annualization factor.
        Returns:
            float: Sortino Ratio.
        """
        downside_returns = strategy_returns[strategy_returns < target]
        downside_deviation = downside_returns.std() * np.sqrt(annual_factor)
        expected_return = (strategy_returns.mean() - target) * annual_factor
        if downside_deviation == 0:
            return np.nan
        return expected_return / downside_deviation

    def calculate_max_drawdown(self, cumulative_returns):
        """
        Calculates the Maximum Drawdown.
        Parameters:
            cumulative_returns (pd.Series): Cumulative strategy returns.
        Returns:
            float: Maximum Drawdown.
        """
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    def calculate_win_rate(self, strategy_returns):
        """
        Calculates the win rate.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
        Returns:
            float: Win rate.
        """
        active_returns = strategy_returns[strategy_returns != 0]
        if active_returns.empty:
            return np.nan
        buy_wins = active_returns > 0
        total_wins = buy_wins.sum()
        total_signals = active_returns.count()
        return total_wins / total_signals

    def calculate_odds_ratio(self, strategy_returns):
        """
        Calculates the Odds Ratio.
        Parameters:
            strategy_returns (pd.Series): Strategy returns.
        Returns:
            float: Odds Ratio.
        """
        active_returns = strategy_returns[strategy_returns != 0]
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        if losses.empty:
            return np.nan
        avg_win = wins.mean()
        avg_loss = losses.mean()
        return avg_win / abs(avg_loss)

    def calculate_kelly_fraction(self, win_rate, odds_ratio):
        """
        Calculates the Kelly Fraction.
        Parameters:
            win_rate (float): Win rate.
            odds_ratio (float): Odds ratio.
        Returns:
            float: Kelly Fraction.
        """
        if np.isnan(odds_ratio) or odds_ratio == 0:
            return 0
        kelly_fraction = (win_rate * (odds_ratio + 1) - 1) / odds_ratio
        return max(kelly_fraction, 0)

    def calculate_trade_counts(self, signals):
        """
        Calculates the number of long and short trades per year.
        Parameters:
            signals (pd.Series): Signal series.
        Returns:
            pd.DataFrame: Trade counts per year.
        """
        long_trades = signals == 1
        short_trades = signals == -1
        annual_long = long_trades.resample(self.time_delta).sum()
        annual_short = short_trades.resample(self.time_delta).sum()
        trade_counts = pd.DataFrame({
            'Annual_Long_Trades': annual_long,
            'Annual_Short_Trades': annual_short
        })
        return trade_counts