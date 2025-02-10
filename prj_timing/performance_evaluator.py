# coding=gbk
# Time Created: 2025/1/14 14:08
# Author  : Lucid
# FileName: performance_evaluator.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

class PerformanceEvaluator:
    def __init__(self, indices_data, signals_columns, frequency='D'):
        """
        Initializes the PerformanceEvaluator.
        ��class����֪����ģ�ֻ���źŽ���������ۡ�ֻ�����ڵ����

        Parameters:
            index_df_with_signal (pd.DataFrame): Dataframe containing price data and signals.
            signals_columns (list): List of signal column names to evaluate.
            frequency (str): Data frequency ('D' for daily, 'M' for monthly).
        """
        self.index_df_with_signal = indices_data.copy()
        self.signals_columns = signals_columns
        self.strategies_results = {}
        self.metrics_df = None
        self.stats_by_each_year = {}
        self.detailed_data = {}
        self.frequency = frequency.upper()

        # Set annualization factor based on frequency
        if self.frequency == 'D':
            self.annual_factor = 252
            self.time_delta = 'Y'  # For yearly metrics
        elif self.frequency == 'M':
            self.annual_factor = 12
            self.time_delta = 'Y'
        else:
            raise ValueError("Unsupported frequency. Use 'D' for daily or 'M' for monthly.")

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        Backtests all specified strategies.

        Parameters:
            start_date (str): Start date for backtesting (format 'YYYY-MM').

        """
        for strategy_signal in self.signals_columns:
            strategy_id = strategy_signal.replace('_signal', '')
            # Extract index name from strategy_id
            index_name, strategy_num = strategy_id.rsplit('_', 1)
            print(f'\n���ڻز���� {strategy_id} ({index_name})...')
            signals = self.index_df_with_signal[strategy_signal]
            cumulative_strategy, cumulative_index, strategy_returns = self.backtest_strategy(signals, start_date)
            # Normalize cumulative returns
            cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
            cumulative_index = cumulative_index / cumulative_index.iloc[0]
            # Store results
            self.strategies_results[strategy_id] = {
                'Cumulative_Strategy': cumulative_strategy,
                'Cumulative_Index': cumulative_index,
                'Strategy_Return': strategy_returns
            }
            # Plot results
            self.plot_results(cumulative_strategy, cumulative_index, strategy_id)
            # Print final net value
            final_strategy = cumulative_strategy.iloc[-1]
            final_index = cumulative_index.iloc[-1]
            print(f'���� {strategy_id} ���վ�ֵ: {final_strategy:.2f}')
            print(f'ָ�� {index_name} ���վ�ֵ: {final_index:.2f}\n')


    def backtest_strategy(self, signals, start_date):
        """
        Backtests a single strategy.

        Parameters:
            signals (pd.Series): Signal series.
            start_date (str): Start date for backtesting.

        Returns:
            pd.Series, pd.Series, pd.Series: Cumulative strategy returns, cumulative index returns, monthly returns.
        """
        df = self.index_df_with_signal
        df['Index_Return'] = df['ָ��:���һ��'].pct_change()

        backtest_start = pd.to_datetime(start_date, format='%Y-%m')
        df = df[df.index >= backtest_start].copy()
        signals = signals[signals.index >= backtest_start].copy()

        # Shift signals to represent holding from previous period
        df['Position'] = signals.shift(1).fillna(0)

        df['Strategy_Return'] = df['Position'] * df['Index_Return']
        df['Strategy_Return'].fillna(0, inplace=True)

        # Calculate cumulative returns
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        df['Cumulative_Index'] = (1 + df['Index_Return']).cumprod()

        self.index_df_with_signal = df
        # �洢ԭʼ����
        original_rows = len(df)
        # ��� 'Cumulative_Strategy', 'Cumulative_Index', 'Strategy_Return' ���Ƿ��� NaN ֵ����ɾ������ NaN ����
        df_before_drop = df.copy()
        df = df.dropna(subset=['Cumulative_Strategy', 'Cumulative_Index', 'Strategy_Return'])
        # �ҳ���ɾ������
        # dropped_rows = df_before_drop[~df_before_drop.index.isin(df.index)]
        # print(f"Deleted rows: {dropped_rows}")
        print(f"Number of deleted rows: {original_rows - len(df)}")
        return df['Cumulative_Strategy'], df['Cumulative_Index'], df['Strategy_Return']

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


    def calculate_metrics_all_strategies(self):
        """
        Calculates performance metrics for all backtested strategies.
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
            strategy_returns = results['Strategy_Return']
            cumulative_strategy = results['Cumulative_Strategy']

            annualized_return = self.calculate_annualized_return(strategy_returns)
            annualized_volatility = self.calculate_annualized_volatility(strategy_returns)
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self.calculate_max_drawdown(cumulative_strategy)
            sortino_ratio = self.calculate_sortino_ratio(strategy_returns)
            win_rate = self.calculate_win_rate(strategy_returns)
            odds_ratio = self.calculate_odds_ratio(strategy_returns)
            kelly_fraction = self.calculate_kelly_fraction(win_rate, odds_ratio)
            average_signals = self.calculate_average_signal_count(strategy_returns)

            metrics['��������'].append(strategy_id[strategy_id.find('_') + 1:])
            metrics['�껯������'].append(annualized_return)
            metrics['�껯������'].append(annualized_volatility)
            metrics['���ձ���'].append(sharpe_ratio)
            metrics['���س�'].append(max_drawdown)
            metrics['����ŵ����'].append(sortino_ratio)
            metrics['ʤ��'].append(win_rate)
            metrics['����'].append(odds_ratio)
            metrics['Kelly��λ'].append(kelly_fraction)
            metrics['����źŴ���'].append(average_signals)

        # Create DataFrame
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
            # ��ȡ���Ե��źŻر�
            if strategy_name not in self.strategies_results:
                raise ValueError(f"�������� '{strategy_name}' �������ڻز����С�")

            strategy_returns = self.strategies_results[strategy_name]['Strategy_Return']
            index_returns = self.index_df_with_signal['Index_Return']

            # �������
            annual_strategy_returns = (1 + strategy_returns).resample('Y').prod() - 1

            # ���ָ������
            annual_index_returns = (1 + index_returns).resample('Y').prod() - 1

            # ��������
            annual_excess_returns = annual_strategy_returns - annual_index_returns

            # ���״���
            trade_counts = self.calculate_trade_counts(self.index_df_with_signal[f'{strategy_name}_signal'])

            # �������ͳ�Ƶ�DataFrame
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
                # �������ͳ������
                if strategy_name in self.stats_by_each_year:
                    self.stats_by_each_year[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_���ͳ��')
                else:
                    print(f"���� {strategy_name} �����ͳ�����ݲ����ڣ�������")

                # ������ϸ����
                if strategy_name in self.detailed_data:
                    self.detailed_data[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_��ϸ����')
                else:
                    print(f"���� {strategy_name} ����ϸ���ݲ����ڣ�������")

            # ��������ָ�꣬����Ϊָ�꣬����Ϊ������
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

    def calculate_annualized_return(self, strategy_returns):
        """
        Calculates the annualized return.

        Parameters:
            strategy_returns (pd.Series): Strategy returns.

        Returns:
            float: Annualized return.
        """
        cumulative_return = (1 + strategy_returns).prod()
        n_periods = strategy_returns.count()
        if n_periods == 0:
            return np.nan
        annualized_return = cumulative_return ** (self.annual_factor / n_periods) - 1
        return annualized_return

    def calculate_annualized_volatility(self, strategy_returns):
        """
        Calculates the annualized volatility.

        Parameters:
            strategy_returns (pd.Series): Strategy returns.

        Returns:
            float: Annualized volatility.
        """
        return strategy_returns.std() * np.sqrt(self.annual_factor)

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0):
        """
        Calculates the Sharpe Ratio.

        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            risk_free_rate (float): Risk-free rate.

        Returns:
            float: Sharpe Ratio.
        """
        excess_returns = strategy_returns.mean() - (risk_free_rate / self.annual_factor)
        volatility = strategy_returns.std()
        if volatility == 0:
            return np.nan
        return (excess_returns / volatility) * np.sqrt(self.annual_factor)

    def calculate_sortino_ratio(self, strategy_returns, target=0):
        """
        Calculates the Sortino Ratio.

        Parameters:
            strategy_returns (pd.Series): Strategy returns.
            target (float): Target return.

        Returns:
            float: Sortino Ratio.
        """
        downside_returns = strategy_returns[strategy_returns < target]
        downside_deviation = downside_returns.std() * np.sqrt(self.annual_factor)
        expected_return = (strategy_returns.mean() - target) * self.annual_factor
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
        # signal_changes = signals.diff()
        # long_trades = signal_changes == 1
        # short_trades = signal_changes == -1
        long_trades = signals == 1
        short_trades = signals == -1
        annual_long = long_trades.resample(self.time_delta).sum()
        annual_short = short_trades.resample(self.time_delta).sum()
        trade_counts = pd.DataFrame({
            'Annual_Long_Trades': annual_long,
            'Annual_Short_Trades': annual_short
        })
        return trade_counts

