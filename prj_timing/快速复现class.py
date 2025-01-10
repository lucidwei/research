# coding=gbk
# Time Created: 2025/1/10 11:13
# Author  : Lucid
# FileName: ���ٸ���class.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy import stats
from utils import process_wind_excel
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class StrategyBacktester:
    def __init__(self, file_path):
        """
        ��ʼ���������������ݲ����г�������

        Parameters:
            file_path (str): Excel�ļ���·����
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.strategies = [1, 2, 3, 4, 5, 6]  # ����1~6
        self.strategies_results = {}

    def load_data(self):
        """
        ����Excel���ݵ�pandas DataFrame��������DateTime������

        Returns:
            pd.DataFrame: ������������ݿ�
        """
        # Read the Excel file
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='ָ������')
        # ��������ת��Ϊ��ֵ����
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        # Sort the DataFrame by date
        df_macro.sort_index(inplace=True)

        # ��ȡSheet2�е���������
        df = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # ͨ������ȫ�������ָͬ�����ݿ�
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        # ���衰��ָ֤�����ǵ�һ�����ݿ�
        if split_indices:
            first_split = split_indices[0]
            z_index_df = df.iloc[:, :first_split].copy()
        else:
            z_index_df = df.copy()

        # ��������ָ֤������
        # ������˳��Ϊ ['����', '���̼�', '�ɽ���\n[��λ]��Ԫ', '�о���PB(LF,�ڵ�)']
        z_index_df.columns = ['����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)']
        # ɾ��ǰ����
        z_index_df = z_index_df[3:]
        # ɾ������ΪNaN����
        z_index_df.dropna(subset=['����'], inplace=True)

        # �� '����' ת��Ϊ datetime ���Ͳ�����Ϊ����
        z_index_df['����'] = pd.to_datetime(z_index_df['����'])
        z_index_df.set_index('����', inplace=True)

        # ����������
        z_index_df.sort_index(inplace=True)

        # ת��������Ϊ��ֵ���ͣ����Դ���
        z_index_df = z_index_df.apply(pd.to_numeric, errors='coerce')

        # ����Ƶ����ת��Ϊ��Ƶ����
        monthly = z_index_df.resample('M').agg({
            '���̼�': 'last',
            '�ɽ���': 'sum',
            '�о���PB(LF,�ڵ�)': 'last'
        })

        # ���������ָ����
        monthly['��֤�ۺ�ָ��:��:���һ��'] = monthly['���̼�']
        monthly['��֤�ۺ�ָ��:��:���һ��:ͬ��'] = monthly['���̼�'].pct_change(12)
        monthly['��֤�ۺ�ָ��:��:���һ��:����'] = monthly['���̼�'].pct_change(1)
        monthly['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ'] = monthly['�ɽ���']
        monthly['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��'] = monthly['�ɽ���'].pct_change(12)
        monthly['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'] = monthly['�ɽ���'].pct_change(1)
        monthly['�о���:��ָ֤��:��:���һ��'] = monthly['�о���PB(LF,�ڵ�)']

        # ѡ������������Ҫ����
        final_df = monthly[[
            '�о���:��ָ֤��:��:���һ��',
            '��֤�ۺ�ָ��:��:���һ��',
            '��֤�ۺ�ָ��:��:���һ��:ͬ��',
            '��֤�ۺ�ָ��:��:���һ��:����',
            '��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��',
            '��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'
        ]]

        # ȷ�� df_macro �� final_df �����������ڣ�����
        if not df_macro.index.equals(final_df.index):
            raise ValueError("df_macro �� final_df ������������ƥ�䣬�޷��ϲ���")

        # �ϲ� df_macro �� final_df
        merged_df = pd.concat([df_macro, final_df], axis=1)

        return merged_df

    def generate_signals_for_all_strategies(self):
        """
        Ϊ����1~6���������źš�
        """
        # Ϊ����1~5�����źŲ��洢�����ݿ���
        for strategy_num in range(1, 6):
            signals = self.generate_strategy_signals(strategy_num)
            self.df[f'strategy{strategy_num}_signal'] = signals

        # Ϊ����6�����ź�
        strategy6_signals = self.generate_strategy6_signals()
        self.df['strategy6_signal'] = strategy6_signals

    def generate_strategy_signals(self, strategy_num):
        """
        Ϊָ���������������źš�

        Parameters:
            strategy_num (int): ���Ա�ţ�1~5����

        Returns:
            pd.Series: �����ź����С�
        """

        def calculate_rolling_percentile_rank(df, column, window):
            """
            ����ָ�����ڹ���������ÿ��ֵ�İٷ�λ������

            ������
                df (pd.DataFrame): ���ݿ�
                column (str): Ҫ����ٷ�λ������������
                window (int): �������ڴ�С���·�������

            ���أ�
                pd.Series: �ٷ�λ������0��1֮�䣩��
            """
            return df[column].rolling(window=window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1], kind='weak') / 100,
                raw=False
            )

        df = self.df
        signals = pd.Series(index=df.index, data=0)

        if strategy_num == 1:
            # Strategy 1: �г��ڴ���ͬ��MA2
            # df['�г��ڴ���ͬ��MA2'] = calculate_moving_average(df, '�г��ڴ���ͬ��MA2', window=2)
            df['�г��ڴ���ͬ��MA2_prev'] = df['�г��ڴ���ͬ��MA2'].shift(1)
            signals = np.where(df['�г��ڴ���ͬ��MA2'] > df['�г��ڴ���ͬ��MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index)

            # ���ź���ǰ�ƶ�һ�����ڣ���ʾ����ĩ����
            signals = signals.shift(1)

        elif strategy_num == 2:
            # Strategy 2: M1ͬ��MA3 and M1-PPIͬ��MA3
            df['M1ͬ��MA3_prev'] = df['M1ͬ��MA3'].shift(1)
            df['M1-PPIͬ��MA3_prev'] = df['M1-PPIͬ��MA3'].shift(1)
            condition1 = df['M1ͬ��MA3'] > df['M1ͬ��MA3_prev']
            condition2 = df['M1-PPIͬ��MA3'] > df['M1-PPIͬ��MA3_prev']
            signals = np.where(condition1 | condition2, 1, 0)
            signals = pd.Series(signals, index=df.index)

            # ���ź���ǰ�ƶ�һ�����ڣ���ʾ����ĩ����
            signals = signals.shift(1)

        elif strategy_num == 3:
            # Strategy 3: ��Ԫָ��MA2
            df['��Ԫָ��MA2_prev'] = df['��Ԫָ��MA2'].shift(1)
            signals = np.where(df['��Ԫָ��MA2'] < df['��Ԫָ��MA2_prev'], 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 4:
            # Strategy 4: Technical Long Strategy
            # # Calculate 5-year (60 months) rolling percentile for �о���:��ָ֤��:��:���һ��
            # df['PER_5Y_PCT'] = calculate_percentile(df, '�о���:��ָ֤��:��:���һ��', window=60, percentile=0.5)
            #
            # # Condition 1: PER percentile below 50%
            # condition_per = df['�о���:��ָ֤��:��:���һ��'] < df['PER_5Y_PCT']

            # �������60���µİٷ�λ����
            df['PER_5Y_Pct_Rank'] = calculate_rolling_percentile_rank(
                df, '�о���:��ָ֤��:��:���һ��', window=60
            )

            # ����1���о��ʵİٷ�λ��������0.5
            condition_per = df['PER_5Y_Pct_Rank'] < 0.5

            # Condition 2: �ɽ����:��:�ϼ�ֵ:ͬ�� and �ɽ����:��:�ϼ�ֵ:���� both increase
            condition_yoy = df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��'] > 0
            condition_mom = df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'] > 0
            condition_volume = condition_yoy & condition_mom

            # Condition 3: ��֤�ۺ�ָ��:��:���һ��:ͬ�� and ��֤�ۺ�ָ��:��:���һ��:���� both increase
            condition_index_yoy = df['��֤�ۺ�ָ��:��:���һ��:ͬ��'] > 0
            condition_index_mom = df['��֤�ۺ�ָ��:��:���һ��:����'] > 0
            condition_price = condition_index_yoy & condition_index_mom

            # Final signal
            signals = np.where(condition_per & condition_volume & condition_price, 1, 0)
            signals = pd.Series(signals, index=df.index)

        elif strategy_num == 5:
            # Strategy 5: Technical Short Strategy
            # �������60���µİٷ�λ����
            df['PER_5Y_Pct_Rank'] = calculate_rolling_percentile_rank(
                df, '�о���:��ָ֤��:��:���һ��', window=60
            )

            # ����1���о��ʵİٷ�λ��������0.5
            condition_per = df['PER_5Y_Pct_Rank'] > 0.5

            # Condition 2a: ���ȷ����µ�
            condition_mom_volume = (df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'] > 0) & (
                    df['��֤�ۺ�ָ��:��:���һ��:����'] < 0)
            # Condition 2b: ͬ����������
            condition_yoy_price = (df['��֤�ۺ�ָ��:��:���һ��:ͬ��'] > 0) & (
                    df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��'] < 0)

            # Final condition: either 2a or 2b
            condition_sell = condition_mom_volume | condition_yoy_price

            # Final signal
            signals = np.where(condition_per & condition_sell, -1, 0)  # -1 indicates sell
            signals = pd.Series(signals, index=df.index)

        return signals

    def generate_strategy6_signals(self):
        """
        Ϊ����6���������źţ����ڲ���1~5���źš�

        Returns:
            pd.Series: ����6�������ź����С�
        """
        df = self.df
        # ����������źţ�����1������2������3����������Ϊ1
        basic_improved = (
                                 (df['strategy1_signal']) +
                                 (df['strategy2_signal']) +
                                 (df['strategy3_signal'])
                         ) >= 2

        # �����������źţ�����5�ź�Ϊ-1
        technical_sell = df['strategy5_signal'] == -1

        # �����������źţ�����4�ź�Ϊ1
        technical_buy = df['strategy4_signal'] == 1

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

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        �����в���1~6���лز⣬����ÿ�»ر����ۼƾ�ֵ��
        """
        for strategy_num in self.strategies:
            print(f'\n���ڻز����{strategy_num}...')
            signals = self.df[f'strategy{strategy_num}_signal']
            cumulative_strategy, cumulative_index, strategy_returns = self.backtest_strategy(signals, start_date)
            # ��һ���ۼƾ�ֵ
            cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
            cumulative_index = cumulative_index / cumulative_index.iloc[0]
            # �洢�ز���
            self.strategies_results[strategy_num] = {
                'Cumulative_Strategy': cumulative_strategy,
                'Cumulative_Index': cumulative_index,
                'Strategy_Return': strategy_returns
            }
            # ��ͼ��������plot_results������
            self.plot_results(cumulative_strategy, cumulative_index, strategy_num)
            # ��ӡ���վ�ֵ
            final_strategy = cumulative_strategy.iloc[-1]
            final_index = cumulative_index.iloc[-1]
            print(f'����{strategy_num} ���վ�ֵ: {final_strategy:.2f}')
            print(f'��֤�ۺ�ָ�� ���վ�ֵ: {final_index:.2f}\n')

    def backtest_strategy(self, signals, start_date):
        """
        �ز�ָ�����ԡ�

        Parameters:
            signals (pd.Series): �����ź����С�
            strategy_num (int): ���Ա�ţ�1~6����

        Returns:
            pd.Series: �ۼƲ��Ծ�ֵ��
            pd.Series: �ۼ�ָ����ֵ��
            pd.Series: ÿ�²��Իر���
        """
        df = self.df
        # Calculate monthly returns of the ��֤�ۺ�ָ��
        df['Index_Return'] = df['��֤�ۺ�ָ��:��:���һ��'].pct_change()

        backtest_start = pd.to_datetime(start_date, format='%Y-%m')
        df = df[df.index >= backtest_start].copy()
        self.df = df
        signals = signals[signals.index >= backtest_start].copy()

        # Shift signals to represent holding the position from previous period
        df['Position'] = signals.shift(1)

        df['Strategy_Return'] = df['Position'] * df['Index_Return']

        # Replace NaN returns with 0
        df['Strategy_Return'].fillna(0, inplace=True)

        # Calculate cumulative returns
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        df['Cumulative_Index'] = df['��֤�ۺ�ָ��:��:���һ��']

        return df['Cumulative_Strategy'], df['Cumulative_Index'], df['Strategy_Return']

    def plot_results(self, cumulative_strategy, cumulative_index, strategy_num):
        """
        ���Ʋ�����ָ�����ۼƾ�ֵ�Ա�ͼ��

        Parameters:
            cumulative_strategy (pd.Series): �����ۼƾ�ֵ��
            cumulative_index (pd.Series): ָ���ۼƾ�ֵ��
            strategy_num (int): ���Ա�š�
        """
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_index, label='��֤�ۺ�ָ��')
        plt.plot(cumulative_strategy, label=f'����{strategy_num} ��ֵ')
        plt.title(f'����{strategy_num} �ز���')
        plt.xlabel('ʱ��')
        plt.ylabel('�ۼ�����')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_metrics_all_strategies(self):
        """
        �������в��Եļ�Чָ�ꡣ
        """
        # ��ʼ��DataFrame���ڴ洢ָ��
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

        for strategy_num, results in self.strategies_results.items():
            print(f'���ڼ������{strategy_num}�ļ�Чָ��...')
            strategy_returns = results['Strategy_Return']
            cumulative_strategy = results['Cumulative_Strategy']

            # �껯������
            annualized_return = self.calculate_annualized_return(strategy_returns)

            # �껯������
            annualized_volatility = self.calculate_annualized_volatility(strategy_returns)

            # ���ձ���
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)

            # ���س�
            max_drawdown = self.calculate_max_drawdown(cumulative_strategy)

            # ����ŵ����
            sortino_ratio = self.calculate_sortino_ratio(strategy_returns)

            # ʤ��
            win_rate = self.calculate_win_rate(strategy_returns)

            # ����
            odds_ratio = self.calculate_odds_ratio(strategy_returns)

            # ������λ
            kelly_fraction = self.calculate_kelly_fraction(win_rate, odds_ratio)

            # ��������źŴ���
            # ͳ��strategy_returns������·�����������ÿ���ƽ������
            annual_signals = (strategy_returns != 0).resample('Y').sum()
            average_signals = annual_signals.mean()

            # ��ӵ�metrics�ֵ�
            metrics['��������'].append(f'����{strategy_num}')
            metrics['�껯������'].append(annualized_return)
            metrics['�껯������'].append(annualized_volatility)
            metrics['���ձ���'].append(sharpe_ratio)
            metrics['���س�'].append(max_drawdown)
            metrics['����ŵ����'].append(sortino_ratio)
            metrics['ʤ��'].append(win_rate)
            metrics['����'].append(odds_ratio)
            metrics['Kelly��λ'].append(kelly_fraction)
            metrics['����źŴ���'].append(average_signals)

        # ����DataFrame
        self.metrics_df = pd.DataFrame(metrics)
        self.metrics_df.set_index('��������', inplace=True)

    def calculate_annual_metrics_strategy6(self):
        """
        �������6ÿ������桢��ָ֤�����桢�������桢ÿ�꽻�׶൥�����Ϳյ�������
        """
        # ��ȡ����6���»ر�
        strategy6_returns = self.strategies_results[6]['Strategy_Return']
        index_returns = self.df['Index_Return']

        # �������
        annual_strategy_returns = self.calculate_annual_returns(strategy6_returns)

        # ���ָ������
        annual_index_returns = (1 + index_returns).resample('Y').prod() - 1

        # ��������
        annual_excess_returns = annual_strategy_returns - annual_index_returns

        # ���״���
        trade_counts = self.calculate_trade_counts(self.df['strategy6_signal'])

        # ����DataFrame
        self.annual_returns_df = pd.DataFrame({
            '�����������': annual_strategy_returns,
            '��ָ֤���������': annual_index_returns,
            '��������': annual_excess_returns,
            'ÿ�꽻�׶൥����': trade_counts['Annual_Long_Trades'],
            'ÿ�꽻�׿յ�����': trade_counts['Annual_Short_Trades']
        })
        self.annual_returns_df.index = self.annual_returns_df.index.year  # ����������Ϊ���

    def generate_excel_reports(self, output_file):
        """
        ���ɲ���������Excelͳ�Ʊ�ͬһ���ļ��������������С�

        Parameters:
            output_file (str): ���Excel�ļ���·����
        """
        with pd.ExcelWriter(output_file) as writer:
            # ��1������6ÿ������桢��ָ֤�����桢�������桢ÿ�꽻�׶൥�����Ϳյ�����
            self.annual_returns_df.to_excel(writer, sheet_name='����6���ͳ��')

            # ��2��������ָ�꣬����Ϊָ�꣬����Ϊ������
            self.metrics_df.to_excel(writer, sheet_name='���Լ�Чָ��')

    def calculate_annualized_return(self, strategy_returns):
        """
        ������Ե��껯�����ʡ�

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���

        Returns:
            float: �껯�����ʡ�
        """
        # �����ۼ�����
        cumulative_return = (1 + strategy_returns).prod()
        # ����������
        n_months = strategy_returns.count()
        # �����껯������
        annualized_return = cumulative_return ** (12 / n_months) - 1
        return annualized_return

    def calculate_annual_returns(self, strategy_returns):
        """
        Calculate annual returns from monthly strategy returns.

        Parameters:
            strategy_returns (pd.Series): Monthly returns of the strategy.

        Returns:
            pd.Series: Annual returns.
        """
        return (1 + strategy_returns).resample('Y').prod() - 1

    # def calculate_excess_returns(self, strategy_returns, index_returns):
    #     """
    #     Calculate annual excess returns of the strategy over the index.
    #
    #     Parameters:
    #         strategy_returns (pd.Series): Monthly returns of the strategy.
    #         index_returns (pd.Series): Monthly returns of the index.
    #
    #     Returns:
    #         pd.Series: Annual excess returns.
    #     """
    #     annual_strategy = self.calculate_annual_returns(strategy_returns)
    #     annual_index = self.calculate_annual_returns(index_returns)
    #     excess_returns = annual_strategy - annual_index
    #     return excess_returns

    def calculate_annualized_volatility(self, strategy_returns):
        """
        ������Ե��껯�����ʡ�

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���

        Returns:
            float: �껯�����ʡ�
        """
        # �����¶Ȳ����ʲ��껯
        annualized_volatility = strategy_returns.std() * np.sqrt(12)
        return annualized_volatility

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0):
        """
        ������Ե����ձ��ʡ�

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���
            risk_free_rate (float): �޷������ʣ�Ĭ��Ϊ0��

        Returns:
            float: ���ձ��ʡ�
        """
        # �����¶ȳ�������
        excess_returns = strategy_returns.mean() - risk_free_rate / 12
        # �������ձ���
        sharpe_ratio = (excess_returns / strategy_returns.std()) * np.sqrt(12)
        return sharpe_ratio

    def calculate_sortino_ratio(self, strategy_returns, target=0):
        """
        ������Ե�����ŵ���ʡ�

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���
            target (float): Ŀ��ر���Ĭ��Ϊ0��

        Returns:
            float: ����ŵ���ʡ�
        """
        # �������з���
        downside_returns = strategy_returns[strategy_returns < target]
        downside_deviation = downside_returns.std() * np.sqrt(12)
        # ����Ŀ��������ʵ��ƽ������Ĳ�
        expected_return = 12 * (strategy_returns.mean() - target)
        # ��ֹ�������
        if downside_deviation == 0:
            return np.nan
        # ��������ŵ����
        sortino_ratio = expected_return / downside_deviation
        return sortino_ratio

    def calculate_max_drawdown(self, cumulative_returns):
        """
        ������Ե����س���

        Parameters:
            cumulative_returns (pd.Series): �����ۼƾ�ֵ��

        Returns:
            float: ���س���
        """
        # ����������ֵ
        rolling_max = cumulative_returns.cummax()
        # ����س�
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        # �������س�
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_win_rate(self, strategy_returns):
        """
        ��������źŵĲ���ʤ�ʡ�
        �����ǲ��Է����źŵ��·ݣ�strategy_returns != 0����
        �����Է��������źţ�strategy_returns > 0��ʱ���ر�Ϊ��Ϊʤ��
        �����Է��������źţ�strategy_returns < 0��ʱ���ر�Ϊ��Ϊʤ��

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���

        Returns:
            float: ʤ�ʡ�
        """
        # �ų����Իر�Ϊ0���·ݣ����ź��·ݣ�
        active_returns = strategy_returns[strategy_returns != 0]

        # ���û�л�Ծ�Ľ��ף�����NaN
        if active_returns.empty:
            return np.nan

        # ʤ�����������Իر� > 0
        buy_wins = active_returns > 0

        # ����ʤ�ʣ�ʤ������ / ���źŴ���
        total_wins = buy_wins.sum()
        total_signals = active_returns.count()
        win_rate = total_wins / total_signals

        return win_rate

    def calculate_odds_ratio(self, strategy_returns):
        """
        ��������źŵĲ������ʣ�ƽ��ӯ�� / ƽ�����𣩡�
        �����ǲ��Է����źŵ��·ݣ�strategy_returns != 0����

        Parameters:
            strategy_returns (pd.Series): ÿ�²��Իر���

        Returns:
            float: ���ʡ�
        """
        # �ų����Իر�Ϊ0���·ݣ����ź��·ݣ�
        active_returns = strategy_returns[strategy_returns != 0]

        # ӯ�����ף����Իر� > 0
        wins = active_returns[active_returns > 0]

        # �����ף����Իر� < 0
        losses = active_returns[active_returns < 0]

        # ���û�п����ף�����NaN
        if losses.empty:
            return np.nan

        # ����ƽ��ӯ����ƽ������
        avg_win = wins.mean()
        avg_loss = losses.mean()

        # ���� = ƽ��ӯ�� / ƽ������ȡ����ֵ��
        odds_ratio = avg_win / abs(avg_loss)

        return odds_ratio

    def calculate_kelly_fraction(self, win_rate, odds_ratio):
        """
        ������ԵĿ�����λ��

        Parameters:
            win_rate (float): ʤ�ʡ�
            odds_ratio (float): ���ʡ�

        Returns:
            float: ������λ��
        """
        # ��������Ƿ���Ч
        if np.isnan(odds_ratio) or odds_ratio == 0:
            return 0
        # ���㿭������
        kelly_fraction = (win_rate * (odds_ratio + 1) - 1) / odds_ratio
        # ȷ��������λ�Ǹ�
        return max(kelly_fraction, 0)

    def calculate_trade_counts(self, signals):
        """
        ����ÿ��Ķ൥�Ϳյ����״�����

        Parameters:
            signals (pd.Series): ���Ե������ź����С�

        Returns:
            pd.DataFrame: ����ÿ��൥�Ϳյ����״�����DataFrame��
        """
        # �����źŵı仯
        signal_changes = signals.diff()

        # �൥���ף��źŴӷǳֲֻ�ղֱ�Ϊ�ֲ֣��źű�Ϊ1��
        long_trades = signal_changes == 1
        # �յ����ף��źŴӷǳֲֻ��ֱ�Ϊ�ղ֣��źű�Ϊ-1��
        short_trades = signal_changes == -1

        # ������ز�����ͳ��ÿ��Ľ��״���
        annual_long = long_trades.resample('Y').sum()
        annual_short = short_trades.resample('Y').sum()

        # �������DataFrame
        trade_counts = pd.DataFrame({
            'Annual_Long_Trades': annual_long,
            'Annual_Short_Trades': annual_short
        })

        return trade_counts




def main():
    # ָ��Excel�ļ�·��
    file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\������ʱ���ٸ���.xlsx"
    output_file = rf"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\���Իز���1.xlsx"

    # ʵ���� StrategyBacktester ��
    backtester = StrategyBacktester(file_path)

    # �������в��Ե������ź�
    backtester.generate_signals_for_all_strategies()

    # �������в��ԵĻز�
    backtester.backtest_all_strategies()

    # �������в��Եļ�Чָ��
    backtester.calculate_metrics_all_strategies()

    # �������6�����ͳ��
    backtester.calculate_annual_metrics_strategy6()

    # ���ɲ�����Excel����
    backtester.generate_excel_reports(output_file)

    print(f'�ز���ɣ�����ѱ��浽 {output_file}')

# 5. Execute the main function
if __name__ == "__main__":
    main()