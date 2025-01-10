# coding=gbk
# Time Created: 2025/1/9 15:18
# Author  : Lucid
# FileName: ���ٸ���.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import process_wind_excel
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


# 1. Load the Excel data
def load_data(file_path):
    """
    Load Excel data into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Read the Excel file
    metadata, df = process_wind_excel(file_path, sheet_name='Sheet1', column_name='ָ������')
    # ��������ת��Ϊ��ֵ����
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Sort the DataFrame by date
    df.sort_index(inplace=True)

    return df


# 2. Define helper functions
def calculate_moving_average(df, column, window):
    """
    Calculate the moving average for a given column.

    Parameters:
        df (pd.DataFrame): The DataFrame.
        column (str): Column name to calculate moving average.
        window (int): Window size for moving average.

    Returns:
        pd.Series: Moving average series.
    """
    return df[column].rolling(window=window).mean()


def calculate_percentile(df, column, window, percentile=0.5):
    """
    Calculate the rolling percentile for a given column.

    Parameters:
        df (pd.DataFrame): The DataFrame.
        column (str): Column name to calculate percentile.
        window (int): Rolling window size.
        percentile (float): The percentile to calculate (0 to 1).

    Returns:
        pd.Series: Rolling percentile series.
    """
    return df[column].rolling(window=window).apply(lambda x: np.percentile(x, percentile * 100), raw=True)


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


def generate_strategy_signals(df, strategy_num):
    """
    Generate buy/sell signals based on the strategy number.

    Parameters:
        df (pd.DataFrame): The DataFrame with all data.
        strategy_num (int): Strategy number (1 to 5).

    Returns:
        pd.Series: Signals where 1 indicates buy and 0 indicates sell.
    """
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
        # df['M1ͬ��MA3'] = calculate_moving_average(df, 'M1ͬ��MA3', window=3)
        # df['M1-PPIͬ��MA3'] = calculate_moving_average(df, 'M1-PPIͬ��MA3', window=3)
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
        # df['��Ԫָ��MA2'] = calculate_moving_average(df, '��Ԫָ��MA2', window=2)
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
        # # Calculate 5-year (60 months) rolling percentile for �о���:��ָ֤��:��:���һ��
        # df['PER_5Y_PCT'] = calculate_percentile(df, '�о���:��ָ֤��:��:���һ��', window=60, percentile=0.5)
        #
        # # Condition 1: PER percentile above 50%
        # condition_per = df['�о���:��ָ֤��:��:���һ��'] > df['PER_5Y_PCT']

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


def generate_strategy6_signals(df):
    """
    Generate buy/sell signals for Strategy 6 based on Strategy 1~5 signals.

    Model Rules:
    1. If any of the following conditions are met, allocate/increase position in the index:
       a. Fundamental indicators show improvement, and technical indicators do not signal sell.
       b. Technical indicators signal a clear buy.
    2. Otherwise, consider reducing/closing the position.

    Parameters:
        df (pd.DataFrame): The DataFrame containing strategy1~5 signals.

    Returns:
        pd.Series: Signals where 1 indicates buy, -1 indicates sell, and 0 indicates hold.
    """
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


def backtest_strategy(df, signals, strategy_num):
    """
    Backtest the strategy based on buy/sell signals.

    Parameters:
        df (pd.DataFrame): The DataFrame with all data.
        signals (pd.Series): Series containing buy(1)/sell(0)/sell_short(-1) signals.
        strategy_num (int): Strategy number (1 to 5).

    Returns:
        pd.Series: Cumulative returns of the strategy.
    """
    # Calculate monthly returns of the ��֤�ۺ�ָ��
    df['Index_Return'] = df['��֤�ۺ�ָ��:��:���һ��'].pct_change()

    # Shift signals to represent holding the position from previous period
    df['Position'] = signals.shift(1)

    df['Strategy_Return'] = df['Position'] * df['Index_Return']

    # Replace NaN returns with 0
    df['Strategy_Return'].fillna(0, inplace=True)

    # Calculate cumulative returns
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Index'] = df['��֤�ۺ�ָ��:��:���һ��']

    return df['Cumulative_Strategy'], df['Cumulative_Index']


# 3. Plotting function
def plot_results(cumulative_strategy, cumulative_index, strategy_num):
    """
    Plot the cumulative returns of the strategy versus the index.

    Parameters:
        cumulative_strategy (pd.Series): Cumulative returns of the strategy.
        cumulative_index (pd.Series): Cumulative returns of the index.
        strategy_num (int): Strategy number.
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


# 4. Main function to execute all strategies
def main():
    # Replace 'your_file.xlsx' with the path to your Excel file
    file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\������ʱ���ٸ���.xlsx"

    # Load data
    df = load_data(file_path)

    # **�ؼ��޸�**�����ûز���ʼʱ��Ϊ2001��12��
    backtest_start = pd.to_datetime('2001-12', format='%Y-%m')
    df_backtest = df[df.index >= backtest_start].copy()

    # List to store strategy numbers
    strategies = [1, 2, 3, 4, 5]
    # ��ʼ�������Դ洢ÿ�����Ե��ź�
    for strategy_num in strategies:
        df_backtest[f'strategy{strategy_num}_signal'] = 0

    for strategy_num in strategies:
        print(f'���ڻز����{strategy_num}...')

        # Generate signals
        signals = generate_strategy_signals(df.copy(), strategy_num)
        signals = signals[signals.index >= backtest_start].copy()

        # **���źŴ洢����Ӧ������**
        df_backtest[f'strategy{strategy_num}_signal'] = signals

        # Backtest strategy
        cumulative_strategy, cumulative_index = backtest_strategy(df_backtest.copy(), signals, strategy_num)

        # �����ۼƾ�ֵ����ʼֵΪ1
        cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
        cumulative_index = cumulative_index / cumulative_index.iloc[0]

        # Plot results
        plot_results(cumulative_strategy, cumulative_index, strategy_num)

        # Optional: Print final cumulative returns
        final_strategy = cumulative_strategy.iloc[-1]
        final_index = cumulative_index.iloc[-1]
        print(f'����{strategy_num} ���վ�ֵ: {final_strategy:.2f}')
        print(f'��֤�ۺ�ָ�� ���վ�ֵ: {final_index:.2f}\n')

    # **��������**����������6
    print('���ڻز����6...')

    # ���ɲ���6���ź�
    df_backtest['strategy6_signal'] = generate_strategy6_signals(df_backtest)

    # Backtest ����6
    cumulative_strategy6, cumulative_index6 = backtest_strategy(
        df_backtest.copy(),
        df_backtest['strategy6_signal'],
        strategy_num=6
    )

    # **�����ۼƾ�ֵ����ʼֵΪ1**
    cumulative_strategy6 = cumulative_strategy6 / cumulative_strategy6.iloc[0]
    cumulative_index6 = cumulative_index6 / cumulative_index6.iloc[0]

    # ���Ʋ���6�Ļز���
    plot_results(cumulative_strategy6, cumulative_index6, strategy_num=6)

    # ��ӡ����6�����վ�ֵ
    final_strategy6 = cumulative_strategy6.iloc[-1]
    final_index6 = cumulative_index6.iloc[-1]
    print(f'����6 ���վ�ֵ: {final_strategy6:.2f}')
    print(f'��֤�ۺ�ָ�� ���վ�ֵ: {final_index6:.2f}\n')


# 5. Execute the main function
if __name__ == "__main__":
    main()