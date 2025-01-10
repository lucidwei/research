# coding=gbk
# Time Created: 2025/1/9 15:18
# Author  : Lucid
# FileName: 快速复现.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import process_wind_excel
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

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
    metadata, df = process_wind_excel(file_path, sheet_name='Sheet1', column_name='指标名称')
    # 将所有列转换为数值类型
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
    计算指定列在滚动窗口内每个值的百分位排名。

    参数：
        df (pd.DataFrame): 数据框。
        column (str): 要计算百分位排名的列名。
        window (int): 滚动窗口大小（月份数）。

    返回：
        pd.Series: 百分位排名（0到1之间）。
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
        # Strategy 1: 中长期贷款同比MA2
        # df['中长期贷款同比MA2'] = calculate_moving_average(df, '中长期贷款同比MA2', window=2)
        df['中长期贷款同比MA2_prev'] = df['中长期贷款同比MA2'].shift(1)
        signals = np.where(df['中长期贷款同比MA2'] > df['中长期贷款同比MA2_prev'], 1, 0)
        signals = pd.Series(signals, index=df.index)

        # 将信号向前移动一个周期，表示次月末建仓
        signals = signals.shift(1)

    elif strategy_num == 2:
        # Strategy 2: M1同比MA3 and M1-PPI同比MA3
        # df['M1同比MA3'] = calculate_moving_average(df, 'M1同比MA3', window=3)
        # df['M1-PPI同比MA3'] = calculate_moving_average(df, 'M1-PPI同比MA3', window=3)
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
        # df['美元指数MA2'] = calculate_moving_average(df, '美元指数MA2', window=2)
        df['美元指数MA2_prev'] = df['美元指数MA2'].shift(1)
        signals = np.where(df['美元指数MA2'] < df['美元指数MA2_prev'], 1, 0)
        signals = pd.Series(signals, index=df.index)

    elif strategy_num == 4:
        # Strategy 4: Technical Long Strategy
        # # Calculate 5-year (60 months) rolling percentile for 市净率:上证指数:月:最后一条
        # df['PER_5Y_PCT'] = calculate_percentile(df, '市净率:上证指数:月:最后一条', window=60, percentile=0.5)
        #
        # # Condition 1: PER percentile below 50%
        # condition_per = df['市净率:上证指数:月:最后一条'] < df['PER_5Y_PCT']

        # 计算滚动60个月的百分位排名
        df['PER_5Y_Pct_Rank'] = calculate_rolling_percentile_rank(
            df, '市净率:上证指数:月:最后一条', window=60
        )

        # 条件1：市净率的百分位排名低于0.5
        condition_per = df['PER_5Y_Pct_Rank'] < 0.5

        # Condition 2: 成交金额:月:合计值:同比 and 成交金额:月:合计值:环比 both increase
        condition_yoy = df['上证综合指数:成交金额:月:合计值:同比'] > 0
        condition_mom = df['上证综合指数:成交金额:月:合计值:环比'] > 0
        condition_volume = condition_yoy & condition_mom

        # Condition 3: 上证综合指数:月:最后一条:同比 and 上证综合指数:月:最后一条:环比 both increase
        condition_index_yoy = df['上证综合指数:月:最后一条:同比'] > 0
        condition_index_mom = df['上证综合指数:月:最后一条:环比'] > 0
        condition_price = condition_index_yoy & condition_index_mom

        # Final signal
        signals = np.where(condition_per & condition_volume & condition_price, 1, 0)
        signals = pd.Series(signals, index=df.index)

    elif strategy_num == 5:
        # Strategy 5: Technical Short Strategy
        # # Calculate 5-year (60 months) rolling percentile for 市净率:上证指数:月:最后一条
        # df['PER_5Y_PCT'] = calculate_percentile(df, '市净率:上证指数:月:最后一条', window=60, percentile=0.5)
        #
        # # Condition 1: PER percentile above 50%
        # condition_per = df['市净率:上证指数:月:最后一条'] > df['PER_5Y_PCT']

        # 计算滚动60个月的百分位排名
        df['PER_5Y_Pct_Rank'] = calculate_rolling_percentile_rank(
            df, '市净率:上证指数:月:最后一条', window=60
        )

        # 条件1：市净率的百分位排名高于0.5
        condition_per = df['PER_5Y_Pct_Rank'] > 0.5

        # Condition 2a: 环比放量下跌
        condition_mom_volume = (df['上证综合指数:成交金额:月:合计值:环比'] > 0) & (
                    df['上证综合指数:月:最后一条:环比'] < 0)
        # Condition 2b: 同比缩量上涨
        condition_yoy_price = (df['上证综合指数:月:最后一条:同比'] > 0) & (
                    df['上证综合指数:成交金额:月:合计值:同比'] < 0)

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
    # 基本面改善信号：策略1、策略2、策略3中至少两个为1
    basic_improved = (
                             (df['strategy1_signal']) +
                             (df['strategy2_signal']) +
                             (df['strategy3_signal'])
                     ) >= 2

    # 技术面卖出信号：策略5信号为-1
    technical_sell = df['strategy5_signal'] == -1

    # 技术面买入信号：策略4信号为1
    technical_buy = df['strategy4_signal'] == 1

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
    # Calculate monthly returns of the 上证综合指数
    df['Index_Return'] = df['上证综合指数:月:最后一条'].pct_change()

    # Shift signals to represent holding the position from previous period
    df['Position'] = signals.shift(1)

    df['Strategy_Return'] = df['Position'] * df['Index_Return']

    # Replace NaN returns with 0
    df['Strategy_Return'].fillna(0, inplace=True)

    # Calculate cumulative returns
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Index'] = df['上证综合指数:月:最后一条']

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
    plt.plot(cumulative_index, label='上证综合指数')
    plt.plot(cumulative_strategy, label=f'策略{strategy_num} 净值')
    plt.title(f'策略{strategy_num} 回测结果')
    plt.xlabel('时间')
    plt.ylabel('累计收益')
    plt.legend()
    plt.grid(True)
    plt.show()


# 4. Main function to execute all strategies
def main():
    # Replace 'your_file.xlsx' with the path to your Excel file
    file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\招商择时快速复现.xlsx"

    # Load data
    df = load_data(file_path)

    # **关键修改**：设置回测起始时间为2001年12月
    backtest_start = pd.to_datetime('2001-12', format='%Y-%m')
    df_backtest = df[df.index >= backtest_start].copy()

    # List to store strategy numbers
    strategies = [1, 2, 3, 4, 5]
    # 初始化新列以存储每个策略的信号
    for strategy_num in strategies:
        df_backtest[f'strategy{strategy_num}_signal'] = 0

    for strategy_num in strategies:
        print(f'正在回测策略{strategy_num}...')

        # Generate signals
        signals = generate_strategy_signals(df.copy(), strategy_num)
        signals = signals[signals.index >= backtest_start].copy()

        # **将信号存储到对应的列中**
        df_backtest[f'strategy{strategy_num}_signal'] = signals

        # Backtest strategy
        cumulative_strategy, cumulative_index = backtest_strategy(df_backtest.copy(), signals, strategy_num)

        # 设置累计净值的起始值为1
        cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
        cumulative_index = cumulative_index / cumulative_index.iloc[0]

        # Plot results
        plot_results(cumulative_strategy, cumulative_index, strategy_num)

        # Optional: Print final cumulative returns
        final_strategy = cumulative_strategy.iloc[-1]
        final_index = cumulative_index.iloc[-1]
        print(f'策略{strategy_num} 最终净值: {final_strategy:.2f}')
        print(f'上证综合指数 最终净值: {final_index:.2f}\n')

    # **新增部分**：构建策略6
    print('正在回测策略6...')

    # 生成策略6的信号
    df_backtest['strategy6_signal'] = generate_strategy6_signals(df_backtest)

    # Backtest 策略6
    cumulative_strategy6, cumulative_index6 = backtest_strategy(
        df_backtest.copy(),
        df_backtest['strategy6_signal'],
        strategy_num=6
    )

    # **设置累计净值的起始值为1**
    cumulative_strategy6 = cumulative_strategy6 / cumulative_strategy6.iloc[0]
    cumulative_index6 = cumulative_index6 / cumulative_index6.iloc[0]

    # 绘制策略6的回测结果
    plot_results(cumulative_strategy6, cumulative_index6, strategy_num=6)

    # 打印策略6的最终净值
    final_strategy6 = cumulative_strategy6.iloc[-1]
    final_index6 = cumulative_index6.iloc[-1]
    print(f'策略6 最终净值: {final_strategy6:.2f}')
    print(f'上证综合指数 最终净值: {final_index6:.2f}\n')


# 5. Execute the main function
if __name__ == "__main__":
    main()