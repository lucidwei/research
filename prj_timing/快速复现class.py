# coding=gbk
# Time Created: 2025/1/10 11:13
# Author  : Lucid
# FileName: 快速复现class.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy import stats
from utils import process_wind_excel
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


class StrategyBacktester:
    def __init__(self, file_path):
        """
        初始化方法：加载数据并进行初步处理。

        Parameters:
            file_path (str): Excel文件的路径。
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.strategies = [1, 2, 3, 4, 5, 6]  # 策略1~6
        self.strategies_results = {}

    def load_data(self):
        """
        加载Excel数据到pandas DataFrame，并设置DateTime索引。

        Returns:
            pd.DataFrame: 经过处理的数据框。
        """
        # Read the Excel file
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='指标名称')
        # 将所有列转换为数值类型
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        # Sort the DataFrame by date
        df_macro.sort_index(inplace=True)

        # 读取Sheet2中的所有数据
        df = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # 通过查找全空列来分割不同的数据块
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        # 假设“上证指数”是第一个数据块
        if split_indices:
            first_split = split_indices[0]
            z_index_df = df.iloc[:, :first_split].copy()
        else:
            z_index_df = df.copy()

        # 重命名上证指数的列
        # 假设列顺序为 ['日期', '收盘价', '成交额\n[单位]亿元', '市净率PB(LF,内地)']
        z_index_df.columns = ['日期', '收盘价', '成交额', '市净率PB(LF,内地)']
        # 删除前三行
        z_index_df = z_index_df[3:]
        # 删除日期为NaN的行
        z_index_df.dropna(subset=['日期'], inplace=True)

        # 将 '日期' 转换为 datetime 类型并设置为索引
        z_index_df['日期'] = pd.to_datetime(z_index_df['日期'])
        z_index_df.set_index('日期', inplace=True)

        # 按日期排序
        z_index_df.sort_index(inplace=True)

        # 转换所有列为数值类型（忽略错误）
        z_index_df = z_index_df.apply(pd.to_numeric, errors='coerce')

        # 将日频数据转换为月频数据
        monthly = z_index_df.resample('M').agg({
            '收盘价': 'last',
            '成交额': 'sum',
            '市净率PB(LF,内地)': 'last'
        })

        # 计算所需的指标列
        monthly['上证综合指数:月:最后一条'] = monthly['收盘价']
        monthly['上证综合指数:月:最后一条:同比'] = monthly['收盘价'].pct_change(12)
        monthly['上证综合指数:月:最后一条:环比'] = monthly['收盘价'].pct_change(1)
        monthly['上证综合指数:成交金额:月:合计值'] = monthly['成交额']
        monthly['上证综合指数:成交金额:月:合计值:同比'] = monthly['成交额'].pct_change(12)
        monthly['上证综合指数:成交金额:月:合计值:环比'] = monthly['成交额'].pct_change(1)
        monthly['市净率:上证指数:月:最后一条'] = monthly['市净率PB(LF,内地)']

        # 选择并重新排列需要的列
        final_df = monthly[[
            '市净率:上证指数:月:最后一条',
            '上证综合指数:月:最后一条',
            '上证综合指数:月:最后一条:同比',
            '上证综合指数:月:最后一条:环比',
            '上证综合指数:成交金额:月:合计值:同比',
            '上证综合指数:成交金额:月:合计值:环比'
        ]]

        # 确保 df_macro 和 final_df 的索引（日期）对齐
        if not df_macro.index.equals(final_df.index):
            raise ValueError("df_macro 和 final_df 的日期索引不匹配，无法合并。")

        # 合并 df_macro 和 final_df
        merged_df = pd.concat([df_macro, final_df], axis=1)

        return merged_df

    def generate_signals_for_all_strategies(self):
        """
        为策略1~6生成买卖信号。
        """
        # 为策略1~5生成信号并存储到数据框中
        for strategy_num in range(1, 6):
            signals = self.generate_strategy_signals(strategy_num)
            self.df[f'strategy{strategy_num}_signal'] = signals

        # 为策略6生成信号
        strategy6_signals = self.generate_strategy6_signals()
        self.df['strategy6_signal'] = strategy6_signals

    def generate_strategy_signals(self, strategy_num):
        """
        为指定策略生成买卖信号。

        Parameters:
            strategy_num (int): 策略编号（1~5）。

        Returns:
            pd.Series: 买卖信号序列。
        """

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

        df = self.df
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

    def generate_strategy6_signals(self):
        """
        为策略6生成买卖信号，基于策略1~5的信号。

        Returns:
            pd.Series: 策略6的买卖信号序列。
        """
        df = self.df
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

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        对所有策略1~6进行回测，计算每月回报和累计净值。
        """
        for strategy_num in self.strategies:
            print(f'\n正在回测策略{strategy_num}...')
            signals = self.df[f'strategy{strategy_num}_signal']
            cumulative_strategy, cumulative_index, strategy_returns = self.backtest_strategy(signals, start_date)
            # 归一化累计净值
            cumulative_strategy = cumulative_strategy / cumulative_strategy.iloc[0]
            cumulative_index = cumulative_index / cumulative_index.iloc[0]
            # 存储回测结果
            self.strategies_results[strategy_num] = {
                'Cumulative_Strategy': cumulative_strategy,
                'Cumulative_Index': cumulative_index,
                'Strategy_Return': strategy_returns
            }
            # 绘图（假设有plot_results方法）
            self.plot_results(cumulative_strategy, cumulative_index, strategy_num)
            # 打印最终净值
            final_strategy = cumulative_strategy.iloc[-1]
            final_index = cumulative_index.iloc[-1]
            print(f'策略{strategy_num} 最终净值: {final_strategy:.2f}')
            print(f'上证综合指数 最终净值: {final_index:.2f}\n')

    def backtest_strategy(self, signals, start_date):
        """
        回测指定策略。

        Parameters:
            signals (pd.Series): 买卖信号序列。
            strategy_num (int): 策略编号（1~6）。

        Returns:
            pd.Series: 累计策略净值。
            pd.Series: 累计指数净值。
            pd.Series: 每月策略回报。
        """
        df = self.df
        # Calculate monthly returns of the 上证综合指数
        df['Index_Return'] = df['上证综合指数:月:最后一条'].pct_change()

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
        df['Cumulative_Index'] = df['上证综合指数:月:最后一条']

        return df['Cumulative_Strategy'], df['Cumulative_Index'], df['Strategy_Return']

    def plot_results(self, cumulative_strategy, cumulative_index, strategy_num):
        """
        绘制策略与指数的累计净值对比图。

        Parameters:
            cumulative_strategy (pd.Series): 策略累计净值。
            cumulative_index (pd.Series): 指数累计净值。
            strategy_num (int): 策略编号。
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

    def calculate_metrics_all_strategies(self):
        """
        计算所有策略的绩效指标。
        """
        # 初始化DataFrame用于存储指标
        metrics = {
            '策略名称': [],
            '年化收益率': [],
            '年化波动率': [],
            '夏普比率': [],
            '最大回撤': [],
            '索提诺比率': [],
            '胜率': [],
            '赔率': [],
            'Kelly仓位': [],
            '年均信号次数': []
        }

        for strategy_num, results in self.strategies_results.items():
            print(f'正在计算策略{strategy_num}的绩效指标...')
            strategy_returns = results['Strategy_Return']
            cumulative_strategy = results['Cumulative_Strategy']

            # 年化收益率
            annualized_return = self.calculate_annualized_return(strategy_returns)

            # 年化波动率
            annualized_volatility = self.calculate_annualized_volatility(strategy_returns)

            # 夏普比率
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)

            # 最大回撤
            max_drawdown = self.calculate_max_drawdown(cumulative_strategy)

            # 索提诺比率
            sortino_ratio = self.calculate_sortino_ratio(strategy_returns)

            # 胜率
            win_rate = self.calculate_win_rate(strategy_returns)

            # 赔率
            odds_ratio = self.calculate_odds_ratio(strategy_returns)

            # 凯利仓位
            kelly_fraction = self.calculate_kelly_fraction(win_rate, odds_ratio)

            # 计算年均信号次数
            # 统计strategy_returns非零的月份数，并计算每年的平均次数
            annual_signals = (strategy_returns != 0).resample('Y').sum()
            average_signals = annual_signals.mean()

            # 添加到metrics字典
            metrics['策略名称'].append(f'策略{strategy_num}')
            metrics['年化收益率'].append(annualized_return)
            metrics['年化波动率'].append(annualized_volatility)
            metrics['夏普比率'].append(sharpe_ratio)
            metrics['最大回撤'].append(max_drawdown)
            metrics['索提诺比率'].append(sortino_ratio)
            metrics['胜率'].append(win_rate)
            metrics['赔率'].append(odds_ratio)
            metrics['Kelly仓位'].append(kelly_fraction)
            metrics['年均信号次数'].append(average_signals)

        # 创建DataFrame
        self.metrics_df = pd.DataFrame(metrics)
        self.metrics_df.set_index('策略名称', inplace=True)

    def calculate_annual_metrics_strategy6(self):
        """
        计算策略6每年的收益、上证指数收益、超额收益、每年交易多单次数和空单次数。
        """
        # 获取策略6的月回报
        strategy6_returns = self.strategies_results[6]['Strategy_Return']
        index_returns = self.df['Index_Return']

        # 年度收益
        annual_strategy_returns = self.calculate_annual_returns(strategy6_returns)

        # 年度指数收益
        annual_index_returns = (1 + index_returns).resample('Y').prod() - 1

        # 超额收益
        annual_excess_returns = annual_strategy_returns - annual_index_returns

        # 交易次数
        trade_counts = self.calculate_trade_counts(self.df['strategy6_signal'])

        # 创建DataFrame
        self.annual_returns_df = pd.DataFrame({
            '策略年度收益': annual_strategy_returns,
            '上证指数年度收益': annual_index_returns,
            '超额收益': annual_excess_returns,
            '每年交易多单次数': trade_counts['Annual_Long_Trades'],
            '每年交易空单次数': trade_counts['Annual_Short_Trades']
        })
        self.annual_returns_df.index = self.annual_returns_df.index.year  # 将索引设置为年份

    def generate_excel_reports(self, output_file):
        """
        生成并保存两个Excel统计表到同一个文件的两个工作表中。

        Parameters:
            output_file (str): 输出Excel文件的路径。
        """
        with pd.ExcelWriter(output_file) as writer:
            # 表1：策略6每年的收益、上证指数收益、超额收益、每年交易多单次数和空单次数
            self.annual_returns_df.to_excel(writer, sheet_name='策略6年度统计')

            # 表2：各评价指标，行名为指标，列名为策略名
            self.metrics_df.to_excel(writer, sheet_name='策略绩效指标')

    def calculate_annualized_return(self, strategy_returns):
        """
        计算策略的年化收益率。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。

        Returns:
            float: 年化收益率。
        """
        # 计算累计收益
        cumulative_return = (1 + strategy_returns).prod()
        # 计算总月数
        n_months = strategy_returns.count()
        # 计算年化收益率
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
        计算策略的年化波动率。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。

        Returns:
            float: 年化波动率。
        """
        # 计算月度波动率并年化
        annualized_volatility = strategy_returns.std() * np.sqrt(12)
        return annualized_volatility

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0):
        """
        计算策略的夏普比率。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。
            risk_free_rate (float): 无风险利率，默认为0。

        Returns:
            float: 夏普比率。
        """
        # 计算月度超额收益
        excess_returns = strategy_returns.mean() - risk_free_rate / 12
        # 计算夏普比率
        sharpe_ratio = (excess_returns / strategy_returns.std()) * np.sqrt(12)
        return sharpe_ratio

    def calculate_sortino_ratio(self, strategy_returns, target=0):
        """
        计算策略的索提诺比率。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。
            target (float): 目标回报，默认为0。

        Returns:
            float: 索提诺比率。
        """
        # 计算下行风险
        downside_returns = strategy_returns[strategy_returns < target]
        downside_deviation = downside_returns.std() * np.sqrt(12)
        # 计算目标收益与实际平均收益的差
        expected_return = 12 * (strategy_returns.mean() - target)
        # 防止除零错误
        if downside_deviation == 0:
            return np.nan
        # 计算索提诺比率
        sortino_ratio = expected_return / downside_deviation
        return sortino_ratio

    def calculate_max_drawdown(self, cumulative_returns):
        """
        计算策略的最大回撤。

        Parameters:
            cumulative_returns (pd.Series): 策略累计净值。

        Returns:
            float: 最大回撤。
        """
        # 计算滚动最大值
        rolling_max = cumulative_returns.cummax()
        # 计算回撤
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        # 返回最大回撤
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_win_rate(self, strategy_returns):
        """
        计算基于信号的策略胜率。
        仅考虑策略发出信号的月份（strategy_returns != 0）。
        当策略发出买入信号（strategy_returns > 0）时，回报为正为胜。
        当策略发出卖出信号（strategy_returns < 0）时，回报为负为胜。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。

        Returns:
            float: 胜率。
        """
        # 排除策略回报为0的月份（无信号月份）
        active_returns = strategy_returns[strategy_returns != 0]

        # 如果没有活跃的交易，返回NaN
        if active_returns.empty:
            return np.nan

        # 胜利条件：策略回报 > 0
        buy_wins = active_returns > 0

        # 计算胜率：胜利次数 / 总信号次数
        total_wins = buy_wins.sum()
        total_signals = active_returns.count()
        win_rate = total_wins / total_signals

        return win_rate

    def calculate_odds_ratio(self, strategy_returns):
        """
        计算基于信号的策略赔率（平均盈利 / 平均亏损）。
        仅考虑策略发出信号的月份（strategy_returns != 0）。

        Parameters:
            strategy_returns (pd.Series): 每月策略回报。

        Returns:
            float: 赔率。
        """
        # 排除策略回报为0的月份（无信号月份）
        active_returns = strategy_returns[strategy_returns != 0]

        # 盈利交易：策略回报 > 0
        wins = active_returns[active_returns > 0]

        # 亏损交易：策略回报 < 0
        losses = active_returns[active_returns < 0]

        # 如果没有亏损交易，返回NaN
        if losses.empty:
            return np.nan

        # 计算平均盈利和平均亏损
        avg_win = wins.mean()
        avg_loss = losses.mean()

        # 赔率 = 平均盈利 / 平均亏损（取绝对值）
        odds_ratio = avg_win / abs(avg_loss)

        return odds_ratio

    def calculate_kelly_fraction(self, win_rate, odds_ratio):
        """
        计算策略的凯利仓位。

        Parameters:
            win_rate (float): 胜率。
            odds_ratio (float): 赔率。

        Returns:
            float: 凯利仓位。
        """
        # 检查赔率是否有效
        if np.isnan(odds_ratio) or odds_ratio == 0:
            return 0
        # 计算凯利分数
        kelly_fraction = (win_rate * (odds_ratio + 1) - 1) / odds_ratio
        # 确保凯利仓位非负
        return max(kelly_fraction, 0)

    def calculate_trade_counts(self, signals):
        """
        计算每年的多单和空单交易次数。

        Parameters:
            signals (pd.Series): 策略的买卖信号序列。

        Returns:
            pd.DataFrame: 包含每年多单和空单交易次数的DataFrame。
        """
        # 计算信号的变化
        signal_changes = signals.diff()

        # 多单交易：信号从非持仓或空仓变为持仓（信号变为1）
        long_trades = signal_changes == 1
        # 空单交易：信号从非持仓或多仓变为空仓（信号变为-1）
        short_trades = signal_changes == -1

        # 按年度重采样并统计每年的交易次数
        annual_long = long_trades.resample('Y').sum()
        annual_short = short_trades.resample('Y').sum()

        # 创建结果DataFrame
        trade_counts = pd.DataFrame({
            'Annual_Long_Trades': annual_long,
            'Annual_Short_Trades': annual_short
        })

        return trade_counts




def main():
    # 指定Excel文件路径
    file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\招商择时快速复现.xlsx"
    output_file = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\策略回测结果1.xlsx"

    # 实例化 StrategyBacktester 类
    backtester = StrategyBacktester(file_path)

    # 生成所有策略的买卖信号
    backtester.generate_signals_for_all_strategies()

    # 进行所有策略的回测
    backtester.backtest_all_strategies()

    # 计算所有策略的绩效指标
    backtester.calculate_metrics_all_strategies()

    # 计算策略6的年度统计
    backtester.calculate_annual_metrics_strategy6()

    # 生成并保存Excel报告
    backtester.generate_excel_reports(output_file)

    print(f'回测完成，结果已保存到 {output_file}')

# 5. Execute the main function
if __name__ == "__main__":
    main()