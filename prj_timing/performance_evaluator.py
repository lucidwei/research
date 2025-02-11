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
    PerformanceEvaluator用于对多种策略信号进行回测和绩效评估。
    支持区分日频、月频策略：
      1. 如果识别到是月频策略，先在月度层面计算净值及指标，以保证胜率、赔率、Kelly等统计的正确性；
      2. 将月度信号逐月映射为日度仓位，以便与其他策略在相同日度坐标系下进行净值曲线合并、作图。
    """
    def __init__(self, data_handler: DataHandler, df_signals: pd.DataFrame, signals_columns=None):
        """
        参数:
            data_handler: 依赖于DataHandler对象，用于获取daily_indices_data和monthly_indices_data。
            df_signals: 包含所有策略信号的DataFrame，行索引为日期（或月份），列名为各策略的 *_signal。
            signals_columns (list): 明示哪些列是策略信号。若为 None，则默认df_signals的全部列。
            frequency (str): 评估频率，'D' 或 'M'。决定最终统计指标（夏普比率、年化收益等）的年化因子。
        """
        self.data_handler = data_handler
        self.df_signals = df_signals.copy()

        if signals_columns is None:
            self.signals_columns = self.df_signals.columns.tolist()
        else:
            self.signals_columns = signals_columns

        # 识别月度/日度。只要列名包含 "_monthly" 就视为月度策略。
        self.is_monthly_signal = {}
        for col in self.signals_columns:
            if "_monthly" in col:
                self.is_monthly_signal[col] = True
            else:
                self.is_monthly_signal[col] = False

        # 存储各策略最终回测结果
        self.strategies_results = {}
        self.metrics_df = None
        self.stats_by_each_year = {}
        self.detailed_data = {}

        # 从 data_handler 获取dict形式的日度/月度数据
        self.daily_data_dict = self.data_handler.daily_indices_data  # {index_name: df_daily}
        self.monthly_data_dict = self.data_handler.monthly_indices_data  # {index_name: df_monthly}

        # 准备一个属性来存放当前指数的日度数据
        self.index_df_daily = None

    def prepare_data(self, index_name):
        """
        从 daily_data_dict 取日度数据，并存到 self.index_df_daily。
        回测时会在这个DataFrame里加入 'Position', 'Strategy_Return' 等列。
        """
        if index_name not in self.daily_data_dict:
            raise ValueError(f"在 data_handler 中未找到日度数据: {index_name}")
        self.index_df_daily = self.daily_data_dict[index_name].copy()

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        对 self.signals_columns 中所有策略进行回测。
        如果某策略名含有上证指数_...，则识别 index_name=上证指数，准备对应的日度数据。
        只做一个简单示例，您可自行扩展更多情况。
        """
        for signal_col in self.signals_columns:
            # 举例：signal_col = "上证指数_macro_loan_monthly_signal"
            # 先去掉"_signal"
            base_name = signal_col.replace("_signal", "")
            # 拿到第一个下划线前的部分，视为index_name
            try:
                index_name = base_name.split("_", 1)[0]
            except:
                index_name = base_name

            # 根据 index_name 准备日度数据
            self.prepare_data(index_name)

            print(f"\n开始回测策略 {base_name} (指数: {index_name})...")

            # 回测单个策略，返回一个字典，包含每日和（如果月频）月度数据
            result = self.backtest_single_strategy(
                index_name=index_name,
                signal_col=signal_col,
                start_date=start_date
            )

            # 存储，并记录该策略是否为月频
            self.strategies_results[base_name] = result

            # 打印最终净值（使用日度净值用于显示）
            final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
            final_index_val = result['Daily_Cumulative_Index'].iloc[-1]
            print(f"策略 {base_name} 最终净值: {final_strategy:.2f}")
            print(f"指数 {index_name} 最终净值: {final_index_val:.2f}")

            # 画图（示例，使用日度数据绘图）
            self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

    def backtest_single_strategy(self, index_name, signal_col, start_date='2001-12'):
        """
        回测单只策略。
        如果是月频策略:
          - 先用月度数据与月度信号，计算月度层面的净值(只为正确计算胜率、赔率、Kelly等)。
          - 再把月度信号映射到日度仓位，做日度净值计算，用于合并和画图。
          返回一个字典，包含：
              'Daily_Strategy_Return', 'Daily_Cumulative_Strategy', 'Daily_Cumulative_Index'
          对于月频策略，还额外包含:
              'Monthly_Strategy_Return', 'Monthly_Cumulative_Strategy'
          以及键 'is_monthly' 指示策略类型。
        如果是日频策略:
          - 直接在日度数据上回测。
        """
        # 日度和月度数据
        df_daily = self.daily_data_dict[index_name].copy()
        df_monthly = self.monthly_data_dict[index_name].copy()

        # 截取起始日期
        df_daily = df_daily[df_daily.index >= pd.to_datetime(start_date)].copy()
        df_monthly = df_monthly[df_monthly.index >= pd.to_datetime(start_date)].copy()

        # 从信号数据中取出对应信号
        current_signal_series = self.df_signals[signal_col].dropna().copy()

        if self.is_monthly_signal[signal_col]:
            # 1) 月度层面回测 -> monthly_net_value, monthly_strategy_returns
            monthly_net_value, monthly_strategy_returns = self._backtest_monthly(
                df_monthly, current_signal_series
            )
            # 2) 将月度信号映射到日度
            daily_positions = self.convert_monthly_signals_to_daily_positions(
                df_daily, current_signal_series
            )
            # 3) 在日度数据上计算策略净值
            df_daily['Position'] = daily_positions.shift(1).fillna(0)
            df_daily['Index_Return'] = df_daily['指数:最后一条'].pct_change()
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
                'is_monthly': True
            }
        else:
            # 日度策略
            df_daily['Position'] = current_signal_series.shift(1).reindex(df_daily.index).fillna(0)
            df_daily['Index_Return'] = df_daily['指数:最后一条'].pct_change()
            df_daily['Strategy_Return'] = df_daily['Position'] * df_daily['Index_Return']
            df_daily['Strategy_Return'].fillna(0, inplace=True)
            df_daily['Cumulative_Strategy'] = (1 + df_daily['Strategy_Return']).cumprod()
            df_daily['Cumulative_Index'] = (1 + df_daily['Index_Return']).cumprod()

            return {
                'Daily_Strategy_Return': df_daily['Strategy_Return'],
                'Daily_Cumulative_Strategy': df_daily['Cumulative_Strategy'],
                'Daily_Cumulative_Index': df_daily['Cumulative_Index'],
                'is_monthly': False
            }

    def _backtest_monthly(self, df_monthly, monthly_signal_series):
        """
        仅在月度层面进行回测，计算出月度层面的策略净值和策略收益。
        用于统计胜率、赔率、Kelly等; 不做日度持仓。
        """
        df_m_temp = df_monthly.copy()
        df_m_temp['Signal'] = monthly_signal_series.reindex(df_m_temp.index).fillna(0)
        df_m_temp['MonthlyReturn'] = df_m_temp['指数:最后一条'].pct_change()

        # 策略收益 = Signal.shift(1) * MonthlyReturn
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
        plt.plot(cumulative_index, label='基准指数')
        plt.plot(cumulative_strategy, label=f'{strategy_id} 净值')
        plt.title(f'{strategy_id} 回测结果')
        plt.xlabel('时间')
        plt.ylabel('累计收益')
        plt.legend()
        plt.grid(True)
        plt.show()

    def convert_monthly_signals_to_daily_positions(self, df_daily, monthly_signal_series,
                                                   shift=True, next_day_open=True):
        """
        将月度信号映射为日度仓位。示例写法：
          shift=True表示当月信号需要到下个月才生效。
          next_day_open=True表示在目标生效日的下一交易日开盘建仓。
        这里仅给一个简单示例逻辑，仅做演示。
        """
        daily_positions = pd.Series(0, index=df_daily.index, dtype=float)
        # 如果shift=True, 则对月度信号做一次shift(1)
        if shift:
            monthly_signal_series = monthly_signal_series.shift(1).fillna(0)

        # 遍历月度信号，根据 next_day_open 决定在下个月首日还是当月末日建仓
        monthly_signal_series = monthly_signal_series.dropna()
        # 月度数据的index是每个月最后一天(或您csv如何定)，我们需要从当月/下月的日度区间筛选

        for month_date, sig_val in monthly_signal_series.items():
            if sig_val == 0:
                continue

            # 月末 (month_date), 下个月第一天
            from pandas.tseries.offsets import MonthBegin
            month_first_day = (month_date + MonthBegin(0)).replace(day=1)
            next_month_first_day = (month_date + MonthBegin(1))

            if next_day_open:
                # 下一个交易日(如下月首日)才建仓
                start_date_for_position = next_month_first_day
            else:
                # 当月末就建仓
                start_date_for_position = month_date

            # 该signal通常有效到下个月末。
            end_date_for_position = (next_month_first_day + MonthBegin(1)) - pd.Timedelta(days=1)

            # 赋值给日度区间
            mask = (df_daily.index >= start_date_for_position) & (df_daily.index <= end_date_for_position)
            daily_positions.loc[mask] = sig_val

        return daily_positions

    def calculate_metrics_all_strategies(self):
        """
        计算所有策略的绩效指标（基于策略收益）。
        对于月度策略：使用月度层面的收益和净值，保证胜率、赔率、Kelly等指标的正确性；
        对于日度策略：使用日度收益计算指标。
        """
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

        for strategy_id, results in self.strategies_results.items():
            print(f'正在计算策略 {strategy_id} 的绩效指标...')
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
            metrics['策略名称'].append(display_name)
            metrics['年化收益率'].append(annualized_return)
            metrics['年化波动率'].append(annualized_volatility)
            metrics['夏普比率'].append(sharpe_ratio)
            metrics['最大回撤'].append(max_drawdown)
            metrics['索提诺比率'].append(sortino_ratio)
            metrics['胜率'].append(win_rate)
            metrics['赔率'].append(odds_ratio)
            metrics['Kelly仓位'].append(kelly_fraction)
            metrics['年均信号次数'].append(average_signals)

        self.metrics_df = pd.DataFrame(metrics)
        self.metrics_df.set_index('策略名称', inplace=True)

    def calculate_annual_metrics_for(self, strategy_names):
        """
        Calculates annual metrics for specific strategies and saves detailed data.
        Parameters:
            strategy_names (list): List of strategy names (e.g., ['strategy6', 'strategy7']).
        """
        if not isinstance(strategy_names, list):
            raise TypeError("strategy_names 应该是一个列表。")

        for strategy_name in strategy_names:
            if strategy_name not in self.strategies_results:
                raise ValueError(f"策略名称 '{strategy_name}' 不存在于回测结果中。")

            # 注意：此处需要保证 self.index_df_with_signal 已经适当构建
            strategy_returns = self.strategies_results[strategy_name].get('Daily_Strategy_Return')
            index_returns = self.index_df_with_signal['Index_Return']

            annual_strategy_returns = (1 + strategy_returns).resample('Y').prod() - 1
            annual_index_returns = (1 + index_returns).resample('Y').prod() - 1
            annual_excess_returns = annual_strategy_returns - annual_index_returns

            trade_counts = self.calculate_trade_counts(self.index_df_with_signal[f'{strategy_name}_signal'])

            self.stats_by_each_year[strategy_name] = pd.DataFrame({
                '策略年度收益': annual_strategy_returns,
                '指数年度收益': annual_index_returns,
                '超额收益': annual_excess_returns,
                '持有多单次数': trade_counts['Annual_Long_Trades'],
                '持有空单次数': trade_counts['Annual_Short_Trades']
            })
            self.stats_by_each_year[strategy_name].index = self.stats_by_each_year[strategy_name].index.year  # 将索引设置为年份

            # 提取用户指定的列
            signal_column = f'{strategy_name}_signal'
            if signal_column not in self.index_df_with_signal.columns:
                raise ValueError(f"信号列 '{signal_column}' 不存在于 index_df_with_signal 中。")

            detailed_df = self.index_df_with_signal[
                [signal_column, 'Position', 'Strategy_Return', 'Cumulative_Strategy', 'Cumulative_Index']].copy()
            detailed_df.rename(columns={
                signal_column: '本策略Signal'
            }, inplace=True)

            # 提取用户指定的列及增量记录
            signal_columns = [col for col in self.index_df_with_signal.columns if col.endswith('_signal')]
            for signal_column in signal_columns:
                # 提取第一个'_'之后的字符串作为列名
                new_col_name = signal_column.split('_', 1)[1]
                detailed_df[f'{new_col_name}'] = self.index_df_with_signal[signal_column]

            self.detailed_data[strategy_name] = detailed_df

    def generate_excel_reports(self, output_file, annual_metrics_strategy_names):
        """
        生成并保存年度统计和详细数据到同一个Excel文件的多个工作表中。
        Parameters:
            output_file (str): 输出Excel文件的路径。
            annual_metrics_strategy_names (list): List of strategy names to generate annual metrics for.
        """
        with pd.ExcelWriter(output_file) as writer:
            for strategy_name in annual_metrics_strategy_names:
                if strategy_name in self.stats_by_each_year:
                    self.stats_by_each_year[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_年度统计')
                else:
                    print(f"策略 {strategy_name} 的年度统计数据不存在，跳过。")

                if strategy_name in self.detailed_data:
                    self.detailed_data[strategy_name].to_excel(writer, sheet_name=f'{strategy_name}_详细数据')
                else:
                    print(f"策略 {strategy_name} 的详细数据不存在，跳过。")

            if self.metrics_df is not None:
                self.metrics_df.to_excel(writer, sheet_name='策略绩效指标')
            else:
                print("策略绩效指标数据不存在，跳过。")

            # # 如果存在月度回测结果，则保存到额外工作表
            # for strategy_name, result in self.strategies_results.items():
            #     if result.get('is_monthly', False):
            #         monthly_df = pd.DataFrame({
            #             'Monthly_Cumulative_Strategy': result['Monthly_Cumulative_Strategy'],
            #             'Monthly_Strategy_Return': result['Monthly_Strategy_Return']
            #         })
            #         monthly_df.index.name = 'Date'
            #         monthly_df.to_excel(writer, sheet_name=f'{strategy_name}_月度回测')

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
            annual_factor (int): Annualization factor. 若为 None，则使用默认值。
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