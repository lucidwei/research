# coding=gbk
# Time Created: 2025/1/14 14:08
# Updated: 2025/02/12
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


class PositionManager:
    """
    PositionManager 负责将月频策略信号转换为日频仓位，并处理与仓位管理相关的操作。
    """

    @staticmethod
    def convert_monthly_signals_to_daily_positions(df_daily, monthly_signal_series, next_day_open=True):
        """
        根据月度信号将仓位映射为日度仓位。

        参数:
            df_daily (pd.DataFrame): 日度数据的 DataFrame，其索引为交易日期。
            monthly_signal_series (pd.Series): 月度信号，索引为月份末日（或对应日期），值为仓位信号。
            next_day_open (bool): 若为 True，则在下一个交易日（如下月的第一天）开盘时建仓；否则在信号当日建仓。

        返回:
            pd.Series: 与日度数据对应的仓位序列。
        """
        daily_positions = pd.Series(0, index=df_daily.index, dtype=float)
        monthly_signal_series = monthly_signal_series.dropna()
        from pandas.tseries.offsets import MonthBegin

        for month_date, sig_val in monthly_signal_series.items():
            if sig_val == 0:
                continue

            # 计算下个月首日
            month_first_day = (month_date + MonthBegin(0)).replace(day=1)
            next_month_first_day = (month_date + MonthBegin(1))

            # 根据 next_day_open 参数决定实际建仓日期
            start_date_for_position = next_month_first_day if next_day_open else month_date

            # 信号有效期：从开始日期到下个月末
            end_date_for_position = (next_month_first_day + MonthBegin(1)) - pd.Timedelta(days=1)
            mask = (df_daily.index >= start_date_for_position) & (df_daily.index <= end_date_for_position)
            daily_positions.loc[mask] = sig_val
        return daily_positions


class PerformanceEvaluator:
    """
    PerformanceEvaluator 用于对多种策略信号进行回测和绩效评估。
    支持区分日频与月频策略：
      1. 若为月频策略，会先在月度层面计算净值及指标，进而映射为日度仓位用于后续计算；
      2. 若为日频策略，则直接基于日度信号计算相应指标。
    """

    def __init__(self, data_handler: DataHandler, signals_dict: dict, signals_columns):
        """
        初始化 PerformanceEvaluator。

        参数:
            data_handler (DataHandler): 用于获取 daily_indices_data 和 monthly_indices_data 的数据处理对象。
            signals_dict (dict): 包含月频和日频信号的字典，示例结构: {"M": merged_monthly, "D": merged_daily}。
            signals_columns (list): 策略信号列的列表。如果为 None，则默认使用信号 DataFrame 的所有列。
        """
        self.data_handler = data_handler
        self.df_signals_monthly = signals_dict.get("M", {})
        self.df_signals_daily = signals_dict.get("D", {})

        self.signals_columns = signals_columns
        self.is_monthly_signal = {}
        for index_name, df in self.df_signals_monthly.items():
            for col in df.columns:
                self.is_monthly_signal[col] = True
        for index_name, df in self.df_signals_daily.items():
            for col in df.columns:
                self.is_monthly_signal[col] = False

        # 配置参数
        self.annual_factor_default = 252
        self.time_delta = 'Y'
        self.strategies_results = {}
        self.metrics_df = None
        self.stats_by_each_year = {}
        self.detailed_data = {}

        self.daily_data_dict = self.data_handler.daily_indices_data  # 日度数据字典
        self.monthly_data_dict = self.data_handler.monthly_indices_data  # 月度数据字典

        self.index_df_daily = None  # 当前指数的日度数据

    def prepare_data(self, index_name):
        """
        根据指数名称获取对应的日度数据，并保存到 self.index_df_daily。

        参数:
            index_name (str): 指数名称，对应数据字典中的键。

        异常:
            ValueError: 若在 data_handler 中未找到对应的日度数据。
        """
        if index_name not in self.daily_data_dict:
            raise ValueError(f"在 data_handler 中未找到日度数据: {index_name}")
        self.index_df_daily = self.daily_data_dict[index_name].copy()

    def backtest_all_strategies(self, start_date='2001-12'):
        """
        对所有月频和日频策略信号分别进行回测评估。

        参数:
            start_date (str): 回测开始日期，格式应为 'YYYY-MM' 或完整日期。
        """
        # 回测月频策略
        for index_name, df_signals in self.df_signals_monthly.items():
            for signal_col in self.signals_columns:
                print(f"\n开始回测月频策略: {signal_col} (指数: {index_name})...")
                self.prepare_data(index_name)
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date,
                                                       signal_series=current_signal_series)
                base_name = signal_col.replace("_monthly_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"策略 {base_name} 最终净值: {final_strategy:.2f}")
                # 可根据需要调用下行代码做图
                # self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

        # 回测日频策略
        for index_name, df_signals in self.df_signals_daily.items():
            for signal_col in self.signals_columns:
                print(f"\n开始回测日频策略: {signal_col} (指数: {index_name})...")
                self.prepare_data(index_name)
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date,
                                                       signal_series=current_signal_series)
                base_name = signal_col.replace("_daily_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"策略 {base_name} 最终净值: {final_strategy:.2f}")
                self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

    def backtest_single_strategy(self, index_name, signal_col, start_date='2001-12', signal_series=None):
        """
        对单一策略进行回测，支持月频和日频策略。

        参数:
            index_name (str): 指数名称，用于读取对应的数据。
            signal_col (str): 策略信号所在的列名称。
            start_date (str): 回测开始日期（格式 'YYYY-MM' 或完整日期）。
            signal_series (pd.Series): 策略信号序列，从月度或日度的信号 DataFrame 中提取。

        返回:
            dict: 包含日度和（若适用）月度的回测结果字典，键包括 'Daily_Strategy_Return'、'Daily_Cumulative_Strategy'、
                  'Daily_Cumulative_Index' 和 'Position'（日仓位信息）。
        """
        df_daily = self.daily_data_dict[index_name].copy()
        df_monthly = self.monthly_data_dict[index_name].copy()

        df_daily = df_daily[df_daily.index >= pd.to_datetime(start_date)].copy()
        df_monthly = df_monthly[df_monthly.index >= pd.to_datetime(start_date)].copy()

        if self.is_monthly_signal.get(signal_col, False):
            # 处理月频策略
            monthly_net_value, monthly_strategy_returns = self._backtest_monthly(df_monthly, signal_series)
            daily_positions = PositionManager.convert_monthly_signals_to_daily_positions(df_daily, signal_series)
            # 仓位滞后一天，确保信号延时生效
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
                'is_monthly': True,
                'Position': df_daily['Position']
            }
        else:
            # 处理日频策略，直接将信号映射为仓位
            df_daily['Position'] = signal_series.shift(1).reindex(df_daily.index).fillna(0)
            df_daily['Index_Return'] = df_daily['指数:最后一条'].pct_change()
            df_daily['Strategy_Return'] = df_daily['Position'] * df_daily['Index_Return']
            df_daily['Strategy_Return'].fillna(0, inplace=True)
            df_daily['Cumulative_Strategy'] = (1 + df_daily['Strategy_Return']).cumprod()
            df_daily['Cumulative_Index'] = (1 + df_daily['Index_Return']).cumprod()
            return {
                'Daily_Strategy_Return': df_daily['Strategy_Return'],
                'Daily_Cumulative_Strategy': df_daily['Cumulative_Strategy'],
                'Daily_Cumulative_Index': df_daily['Cumulative_Index'],
                'is_monthly': False,
                'Position': df_daily['Position']
            }

    def _backtest_monthly(self, df_monthly, monthly_signal_series):
        """
        在月度层面计算策略净值和月度收益，用于统计胜率、赔率、Kelly 等指标。

        参数:
            df_monthly (pd.DataFrame): 月度数据的 DataFrame。
            monthly_signal_series (pd.Series): 月度仓位信号序列。

        返回:
            tuple: (monthly_net_value, monthly_strategy_returns)
                   monthly_net_value (pd.Series): 月累计策略净值；
                   monthly_strategy_returns (pd.Series): 月策略收益。
        """
        df_m_temp = df_monthly.copy()
        df_m_temp['Signal'] = monthly_signal_series.reindex(df_m_temp.index).fillna(0)
        df_m_temp['MonthlyReturn'] = df_m_temp['指数:最后一条'].pct_change()
        df_m_temp['MonthlyStrategyReturn'] = df_m_temp['Signal'].shift(1).fillna(0) * df_m_temp['MonthlyReturn']
        df_m_temp['MonthlyStrategyReturn'].fillna(0, inplace=True)
        df_m_temp['MonthlyCumStrategy'] = (1 + df_m_temp['MonthlyStrategyReturn']).cumprod()
        monthly_net_value = df_m_temp['MonthlyCumStrategy']
        monthly_strategy_returns = df_m_temp['MonthlyStrategyReturn']
        return monthly_net_value, monthly_strategy_returns

    def plot_results(self, cumulative_strategy, cumulative_index, strategy_id):
        """
        绘制策略净值与基准指数的累计收益曲线对比图。

        参数:
            cumulative_strategy (pd.Series): 策略累计净值序列。
            cumulative_index (pd.Series): 指数累计收益序列。
            strategy_id (str): 策略标识（用于图例和标题）。
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

    def compose_strategies_by_kelly(self, method='sum_to_one'):
        """
        根据每个策略的 Kelly 仓位及每日持仓（由 Position 字段给出）组合成一个总策略。

        参数:
            method (str): 组合收益计算方法。
                          'sum_to_one'：使用当日所有活跃策略 Kelly 仓位之和归一化；
                          'avg_to_one'：扩展方案，目前实现与 'sum_to_one' 类似。

        返回:
            pd.DataFrame: 包含综合策略每日收益、累计净值、各策略贡献收益、有效 Kelly 仓位以及总仓位（Composite_Position）的数据表。
        """
        if not self.strategies_results:
            print("无策略回测结果。")
            return None

        # 以第一个策略的 Daily_Strategy_Return 索引作为基础日期序列
        base_index = list(self.strategies_results.values())[0]['Daily_Strategy_Return'].index
        combined_df = pd.DataFrame(index=base_index)
        kelly_dict = self.load_kelly_fractions()

        composite_raw = pd.Series(0.0, index=base_index)  # 用于累计各策略贡献的日收益
        active_kelly_sum = pd.Series(0.0, index=base_index)  # 当日活跃策略的 Kelly 仓位加总

        for strategy_id, results in self.strategies_results.items():
            # 根据策略名称去除前缀，例如 '上证指数_tech_sell' 得到 "tech_sell"
            strategy_id_ = "_".join(strategy_id.split("_")[1:])
            if strategy_id_ not in kelly_dict:
                print(f"策略 {strategy_id_} 不在 Kelly 数据中，跳过。")
                continue
            fraction = kelly_dict[strategy_id_]
            # 获取该策略的日收益序列
            daily_return = results['Daily_Strategy_Return'].reindex(base_index).fillna(0)
            effective_fraction = fraction * results['Position'].reindex(base_index).fillna(0)

            active_kelly_sum += effective_fraction
            composite_raw += daily_return * effective_fraction

            # 保存各策略的贡献收益和每日有效仓位到结果 DataFrame 中
            combined_df[f'{strategy_id_}_ret'] = daily_return * effective_fraction
            combined_df[f'{strategy_id_}_position'] = effective_fraction

        # 以第一个策略的基准指数累计净值作为对比基准
        first_result = list(self.strategies_results.values())[0]
        combined_df['Daily_Cumulative_Index'] = first_result['Daily_Cumulative_Index'].reindex(base_index).fillna(
            method='ffill')

        # 当日若无活跃策略（active_kelly_sum 为 0），则综合收益记为 0
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

        # 保存当日所有活跃策略 Kelly 仓位之和为综合仓位
        combined_df['Composite_Position'] = active_kelly_sum

        # 利用已有指标计算函数（假设这些函数已在类中实现）计算综合策略业绩指标，年化因子对日频数据取 252
        annual_return = self.calculate_annualized_return(composite_daily_return, annual_factor=252)
        annual_vol = self.calculate_annualized_volatility(composite_daily_return, annual_factor=252)
        sharpe = self.calculate_sharpe_ratio(composite_daily_return, risk_free_rate=0, annual_factor=252)
        sortino = self.calculate_sortino_ratio(composite_daily_return, target=0, annual_factor=252)
        max_dd = self.calculate_max_drawdown(composite_cum)
        win_rate = self.calculate_win_rate(composite_daily_return)
        odds = self.calculate_odds_ratio(composite_daily_return)

        metrics = {
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe,
            '索提诺比率': sortino,
            '最大回撤': max_dd,
            '胜率': win_rate,
            '赔率': odds,
            'Composite_Position_Today': active_kelly_sum.iloc[-1]
        }
        print("综合策略业绩指标:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return combined_df

    def load_kelly_fractions(self):
        """
        从指标数据中提取各策略的 Kelly 仓位信息。

        返回:
            dict: 键为策略标识，值为对应的 Kelly 仓位值；若指标数据不存在则返回空字典。
        """
        if self.metrics_df is None:
            return {}
        return self.metrics_df['Kelly仓位'].to_dict()

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

            display_name = strategy_id[strategy_id.find('_') + 1:] if '_' in strategy_id else strategy_id
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
        为指定策略计算年度绩效指标，并保存详细数据。

        参数:
            strategy_names (list): 策略名称列表，例如 ['strategy6', 'strategy7']。

        注意:
            需要确保 self.index_df_with_signal 已经被正确构建，否则可能导致错误。
        """
        if not isinstance(strategy_names, list):
            raise TypeError("strategy_names 应该是一个列表。")

        for strategy_name in strategy_names:
            if strategy_name not in self.strategies_results:
                raise ValueError(f"策略名称 '{strategy_name}' 不存在于回测结果中。")

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
            self.stats_by_each_year[strategy_name].index = self.stats_by_each_year[strategy_name].index.year

            signal_column = f'{strategy_name}_signal'
            if signal_column not in self.index_df_with_signal.columns:
                raise ValueError(f"信号列 '{signal_column}' 不存在于 index_df_with_signal 中。")

            detailed_df = self.index_df_with_signal[
                [signal_column, 'Position', 'Strategy_Return', 'Cumulative_Strategy', 'Cumulative_Index']].copy()
            detailed_df.rename(columns={signal_column: '本策略Signal'}, inplace=True)

            # 提取用户指定的列及增量记录
            signal_columns = [col for col in self.index_df_with_signal.columns if col.endswith('_signal')]
            for signal_column in signal_columns:
                # 提取第一个'_'之后的字符串作为列名
                new_col_name = signal_column.split('_', 1)[1]
                detailed_df[f'{new_col_name}'] = self.index_df_with_signal[signal_column]

            self.detailed_data[strategy_name] = detailed_df

    def generate_excel_reports(self, output_file, annual_metrics_strategy_names):
        """
        将年度统计数据和详细数据输出到同一个 Excel 文件的各个工作表中。

        参数:
            output_file (str): 输出 Excel 文件的完整路径。
            annual_metrics_strategy_names (list): 需要生成年度绩效指标的策略名称列表。
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

    def calculate_average_signal_count(self, strategy_returns):
        """
        计算每年平均的信号次数。

        参数:
            strategy_returns (pd.Series): 策略收益序列（非零代表有信号）。

        返回:
            float: 每年平均的信号次数。
        """
        signals = strategy_returns != 0
        annual_signals = signals.resample(self.time_delta).sum()
        average_signals = annual_signals.mean()
        return average_signals

    def calculate_annualized_return(self, strategy_returns, annual_factor=None):
        """
        计算策略的年化收益率。

        参数:
            strategy_returns (pd.Series): 策略收益序列。
            annual_factor (int): 年化因子。如果为 None，则使用默认值。

        返回:
            float: 年化收益率。
        """
        cumulative_return = (1 + strategy_returns).prod()
        n_periods = strategy_returns.count()
        if n_periods == 0:
            return np.nan
        annual_factor = annual_factor if annual_factor is not None else self.annual_factor_default
        annualized_return = cumulative_return ** (annual_factor / n_periods) - 1
        return annualized_return

    def calculate_annualized_volatility(self, strategy_returns, annual_factor=None):
        """
        计算策略收益的年化波动率。

        参数:
            strategy_returns (pd.Series): 策略收益序列。
            annual_factor (int): 年化因子。

        返回:
            float: 年化波动率。
        """
        annual_factor = annual_factor if annual_factor is not None else self.annual_factor_default
        return strategy_returns.std() * np.sqrt(annual_factor)

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0, annual_factor=None):
        """
        计算策略的夏普比率。

        参数:
            strategy_returns (pd.Series): 策略收益序列。
            risk_free_rate (float): 无风险收益率。
            annual_factor (int): 年化因子。

        返回:
            float: 夏普比率。
        """
        annual_factor = annual_factor if annual_factor is not None else self.annual_factor_default
        excess_returns = strategy_returns.mean() - (risk_free_rate / annual_factor)
        volatility = strategy_returns.std()
        if volatility == 0:
            return np.nan
        return (excess_returns / volatility) * np.sqrt(annual_factor)

    def calculate_sortino_ratio(self, strategy_returns, target=0, annual_factor=None):
        """
        计算策略的索提诺比率。

        参数:
            strategy_returns (pd.Series): 策略收益序列。
            target (float): 目标收益率。
            annual_factor (int): 年化因子。

        返回:
            float: 索提诺比率。
        """
        annual_factor = annual_factor if annual_factor is not None else self.annual_factor_default
        downside_returns = strategy_returns[strategy_returns < target]
        downside_deviation = downside_returns.std() * np.sqrt(annual_factor)
        expected_return = (strategy_returns.mean() - target) * annual_factor
        if downside_deviation == 0:
            return np.nan
        return expected_return / downside_deviation

    def calculate_max_drawdown(self, cumulative_returns):
        """
        计算策略累计净值的最大回撤。

        参数:
            cumulative_returns (pd.Series): 累计收益序列。

        返回:
            float: 最大回撤值。
        """
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    def calculate_win_rate(self, strategy_returns):
        """
        计算策略的胜率，即正收益信号占所有交易信号的比例。

        参数:
            strategy_returns (pd.Series): 策略收益序列。

        返回:
            float: 胜率。
        """
        active_returns = strategy_returns[strategy_returns != 0]
        if active_returns.empty:
            return np.nan
        total_wins = (active_returns > 0).sum()
        total_signals = active_returns.count()
        return total_wins / total_signals

    def calculate_odds_ratio(self, strategy_returns):
        """
        计算策略的赔率，即平均盈利与平均亏损之比。

        参数:
            strategy_returns (pd.Series): 策略收益序列。

        返回:
            float: 赔率；若亏损为空，则返回 NaN。
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
        计算策略的 Kelly 仓位分配比例。

        参数:
            win_rate (float): 胜率。
            odds_ratio (float): 赔率。

        返回:
            float: Kelly 分配比例。若赔率无效，则返回 0；否则返回计算值与 0 的较大者。
        """
        if np.isnan(odds_ratio) or odds_ratio == 0:
            return 0
        kelly_fraction = (win_rate * (odds_ratio + 1) - 1) / odds_ratio
        return max(kelly_fraction, 0)

    def calculate_trade_counts(self, signals):
        """
        计算每年多头和空头交易的次数。

        参数:
            signals (pd.Series): 信号序列，其中1代表多头信号，-1代表空头信号。

        返回:
            pd.DataFrame: 包含每年多空交易笔数的数据表，列名为 'Annual_Long_Trades' 和 'Annual_Short_Trades'。
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