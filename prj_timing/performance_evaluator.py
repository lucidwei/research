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
    PositionManager 负责将不同频率（如日频、月频）的信号数据转换为每日实际仓位。

    主要职责：
      - 针对日频信号，保证信号与每日交易数据对齐，且根据参数决定是否延后一天生效；
      - 针对月频信号，将每个月信号映射到对应的日交易区间，确保信号在该区间内生效。

    注意：
      - 此模块仅专注于仓位转换，不涉及业绩指标计算。
    """

    def __init__(self, next_day_open=True):
        """
        初始化 PositionManager。

        参数：
            next_day_open (bool): 若为 True，则信号生效日期延迟至下一交易日；否则在信号当日生效。
        """
        self.next_day_open = next_day_open

    def convert_signal_to_daily_position(self, signal_series, df_daily, frequency):
        """
        将给定信号转换成每日实际仓位。

        参数：
            signal_series (pd.Series): 信号序列。若为日频，索引应为交易日期；若为月频，索引通常为月份末日。
            df_daily (pd.DataFrame): 日交易数据 DataFrame，其索引为交易日期。
            frequency (str): 信号频率，目前支持 "daily" 或 "monthly"。

        返回：
            pd.Series: 对应 df_daily 索引的每日仓位序列。

        异常：
            ValueError: 若 frequency 非 "daily" 或 "monthly"。
        """
        if frequency == "daily":
            return self._convert_daily_signal(signal_series, df_daily)
        elif frequency == "monthly":
            return self._convert_monthly_signal(signal_series, df_daily)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def _convert_daily_signal(self, signal_series, df_daily):
        """
        将日频信号转换为每日仓位。
        如果 next_day_open 为 True，则信号延后一天生效。
        """
        if self.next_day_open:
            aligned = signal_series.reindex(df_daily.index)
            daily_positions = aligned.shift(1).fillna(0)
        else:
            daily_positions = signal_series.reindex(df_daily.index).fillna(0)
        return daily_positions

    def _convert_monthly_signal(self, signal_series, df_daily):
        """
        将月频信号转换为每日仓位。
        对于每个月的信号，根据 next_day_open 参数确定生效区间，再赋值信号值到该区间内。
        """
        daily_positions = pd.Series(0, index=df_daily.index, dtype=float)
        sorted_signals = signal_series.sort_index()

        for current_date, signal_value in sorted_signals.items():
            if signal_value == 0:
                continue

            # 确定起始生效日期
            if self.next_day_open:
                subsequent = df_daily.index[df_daily.index > current_date]
                if len(subsequent) == 0:
                    continue
                start_date = subsequent[0]
            else:
                if current_date in df_daily.index:
                    start_date = current_date
                else:
                    start_date = df_daily.index[df_daily.index >= current_date][0]

            # 确定结束日期：下一个信号生效前一天；若当前信号最后一个，则延续到最后一个交易日
            next_signals = sorted_signals.loc[sorted_signals.index > current_date]
            if len(next_signals) > 0:
                next_date = next_signals.index[0]
                if self.next_day_open:
                    subsequent_next = df_daily.index[df_daily.index > next_date]
                    if len(subsequent_next) > 0:
                        end_date = subsequent_next[0] - pd.Timedelta(days=1)
                    else:
                        end_date = df_daily.index[-1]
                else:
                    if next_date in df_daily.index:
                        end_date = next_date - pd.Timedelta(days=1)
                    else:
                        end_date = df_daily.index[df_daily.index >= next_date][0] - pd.Timedelta(days=1)
            else:
                end_date = df_daily.index[-1]

            mask = (df_daily.index >= start_date) & (df_daily.index <= end_date)
            daily_positions.loc[mask] = signal_value
        return daily_positions

class PerformanceEvaluator:
    """
    PerformanceEvaluator 根据传入信号对各策略进行回测和绩效评估，
    并利用 PositionManager 将不同频率信号转换为每日实际仓位。

    重点：
      - 内部包含基于各策略凯利公式计算仓位的功能，用于合成综合策略；
      - 提供 compose_strategies_by_kelly、get_composite_positions、calculate_composite_metrics 三个接口，
        分别用于生成综合策略、提取综合策略每日仓位、计算综合策略业绩指标。
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

        self.position_manager = PositionManager(next_day_open=True)
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
            daily_positions = self.position_manager.convert_signal_to_daily_position(signal_series, df_daily, "monthly")
            df_daily['Position'] = daily_positions
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
            # 日频策略
            daily_positions = self.position_manager.convert_signal_to_daily_position(signal_series, df_daily, "daily")
            df_daily['Position'] = daily_positions
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
            if strategy_id not in kelly_dict:
                print(f"策略 {strategy_id} 不在 Kelly 数据中，跳过。")
                continue
            fraction = kelly_dict[strategy_id]
            # 获取该策略的日收益序列
            daily_return = results['Daily_Strategy_Return'].reindex(base_index).fillna(0)
            effective_fraction = fraction * results['Position'].reindex(base_index).fillna(0)

            active_kelly_sum += effective_fraction
            composite_raw += daily_return * effective_fraction

            # 保存各策略的贡献收益和每日有效仓位到结果 DataFrame 中
            combined_df[f'{strategy_id}_ret'] = daily_return * effective_fraction
            combined_df[f'{strategy_id}_position'] = effective_fraction

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

        print("综合策略业绩指标（基于 Kelly 加权计算）：")
        composite_indicators = self.calculate_indicators(composite_daily_return, composite_cum, annual_factor=252)
        for key, value in composite_indicators.items():
            print(f"{key}: {value}")

        self.composite_df = combined_df.copy()
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

    def calculate_indicators(self, returns_series, cumulative_series, annual_factor):
        """
        根据传入的收益序列和累计净值序列计算常用绩效指标。

        参数：
          returns_series (pd.Series): 策略日收益序列；
          cumulative_series (pd.Series): 策略累计净值序列；
          annual_factor (int): 年化因子（例如日频用 252，月频用 12）。

        返回：
          dict: 包含以下指标：
                - 年化收益率
                - 年化波动率
                - 夏普比率
                - 索提诺比率
                - 最大回撤
                - 胜率
                - 赔率
        """
        annual_return = self.calculate_annualized_return(returns_series, annual_factor=annual_factor)
        annual_vol = self.calculate_annualized_volatility(returns_series, annual_factor=annual_factor)
        sharpe = self.calculate_sharpe_ratio(returns_series, risk_free_rate=0, annual_factor=annual_factor)
        sortino = self.calculate_sortino_ratio(returns_series, target=0, annual_factor=annual_factor)
        max_dd = self.calculate_max_drawdown(cumulative_series)
        win_rate = self.calculate_win_rate(returns_series)
        odds = self.calculate_odds_ratio(returns_series)
        return {
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe,
            '索提诺比率': sortino,
            '最大回撤': max_dd,
            '胜率': win_rate,
            '赔率': odds
        }

    def calculate_metrics_all_strategies(self):
        """
        对所有策略计算绩效指标。对每个策略调用 calculate_indicators 得到主要指标，
        同时计算 Kelly 仓位和年均信号次数，最后整合成 DataFrame 保存在 self.metrics_df 中。
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
            if results.get('is_monthly', False):
                ret_series = results['Monthly_Strategy_Return']
                cumulative = results['Monthly_Cumulative_Strategy']
                factor = 12
            else:
                ret_series = results['Daily_Strategy_Return']
                cumulative = results['Daily_Cumulative_Strategy']
                factor = 252

            inds = self.calculate_indicators(ret_series, cumulative, annual_factor=factor)
            # 通过凯利公式计算仓位（使用胜率和赔率）
            kelly_fraction = self.calculate_kelly_fraction(inds['胜率'], inds['赔率'])
            avg_signals = self.calculate_average_signal_count(ret_series)

            metrics['策略名称'].append(strategy_id)
            metrics['年化收益率'].append(inds['年化收益率'])
            metrics['年化波动率'].append(inds['年化波动率'])
            metrics['夏普比率'].append(inds['夏普比率'])
            metrics['最大回撤'].append(inds['最大回撤'])
            metrics['索提诺比率'].append(inds['索提诺比率'])
            metrics['胜率'].append(inds['胜率'])
            metrics['赔率'].append(inds['赔率'])
            metrics['Kelly仓位'].append(kelly_fraction)
            metrics['年均信号次数'].append(avg_signals)

        self.metrics_df = pd.DataFrame(metrics).set_index("策略名称")
        print("所有策略绩效指标：")
        print(self.metrics_df)

    def calculate_composite_metrics(self, composite_return, composite_cum):
        """
        对综合策略计算绩效指标。内部直接调用 calculate_indicators，
        参数 annual_factor 固定为 252（日频数据）。

        返回：
          dict: 包含综合策略各项指标。
        """
        return self.calculate_indicators(composite_return, composite_cum, annual_factor=252)

    def get_composite_positions(self):
        """
        返回综合策略每日仓位（各策略权重合成）。
        """
        if self.composite_df is not None:
            return self.composite_df['Composite_Position']
        else:
            print("尚未生成综合策略，请先调用 compose_strategies_by_kelly()。")
            return None

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