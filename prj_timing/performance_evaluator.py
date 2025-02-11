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
    def __init__(self, data_handler: DataHandler, signals_dict: dict, signals_columns):
        """
        参数:
            data_handler: 依赖于DataHandler对象，用于获取daily_indices_data和monthly_indices_data。
            df_signals: 来自 generate_signals_for_all_strategies，结构形如：
                 {"M": merged_monthly, "D": merged_daily}。
            signals_columns (list): 明示哪些列是策略信号。若为 None，则默认df_signals的全部列。
        """
        self.data_handler = data_handler
        # 分别保存月频与日频的合并信号（均为字典：键为指数名称，值为对应的 DataFrame）
        self.df_signals_monthly = signals_dict.get("M", {})
        self.df_signals_daily = signals_dict.get("D", {})

        # 构造所有信号列和频率映射
        self.signals_columns = signals_columns
        self.is_monthly_signal = {}
        # 遍历月频信号，列名中统一附加 "_monthly_signal"
        for index_name, df in self.df_signals_monthly.items():
            for col in df.columns:
                self.is_monthly_signal[col] = True
        # 遍历日频信号，列名中统一附加 "_daily_signal"
        for index_name, df in self.df_signals_daily.items():
            for col in df.columns:
                self.is_monthly_signal[col] = False

        # 其它初始化设置保持不变
        # 默认年化因子针对日频策略，月频策略在计算绩效指标时使用 12
        self.annual_factor_default = 252
        self.time_delta = 'Y'
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
        分别对月频和日频信号进行回测。
        """
        # 处理月频信号回测
        for index_name, df_signals in self.df_signals_monthly.items():
            for signal_col in self.signals_columns:
                print(f"\n开始回测月频策略: {signal_col} (指数: {index_name})...")
                self.prepare_data(index_name)  # 从 daily_data_dict 中获取日度数据（回测时需要日频数据作图）
                # 直接从月频信号 DataFrame 中提取对应信号
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date, signal_series=current_signal_series)
                base_name = signal_col.replace("_monthly_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"策略 {base_name} 最终净值: {final_strategy:.2f}")
                # self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

        # 处理日频信号回测
        for index_name, df_signals in self.df_signals_daily.items():
            for signal_col in self.signals_columns:
                print(f"\n开始回测日频策略: {signal_col} (指数: {index_name})...")
                self.prepare_data(index_name)
                current_signal_series = df_signals[signal_col].dropna().copy()
                result = self.backtest_single_strategy(index_name, signal_col, start_date, signal_series=current_signal_series)
                base_name = signal_col.replace("_daily_signal", "")
                self.strategies_results[base_name] = result
                final_strategy = result['Daily_Cumulative_Strategy'].iloc[-1]
                print(f"策略 {base_name} 最终净值: {final_strategy:.2f}")
                self.plot_results(result['Daily_Cumulative_Strategy'], result['Daily_Cumulative_Index'], base_name)

    def backtest_single_strategy(self, index_name, signal_col, start_date='2001-12', signal_series=None):
        """
        增加 signal_series 参数，直接使用传入的信号数据（来自 df_signals_monthly 或 df_signals_daily）。
        对月频策略：
            1. 在月度数据上计算月频净值及收益；
            2. 调用 convert_monthly_signals_to_daily_positions 将月频信号映射为日频仓位，再计算日频净值。
        对日频策略：
            直接在日度数据上计算策略净值。
        修改点：增加返回结果中的 'Position' 字段，用于保存每日仓位信息（增量信息）。
        """
        df_daily = self.daily_data_dict[index_name].copy()
        df_monthly = self.monthly_data_dict[index_name].copy()

        df_daily = df_daily[df_daily.index >= pd.to_datetime(start_date)].copy()
        df_monthly = df_monthly[df_monthly.index >= pd.to_datetime(start_date)].copy()

        if self.is_monthly_signal.get(signal_col, False):
            # 月频策略处理
            monthly_net_value, monthly_strategy_returns = self._backtest_monthly(df_monthly, signal_series)
            daily_positions = self.convert_monthly_signals_to_daily_positions(df_daily, signal_series)
            # 注意：仓位设置后要向后移一日，保证信号滞后生效
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
                'Position': df_daily['Position']  # 保存增量信息（仓位）
            }
        else:
            # 日频策略处理：直接使用日频信号进行回测
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
                'Position': df_daily['Position']  # 保存增量信息（仓位）
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
                                                   next_day_open=True):
        """
        将月度信号映射为日度仓位。示例写法：
          next_day_open=True表示在目标生效日的下一交易日开盘建仓。
        """
        daily_positions = pd.Series(0, index=df_daily.index, dtype=float)

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

    def load_kelly_fractions(self):
        """
        从 self.metrics_df 中读取各策略 Kelly 仓位信息
        返回一个字典，键为策略标识，值为对应的 Kelly 仓位。
        """
        if self.metrics_df is None:
            return {}
        kelly_dict = self.metrics_df['Kelly仓位'].to_dict()
        return kelly_dict

    def compose_strategies_by_kelly(self, method='sum_to_one'):
        """
        根据各策略的 Kelly 仓位以及当日是否持仓（通过 Daily_Strategy_Return 判断）
        组合成一个总策略：
          - 当日若该策略的 Daily_Strategy_Return 为 0，则认为无持仓，其 Kelly 仓位不计入当日加权。
          - 对于有持仓的策略，其贡献收益为 Daily_Strategy_Return 乘以 Kelly 仓位，
            当日有效 Kelly 仓位为 Kelly 仓位（即常量）乘以持仓指示器（非零 -> 1，否则 0）。
          - 日综合收益为所有策略贡献收益总和除以当日所有活跃策略 Kelly 仓位之和，
            若当日无活跃策略，则综合收益记为 0。

        特别说明：
          对于策略 '上证指数_tech_sell' 和 '上证指数_composite_basic_tech'（做空策略），
          其有效仓位将取负值，从而在总仓位中“减去”它们的仓位（总仓位可以为负，代表做空）。

        参数:
            method:
                'sum_to_one'：使用当日所有活跃策略的 Kelly 仓位求和归一化计算综合收益；
                'avg_to_one'：扩展方案（目前示例中与 sum_to_one 用法一致）。

        返回:
            pd.DataFrame: 包含综合策略日收益、累计净值、各策略当日贡献收益及其有效 Kelly 仓位，
                          以及每日综合仓位（Composite_Position）。
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

        # 遍历每个策略，判断当日是否持仓（通过 Daily_Strategy_Return 是否为 0 判断）
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