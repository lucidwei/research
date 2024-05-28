# coding=gbk
# Time Created: 2023/4/26 8:11
# Author  : Lucid
# FileName: asset_allocator.py
# Software: PyCharm
import time

from prj_risk_parity.db_interact import DatabaseReader
from prj_risk_parity.signal_generator import StockSignalGenerator, GoldSignalGenerator
import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping
from utils_risk_parity import align_signals
import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")


class AssetAllocator:
    def __init__(self, data: DatabaseReader, parameters: dict):
        self.data = data
        self.parameters = parameters
        self.strategic_allocator = StrategicAllocator(self.data, parameters['risk_budget'])
        self.tactical_allocator = TacticalAllocator(self.data)
        self.signals = self.tactical_allocator.signals
        self.calculate_daily_win_rate()
        self.generate_asset_positions()

    def calculate_daily_win_rate(self):
        def calculate_signal_win_rate(signal_data, returns_data, signal_col, return_col):
            """
            计算指定信号的胜率，排除 signal 为 0 的情况。

            :param signal_data: 包含信号的 DataFrame
            :param returns_data: 包含收益率的 DataFrame
            :param signal_col: 信号列的名称
            :param return_col: 收益率列的名称
            :return: 当日胜率、次日胜率、累计净值
            """
            # 确保索引为日期
            signal_data.index = pd.to_datetime(signal_data.index)
            returns_data.index = pd.to_datetime(returns_data.index)

            # 合并数据
            merged_data = pd.concat([signal_data, returns_data], axis=1, join='inner')

            # 筛选出索引在 self.data.tradedays 中的行
            filtered_index = pd.Index(merged_data.index.date).isin(self.data.tradedays)
            merged_data = merged_data[filtered_index]

            # 计算次日收益率
            merged_data['next_day_return'] = merged_data[return_col].shift(-1)

            # 过滤掉 signal 为 0 的数据
            filtered_data = merged_data[merged_data[signal_col] != 0]

            # 计算信号的正确性（当日）
            filtered_data.loc[:, 'correct'] = ((filtered_data[signal_col] > 0) & (filtered_data[return_col] > 0)) | \
                                       ((filtered_data[signal_col] < 0) & (filtered_data[return_col] < 0))

            # 计算信号当日胜率
            win_rate = filtered_data['correct'].mean()

            # 计算信号的正确性（次日）
            filtered_data['next_day_correct'] = ((filtered_data[signal_col] > 0) & (
                        filtered_data['next_day_return'] > 0)) | \
                                                ((filtered_data[signal_col] < 0) & (
                                                            filtered_data['next_day_return'] < 0))

            # 计算信号次日胜率
            next_day_win_rate = filtered_data['next_day_correct'].mean()

            # 计算累计净值
            filtered_data.loc[:, 'strategy_return'] = filtered_data[signal_col] * filtered_data['next_day_return']
            filtered_data.loc[:, 'cumulative_net_value'] = (1 + filtered_data['strategy_return']).cumprod()

            # 计算并打印 'correct' 列分别为 True 和 False 时 'next_day_correct' 为 True 的正确率
            correct_true_next_day_correct_rate = filtered_data[filtered_data['correct']]['next_day_correct'].mean()
            correct_false_next_day_correct_rate = filtered_data[~filtered_data['correct']]['next_day_correct'].mean()

            print(f"Next day correct rate when 'correct' is True: {correct_true_next_day_correct_rate}")
            print(f"Next day correct rate when 'correct' is False: {correct_false_next_day_correct_rate}")

            return win_rate, next_day_win_rate, filtered_data['cumulative_net_value']

        # 计算股票信号的胜率
        stock_win_rate, stock_next_day_win_rate, stock_cumulative_net_value = calculate_signal_win_rate(
            self.signals['combined'], self.strategic_allocator.returns_data,
            'combined_stock_signal', 'stock'
        )
        print(f"Stock Signal Win Rate: {stock_win_rate:.2%}")
        print(f"Stock Signal Next Day Win Rate: {stock_next_day_win_rate:.2%}")
        print(f"Stock Cumulative Net Value: {stock_cumulative_net_value}")

        # 计算黄金信号的胜率
        gold_win_rate, gold_next_day_win_rate, gold_cumulative_net_value = calculate_signal_win_rate(
            self.signals['combined'], self.strategic_allocator.returns_data,
            'combined_gold_signal', 'gold'
        )
        print(f"Gold Signal Win Rate: {gold_win_rate:.2%}")
        print(f"Gold Signal Next Day Win Rate: {gold_next_day_win_rate:.2%}")
        print(f"Gold Cumulative Net Value: {gold_cumulative_net_value}")

    def limit_stock_weight(self, weights):
        stock_weight_limit = 0.3
        if weights[0] > stock_weight_limit:
            diff = weights[0] - stock_weight_limit
            weights[0] = stock_weight_limit
            weights[1] += diff  # 将多余的股票仓位分配给债券
        return weights

    def generate_asset_positions(self):

        self.strategic_monthly_weights = self.strategic_allocator.calc_strategic_monthly_weight(self.parameters['start_date'], self.parameters['end_date'])

        combined_signal_report = self.tactical_allocator.signals
        combined_signal_report = combined_signal_report['combined'].resample('M').mean().round()

        def adjust_weights_by_signal(row, combined_signal_report):
            original_row = row.copy()
            signal_row = combined_signal_report.loc[row.name]
            if signal_row['combined_stock_signal'] == 1:
                row[0] *= 1.3
                print(
                    f"{row.name.strftime('%Y-%m-%d')}: Increased stock weight due to positive combined_stock_signal. Original weights: {original_row.to_string(index=False)}, New weights: {row.to_string(index=False)}")
            if signal_row['combined_gold_signal'] == 1:
                row[2] *= 1.3
                print(
                    f"{row.name.strftime('%Y-%m-%d')}: Increased gold weight due to positive combined_gold_signal. Original weights: {original_row.to_string(index=False)}, New weights: {row.to_string(index=False)}")
            if signal_row['combined_stock_signal'] == -1:
                row[0] *= 0.7
                print(
                    f"{row.name.strftime('%Y-%m-%d')}: Decreased stock weight due to negative combined_stock_signal. Original weights: {original_row.to_string(index=False)}, New weights: {row.to_string(index=False)}")
            if signal_row['combined_gold_signal'] == -1:
                row[2] *= 0.7
                print(
                    f"{row.name.strftime('%Y-%m-%d')}: Decreased gold weight due to negative combined_gold_signal. Original weights: {original_row.to_string(index=False)}, New weights: {row.to_string(index=False)}")
            # 标准化总仓位之和为1
            row /= row.sum()
            return row

        adjusted_monthly_weights = self.strategic_monthly_weights.apply(adjust_weights_by_signal, axis=1,
                                                                   combined_signal_report=combined_signal_report)

        self.adjusted_monthly_weights = adjusted_monthly_weights.apply(self.limit_stock_weight, axis=1,
                                                                       result_type='expand')


class StrategicAllocator:
    def __init__(self, data: DatabaseReader, stk_bond_gold_risk_budget: list):
        self.data = data
        self.risk_budget = stk_bond_gold_risk_budget
        self.get_returns_data()

    def get_returns_data(self):
        stock_prices = self.data.data_dict['csi']['close']
        bond_prices = self.data.data_dict['cba']['close']
        gold_prices = self.data.data_dict['gold']['close']

        stock_returns = stock_prices.pct_change().dropna()
        bond_returns = bond_prices.pct_change().dropna()
        gold_returns = gold_prices.pct_change().dropna()

        returns_data = pd.concat([stock_returns, bond_returns, gold_returns], axis=1)
        returns_data.columns = ['stock', 'bond', 'gold']
        self.returns_data = returns_data

    def calc_strategic_monthly_weight(self, start_date, end_date):
        returns_data = self.returns_data

        # 计算月度收益率
        returns_data.index = pd.to_datetime(returns_data.index)
        monthly_returns_data = returns_data.resample('M').apply(lambda x: (x + 1).prod() - 1)

        # 计算滚动协方差矩阵（使用 12 个月数据）
        rolling_cov_matrix = self.calculate_rolling_cov_matrix(monthly_returns_data, window=12)

        # 创建一个空 DataFrame 用于存储每月的资产权重
        monthly_asset_weights = pd.DataFrame(columns=['stock', 'bond', 'gold'])

        # 在给定的日期范围内迭代
        for date in pd.date_range(start_date, end_date, freq='M'):
            if date in rolling_cov_matrix.index.get_level_values(0):
                cov_matrix = rolling_cov_matrix.loc[date].unstack(level=-1)
                # 将协方差矩阵转换为 3x3 的形式
                cov_matrix_3x3 = cov_matrix.reset_index().pivot_table(index='level_0', columns='level_1', values=0)
                cov_matrix_3x3 = cov_matrix_3x3.reindex(index=['stock', 'bond', 'gold'],
                                                        columns=['stock', 'bond', 'gold'])
                print(f'Calculating strategic weight for {date}')
                asset_weights = self.risk_budget_allocation(cov_matrix_3x3, self.risk_budget)
                monthly_asset_weights.loc[date] = asset_weights
            else:
                # 如果日期不在协方差矩阵的索引中，保持权重不变
                try:
                    monthly_asset_weights.loc[date] = monthly_asset_weights.iloc[-1]
                except:
                    continue

        return monthly_asset_weights

    def risk_budget_allocation(self, cov, risk_budget):
        def risk_budget_objective(weights, cov, risk_budget):
            sigma = np.sqrt(weights @ cov @ weights)
            MRC = np.dot(cov, weights) / sigma
            TRC = weights * MRC
            risk_contribution_percent = TRC / np.sum(TRC)
            # 风险平价
            # delta_TRC = [sum((i - risk_contribution_percent) ** 2) for i in risk_contribution_percent]
            # delta_TRC = [sum((i - TRC) ** 2) for i in TRC] #原始代码
            # 风险预算
            delta_TRC = np.sum((risk_contribution_percent - risk_budget) ** 2)
            return delta_TRC

        def total_weight_constraint(x):
            return np.sum(x) - 1.0

        initial_weights = np.array([0.15, 0.75, 0.1])
        cons = ({'type': 'eq', 'fun': total_weight_constraint},)
        bounds = ((0.02, 0.4), (0.1, 1), (0.02, 0.4))
        options = {"maxiter": 1000, "ftol": 1e-10}
        minimizer_kwargs = {
            "method": "SLSQP",
            "args": (cov, risk_budget),
            "constraints": cons,
            "bounds": bounds,  # 添加边界参数
            "options": options,
        }
        # 求解出权重
        solution = basinhopping(risk_budget_objective, x0=initial_weights, minimizer_kwargs=minimizer_kwargs)
        weight = solution.x

        return weight

    def calculate_rolling_cov_matrix(self, returns_data, window):

        cov_matrix = returns_data.rolling(window=window).cov().dropna()

        return cov_matrix


class TacticalAllocator:
    def __init__(self, data: DatabaseReader):
        self.initialize_data(data)
        self.stock_signal_generator = StockSignalGenerator(self.stock_prices, self.stock_volume, self.pe_ttm,
                                                           self.bond_yields)
        self.gold_signal_generator = GoldSignalGenerator(self.tips_10y, self.vix, self.gold_prices)
        # self.adjust_date_index()
        self.generate_signal_report()

    def initialize_data(self, data):
        self.data = data
        self.stock_prices = self.data.data_dict['csi']['close']
        self.stock_volume = self.data.data_dict['csi']['volume']
        self.pe_ttm = self.data.data_dict['csi']['pe_ttm']
        self.bond_yields = self.data.data_dict['high_freq_view']['china_t_yield']
        self.tips_10y = self.data.data_dict['high_freq_view']['us_tips_10y']
        self.vix = self.data.data_dict['vix']
        self.gold_prices = self.data.data_dict['gold']

    def generate_signal_report(self):
        # 生成个别信号
        erp_signal = self.stock_signal_generator.erp_signal()
        ma_signal = self.stock_signal_generator.ma_signal()
        volume_signal = self.stock_signal_generator.volume_signal()
        volume_ma_signal = self.stock_signal_generator.volume_ma_signal()
        us_tips_signal = self.gold_signal_generator.us_tips_signal()
        vix_signal = self.gold_signal_generator.vix_signal()
        gold_momentum_signal = self.gold_signal_generator.gold_momentum_signal()

        # 调整日期
        aligned_signals = align_signals(erp_signal=erp_signal, us_tips_signal=us_tips_signal, volume_ma_signal=volume_ma_signal)
        erp_signal = aligned_signals['erp_signal']
        us_tips_signal = aligned_signals['us_tips_signal']
        volume_ma_signal = aligned_signals['volume_ma_signal']

        # 将个别信号整合到一个 DataFrame 中
        individual_signal_report = pd.concat([erp_signal, ma_signal, volume_signal, volume_ma_signal,
                                              us_tips_signal, vix_signal, gold_momentum_signal], axis=1)
        individual_signal_report.columns = ['erp_signal', 'ma_signal', 'volume_signal', 'volume_ma_signal',
                                            'us_tips_signal', 'vix_signal', 'gold_momentum_signal']
        individual_signal_report.index = pd.to_datetime(individual_signal_report.index)

        # 生成组合信号
        combined_stock_signal = self.stock_signal_generator.combined_stock_signal()
        combined_gold_signal = self.gold_signal_generator.combined_gold_signal()

        # 将组合信号整合到一个 DataFrame 中
        combined_signal_report = pd.concat([combined_stock_signal, combined_gold_signal], axis=1)
        combined_signal_report.columns = ['combined_stock_signal', 'combined_gold_signal']
        combined_signal_report.index = pd.to_datetime(combined_signal_report.index)

        self.signals = {'individual': individual_signal_report, 'combined': combined_signal_report}

    # def allocate_tactical_assets(self, start_date, end_date, frequency='M'):
    #
    #     if frequency == 'M':
    #         erp_signal = erp_signal.resample('M').last()
    #     elif frequency == 'W':
    #         erp_signal = erp_signal.resample('W').last()
    #     else:
    #         raise ValueError("Invalid frequency. Use 'M' for monthly or 'W' for weekly.")
    #
    #     tactical_allocation = erp_signal.loc[start_date:end_date]
    #
    #     return tactical_allocation

