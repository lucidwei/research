# coding=gbk
# Time Created: 2024/10/30 14:56
# Author  : Lucid
# FileName: strategy_momentum.py
# Software: PyCharm
# strategy_momentum.py

from strategy_pure_passive import BaseStrategy
from evaluator import Evaluator
import pandas as pd
import numpy as np
import os
import pickle


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def run_strategy(self, data_dict, start_date, end_date, parameters):
        price_data = data_dict['close_prices']
        # 策略特定参数
        asset_class_mapping = parameters.get('asset_class_mapping', {})
        self.asset_class_mapping = asset_class_mapping
        rebalance_frequency = parameters.get('rebalance_frequency', 'M')
        lookback_periods = parameters.get('lookback_periods', [252, 126, 63])  # 1 年、6 个月、3 个月
        top_n_assets = parameters.get('top_n_assets', 10)
        risk_budget = parameters.get('risk_budget')

        # 确定预算类型
        if risk_budget:
            budget_type = self.check_budget_type(risk_budget, asset_class_mapping)
        else:
            budget_type = None  # 使用风险平价

        all_assets = list(asset_class_mapping.keys())
        price_data = price_data.loc[:, all_assets]

        if not end_date:
            end_date = price_data.index.max()

        date_index = price_data.loc[start_date:end_date].index

        rebalance_dates = self.calc_rebalance_dates(start_date, end_date, date_index, rebalance_frequency)

        weights_history = pd.DataFrame(index=date_index, columns=all_assets)

        previous_weights = None

        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"\nStarting rebalancing calculations from {start_date} to {end_date}")
        for i, date in enumerate(rebalance_dates):
            # 检查当前日期是否是暂停日期
            if date == pd.Timestamp('2024-10-31 00:00:00'):
                print("Pausing on", date.strftime('%Y-%m-%d'))
                # 在这里可以进行调试或暂停操作，例如使用断点或打印调试信息
            print(f"Processing rebalance date {date.strftime('%Y-%m-%d')} ({i + 1}/{len(rebalance_dates)})")
            # 生成缓存文件名
            cache_key_elements = {
                'date': date.strftime('%Y%m%d'),
                'top_n_assets': top_n_assets,
                'lookbacks': '_'.join(map(str, lookback_periods)),
                'risk_budget': str(sorted(risk_budget.items())) if risk_budget else 'None',
                'budget_type': budget_type if budget_type else 'risk_parity'
            }
            cache_filename = self.generate_cache_filename(cache_key_elements)

            if os.path.exists(cache_filename):
                with open(cache_filename, 'rb') as f:
                    weights = pickle.load(f)
            else:
                selected_assets = self.select_assets(price_data, date, lookback_periods, top_n_assets)

                if not selected_assets:
                    print(f"  No assets selected for date {date.strftime('%Y-%m-%d')}. Using previous weights.")
                    if previous_weights is not None:
                        weights = previous_weights
                    else:
                        weights = pd.Series(1.0 / len(all_assets), index=all_assets)
                else:
                    weights_list = []
                    for lookback_period in lookback_periods:
                        print(f"  Calculating weights using lookback period: {lookback_period} trading days")
                        available_dates = price_data.index
                        if date in available_dates:
                            cov_end_date = date
                        else:
                            cov_end_date = available_dates[available_dates.get_loc(date, method='ffill')]
                        cov_end_idx = available_dates.get_loc(cov_end_date)
                        cov_start_idx = cov_end_idx - (lookback_period - 1)
                        if cov_start_idx < 0:
                            print(f"  Not enough data for lookback period of {lookback_period} trading days.")
                            continue
                        cov_start_date = available_dates[cov_start_idx]
                        cov_data = price_data.loc[cov_start_date:cov_end_date, selected_assets]

                        if cov_data.shape[0] < lookback_period * 0.9:
                            print(f"  Not enough data for lookback period of {lookback_period} trading days.")
                            continue

                        daily_returns = cov_data.pct_change().dropna()
                        cov_matrix = daily_returns.cov()

                        if risk_budget and budget_type:
                            if budget_type == 'class_budget':
                                print(f"  Performing class-level risk budget allocation")
                                cov_matrix_class = self.aggregate_covariance_by_class(cov_matrix, asset_class_mapping)
                                class_weights = self.risk_budget_allocation(cov_matrix_class, risk_budget)
                                weights_period = self.distribute_class_weights_to_assets(
                                    class_weights, selected_assets, asset_class_mapping
                                )
                            elif budget_type == 'asset_budget':
                                print(f"  Performing asset-level risk budget allocation")
                                weights_period = self.risk_budget_allocation(cov_matrix, risk_budget)
                            else:
                                raise ValueError("Invalid budget type detected.")
                        else:
                            # 使用风险平价
                            print(f"  Performing risk parity allocation")
                            weights_period = self.risk_parity_weights(cov_matrix)

                        weights_list.append(weights_period)

                    if len(weights_list) == 0:
                        if previous_weights is not None:
                            weights = previous_weights
                        else:
                            weights = pd.Series(1.0 / len(all_assets), index=all_assets)
                    else:
                        # 平均权重
                        avg_weights = pd.concat(weights_list, axis=1).mean(axis=1)
                        avg_weights /= avg_weights.sum()
                        weights = pd.Series(0, index=all_assets)
                        weights[selected_assets] = avg_weights

                    with open(cache_filename, 'wb') as f:
                        pickle.dump(weights, f)

            previous_weights = weights

            if i + 1 < len(rebalance_dates):
                next_rebalance_date = rebalance_dates.iloc[i + 1]
            else:
                next_rebalance_date = date_index[-1] + pd.Timedelta(days=1)

            weight_dates = date_index[(date_index >= date) & (date_index < next_rebalance_date)]
            weights_history.loc[weight_dates] = weights.values

        # 处理第一个调仓日前的时期
        first_rebalance_date = rebalance_dates.iloc[0]
        initial_dates = date_index[date_index < first_rebalance_date]
        if not initial_dates.empty:
            if previous_weights is not None:
                weights_history.loc[initial_dates] = previous_weights.values
            else:
                equal_weights = pd.Series(1.0 / len(all_assets), index=all_assets)
                weights_history.loc[initial_dates] = equal_weights.values

        weights_history.ffill(inplace=True)

        # 初始化评估器
        evaluator = Evaluator('MomentumStrategy', price_data.loc[start_date:end_date], weights_history)
        return evaluator

    def select_assets(self, price_data, current_date, lookback_periods, top_n_assets):
        """
        根据动量指标选择资产。
        """
        assets = price_data.columns.tolist()
        ranking_df = pd.DataFrame(index=assets)

        period_days = {'1Y': 252, '6M': 126, '3M': 63}

        # 步骤 1：筛选最近1或3个月收益为正的资产
        returns_3m = self.calculate_return(price_data, current_date, 63)
        returns_1m = self.calculate_return(price_data, current_date, 21)
        # trending_assets = returns_3m[returns_3m > 0].dropna().index.tolist()
        trending_assets = returns_3m[(returns_3m > 0) | (returns_1m > 0)].dropna().index.tolist()
        if not trending_assets:
            print(f"  No assets have positive 3M returns on {current_date.strftime('%Y-%m-%d')}")
            return []

        # 步骤 2：计算排名
        for period_name, days in period_days.items():
            returns = self.calculate_return(price_data, current_date, days)
            sharpe_ratios = self.calculate_sharpe_ratio(price_data, current_date, days)

            returns = returns.loc[trending_assets]
            sharpe_ratios = sharpe_ratios.loc[trending_assets]

            ranking_df[f'Return_Rank_{period_name}'] = returns.rank(ascending=False)
            ranking_df[f'Sharpe_Rank_{period_name}'] = sharpe_ratios.rank(ascending=False)

        ranking_df['Average_Rank'] = ranking_df.mean(axis=1)

        ranking_df = ranking_df.sort_values('Average_Rank')
        number_of_selected = min(top_n_assets, len(trending_assets))
        selected_assets = ranking_df.index.tolist()[:number_of_selected]

        print(f"  Selected assets on {current_date.strftime('%Y-%m-%d')}: {selected_assets}")

        return selected_assets

    def calculate_return(self, price_data, current_date, lookback_period):
        available_dates = price_data.index

        if current_date not in available_dates:
            current_date = available_dates[available_dates.get_loc(current_date, method='ffill')]

        current_idx = available_dates.get_loc(current_date)
        start_idx = current_idx - (lookback_period - 1)
        if start_idx < 0:
            return pd.Series(dtype=float)

        past_date = available_dates[start_idx]
        current_price = price_data.loc[current_date]
        past_price = price_data.loc[past_date]
        returns = (current_price / past_price) - 1
        return returns

    def calculate_sharpe_ratio(self, price_data, current_date, lookback_period):
        available_dates = price_data.index

        if current_date not in available_dates:
            current_date = available_dates[available_dates.get_loc(current_date, method='ffill')]

        current_idx = available_dates.get_loc(current_date)
        start_idx = current_idx - (lookback_period - 1)
        if start_idx < 0:
            return pd.Series(dtype=float)

        returns = price_data.iloc[start_idx:current_idx + 1].pct_change().dropna()
        mean_returns = returns.mean()
        std_returns = returns.std()
        sharpe_ratio = mean_returns / std_returns * np.sqrt(252)
        return sharpe_ratio