# coding=gbk
# Time Created: 2024/10/30 14:56
# Author  : Lucid
# FileName: strategy_pure_passive.py
# Software: PyCharm
# strategy_risk_parity.py
from evaluator import Evaluator
import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping
import os
import hashlib
import pickle
from utils import get_tradedays
import warnings

warnings.filterwarnings("ignore")


class FixedWeightStrategy:
    def __init__(self):
        pass

    def run_strategy(self, data_dict, start_date, end_date, parameters):
        price_data = data_dict['close_prices']
        weights = parameters['weights']  # 字典形式 {'Asset1': 0.3, ...}

        # 仅处理必要的资产列和日期范围
        relevant_assets = list(weights.keys())  # 只取weights中涉及的资产

        if not end_date:
            end_date = price_data.index.max()
        filtered_data = price_data.loc[start_date:end_date, relevant_assets]

        weights_history = pd.DataFrame(index=filtered_data.index, columns=filtered_data.columns)
        for date in weights_history.index:
            weights_history.loc[date] = pd.Series(weights)

        # 初始化评估器
        evaluator = Evaluator('FixedWeight', filtered_data, weights_history)
        return evaluator


class BaseStrategy:
    def __init__(self):
        self.cache_dir = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\专题-风险预算的资产配置策略\cache"
        pass

    def calculate_initial_positions(self, cov_matrix, risk_budget):
        """
        计算基于风险平价的资产初始仓位。

        参数:
        cov_matrix : np.array
            资产的协方差矩阵。
        risk_budget : np.array
            每个资产的风险预算。

        返回:
        np.array
            每个资产的优化前初始仓位。
        """
        # 计算每个资产的波动率（标准差）
        volatilities = np.sqrt(np.diag(cov_matrix))

        # 计算风险调整后的权重
        risk_adjusted_weights = risk_budget / volatilities

        # 计算最终的资产配置比例
        final_positions = risk_adjusted_weights / np.sum(risk_adjusted_weights)

        return final_positions

    def risk_budget_allocation(self, cov_matrix, risk_budget_dict, initial_weights=None, bounds=None):
        """
        计算基于风险预算的权重。

        :param cov_matrix: 协方差矩阵（DataFrame），索引和列是资产名称或资产类别。
        :param risk_budget_dict: 目标风险预算的字典，键为资产名称或资产类别，与 cov_matrix 匹配。
        :param initial_weights: 初始权重猜测。
        :param bounds: 权重的边界。
        :return: 权重的 Series。
        """
        assets = cov_matrix.columns.tolist()
        num_assets = len(assets)

        # 按照资产顺序安排风险预算向量
        risk_budget = np.array([risk_budget_dict[asset] for asset in assets])

        # 规范化风险预算，使其和为 1
        risk_budget = risk_budget / np.sum(risk_budget)

        if initial_weights is None:
            initial_weights = self.calculate_initial_positions(cov_matrix, risk_budget)

            # # 处理单一资产的情况
            # if num_assets == 1:
            #     initial_weights[0] = 1.0
            #
            # # 处理两个资产的情况
            # elif num_assets == 2:
            #     if 'Equity' in assets and 'Commodity' in assets:
            #         equity_index = assets.index('Equity')
            #         commodity_index = assets.index('Commodity')
            #         initial_weights[equity_index] = 0.6
            #         initial_weights[commodity_index] = 0.4
            #     elif 'Equity' in assets and 'Bond' in assets:
            #         equity_index = assets.index('Equity')
            #         bond_index = assets.index('Bond')
            #         initial_weights[equity_index] = 0.2
            #         initial_weights[bond_index] = 0.8
            #     elif 'Commodity' in assets and 'Bond' in assets:
            #         commodity_index = assets.index('Commodity')
            #         bond_index = assets.index('Bond')
            #         initial_weights[commodity_index] = 0.2
            #         initial_weights[bond_index] = 0.8
            #
            # # 处理三个资产的情况
            # elif num_assets == 3 and set(assets) == {'Equity', 'Commodity', 'Bond'}:
            #     equity_index = assets.index('Equity')
            #     commodity_index = assets.index('Commodity')
            #     bond_index = assets.index('Bond')
            #     initial_weights[equity_index] = 0.2
            #     initial_weights[commodity_index] = 0.1
            #     initial_weights[bond_index] = 0.7
            # # 处理其他资产数量情况或资产不匹配预设组合的情况
            # else:
            #     equally_distributed = 1.0 / num_assets
            #     initial_weights = np.full(num_assets, equally_distributed)

        else:
            initial_weights = np.array(initial_weights)

        if bounds is None:
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        def risk_budget_objective(weights, cov_mat, risk_budget):
            # 计算组合方差和标准差
            portfolio_variance = np.dot(weights, np.dot(cov_mat, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # 避免除以零
            if portfolio_std == 0:
                return np.inf

            # 计算边际风险贡献 (Marginal Risk Contribution)
            mrc = np.dot(cov_mat, weights) / portfolio_std

            # 计算总体风险贡献 (Total Risk Contribution)
            trc = weights * mrc

            # 计算风险贡献百分比
            rc_percent = trc / trc.sum()

            # 目标函数：最小化实际风险贡献百分比与目标风险预算之间的平方差
            objective_value = np.sum((rc_percent - risk_budget) ** 2)

            # 调试信息
            # print(f"Weights: {weights}, RC Percent: {rc_percent}, Objective: {objective_value}")

            return objective_value

        constraints = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        }

        result = minimize(
            fun=risk_budget_objective,
            x0=initial_weights,
            args=(cov_matrix.values, risk_budget),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            # options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
        )

        if not result.success:
            raise ValueError("优化未能收敛： " + result.message)

        weights = pd.Series(result.x, index=assets)
        return weights

    def generate_cache_filename(self, cache_key_elements):
        """
        生成缓存文件名。

        :param cache_key_elements: 字典，包含生成缓存文件名所需的关键元素。
        :return: 缓存文件名字符串。
        """
        cache_key_str = '_'.join([f"{k}_{v}" for k, v in cache_key_elements.items()])
        # 创建缓存键字符串的哈希值，避免文件名过长
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_filename = os.path.join(self.cache_dir, f"weights_{cache_hash}.pkl")
        return cache_filename

    def check_budget_type(self, risk_budget, asset_class_mapping):
        risk_budget_keys = set(risk_budget.keys())
        mapping_keys = set(asset_class_mapping.keys())
        mapping_values = set(asset_class_mapping.values())

        if risk_budget_keys.issubset(mapping_keys):
            return "asset_budget"
        elif risk_budget_keys.issubset(mapping_values):
            return "class_budget"
        else:
            raise ValueError("risk_budget keys must all exist in either keys or values of asset_class_mapping")

    def aggregate_covariance_by_class(self, cov_matrix, asset_class_mapping):
        """
        Aggregate the covariance matrix to the asset class level.

        :param cov_matrix: Covariance matrix of individual assets (DataFrame)
        :param asset_class_mapping: Dictionary mapping asset names to asset classes
        :return: Aggregated covariance matrix at the class level (DataFrame)
        """
        # Map assets to classes
        asset_classes = pd.Series(asset_class_mapping)
        # Keep only assets present in cov_matrix
        asset_classes = asset_classes[asset_classes.index.isin(cov_matrix.index)]

        if asset_classes.empty:
            raise ValueError("No overlapping assets between cov_matrix and asset_class_mapping.")

        class_list = asset_classes.unique()
        # Initialize class-level covariance matrix
        cov_matrix_class = pd.DataFrame(index=class_list, columns=class_list, dtype=float)

        for class_i in class_list:
            assets_i = asset_classes[asset_classes == class_i].index
            for class_j in class_list:
                assets_j = asset_classes[asset_classes == class_j].index
                # Ensure assets_i and assets_j are in cov_matrix
                assets_i_in_cov = [asset for asset in assets_i if asset in cov_matrix.index]
                assets_j_in_cov = [asset for asset in assets_j if asset in cov_matrix.columns]

                if not assets_i_in_cov or not assets_j_in_cov:
                    cov_value = 0  # If no overlapping assets, set covariance to zero
                    print(f"No overlapping assets for classes {class_i} and {class_j}. Setting covariance to zero.")
                else:
                    sub_cov = cov_matrix.loc[assets_i_in_cov, assets_j_in_cov].values
                    cov_value = np.sum(sub_cov)
                cov_matrix_class.loc[class_i, class_j] = cov_value

        return cov_matrix_class

    def distribute_class_weights_to_assets(self, class_weights, selected_assets, asset_class_mapping):
        """
        将类别权重分配到类别内的个别资产。

        :param class_weights: 分配给每个资产类别的权重 Series
        :param selected_assets: 资产名称列表
        :param asset_class_mapping: 资产名称到资产类别的字典映射
        :return: 分配给个别资产的权重 Series
        """
        # 初始化资产权重
        asset_weights = pd.Series(0, index=selected_assets)

        # 对于每个资产类别，将类别权重分配到该类别内的资产
        for asset_class, class_weight in class_weights.items():
            class_assets = [asset for asset in selected_assets if asset_class_mapping[asset] == asset_class]
            num_assets_in_class = len(class_assets)
            if num_assets_in_class > 0:
                # 在类别内等权分配
                asset_weight = class_weight / num_assets_in_class
                asset_weights[class_assets] = asset_weight
            else:
                # 该类别下没有资产
                continue

        return asset_weights

    def risk_parity_weights(self, cov_matrix):
        """
        计算给定协方差矩阵的风险平价权重。
        """
        assets = cov_matrix.columns.tolist()
        num_assets = len(assets)
        x0 = np.array([1 / num_assets] * num_assets)
        bounds = [(0, 1) for _ in assets]
        constraints = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        }

        def objective(weights):
            portfolio_variance = weights.T @ cov_matrix.values @ weights
            sigma = np.sqrt(portfolio_variance)
            MRC = cov_matrix.values @ weights / sigma
            TRC = weights * MRC
            risk_contribution = TRC / np.sum(TRC)
            target = np.ones(num_assets) / num_assets
            return np.sum((risk_contribution - target) ** 2)

        result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        weights = pd.Series(result.x, index=assets)
        return weights

    def calc_rebalance_dates(self, start_date, end_date, date_index, rebalance_frequency):
        # 获取交易日列表
        all_trading_days_list = get_tradedays(start_date, end_date)

        # 将交易日列表转换为 DateTime Series
        all_trading_days = pd.to_datetime(all_trading_days_list)
        all_trading_days = pd.Series(all_trading_days).sort_values()
        all_trading_days.index = all_trading_days  # 设置日期为索引

        # 用于找到每月的最后一个交易日
        end_of_month_trading_days = all_trading_days.resample('M').last()

        # 根据给定的调仓频率和日期索引找到调仓日期
        rebalance_dates = date_index.to_series().resample(rebalance_frequency).last().dropna()

        # 确保调仓日期是有效的交易日
        # rebalance_dates = rebalance_dates[rebalance_dates.isin(all_trading_days)]

        # 检查最后一个调仓日期是否为其所在月的最后一个交易日
        if not rebalance_dates.empty:
            last_rebalance_date = rebalance_dates.iloc[-1]
            if last_rebalance_date not in end_of_month_trading_days.index:
                # 如果最后一个调仓日期不是月末交易日，则剔除
                rebalance_dates = rebalance_dates[rebalance_dates != last_rebalance_date]
                print(f"最后一个调仓日期 {last_rebalance_date} 不是月末交易日，将被剔除。")

        return rebalance_dates


class RiskParityStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def run_strategy(self, data_dict, start_date, end_date, parameters):
        price_data = data_dict['close_prices']
        selected_assets = parameters['selected_assets']
        asset_class_mapping = parameters['asset_class_mapping']
        rebalance_frequency = parameters.get('rebalance_frequency', 'M')
        lookback_periods = parameters.get('lookback_periods', [63])  # 默认使用 63 个交易日
        risk_budget = parameters.get('risk_budget')

        budget_type = self.check_budget_type(risk_budget, asset_class_mapping)

        # 确保 selected_assets 在 price_data 中
        missing_assets = [asset for asset in selected_assets if asset not in price_data.columns]
        if missing_assets:
            raise ValueError(f"The following selected assets are missing from price_data: {missing_assets}")

        # 使用完整的价格数据
        price_data_full = price_data.loc[:, selected_assets].copy(deep=True)
        if not end_date:
            end_date = price_data.index.max()
        price_data = price_data.loc[start_date:end_date, selected_assets]

        # 生成调仓日期
        date_index = price_data.loc[start_date:end_date].index
        rebalance_dates = self.calc_rebalance_dates(start_date, end_date, date_index, rebalance_frequency)

        weights_history = pd.DataFrame(index=date_index, columns=price_data.columns)

        previous_weights = None

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"\nStarting rebalancing calculations from {start_date} to {end_date}")
        for i, date in enumerate(rebalance_dates):
            print(f"Processing rebalance date {date.strftime('%Y-%m-%d')} ({i + 1}/{len(rebalance_dates)})")
            # 生成缓存文件名
            cache_key_elements = {
                'date': date.strftime('%Y%m%d'),
                'assets': '_'.join(sorted(selected_assets)),
                'risk_budget': str(sorted(risk_budget.items())),
                'lookbacks': '_'.join(map(str, lookback_periods)),
                'budget_type': budget_type
            }
            cache_filename = self.generate_cache_filename(cache_key_elements)

            if os.path.exists(cache_filename):
                # 从缓存中加载权重
                with open(cache_filename, 'rb') as f:
                    weights = pickle.load(f)
            else:
                # 计算多期望回溯期的权重
                weights_list = []
                for lookback_period in lookback_periods:
                    print(f"  Calculating for lookback period: {lookback_period} days")
                    # 处理非交易日
                    available_dates = price_data_full.index
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
                    cov_data = price_data_full.loc[cov_start_date:cov_end_date]

                    if cov_data.shape[0] < lookback_period * 0.9:
                        print(f"  Not enough data for lookback period of {lookback_period} trading days.")
                        continue

                    daily_returns = cov_data.pct_change().dropna()
                    cov_matrix = daily_returns.cov()

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

                    weights_list.append(weights_period)

                if len(weights_list) == 0:
                    if previous_weights is not None:
                        weights = previous_weights
                    else:
                        weights = pd.Series(1.0 / len(price_data.columns), index=price_data.columns)
                else:
                    avg_weights = pd.concat(weights_list, axis=1).mean(axis=1)
                    avg_weights /= avg_weights.sum()
                    weights = avg_weights

                # 保存权重到缓存
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
                equal_weights = pd.Series(1.0 / len(price_data.columns), index=price_data.columns)
                weights_history.loc[initial_dates] = equal_weights.values

        weights_history.ffill(inplace=True)

        # 初始化评估器
        evaluator = Evaluator('RiskParity', price_data, weights_history)
        return evaluator
