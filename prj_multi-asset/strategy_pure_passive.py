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
import warnings
warnings.filterwarnings("ignore")

class FixedWeightStrategy:
    def __init__(self):
        pass

    def run_strategy(self, price_data, start_date, end_date, parameters):
        weights = parameters['weights']  # 字典形式 {'Asset1': 0.3, ...}

        # 仅处理必要的资产列和日期范围
        relevant_assets = list(weights.keys())  # 只取weights中涉及的资产
        filtered_data = price_data.loc[start_date:end_date, relevant_assets]

        weights_history = pd.DataFrame(index=filtered_data.index, columns=filtered_data.columns)
        for date in weights_history.index:
            weights_history.loc[date] = pd.Series(weights)

        # 初始化评估器
        evaluator = Evaluator('FixedWeight', filtered_data, weights_history)
        return evaluator


class RiskParityStrategy:
    def __init__(self):
        self.cache_dir = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\专题-风险预算的资产配置策略\cache"
        pass

    def run_strategy(self, price_data, start_date, end_date, parameters):
        selected_assets = parameters['selected_assets']
        asset_class_mapping = parameters['asset_class_mapping']
        rebalance_frequency = parameters.get('rebalance_frequency', 'M')
        lookback_periods = parameters.get('lookback_periods', [63])  # Default to [63] trading days
        risk_budget = parameters.get('risk_budget')
        cache_dir = self.cache_dir

        def check_budget_type(risk_budget, asset_class_mapping):
            risk_budget_keys = set(risk_budget.keys())
            mapping_keys = set(asset_class_mapping.keys())
            mapping_values = set(asset_class_mapping.values())

            if risk_budget_keys.issubset(mapping_keys):
                return "asset_budget"
            elif risk_budget_keys.issubset(mapping_values):
                return "class_budget"
            else:
                raise ValueError("risk_budget keys must all exist in either keys or values of asset_class_mapping")

        budget_type = check_budget_type(risk_budget, asset_class_mapping)

        # Ensure selected_assets are in price_data
        missing_assets = [asset for asset in selected_assets if asset not in price_data.columns]
        if missing_assets:
            raise ValueError(f"The following selected assets are missing from price_data: {missing_assets}")

        # Proceed with only the selected assets
        price_data = price_data.loc[start_date:end_date, selected_assets]

        # Generate rebalancing dates (e.g., monthly)
        date_index = price_data.loc[start_date:end_date].index
        rebalance_dates = date_index.to_series().resample(rebalance_frequency).last().dropna()

        weights_history = pd.DataFrame(index=date_index, columns=price_data.columns)

        previous_weights = None

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        print(f"\nStarting rebalancing calculations from {start_date} to {end_date}")
        for i, date in enumerate(rebalance_dates):
            print(f"Processing rebalance date {date.strftime('%Y-%m-%d')} ({i + 1}/{len(rebalance_dates)})")
            # Generate cache filename based on parameters
            cache_filename = self.generate_cache_filename(
                date, selected_assets, risk_budget, lookback_periods, budget_type, cache_dir
            )

            if os.path.exists(cache_filename):
                # Load weights from cache
                with open(cache_filename, 'rb') as f:
                    weights = pickle.load(f)
            else:
                # Compute weights for multiple lookback periods
                weights_list = []
                for lookback_period in lookback_periods:
                    print(f"  Calculating for lookback period: {lookback_period} days")
                    cov_end_date = date - pd.Timedelta(days=1)
                    cov_start_date = cov_end_date - pd.Timedelta(days=lookback_period - 1)
                    cov_data = price_data.loc[cov_start_date:cov_end_date]
                    if cov_data.shape[0] < lookback_period:
                        continue
                    daily_returns = cov_data.pct_change().dropna()
                    cov_matrix = daily_returns.cov()

                    if budget_type == 'class_budget':
                        print(
                            f"  Performing {'class-level' if budget_type == 'class_budget' else 'asset-level'} risk budget allocation")
                        # Aggregate covariance matrix to class level
                        cov_matrix_class = self.aggregate_covariance_by_class(cov_matrix, asset_class_mapping)
                        # Perform risk budget allocation at class level
                        class_weights = self.risk_budget_allocation(cov_matrix_class, risk_budget)
                        # Distribute class weights to individual assets
                        weights_period = self.distribute_class_weights_to_assets(
                            class_weights, selected_assets, asset_class_mapping
                        )
                    elif budget_type == 'asset_budget':
                        # Perform risk budget allocation at asset level
                        weights_period = self.risk_budget_allocation(cov_matrix, risk_budget)
                    else:
                        raise ValueError("Invalid budget type detected.")

                    weights_list.append(weights_period)

                if len(weights_list) == 0:
                    # Not enough data to compute weights; use previous weights or equal weights
                    if previous_weights is not None:
                        weights = previous_weights
                    else:
                        weights = pd.Series(1.0 / len(price_data.columns), index=price_data.columns)
                else:
                    # Average weights
                    avg_weights = pd.concat(weights_list, axis=1).mean(axis=1)
                    # Normalize weights
                    avg_weights /= avg_weights.sum()
                    weights = avg_weights

                # Save weights to cache
                with open(cache_filename, 'wb') as f:
                    pickle.dump(weights, f)

            previous_weights = weights

            # Apply weights from current rebalancing date until the next rebalancing date
            if i + 1 < len(rebalance_dates):
                next_rebalance_date = rebalance_dates.iloc[i + 1]
            else:
                next_rebalance_date = date_index[-1] + pd.Timedelta(days=1)  # Include the last date

            # Get the date range for applying weights
            weight_dates = date_index[(date_index >= date) & (date_index < next_rebalance_date)]
            weights_history.loc[weight_dates] = weights.values

        # Handle initial period before the first rebalancing date
        first_rebalance_date = rebalance_dates.iloc[0]
        initial_dates = date_index[date_index < first_rebalance_date]
        if not initial_dates.empty:
            if previous_weights is not None:
                weights_history.loc[initial_dates] = previous_weights.values
            else:
                # If no previous weights, use equal weights
                equal_weights = pd.Series(1.0 / len(price_data.columns), index=price_data.columns)
                weights_history.loc[initial_dates] = equal_weights.values

        # Forward-fill any remaining NaN weights
        weights_history.ffill(inplace=True)

        # Initialize evaluator
        evaluator = Evaluator('RiskParity', price_data, weights_history)
        return evaluator

    def generate_cache_filename(self, date, selected_assets, risk_budget, lookback_periods, budget_type, cache_dir):
        cache_key_elements = {
            'date': date.strftime('%Y%m%d'),
            'assets': '_'.join(sorted(selected_assets)),
            'risk_budget': str(sorted(risk_budget.items())),
            'lookbacks': '_'.join(map(str, lookback_periods)),
            'budget_type': budget_type
        }
        cache_key_str = '_'.join([f"{k}_{v}" for k, v in cache_key_elements.items()])
        # Create a hash of the cache_key_str to keep filename manageable
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_filename = os.path.join(cache_dir, f"weights_{cache_hash}.pkl")
        return cache_filename

    def risk_budget_allocation(self, cov_matrix, risk_budget_dict, initial_weights=None, bounds=None):
        """
        Calculate weights based on risk budgeting.

        :param cov_matrix: Covariance matrix (DataFrame), index and columns are asset names or asset classes.
        :param risk_budget_dict: Dict of target risk budgets, keys are asset names or asset classes matching cov_matrix.
        :param initial_weights: Initial guess for weights.
        :param bounds: Bounds for weights.
        :return: Series of weights.
        """
        assets = cov_matrix.columns.tolist()
        num_assets = len(assets)
        if initial_weights is None:
            initial_weights = np.array([1 / num_assets] * num_assets)
        if bounds is None:
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        # Arrange risk_budget vector in the same order as assets
        risk_budget = np.array([risk_budget_dict[asset] for asset in assets])

        # Normalize risk_budget to sum to 1
        risk_budget = risk_budget / np.sum(risk_budget)

        def risk_budget_objective(weights, cov_mat, risk_budget):
            sigma = np.sqrt(weights.T @ cov_mat @ weights)
            MRC = cov_mat @ weights / sigma  # Marginal Risk Contribution
            TRC = weights * MRC  # Total Risk Contribution
            risk_contribution_percent = TRC / np.sum(TRC)
            delta = risk_contribution_percent - risk_budget
            return np.sum(delta ** 2)

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
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        weights = pd.Series(result.x, index=assets)
        return weights

    def aggregate_covariance_by_class(self, cov_matrix, asset_class_mapping):
        """
        Aggregate the covariance matrix to the asset class level.

        :param cov_matrix: Covariance matrix of individual assets (DataFrame)
        :param asset_class_mapping: Dictionary mapping asset names to asset classes
        :return: Aggregated covariance matrix at the class level (DataFrame)
        """
        # Map assets to classes
        asset_classes = pd.Series(asset_class_mapping)
        class_list = asset_classes.unique()
        # Initialize class-level covariance matrix
        cov_matrix_class = pd.DataFrame(index=class_list, columns=class_list, dtype=float)

        for class_i in class_list:
            assets_i = asset_classes[asset_classes == class_i].index
            for class_j in class_list:
                assets_j = asset_classes[asset_classes == class_j].index
                sub_cov = cov_matrix.loc[assets_i, assets_j].values
                cov_sum = np.sum(sub_cov)
                cov_matrix_class.loc[class_i, class_j] = cov_sum

        return cov_matrix_class

    def distribute_class_weights_to_assets(self, class_weights, selected_assets, asset_class_mapping):
        """
        Distribute class weights to individual assets within each class.

        :param class_weights: Series of weights allocated to each asset class
        :param selected_assets: List of asset names
        :param asset_class_mapping: Dictionary mapping asset names to asset classes
        :return: Series of weights allocated to individual assets
        """
        # Initialize asset weights
        asset_weights = pd.Series(0, index=selected_assets)

        # For each asset class, distribute class weight to assets within that class
        for asset_class, class_weight in class_weights.items():
            class_assets = [asset for asset in selected_assets if asset_class_mapping[asset] == asset_class]
            num_assets_in_class = len(class_assets)
            if num_assets_in_class > 0:
                # Equal weighting within the class
                asset_weight = class_weight / num_assets_in_class
                asset_weights[class_assets] = asset_weight
            else:
                # No assets in this class
                continue

        return asset_weights
