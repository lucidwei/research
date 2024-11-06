# coding=gbk
# Time Created: 2024/10/30 14:56
# Author  : Lucid
# FileName: strategy_momentum.py
# Software: PyCharm
# strategy_momentum.py
from evaluator import Evaluator
import pandas as pd
import numpy as np

class MomentumStrategy:
    def __init__(self):
        pass

    def run_strategy(self, price_data, start_date, end_date, parameters):
        asset_class_mapping = parameters['asset_class_mapping']
        risk_prefs = parameters.get('risk_prefs')
        macro_adj = parameters.get('macro_adj')
        subjective_adj = parameters.get('subjective_adj')

        weights_history = self.backtest(price_data, start_date, end_date, asset_class_mapping, risk_prefs, macro_adj, subjective_adj)
        evaluator = Evaluator('Momentum', price_data, weights_history)
        return evaluator

    def backtest(self, price_data, start_date, end_date, asset_class_mapping, risk_prefs, macro_adj, subjective_adj):
        weights_history = pd.DataFrame(index=price_data.loc[start_date:end_date].index, columns=price_data.columns)
        current_weights = pd.Series(0, index=price_data.columns)
        rebalance_dates = price_data.loc[start_date:end_date].resample('M').last().index
        for date in weights_history.index:
            if date in rebalance_dates:
                returns_dict, sharpe_dict = self.calculate_indicators(price_data, date)
                selected_assets = self.select_assets(price_data, date, returns_dict)
                if not selected_assets:
                    print(f"{date.strftime('%Y-%m-%d')} �޷����������ʲ���ά�ֿղ֡�")
                    current_weights = pd.Series(0, index=price_data.columns)
                else:
                    initial_weights = self.calculate_weights(price_data, date, selected_assets)
                    # ����Ȩ��
                    adjusted_weights = self.adjust_weights(initial_weights, asset_class_mapping, risk_prefs, macro_adj, subjective_adj)
                    current_weights = adjusted_weights
            weights_history.loc[date] = current_weights
        return weights_history

    def calculate_indicators(self, price_data, current_date):
        periods = {'1M': 21, '3M': 63, '6M': 126}
        assets = price_data.columns.tolist()
        returns_dict = {}
        sharpe_dict = {}
        for period_name, period_days in periods.items():
            if price_data.index.get_loc(current_date) >= period_days:
                # ������
                past_price = price_data.loc[:current_date].iloc[-(period_days+1)]
                current_price = price_data.loc[current_date]
                returns = (current_price / past_price - 1)
                returns_dict[period_name] = returns
                # ���ձ���
                daily_returns = price_data.loc[:current_date].iloc[-(period_days+1):].pct_change().dropna()
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                sharpe_dict[period_name] = sharpe_ratio
            else:
                returns_dict[period_name] = pd.Series(np.nan, index=assets)
                sharpe_dict[period_name] = pd.Series(np.nan, index=assets)
        return returns_dict, sharpe_dict

    def select_assets(self, price_data, current_date, returns_dict):
        assets = price_data.columns.tolist()
        # ɸѡ���3��������Ϊ�����ʲ�
        returns_3m = returns_dict['3M']
        trending_assets = returns_3m[returns_3m > 0].dropna().index.tolist()
        if not trending_assets:
            return []
        # �������ʺ����ձ��ʽ�������
        rank_df = pd.DataFrame(index=trending_assets)
        for period in ['1M', '3M', '6M']:
            rank_df[f'Return_Rank_{period}'] = returns_dict[period][trending_assets].rank(ascending=False)
            rank_df[f'Sharpe_Rank_{period}'] = returns_dict[period][trending_assets].rank(ascending=False)
        rank_df['Average_Rank'] = rank_df.mean(axis=1)
        # ѡȡƽ��������ǰ��ǰ10���ʲ�
        rank_df = rank_df.sort_values('Average_Rank')
        selected_assets = rank_df.index.tolist()[:10]
        return selected_assets

    def calculate_weights(self, price_data, current_date, selected_assets):
        periods = {'1M': 21, '3M': 63, '6M': 126}
        weight_list = []
        for period_name, period_days in periods.items():
            if price_data.index.get_loc(current_date) >= period_days:
                # Э�������
                daily_returns = price_data.loc[:current_date].iloc[-(period_days+1):][selected_assets].pct_change().dropna()
                cov_matrix = daily_returns.cov()
                # ����ƽ��Ȩ��
                weights = risk_parity_weights(cov_matrix)
                weight_list.append(weights)
        # ƽ������Ȩ��
        avg_weights = pd.concat(weight_list, axis=1).mean(axis=1)
        # ��һ��
        final_weights = avg_weights / avg_weights.sum()
        return final_weights

    def adjust_weights(self, weights, asset_class_mapping, risk_prefs=None, macro_adj=None, subjective_adj=None):
        weights = weights.copy()
        # �����ʲ�������Ȩ��
        weights_by_class = weights.groupby(asset_class_mapping).sum()
        # Ӧ�÷���ƫ�õ���
        if risk_prefs:
            for asset_class, target_weight in risk_prefs.items():
                if asset_class in weights_by_class.index:
                    scaling_factor = target_weight / weights_by_class[asset_class]
                    asset_indices = [asset for asset in weights.index if asset_class_mapping[asset] == asset_class]
                    weights[asset_indices] *= scaling_factor
        # Ӧ�ú�۵���
        if macro_adj:
            for asset_class, adj in macro_adj.items():
                asset_indices = [asset for asset in weights.index if asset_class_mapping[asset] == asset_class]
                weights[asset_indices] *= (1 + adj)
        # Ӧ�����۵���
        if subjective_adj:
            for asset_class, adj in subjective_adj.items():
                asset_indices = [asset for asset in weights.index if asset_class_mapping[asset] == asset_class]
                weights[asset_indices] *= (1 + adj)
        # ��һ��
        adjusted_weights = weights / weights.sum()
        return adjusted_weights