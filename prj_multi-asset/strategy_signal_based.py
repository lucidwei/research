# coding=gbk
# Time Created: 2024/10/30 14:55
# Author  : Lucid
# FileName: strategy_signal_based.py
# Software: PyCharm
# strategy_signal_based.py

import pandas as pd
from evaluator import Evaluator
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

class SignalBasedStrategy:
    def __init__(self):
        pass

    def run_strategy(self, price_data, start_date, end_date, parameters):
        # Extract parameters specific to the signal-based strategy
        asset_class_mapping = parameters['asset_class_mapping']
        signal_parameters = parameters.get('signal_parameters', {})
        risk_budget = parameters.get('risk_budget', [0.4, 0.4, 0.2])  # Default risk budget
        # Create instances of StrategicAllocator and TacticalAllocator
        strategic_allocator = StrategicAllocator(price_data, asset_class_mapping, risk_budget)
        tactical_allocator = TacticalAllocator(price_data, signal_parameters)
        # Generate signals
        signals = tactical_allocator.generate_signal_report()
        # Generate asset positions
        weights_history = strategic_allocator.generate_asset_positions(tactical_allocator, signals, start_date, end_date)
        # Initialize evaluator
        evaluator = Evaluator('SignalBased', price_data, weights_history)
        return evaluator

class StrategicAllocator:
    def __init__(self, price_data, asset_class_mapping, risk_budget):
        self.price_data = price_data
        self.asset_class_mapping = asset_class_mapping
        self.risk_budget = risk_budget
        self.assets = list(asset_class_mapping.keys())
        self.returns_data = self.calculate_returns()

    def calculate_returns(self):
        returns_data = self.price_data[self.assets].pct_change().dropna()
        returns_data.columns = self.assets
        return returns_data

    def calc_strategic_monthly_weight(self, start_date, end_date):
        returns_data = self.returns_data
        # Calculate monthly returns
        monthly_returns = (returns_data + 1).resample('M').apply(lambda x: x.prod() - 1)
        # Calculate rolling covariance matrices (12 months window)
        rolling_cov_matrix = self.calculate_rolling_cov_matrix(monthly_returns, window=12)
        # Initialize DataFrame to store monthly weights
        monthly_weights = pd.DataFrame(index=monthly_returns.index, columns=self.assets)
        for date in monthly_returns.index:
            if date in rolling_cov_matrix.index.get_level_values(0):
                cov_matrix = rolling_cov_matrix.loc[date].unstack(level=-1)
                cov_matrix = cov_matrix.reindex(index=self.assets, columns=self.assets)
                # Calculate weights using risk_budget_allocation
                weights = risk_budget_allocation(cov_matrix, self.risk_budget)
                monthly_weights.loc[date] = weights
            else:
                # If covariance matrix not available, use previous weights
                if not monthly_weights.iloc[:-1].empty:
                    monthly_weights.loc[date] = monthly_weights.iloc[-1]
                else:
                    monthly_weights.loc[date] = np.nan
        self.strategic_monthly_weights = monthly_weights.dropna()
        return self.strategic_monthly_weights

    def generate_asset_positions(self, tactical_allocator, signals, start_date, end_date):
        # Calculate strategic weights
        self.calc_strategic_monthly_weight(start_date, end_date)
        # Adjust weights based on signals
        adjusted_monthly_weights = self.adjust_weights_by_signals(self.strategic_monthly_weights, signals)
        # Return adjusted weights history
        return adjusted_monthly_weights

    def adjust_weights_by_signals(self, strategic_weights, signals):
        combined_signals = signals['combined'].resample('M').last()
        adjusted_weights = strategic_weights.copy()
        for date in strategic_weights.index:
            if date in combined_signals.index:
                signal_row = combined_signals.loc[date]
                original_weights = adjusted_weights.loc[date]
                adjusted_weights.loc[date] = self.adjust_weights(original_weights, signal_row)
            else:
                # If no signal for the date, keep the weights unchanged
                pass
        return adjusted_weights

    def adjust_weights(self, weights, signal_row):
        adjusted_weights = weights.copy()
        # Adjust weights based on signals
        # Example: Increase or decrease weights by 30% based on signal
        for asset_class in ['Equity', 'Gold']:
            signal = signal_row.get(f'combined_{asset_class.lower()}_signal', 0)
            assets_in_class = [asset for asset, cls in self.asset_class_mapping.items() if cls == asset_class]
            if signal == 1:
                adjusted_weights[assets_in_class] *= 1.3
            elif signal == -1:
                adjusted_weights[assets_in_class] *= 0.7
            # Ensure weights don't exceed limits
            adjusted_weights[assets_in_class] = np.clip(adjusted_weights[assets_in_class], 0, 0.5)
        # Normalize weights to sum to 1
        adjusted_weights /= adjusted_weights.sum()
        return adjusted_weights

    def calculate_rolling_cov_matrix(self, returns_data, window):
        cov_matrix = returns_data.rolling(window=window).cov().dropna()
        return cov_matrix

class TacticalAllocator:
    def __init__(self, price_data, signal_parameters):
        self.price_data = price_data
        self.signal_parameters = signal_parameters
        # Extract necessary data for signal generation
        self.initialize_data()

    def initialize_data(self):
        # Ensure necessary data columns are available
        required_columns = ['Stock_Price', 'Stock_Volume', 'PE_TTM', 'Bond_Yield', 'Gold_Price', 'VIX']
        missing_columns = [col for col in required_columns if col not in self.price_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required data columns: {missing_columns}")
        self.stock_prices = self.price_data['Stock_Price']
        self.stock_volume = self.price_data['Stock_Volume']
        self.pe_ttm = self.price_data['PE_TTM']
        self.bond_yields = self.price_data['Bond_Yield']
        self.gold_prices = self.price_data['Gold_Price']
        self.vix = self.price_data['VIX']

    def generate_signal_report(self):
        # Generate individual signals
        stock_signal_generator = StockSignalGenerator(self.stock_prices, self.stock_volume, self.pe_ttm, self.bond_yields)
        gold_signal_generator = GoldSignalGenerator(self.gold_prices, self.vix)
        erp_signal = stock_signal_generator.erp_signal()
        ma_signal = stock_signal_generator.ma_signal()
        volume_signal = stock_signal_generator.volume_signal()
        volume_ma_signal = stock_signal_generator.volume_ma_signal()
        us_tips_signal = gold_signal_generator.us_tips_signal()
        vix_signal = gold_signal_generator.vix_signal()
        gold_momentum_signal = gold_signal_generator.gold_momentum_signal()
        # Align signals
        aligned_signals = align_signals(erp_signal=erp_signal, us_tips_signal=us_tips_signal, volume_ma_signal=volume_ma_signal)
        erp_signal = aligned_signals['erp_signal']
        us_tips_signal = aligned_signals['us_tips_signal']
        volume_ma_signal = aligned_signals['volume_ma_signal']
        # Combine signals
        combined_stock_signal = stock_signal_generator.combined_stock_signal()
        combined_gold_signal = gold_signal_generator.combined_gold_signal()
        # Aggregate signals into DataFrame
        individual_signals = pd.concat([erp_signal, ma_signal, volume_signal, volume_ma_signal,
                                        us_tips_signal, vix_signal, gold_momentum_signal], axis=1)
        individual_signals.columns = ['erp_signal', 'ma_signal', 'volume_signal', 'volume_ma_signal',
                                      'us_tips_signal', 'vix_signal', 'gold_momentum_signal']
        combined_signals = pd.concat([combined_stock_signal, combined_gold_signal], axis=1)
        combined_signals.columns = ['combined_stock_signal', 'combined_gold_signal']
        signals = {'individual': individual_signals, 'combined': combined_signals}
        return signals

def align_signals(**signals):
    """
    Align multiple pandas Series based on their common index.
    """
    common_index = None
    for signal in signals.values():
        if common_index is None:
            common_index = signal.index
        else:
            common_index = common_index.intersection(signal.index)
    aligned_signals = {name: signal.reindex(common_index) for name, signal in signals.items()}
    return aligned_signals

class StockSignalGenerator:
    def __init__(self, stock_prices, stock_volume, pe_ttm, bond_yields):
        self.stock_prices = stock_prices
        self.stock_volume = stock_volume
        self.pe_ttm = pe_ttm
        self.bond_yields = bond_yields
        self.calculate_erp()

    def calculate_erp(self):
        # Calculate Equity Risk Premium (ERP)
        self.erp = (1 / self.pe_ttm) - self.bond_yields

    def erp_signal(self, window=120, upper_quantile=0.7, lower_quantile=0.3):
        erp_weighted = self.erp.ewm(span=window).mean()
        lower_quantile_value = erp_weighted.rolling(window=252 * 5).quantile(lower_quantile)
        upper_quantile_value = erp_weighted.rolling(window=252 * 5).quantile(upper_quantile)
        signal = pd.Series(0, index=erp_weighted.index)
        signal[erp_weighted > upper_quantile_value] = 1
        signal[erp_weighted < lower_quantile_value] = -1
        return signal.dropna()

    def ma_signal(self, window=5):
        ma = self.stock_prices.rolling(window=window).mean()
        signal = pd.Series(np.where(self.stock_prices > ma, 1, -1), index=self.stock_prices.index)
        return signal.dropna()

    def volume_signal(self, window=5):
        mean_volume = self.stock_volume.rolling(window=window).mean()
        signal = pd.Series(np.where(self.stock_volume > mean_volume, 1, -1), index=self.stock_volume.index)
        return signal.dropna()

    def volume_ma_signal(self, ma_window=5, volume_window=5):
        ma_signal = self.ma_signal(ma_window)
        volume_signal = self.volume_signal(volume_window)
        aligned_signals = align_signals(ma_signal=ma_signal, volume_signal=volume_signal)
        ma_signal = aligned_signals['ma_signal']
        volume_signal = aligned_signals['volume_signal']
        combined_signal = pd.Series(np.where((ma_signal == 1) & (volume_signal == 1), 1, -1), index=ma_signal.index)
        return combined_signal.dropna()

    def combined_stock_signal(self):
        erp_signal = self.erp_signal()
        volume_ma_signal = self.volume_ma_signal()
        aligned_signals = align_signals(erp_signal=erp_signal, volume_ma_signal=volume_ma_signal)
        erp_signal = aligned_signals['erp_signal']
        volume_ma_signal = aligned_signals['volume_ma_signal']
        combined_signal = pd.Series(0, index=erp_signal.index)
        combined_signal[(erp_signal == 1) & (volume_ma_signal == 1)] = 1
        combined_signal[(erp_signal == -1) & (volume_ma_signal == -1)] = -1
        return combined_signal.dropna()

class GoldSignalGenerator:
    def __init__(self, gold_prices, vix):
        self.gold_prices = gold_prices
        self.vix = vix

    def us_tips_signal(self, window=20):
        # Placeholder for US TIPS data
        tips_10y = pd.Series(0, index=self.gold_prices.index)  # Replace with actual TIPS data if available
        ma = tips_10y.rolling(window=window).mean()
        signal = pd.Series(np.where(tips_10y > ma, -1, 1), index=tips_10y.index)
        return signal.dropna()

    def vix_signal(self, window=15, threshold=20):
        vix_max = self.vix.rolling(window=window).max()
        signal = pd.Series(np.where(vix_max > threshold, 1, 0), index=self.vix.index)
        return signal.dropna()

    def gold_momentum_signal(self, window=250):
        gold_momentum = (self.gold_prices / self.gold_prices.shift(window) - 1)
        signal = pd.Series(np.where(gold_momentum > 0, 1, 0), index=self.gold_prices.index)
        return signal.dropna()

    def combined_gold_signal(self):
        us_tips_signal = self.us_tips_signal()
        vix_signal = self.vix_signal()
        gold_momentum_signal = self.gold_momentum_signal()
        aligned_signals = align_signals(us_tips_signal=us_tips_signal, vix_signal=vix_signal, gold_momentum_signal=gold_momentum_signal)
        us_tips_signal = aligned_signals['us_tips_signal']
        vix_signal = aligned_signals['vix_signal']
        gold_momentum_signal = aligned_signals['gold_momentum_signal']
        combined_signal = us_tips_signal + vix_signal + gold_momentum_signal
        signal = pd.Series(0, index=combined_signal.index)
        signal[combined_signal >= 2] = 1
        signal[combined_signal < 0] = -1
        return signal.dropna()