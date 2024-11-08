# -*- coding: utf-8 -*-
# strategy_signal_based.py

from strategy_pure_passive import BaseStrategy
from evaluator import Evaluator
import pandas as pd
import numpy as np

"""
### **Explanation and Detailed Documentation**

#### **1. Data Loading**

- **Data Sources:**
  - `close_prices`: Asset close prices.
  - `edb_data`: Economic data (e.g., bond yields, TIPS yields, VIX).
  - `composite_data`: Additional data (e.g., stock volume, PE ratios).

- **Reusing Existing Data Loader:**
  - The data is loaded using your existing code framework.
  - The `run_strategy` method now accepts `data_dict`, which contains all the required data.

#### **2. Signal-Based Strategy Logic**

- **Baseline Allocation:**
  - Uses the **risk budgeting model** to determine baseline weights.
  - **Risk budgets** are specified in `parameters['risk_budget']`.
  - The **budget type** is determined (`asset_budget` or `class_budget`) based on `risk_budget` and `asset_class_mapping`.

- **Signal Generation:**
  - **Stock Signals:**
    - **ERP Signal (`erp_signal`):**
      - **Calculation:**
        - ERP = (1 / PE Ratio) - Bond Yield.
        - Uses exponential weighted moving average (EWMA) over 120 days.
        - Signals generated based on ERP's position relative to rolling quantiles over the past 5 years.
      - **Signal Meanings:**
        - **1 (Buy):** ERP above the 70th percentile (upper quantile).
        - **-1 (Sell):** ERP below the 30th percentile (lower quantile).
        - **0 (Hold):** ERP between the quantiles.
      - **Economic Interpretation:**
        - A high ERP suggests stocks are undervalued relative to bonds, indicating a potential buying opportunity.
    - **Volume and MA Signal (`volume_ma_signal`):**
      - **Calculation:**
        - **Price MA Signal:** Stock price compared to its 5-day moving average.
        - **Volume MA Signal:** Stock volume compared to its 5-day moving average.
        - **Combined Signal:** Buy when both price and volume are above their MAs.
      - **Signal Meanings:**
        - **1 (Buy):** Both price and volume signals are positive.
        - **-1 (Sell):** Otherwise.
      - **Economic Interpretation:**
        - Rising prices with increasing volume may indicate strong market interest and momentum.
    - **Combined Stock Signal (`combined_stock_signal`):**
      - **Logic:**
        - Buy when both ERP and volume MA signals are positive.
        - Sell when both are negative.
        - Hold when signals disagree.
  - **Gold Signals:**
    - **US TIPS Signal (`us_tips_signal`):**
      - **Calculation:**
        - TIPS yield compared to its 20-day moving average.
      - **Signal Meanings:**
        - **1 (Buy):** TIPS yield below MA (indicating lower real yields).
        - **-1 (Sell):** TIPS yield above MA.
      - **Economic Interpretation:**
        - Lower real yields reduce the opportunity cost of holding gold, making it more attractive.
    - **VIX Signal (`vix_signal`):**
      - **Calculation:**
        - Max VIX over 15 days compared to a threshold (e.g., 20).
      - **Signal Meanings:**
        - **1 (Buy):** VIX exceeds threshold (indicating higher market volatility).
        - **0 (Hold):** VIX below threshold.
      - **Economic Interpretation:**
        - High volatility may drive investors toward safe-haven assets like gold.
    - **Gold Momentum Signal (`gold_momentum_signal`):**
      - **Calculation:**
        - Gold price momentum over 250 trading days.
      - **Signal Meanings:**
        - **1 (Buy):** Positive momentum.
        - **0 (Hold):** Negative momentum.
      - **Economic Interpretation:**
        - Positive momentum suggests a continuing upward trend in gold prices.
    - **Combined Gold Signal (`combined_gold_signal`):**
      - **Logic:**
        - Buy when at least two signals are positive.
        - Sell when all signals are negative.
        - Hold otherwise.

#### **3. Adjusting Weights Based on Signals**

- **Adjustment Logic:**
  - **Adjustment Factor:** +/- 30% of the baseline weight.
  - **Stock Assets:**
    - Increase weights if `combined_stock_signal` is positive.
    - Decrease weights if negative.
  - **Gold Assets:**
    - Increase weights if `combined_gold_signal` is positive.
    - Decrease weights if negative.
- **Normalization:**
  - After adjustments, weights are normalized to sum to 1.

#### **4. Strategy Execution Flow**

- **Rebalancing:**
  - Weights are adjusted at each rebalancing date based on the signals.
  - Between rebalancing dates, the weights remain constant.
- **Baseline Risk Budgeting:**
  - Provides a systematic approach to allocate capital based on risk contributions.
- **Signal Integration:**
  - Allows for tactical adjustments to the strategic allocation based on market signals.
"""


class SignalBasedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def run_strategy(self, data_dict, start_date, end_date, parameters):
        """
        Run the signal-based strategy.

        :param data_dict: Dictionary containing 'close_prices', 'edb_data', 'composite_data'
        :param start_date: Strategy start date (string or datetime)
        :param end_date: Strategy end date (string or datetime)
        :param parameters: Dictionary of strategy parameters
        :return: Evaluator instance with performance metrics
        """
        price_data = data_dict['close_prices']
        self.price_data_full = price_data
        edb_data = data_dict['edb_data']
        composite_data = data_dict['composite_data']
        asset_class_mapping = parameters['asset_class_mapping']
        risk_budget = parameters['risk_budget']
        selected_assets = list(risk_budget.keys())
        self.all_assets = price_data.columns.tolist()
        rebalance_frequency = parameters.get('rebalance_frequency', 'M')

        # Determine budget type
        budget_type = self.check_budget_type(risk_budget, asset_class_mapping)

        # Generate rebalancing dates
        date_index = price_data.loc[start_date:end_date].index
        rebalance_dates = date_index.to_series().resample(rebalance_frequency).last().dropna()

        weights_history = pd.DataFrame(index=date_index, columns=selected_assets)
        previous_weights = None

        # Generate signals
        self.generate_signals(price_data, edb_data, composite_data)

        print(f"\nStarting strategy from {start_date} to {end_date}")
        for i, date in enumerate(rebalance_dates):
            print(f"Processing date: {date.strftime('%Y-%m-%d')} ({i + 1}/{len(rebalance_dates)})")

            # Adjust weights based on signals
            weights = self.adjust_weights_based_on_signals(
                date, selected_assets, asset_class_mapping, risk_budget, budget_type, previous_weights
            )

            previous_weights = weights

            # Apply weights until the next rebalance date
            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates.iloc[i + 1]
            else:
                next_date = date_index[-1] + pd.Timedelta(days=1)

            weight_dates = date_index[(date_index >= date) & (date_index < next_date)]
            weights_history.loc[weight_dates] = weights.values

        # Handle initial period before the first rebalancing date
        first_rebalance_date = rebalance_dates.iloc[0]
        initial_dates = date_index[date_index < first_rebalance_date]
        if not initial_dates.empty:
            if previous_weights is not None:
                weights_history.loc[initial_dates] = previous_weights.values
            else:
                equal_weights = pd.Series(1.0 / len(selected_assets), index=price_data.columns)
                weights_history.loc[initial_dates] = equal_weights.values

        # Forward-fill any remaining NaN weights
        weights_history.ffill(inplace=True)

        # Initialize evaluator
        evaluator = Evaluator('SignalBasedStrategy', price_data.loc[start_date:end_date], weights_history)
        return evaluator

    def generate_signals(self, price_data, edb_data, composite_data):
        """
        Generate signals required for the strategy.

        :param price_data: DataFrame of asset close prices
        :param edb_data: DataFrame of economic data
        :param composite_data: DataFrame of composite data (e.g., volume, PE ratios)
        """
        # Initialize signal generators with required data
        self.stock_signal_generator = StockSignalGenerator(
            stock_prices=price_data['中证800'],
            stock_volume=composite_data['成交额\n[单位]亿元'],
            pe_ttm=composite_data['市盈率PE(TTM)'],
            bond_yields=edb_data['中债国债到期收益率:10年']
        )
        self.gold_signal_generator = GoldSignalGenerator(
            tips_10y=edb_data['美国:国债实际收益率:10年'],
            vix=price_data['CBOE波动率'],
            gold_prices=price_data['SHFE黄金']
        )

        # Generate signals
        self.erp_signal = self.stock_signal_generator.erp_signal()
        self.volume_ma_signal = self.stock_signal_generator.volume_ma_signal()
        self.combined_stock_signal = self.stock_signal_generator.combined_stock_signal(
            erp_signal=self.erp_signal,
            volume_ma_signal=self.volume_ma_signal
        )

        self.us_tips_signal = self.gold_signal_generator.us_tips_signal()
        self.vix_signal = self.gold_signal_generator.vix_signal()
        self.gold_momentum_signal = self.gold_signal_generator.gold_momentum_signal()
        self.combined_gold_signal = self.gold_signal_generator.combined_gold_signal(
            us_tips_signal=self.us_tips_signal,
            vix_signal=self.vix_signal,
            gold_momentum_signal=self.gold_momentum_signal
        )

    def adjust_weights_based_on_signals(self, current_date, selected_assets, asset_class_mapping,
                                        risk_budget, budget_type, previous_weights):
        """
        Adjust weights based on signals at the current date.

        :param current_date: Date for which to adjust weights
        :param selected_assets: List of asset names
        :param asset_class_mapping: Dictionary mapping assets to asset classes
        :param risk_budget: Dictionary of risk budgets
        :param budget_type: 'asset_budget' or 'class_budget'
        :param previous_weights: Previous weights (Series)
        :return: Adjusted weights (Series)
        """
        # Baseline allocation using risk budgeting with actual covariance matrix
        lookback_period = 252  # One year of trading days

        available_dates = self.price_data_full.index
        if current_date not in available_dates:
            current_date = available_dates[available_dates.get_loc(current_date, method='ffill')]

        current_idx = available_dates.get_loc(current_date)
        start_idx = current_idx - lookback_period
        if start_idx < 0:
            print(f"  Not enough data for covariance calculation at {current_date}.")
            # Use previous weights or equal weights
            if previous_weights is not None:
                weights = previous_weights
            else:
                weights = pd.Series(1.0 / len(self.all_assets), index=self.all_assets)
            return weights

        # Get historical price data
        price_data_window = self.price_data_full.iloc[start_idx:current_idx][selected_assets]
        daily_returns = price_data_window.pct_change().dropna()

        # Compute covariance matrix
        cov_matrix = daily_returns.cov()

        # Risk budgeting allocation
        weights = self.risk_budget_allocation(cov_matrix, risk_budget)

        # Adjust weights based on signals
        adjustment_factor = 0.3  # Adjust by +/-30%
        # Stock signal
        stock_signal_value = self.combined_stock_signal.loc[
            current_date] if current_date in self.combined_stock_signal.index else 0
        # Gold signal
        gold_signal_value = self.combined_gold_signal.loc[
            current_date] if current_date in self.combined_gold_signal.index else 0

        weights = weights.copy()

        # Adjust stock weights
        stock_assets = [asset for asset, cls in asset_class_mapping.items() if
                        cls == 'Equity' and asset in weights.index]
        if stock_signal_value == 1:
            weights[stock_assets] *= (1 + adjustment_factor)
            print(f"  Increased weights for stock assets due to positive signal")
        elif stock_signal_value == -1:
            weights[stock_assets] *= (1 - adjustment_factor)
            print(f"  Decreased weights for stock assets due to negative signal")

        # Adjust gold weights
        gold_assets = [asset for asset, cls in asset_class_mapping.items() if
                       cls == 'Commodity' and asset in weights.index]
        if gold_signal_value == 1:
            weights[gold_assets] *= (1 + adjustment_factor)
            print(f"  Increased weights for gold assets due to positive signal")
        elif gold_signal_value == -1:
            weights[gold_assets] *= (1 - adjustment_factor)
            print(f"  Decreased weights for gold assets due to negative signal")

        # Normalize weights
        weights = weights / weights.sum()

        return weights


# Signal Generators
class StockSignalGenerator:
    def __init__(self, stock_prices, stock_volume, pe_ttm, bond_yields):
        self.stock_prices = stock_prices.sort_index()
        self.stock_volume = stock_volume.sort_index()
        self.pe_ttm = pe_ttm.sort_index()
        self.bond_yields = bond_yields.sort_index()
        self.calculate_erp()

    def calculate_erp(self):
        """
        Calculate Equity Risk Premium (ERP).

        ERP = Earnings Yield - Bond Yield
            = (1 / PE Ratio) - Bond Yield
        """
        self.erp = (1 / self.pe_ttm) - self.bond_yields

    def erp_signal(self, window=120, upper_quantile=0.7, lower_quantile=0.3):
        """
        Generate ERP signal based on weighted ERP quantiles.

        Signal meanings:
        - 1: Buy signal (ERP above upper quantile)
        - -1: Sell signal (ERP below lower quantile)
        - 0: Hold signal (ERP between quantiles)
        """
        erp_weighted = self.erp.ewm(span=window).mean()
        lower_erp_quantile = erp_weighted.rolling(window=252 * 5).quantile(lower_quantile)
        upper_erp_quantile = erp_weighted.rolling(window=252 * 5).quantile(upper_quantile)

        signal = pd.Series(0, index=erp_weighted.index)
        signal[erp_weighted > upper_erp_quantile] = 1
        signal[erp_weighted < lower_erp_quantile] = -1
        return signal.dropna()

    def volume_ma_signal(self, ma_window=5, volume_window=5):
        """
        Generate combined signal based on price MA and volume MA.

        Signal meanings:
        - 1: Buy signal (price above MA and volume above MA)
        - -1: Sell signal (otherwise)
        """
        # Price MA signal
        price_ma = self.stock_prices.rolling(window=ma_window).mean()
        ma_signal = pd.Series(np.where(self.stock_prices > price_ma, 1, -1), index=self.stock_prices.index)

        # Volume MA signal
        volume_ma = self.stock_volume.rolling(window=volume_window).mean()
        volume_signal = pd.Series(np.where(self.stock_volume > volume_ma, 1, -1), index=self.stock_volume.index)

        # Combined signal
        combined_signal = pd.Series(np.where((ma_signal == 1) & (volume_signal == 1), 1, -1),
                                    index=self.stock_prices.index)
        return combined_signal.dropna()

    def combined_stock_signal(self, erp_signal, volume_ma_signal):
        """
        Combine ERP signal and volume MA signal.

        Signal meanings:
        - 1: Buy signal (both ERP and volume MA signals are positive)
        - -1: Sell signal (both signals are negative)
        - 0: Hold signal (signals disagree)
        """
        combined_signal = pd.Series(0, index=erp_signal.index)
        combined_signal[(erp_signal == 1) & (volume_ma_signal == 1)] = 1
        combined_signal[(erp_signal == -1) & (volume_ma_signal == -1)] = -1
        return combined_signal.dropna()


class GoldSignalGenerator:
    def __init__(self, tips_10y, vix, gold_prices):
        self.tips_10y = tips_10y.sort_index()
        self.vix = vix.sort_index()
        self.gold_prices = gold_prices.sort_index()

    def us_tips_signal(self, window=20):
        """
        Generate signal based on US 10-year TIPS yield moving average.

        Signal meanings:
        - 1: Buy signal (TIPS yield below MA)
        - -1: Sell signal (TIPS yield above MA)
        """
        ma = self.tips_10y.rolling(window=window).mean()
        signal = pd.Series(np.where(self.tips_10y < ma, 1, -1), index=self.tips_10y.index)
        return signal.dropna()

    def vix_signal(self, window=15, threshold=20):
        """
        Generate signal based on VIX index.

        Signal meanings:
        - 1: Buy signal (max VIX over window exceeds threshold)
        - 0: Hold signal (max VIX below threshold)
        """
        vix_max = self.vix.rolling(window=window).max()
        signal = pd.Series(np.where(vix_max > threshold, 1, 0), index=self.vix.index)
        return signal.dropna()

    def gold_momentum_signal(self, window=250):
        """
        Generate signal based on gold momentum over the given window.

        Signal meanings:
        - 1: Buy signal (positive momentum)
        - 0: Hold signal (negative momentum)
        """
        momentum = (self.gold_prices / self.gold_prices.shift(window) - 1)
        signal = pd.Series(np.where(momentum > 0, 1, 0), index=self.gold_prices.index)
        return signal.dropna()

    def combined_gold_signal(self, us_tips_signal, vix_signal, gold_momentum_signal):
        """
        Combine US TIPS signal, VIX signal, and gold momentum signal.

        Signal meanings:
        - 1: Buy signal (at least two positive signals)
        - -1: Sell signal (all signals negative)
        - 0: Hold signal (signals disagree)
        """
        aligned_signals = us_tips_signal.align(vix_signal, join='inner')[0].align(gold_momentum_signal, join='inner')[0]
        combined = us_tips_signal + vix_signal + gold_momentum_signal
        combined_signal = pd.Series(0, index=combined.index)
        combined_signal[combined >= 2] = 1
        combined_signal[combined < 0] = -1
        return combined_signal.dropna()
