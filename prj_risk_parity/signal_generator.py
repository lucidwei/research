# coding=gbk
# Time Created: 2023/4/27 16:15
# Author  : Lucid
# FileName: signal_generator.py
# Software: PyCharm
import numpy as np
import pandas as pd
from utils_risk_parity import align_signals


class StockSignalGenerator:

    def __init__(self, stock_prices, stock_volume, pe_ttm, bond_yields):
        self.stock_prices = stock_prices.dropna().sort_index()
        self.stock_volume = stock_volume.dropna().sort_index()
        self.pe_ttm = pe_ttm.dropna().sort_index()
        self.bond_yields = bond_yields.dropna().sort_index()
        self.calculate_erp()

    def calculate_erp(self):
        # 计算 ERP
        self.erp = (1 / self.pe_ttm) - self.bond_yields

    def erp_signal(self, window=120, upper_quantile=0.7, lower_quantile=0.3):
        # 计算加权 ERP 分位数
        erp_weighted = self.erp.ewm(span=window).mean()

        # 计算 5 年分位数
        lower_erp_quantile = erp_weighted.rolling(window=252 * 5).quantile(lower_quantile)
        upper_erp_quantile = erp_weighted.rolling(window=252 * 5).quantile(upper_quantile)

        # 生成信号
        signal = pd.Series(index=self.erp.index, dtype='float64')
        signal[(erp_weighted > upper_erp_quantile)] = 1
        signal[(erp_weighted < lower_erp_quantile)] = -1
        signal[(erp_weighted >= lower_erp_quantile) & (erp_weighted <= upper_erp_quantile)] = 0

        return signal

    def ma_signal(self, window=5):
        ma = self.stock_prices.rolling(window=window).mean()
        signal = pd.Series(np.where(self.stock_prices > ma, 1, -1), index=self.stock_prices.index)
        return signal

    def volume_signal(self, window=5):
        mean_volume = self.stock_volume.rolling(window=window).mean()
        signal = pd.Series(np.where(self.stock_volume > mean_volume, 1, -1), index=self.stock_volume.index)
        return signal

    def volume_ma_signal(self, ma_window=5, volume_window=5):
        ma_signal = self.ma_signal(ma_window)
        volume_signal = self.volume_signal(volume_window)
        combined_signal = pd.Series(np.where((ma_signal == 1) & (volume_signal == 1), 1, -1),
                                    index=self.stock_prices.index)
        return combined_signal

    def combined_stock_signal(self, erp_window=120, erp_upper_quantile=0.7, erp_lower_quantile=0.3, ma_window=5,
                              volume_window=5):
        erp_signal = self.erp_signal(erp_window, erp_upper_quantile, erp_lower_quantile)
        volume_ma_signal = self.volume_ma_signal(ma_window, volume_window)
        erp_signal = erp_signal[volume_ma_signal.index]

        combined_signal = pd.Series(index=volume_ma_signal.index, dtype='float64')

        # 如果 ERP 和量价信号都发出看多信号，发出加仓信号
        combined_signal[(erp_signal == 1) & (volume_ma_signal == 1)] = 1

        # 如果 ERP 和量价信号都发出看空信号，发出减仓信号
        combined_signal[(erp_signal == -1) & (volume_ma_signal == -1)] = -1

        # 其他情况不发出信号
        combined_signal[(erp_signal != volume_ma_signal)] = 0

        return combined_signal.dropna()


class GoldSignalGenerator:

    def __init__(self, tips_10y, vix, gold_prices):
        self.tips_10y = tips_10y.dropna().sort_index()
        self.vix = vix.dropna().sort_index()
        self.gold_prices = gold_prices.dropna().sort_index()

    def us_tips_signal(self, window=20):
        ma = self.tips_10y.rolling(window=window).mean()
        signal = pd.Series(np.where(self.tips_10y > ma, -1, 1), index=self.tips_10y.index)
        return signal

    def vix_signal(self, window=15, threshold=20):
        vix_max = self.vix.rolling(window=window).max().squeeze()
        signal = pd.Series(np.where(vix_max > threshold, 1, 0), index=self.vix.index)
        return signal

    def gold_momentum_signal(self, window=250):
        gold_prices = self.gold_prices
        gold_momentum = (gold_prices / gold_prices.shift(window) - 1).squeeze()
        signal = pd.Series(np.where(gold_momentum > 0, 1, 0), index=gold_prices.index)
        return signal

    def combined_gold_signal(self, us_tips_window=20, vix_window=15, vix_threshold=20, gold_momentum_window=250):
        us_tips_signal = self.us_tips_signal(us_tips_window)
        vix_signal = self.vix_signal(vix_window, vix_threshold)
        gold_momentum_signal = self.gold_momentum_signal(gold_momentum_window)

        aligned_signals = align_signals(us_tips_signal=us_tips_signal, vix_signal=vix_signal)
        vix_signal = aligned_signals['vix_signal']
        us_tips_signal = aligned_signals['us_tips_signal']

        combined_signal = us_tips_signal + vix_signal + gold_momentum_signal
        signal = pd.Series(index=combined_signal.index, dtype='float64')
        signal[combined_signal >= 2] = 1
        signal[combined_signal < 0] = -1
        signal[(combined_signal >= 0) & (combined_signal < 2)] = 0

        return signal.dropna()
