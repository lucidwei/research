# coding=gbk
# Time Created: 2023/4/26 8:11
# Author  : Lucid
# FileName: db_interact.py
# Software: PyCharm
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase


class DatabaseReader(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.read_data()
        self.initialize_data()

    def read_data(self):
        # 使用 `read_from_high_freq_view` 方法读取数据
        high_freq_view_data = self.read_from_high_freq_view(['S0059749', 'G0005428'])

        # 使用 `read_from_markets_daily_long` 方法读取数据
        vix_data = self.read_from_markets_daily_long("VIX.GI", "close")
        cba_data = self.read_from_markets_daily_long("CBA00301.CS", "close")
        csi_data = self.read_from_markets_daily_long("000906.SH", "close,volume,pe_ttm")
        gold_data = self.read_from_markets_daily_long("AU9999.SGE", "close")

        # 将数据保存为字典
        self.data_dict = {
            'high_freq_view': high_freq_view_data,
            'vix': vix_data,
            'cba': cba_data,
            'csi': csi_data,
            'gold': gold_data,
        }

    def initialize_data(self):
        self.data_easy_dict = {
            'stock_prices': self.data_dict['csi']['close'],
            'stock_volume': self.data_dict['csi']['volume'],
            'pe_ttm': self.data_dict['csi']['pe_ttm'],
            'bond_yields': self.data_dict['high_freq_view']['china_t_yield'],
            'tips_10y': self.data_dict['high_freq_view']['us_tips_10y'],
            'vix': self.data_dict['vix'],
            'gold_prices': self.data_dict['gold'].squeeze(),

        }


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_high_freq_by_edb_id('S0059749')
        self.update_high_freq_by_edb_id('G0005428')
        self.update_markets_daily_by_wsd_id_fields("VIX.GI", "close")
        self.update_markets_daily_by_wsd_id_fields("CBA00301.CS", "close")
        self.update_markets_daily_by_wsd_id_fields("000906.SH", "close,volume,pe_ttm")
        self.update_markets_daily_by_wsd_id_fields("AU9999.SGE", "close")
        # self.set_all_nan_to_null()
        self.close()