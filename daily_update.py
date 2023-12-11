# coding=gbk
# Time Created: 2023/8/7 16:42
# Author  : Lucid
# FileName: daily_update.py
# Software: PyCharm
from base_config import BaseConfig
from prj_equity_liquidity.db_updater import DatabaseUpdater as LiquidityDatabaseUpdater
from prj_equity_sentiment.db_updater import DatabaseUpdater as SentimentDatabaseUpdater
from prj_equity_sentiment.processor import Processor

base_config_liquidity = BaseConfig('equity_liquidity')
data_updater = LiquidityDatabaseUpdater(base_config_liquidity)
# 仅运行部分更新逻辑
data_updater.all_funds_info_updater.update_all_funds_info()
data_updater.etf_lof_updater.logic_etf_lof_funds()
# 北向晚上8点还没出，估计也得第二天出
data_updater.north_inflow_updater.logic_north_inflow_by_industry()
data_updater.margin_trade_by_industry_updater.logic_margin_trade_by_industry()

base_config_sentiment = BaseConfig('equity_sentiment')
SentimentDatabaseUpdater(base_config_sentiment)
processor = Processor(base_config_sentiment)
processor.upload_indicators()

