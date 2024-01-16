# coding=gbk
# Time Created: 2023/8/7 16:42
# Author  : Lucid
# FileName: daily_update.py
# Software: PyCharm
import datetime
from base_config import BaseConfig
from prj_equity_liquidity.db_updater import DatabaseUpdater as LiquidityDatabaseUpdater
from prj_equity_sentiment.db_updater import DatabaseUpdater as SentimentDatabaseUpdater
from prj_equity_sentiment.processor import Processor

# ��ȡ��ǰʱ��
current_time = datetime.datetime.now()
# ������ϣ���Ƚϵ�ʱ�䣨����9�㣩
target_time = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

base_config_liquidity = BaseConfig('equity_liquidity')
data_updater = LiquidityDatabaseUpdater(base_config_liquidity)


# �Ƚϵ�ǰʱ���Ƿ����ڻ��������9�㣬9��֮���������
if current_time >= target_time:
    data_updater.margin_trade_by_industry_updater.logic_margin_trade_by_industry()
else:
    # �����в��ָ����߼�
    data_updater.all_funds_info_updater.update_all_funds_info()
    data_updater.etf_lof_updater.logic_etf_lof_funds()
    # ��������8�㻹û��������Ҳ�õڶ����
    data_updater.north_inflow_updater.logic_north_inflow_by_industry()

    base_config_sentiment = BaseConfig('equity_sentiment')
    SentimentDatabaseUpdater(base_config_sentiment).run_all_updater()
    processor = Processor(base_config_sentiment)
    processor.upload_indicators()