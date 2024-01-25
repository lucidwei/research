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
# �����������ʱ��
rongzi_time = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
# ����ʱ��
close_time = datetime.datetime.now().replace(hour=15, minute=10, second=0, microsecond=0)

base_config_liquidity = BaseConfig('equity_liquidity')
data_updater = LiquidityDatabaseUpdater(base_config_liquidity)


# �Ƚϵ�ǰʱ���Ƿ����ڻ��������9�㣬9��֮���������
# ÿ������������ݣ���䣬9������̺�
if current_time < rongzi_time:
    print('����ETF�ͱ���')
    data_updater.all_funds_info_updater.update_all_funds_info()
    data_updater.etf_lof_updater.logic_etf_lof_funds()
    # ��������8�㻹û��������Ҳ�õڶ����
    data_updater.north_inflow_updater.logic_north_inflow_by_industry()
elif rongzi_time < current_time < close_time:
    print('��������')
    data_updater.margin_trade_by_industry_updater.logic_margin_trade_by_industry()
elif current_time > close_time:
    print('���¸������ݺ�����ָ��')
    base_config_sentiment = BaseConfig('equity_sentiment')
    SentimentDatabaseUpdater(base_config_sentiment).run_all_updater()
    processor = Processor(base_config_sentiment)
    processor.upload_indicators()
else:
    print('��Ӧ����������ʱ��')
