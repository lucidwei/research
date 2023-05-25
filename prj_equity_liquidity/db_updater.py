# coding=gbk
# Time Created: 2023/5/25 9:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from utils import timeit, get_nearest_dates_from_contract, check_wind
from WindPy import w


class DatabaseUpdater(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_dates()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str

    def update_reopened_funds(self):
        """
        1.���product_static_info�д��ڵĻ�����Ϣ��¼����ȡ��Ҫ���µ�����
            - product_static_info��internal_id��Ϊmarkets_daily_long��product_static_info_id�е����
            - ��ѯmarkets_daily_long����field��Ϊstartdate��Ӧ��value�е����ֵ����Сֵ����ȡ�������ݵ����䣨��ð�װ��һ���ຯ�����Ա��ظ�ʹ�ã�
            - ����������ݵ�����Ϊ�գ�missing_datesΪself.all_dates
            - self.all_dates[0]���������ݵ����������Ϊ��һ��missing_dates���������ݵ���������޵�self.all_dates[-1]Ϊ�ڶ���missing_dates
        2.����missing_datesִ��w.wset�������ݣ����ڿ�missing_dates����ִ��w.wset
        3.�������صõ������ݽ������ϴ������ݿ�
            - windcode�ϴ���product_static_info��code��
            - name�ϴ���product_static_info��chinese_name��
            - buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate��Щ��(��Щ�ַ�����Ϊfield)�ϴ���markets_daily_long���Գ���ʽ���ݴ��档
            - markets_daily_long��ĳ���ʽ���ݰ��������У�date��product_name��field��value��product_static_info_id
        """
        # ��ȡ��Ҫ���µ���������
        date_range = self.get_existing_dates_from_db('markets_daily_long', field='startdate')

        if date_range is None:
            missing_dates = self.all_dates
        else:
            missing_dates = self.get_missing_dates(all_dates=self.all_dates, existing_dates=date_range)

        if not missing_dates:
            print("No missing dates for update_reopened_funds")
            return

        # ִ����������
        downloaded_df = None
        if missing_dates:
            start_date = missing_dates[0]
            end_date = missing_dates[-1]
            downloaded_df = w.wset("fundissuegeneralview",
                                   f"startdate={start_date};enddate={end_date};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate")

        if downloaded_df is None or downloaded_df.Data is None or not downloaded_df.Data:
            print("No data downloaded for update_reopened_funds")
            return

        # �������ص����ݲ��ϴ������ݿ�
        windcode = downloaded_df.Data[0]
        name = downloaded_df.Data[1]
        buystartdate = downloaded_df.Data[2]
        issueshare = downloaded_df.Data[3]
        fundfounddate = downloaded_df.Data[4]
        closemonth = downloaded_df.Data[5]
        openbuystartdate = downloaded_df.Data[6]
        openrepurchasestartdate = downloaded_df.Data[7]

        # �ϴ���product_static_info��
        self.upload_to_product_static_info(windcode, name)

        # �ϴ���markets_daily_long��
        self.upload_to_markets_daily_long('buystartdate', buystartdate)
        self.upload_to_markets_daily_long('issueshare', issueshare)
        self.upload_to_markets_daily_long('fundfounddate', fundfounddate)
        self.upload_to_markets_daily_long('closemonth', closemonth)
        self.upload_to_markets_daily_long('openbuystartdate', openbuystartdate)
        self.upload_to_markets_daily_long('openrepurchasestartdate', openrepurchasestartdate)
        # ���Ϲ���ʼ����Ϊɸѡ������ѡȡ�����ݸ�����������ǰհ�ԡ�ֻѡȡ�ϸ������ϵ��·�����
        downloaded_df = w.wset("fundissuegeneralview",
               f"startdate={self.all_dates_str[0]};enddate={self.all_dates_str[-1]};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate")
