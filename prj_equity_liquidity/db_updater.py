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
        1.检查product_static_info中存在的基金信息记录，获取需要更新的日期
            - product_static_info的internal_id列为markets_daily_long表product_static_info_id列的外键
            - 查询markets_daily_long表中field列为startdate对应的value列的最大值和最小值，获取存在数据的区间（最好包装成一个类函数，以便重复使用）
            - 如果存在数据的区间为空，missing_dates为self.all_dates
            - self.all_dates[0]到存在数据的区间的下限为第一段missing_dates，存在数据的区间的上限到self.all_dates[-1]为第二段missing_dates
        2.根据missing_dates执行w.wset下载数据，对于空missing_dates跳过执行w.wset
        3.利用下载得到的数据将数据上传至数据库
            - windcode上传至product_static_info的code列
            - name上传至product_static_info的chinese_name列
            - buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate这些列(这些字符串作为field)上传至markets_daily_long表，以长格式数据储存。
            - markets_daily_long表的长格式数据包括以下列：date，product_name，field，value，product_static_info_id
        """
        # 获取需要更新的日期区间
        date_range = self.get_existing_dates_from_db('markets_daily_long', field='startdate')

        if date_range is None:
            missing_dates = self.all_dates
        else:
            missing_dates = self.get_missing_dates(all_dates=self.all_dates, existing_dates=date_range)

        if not missing_dates:
            print("No missing dates for update_reopened_funds")
            return

        # 执行数据下载
        downloaded_df = None
        if missing_dates:
            start_date = missing_dates[0]
            end_date = missing_dates[-1]
            downloaded_df = w.wset("fundissuegeneralview",
                                   f"startdate={start_date};enddate={end_date};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate")

        if downloaded_df is None or downloaded_df.Data is None or not downloaded_df.Data:
            print("No data downloaded for update_reopened_funds")
            return

        # 解析下载的数据并上传至数据库
        windcode = downloaded_df.Data[0]
        name = downloaded_df.Data[1]
        buystartdate = downloaded_df.Data[2]
        issueshare = downloaded_df.Data[3]
        fundfounddate = downloaded_df.Data[4]
        closemonth = downloaded_df.Data[5]
        openbuystartdate = downloaded_df.Data[6]
        openrepurchasestartdate = downloaded_df.Data[7]

        # 上传至product_static_info表
        self.upload_to_product_static_info(windcode, name)

        # 上传至markets_daily_long表
        self.upload_to_markets_daily_long('buystartdate', buystartdate)
        self.upload_to_markets_daily_long('issueshare', issueshare)
        self.upload_to_markets_daily_long('fundfounddate', fundfounddate)
        self.upload_to_markets_daily_long('closemonth', closemonth)
        self.upload_to_markets_daily_long('openbuystartdate', openbuystartdate)
        self.upload_to_markets_daily_long('openrepurchasestartdate', openrepurchasestartdate)
        # 以认购起始日作为筛选条件，选取的数据更完整、更有前瞻性。只选取严格意义上的新发基金。
        downloaded_df = w.wset("fundissuegeneralview",
               f"startdate={self.all_dates_str[0]};enddate={self.all_dates_str[-1]};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,closemonth,openbuystartdate,openrepurchasestartdate")
