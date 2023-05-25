# coding=gbk
# Time Created: 2023/5/25 9:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from utils import timeit, get_nearest_dates_from_contract, check_wind
from WindPy import w


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_dates()
        self.update_all_funds_info()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str

    def update_all_funds_info(self):
        """
        1.检查product_static_info中存在的基金信息记录，获取需要更新的日期
            - product_static_info的internal_id列为markets_daily_long表product_static_info_id列的外键，将两个表连接起来，然后通过type=fund筛选出关注的行
            - 针对筛选出的数据行，查询markets_daily_long表中field列为startdate对应的value列的最大值和最小值，获取存在数据的区间
            - 如果存在数据的区间为空，missing_dates为self.all_dates
            - self.all_dates[0]到存在数据的区间的下限为第一段missing_dates，存在数据的区间的上限到self.all_dates[-1]为第二段missing_dates
        2.根据missing_dates执行w.wset下载数据，对于空missing_dates跳过执行w.wset
        3.利用下载得到的数据将数据上传至数据库
            - windcode上传至product_static_info的code列
            - name上传至product_static_info的chinese_name列
            - buystartdate,issueshare,fundfounddate,openbuystartdate,openrepurchasestartdate这些列(这些字符串作为field)上传至markets_daily_long表，以长格式数据储存。
            - markets_daily_long表的长格式数据包括以下列：date，product_name，field，value，product_static_info_id
        """
        # 获取需要更新的日期区间
        existing_dates = self.select_column_from_joined_table(
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column='date_value',
            filter_condition="product_static_info.type = 'fund'"
        )

        if len(existing_dates) == 0:
            missing_dates = self.all_dates
        else:
            missing_dates = self.get_missing_dates(all_dates=self.all_dates, existing_dates=existing_dates)

        if not missing_dates:
            print("No missing dates for update_all_funds_info")
            return

        # 执行数据下载
        date_start = missing_dates[0]
        date_end = missing_dates[-1]
        # 以认购起始日作为筛选条件，选取的数据更完整、更有前瞻性。只选取严格意义上的新发基金。
        downloaded_df = w.wset("fundissuegeneralview",
                               f"startdate={date_start};enddate={date_end};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,openbuystartdate,openrepurchasestartdate",
                               usedf=True)[1]

        if downloaded_df.empty:
            print(f"Missing dates from {date_start} and {date_end}, but no data downloaded for update_all_funds_info")
            return

        # 解析下载的数据并上传至数据库
        product_metric_upload_df = downloaded_df[['windcode', 'name', 'buystartdate', 'fundfounddate', 'issueshare']].rename(
            columns={'windcode': 'code', 'name': 'chinese_name'})
        # 添加source和type列并上传
        product_metric_upload_df['english_name'] = ''
        product_metric_upload_df['source'] = 'wind'
        product_metric_upload_df['type'] = 'fund'
        for _, row in product_metric_upload_df.iterrows():
            self.insert_product_static_info(row)

        markets_daily_long_upload_df = downloaded_df[
            ['name', 'openbuystartdate', 'openrepurchasestartdate']].rename(
            columns={'name': 'chinese_name'})
        markets_daily_long_upload_df = markets_daily_long_upload_df.melt(id_vars=['chinese_name'], var_name='field',
                                                                           value_name='date_value')
        markets_daily_long_upload_df = markets_daily_long_upload_df.dropna(subset=['date_value']).rename(
            columns={'chinese_name': 'product_name'})
        markets_daily_long_upload_df['date'] = self.all_dates[-1]

        # 上传前要剔除已存在的product
        existing_products = self.select_column_from_joined_table(
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column='chinese_name',
            filter_condition="product_static_info.type = 'fund'"
        )
        filtered_df = markets_daily_long_upload_df[
            ~markets_daily_long_upload_df['product_name'].isin(existing_products)]
        filtered_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

