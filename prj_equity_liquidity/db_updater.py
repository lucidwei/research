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
        1.���product_static_info�д��ڵĻ�����Ϣ��¼����ȡ��Ҫ���µ�����
            - product_static_info��internal_id��Ϊmarkets_daily_long��product_static_info_id�е������������������������Ȼ��ͨ��type=fundɸѡ����ע����
            - ���ɸѡ���������У���ѯmarkets_daily_long����field��Ϊstartdate��Ӧ��value�е����ֵ����Сֵ����ȡ�������ݵ�����
            - ����������ݵ�����Ϊ�գ�missing_datesΪself.all_dates
            - self.all_dates[0]���������ݵ����������Ϊ��һ��missing_dates���������ݵ���������޵�self.all_dates[-1]Ϊ�ڶ���missing_dates
        2.����missing_datesִ��w.wset�������ݣ����ڿ�missing_dates����ִ��w.wset
        3.�������صõ������ݽ������ϴ������ݿ�
            - windcode�ϴ���product_static_info��code��
            - name�ϴ���product_static_info��chinese_name��
            - buystartdate,issueshare,fundfounddate,openbuystartdate,openrepurchasestartdate��Щ��(��Щ�ַ�����Ϊfield)�ϴ���markets_daily_long���Գ���ʽ���ݴ��档
            - markets_daily_long��ĳ���ʽ���ݰ��������У�date��product_name��field��value��product_static_info_id
        """
        # ��ȡ��Ҫ���µ���������
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

        # ִ����������
        date_start = missing_dates[0]
        date_end = missing_dates[-1]
        # ���Ϲ���ʼ����Ϊɸѡ������ѡȡ�����ݸ�����������ǰհ�ԡ�ֻѡȡ�ϸ������ϵ��·�����
        downloaded_df = w.wset("fundissuegeneralview",
                               f"startdate={date_start};enddate={date_end};datetype=startdate;isvalid=yes;deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,openbuystartdate,openrepurchasestartdate",
                               usedf=True)[1]

        if downloaded_df.empty:
            print(f"Missing dates from {date_start} and {date_end}, but no data downloaded for update_all_funds_info")
            return

        # �������ص����ݲ��ϴ������ݿ�
        product_metric_upload_df = downloaded_df[['windcode', 'name', 'buystartdate', 'fundfounddate', 'issueshare']].rename(
            columns={'windcode': 'code', 'name': 'chinese_name'})
        # ���source��type�в��ϴ�
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

        # �ϴ�ǰҪ�޳��Ѵ��ڵ�product
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

