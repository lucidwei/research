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
        self.update_funds_name()
        self.update_reopened_dk_funds()
        self.update_reopened_cyq_funds()
        # self.update_etf_lof_funds()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str

    def update_reopened_dk_funds(self):
        """
        1. ��ȡfull_name�д���'���ڿ���'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        dk_funds_df = self.select_rows_by_column_strvalue(table_name='product_static_info', column_name='fund_fullname',
                                                          search_value='���ڿ���', selected_columns=['code', 'chinese_name', 'fund_fullname'],
                                                          filter_condition="type='fund' AND fund_fullname NOT LIKE '%ծ%'")
        dk_funds_df = dk_funds_df[~dk_funds_df['fund_fullname'].str.contains('ծ')]

        today_updated = self.is_markets_daily_long_updated_today('����')
        if not today_updated:
            for _, row in dk_funds_df.iterrows():
                downloaded = w.wsd(row['code'],
                                   "NAV_adj,fund_expectedopenday,netasset_total,fund_fundscale,fund_info_name",
                                   self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
                downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'NAV_ADJ': 'nav_adj',
                                                                      'FUND_EXPECTEDOPENDAY': 'fund_expectedopenday',
                                                                      'FUND_FUNDSCALE': 'fund_fundscale',
                                                                      'NETASSET_TOTAL': 'netasset_total',
                                                                      'FUND_INFO_NAME': 'product_name'})
                upload_date_value = downloaded[['product_name', 'fund_expectedopenday']].melt(id_vars=['product_name'], var_name='field', value_name='date_value')
                upload_value = downloaded[['product_name', 'nav_adj', 'fund_fundscale', 'netasset_total']].melt(id_vars=['product_name'], var_name='field', value_name='value')
                upload_date_value['date'] = self.all_dates[-1]
                upload_value['date'] = self.all_dates[-1]
                upload_date_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)
                upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

        self.upload_joined_products_wide_table(full_name_keyword='���ڿ���')

    def update_reopened_cyq_funds(self):
        """
        1. ��ȡfull_name�д���'������'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        cyq_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='������', selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare'],
            filter_condition="type='fund' AND fund_fullname NOT LIKE '%ծ%'")

        today_updated = self.is_markets_daily_long_updated_today('����')
        if not today_updated:
            for _, row in cyq_funds_df.iterrows():
                downloaded = w.wsd(row['code'],
                                   "NAV_adj,fund_fundscale,fund_info_name,fund_minholdingperiod",
                                   self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
                downloaded['fundfounddate'] = row['fundfounddate']
                downloaded['issueshare'] = row['issueshare']
                downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'NAV_ADJ': 'nav_adj',
                                                                      'FUND_FUNDSCALE': 'fund_fundscale',
                                                                      'FUND_INFO_NAME': 'product_name',
                                                                      'FUND_MINHOLDINGPERIOD': 'fund_minholdingperiod'})
                upload_value = downloaded[['product_name', 'nav_adj', 'fund_fundscale', 'fund_minholdingperiod']].melt(id_vars=['product_name'], var_name='field', value_name='value')
                upload_value['date'] = self.all_dates[-1]
                upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

        self.upload_joined_products_wide_table(full_name_keyword='������')

    def update_funds_name(self):
        """
        �ú���ֻ��ִ��һ�Ρ�
        :return:
        """
        code_set = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('type', 'fund'),
            ('fund_fullname', None)
        )

        for code in code_set:
            print('start download')
            downloaded = w.wsd(code,
                               "fund_fullname,fund_fullnameen",
                               self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
            # ����������������Ϊһ��, ����������
            downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'FUND_FULLNAME': 'fund_fullname',
                                                                  'FUND_FULLNAMEEN': 'english_name'})
            print(f'Updating {code} name')
            self.update_product_static_info(downloaded.squeeze(), task='fund_name')

    def update_all_funds_info(self):
        """
        1.���product_static_info�д��ڵĻ�����Ϣ��¼����ȡ��Ҫ���µ�����
            - product_static_info��internal_id��Ϊmarkets_daily_long��product_static_info_id�е������������������������Ȼ��ͨ��type=fundɸѡ����ע����
            - ���ɸѡ���������У���ѯproduct_static_info����buystartdate�е����ֵ����Сֵ����ȡ�������ݵ�����
            - ����������ݵ�����Ϊ�գ�missing_datesΪself.all_dates
            - self.all_dates[0]���������ݵ����������Ϊ��һ��missing_dates���������ݵ���������޵�self.all_dates[-1]Ϊ�ڶ���missing_dates
        2.����missing_datesִ��w.wset�������ݣ����ڿ�missing_dates����ִ��w.wset
        3.�������صõ������ݽ������ϴ������ݿ�
            - windcode�ϴ���product_static_info��code��
            - name�ϴ���product_static_info��chinese_name��
            - buystartdate,issueshare,fundfounddate��Ϊ�ǹ̶��ģ��ϴ���product_static_info�ĸ�����
            - openbuystartdate,openrepurchasestartdate��Щ��(��Щ�ַ�����Ϊfield)�ϴ���markets_daily_long���Գ���ʽ���ݴ��档
        """
        # ��ȡ��Ҫ���µ���������
        existing_dates = self.select_column_from_joined_table(
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column='buystartdate',
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
        product_metric_upload_df = downloaded_df[
            ['windcode', 'name', 'buystartdate', 'fundfounddate', 'issueshare']].rename(
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
