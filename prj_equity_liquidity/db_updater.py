# coding=gbk
# Time Created: 2023/5/25 9:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import numpy as np
import pandas as pd

from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from WindPy import w


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        # self.update_all_funds_info()
        # self.update_funds_name()
        # self.update_reopened_dk_funds()
        # self.update_reopened_cyq_funds()
        # self.update_etf_lof_funds()
        self.logic_margin_trade_by_industry()
        self.logic_north_inflow_by_industry()
        self.logic_major_holder()

    def logic_margin_trade_by_industry(self):
        """
        1. ���
        :return:
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name', type_identifier='margin_by_industry')
        if need_update_meta_table:
            for industry in self.today_industries_df['industryname'].tolist():
                self.insert_metric_static_info(source_code=f'wind_tradingstatisticsbyindustry_{industry}',
                                               chinese_name=f'������ȯ��ҵ����ͳ��_{industry}', english_name='',
                                               type_identifier='margin_by_industry', unit='')
        # �������data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='margin_by_industry')
        self._upload_missing_data_industry_margin(missing_dates)
        # self._upload_wide_data_industry_margin()

    def _check_data_table(self, table_name, type_identifier, **kwargs):
        # ��ȡ��Ҫ���µ���������
        if type_identifier == 'major_holder':
            existing_dates = self.select_column_from_joined_table(
                target_table_name='product_static_info',
                target_join_column='internal_id',
                join_table_name=table_name,
                join_column='product_static_info_id',
                selected_column=f'date',
                filter_condition=f"product_static_info.type_identifier = '{type_identifier}'"
            )
        else:
            existing_dates = self.select_column_from_joined_table(
                target_table_name='metric_static_info',
                target_join_column='internal_id',
                join_table_name=table_name,
                join_column='metric_static_info_id',
                selected_column=f'date',
                filter_condition=f"metric_static_info.type_identifier = '{type_identifier}'"
            )

        if len(existing_dates) == 0:
            missing_dates = self.tradedays
        else:
            missing_dates = self.get_missing_dates(all_dates=self.tradedays, existing_dates=existing_dates)

        if not missing_dates:
            print(f"No missing dates for check_data_table, type_identifier={type_identifier}")
            return False
        return missing_dates

    def _check_meta_table(self, table_name, check_column, type_identifier):
        match type_identifier:
            case 'margin_by_industry':
                print(f'Wind downloading tradingstatisticsbyindustry industryname for _check_meta_table')
                self.today_industries_df = w.wset("tradingstatisticsbyindustry",
                                                  f"exchange=citic;startdate={self.tradedays_str[-2]};enddate={self.tradedays_str[-2]};"
                                                  "field=industryname", usedf=True)[1]
                industry_list = self.today_industries_df['industryname'].tolist()
                required_value = ['������ȯ��ҵ����ͳ��_' + str(value) for value in industry_list]

            case 'north_inflow':
                print(f'Wind downloading shscindustryfundflow industry for _check_meta_table')
                self.today_industries_df = w.wset("shscindustryfundflow",
                                                  f"industrytype=citic;date={self.tradedays_str[-2]};"
                                                  "field=industry", usedf=True)[1]
                industry_list = self.today_industries_df['industry'].tolist()
                required_value = ['�����ʽ�_' + str(value) for value in industry_list]

            case 'major_holder':
                # �����ճ��ֵĹ�Ʊ�Ƿ������product_static_info (type_identifier='major_shareholder')
                print(f'Wind downloading shareplanincreasereduce for {self.tradedays_str[-1]}')
                downloaded_df = w.wset("shareplanincreasereduce",
                                       f"startdate={self.tradedays_str[-1]};enddate={self.tradedays_str[-1]};"
                                       f"datetype=firstannouncementdate;type=all;field=windcode", usedf=True)[1]
                required_value = downloaded_df['windcode'].drop_duplicates().tolist()

            case _:
                raise Exception(f'type_identifier {type_identifier} not supported')

        existing_value = self.select_existing_values_in_target_column(table_name, check_column,
                                                                      ('type_identifier', type_identifier))
        missing_value = set(required_value) - set(existing_value)
        if missing_value:
            return True
        else:
            return False

    def _upload_missing_data_industry_margin(self, missing_dates):
        if len(missing_dates) == 0:
            return

        for date in missing_dates[-100:-1]:
            print(f'Wind downloading tradingstatisticsbyindustry for {date}')
            downloaded_df = w.wset("tradingstatisticsbyindustry",
                                   f"exchange=citic;startdate={date};enddate={date};"
                                   "field=industryname,totalbalance,financingbuybetween,"
                                   "securiesnetsellvolume,financingbuybetweenrate,securiesnetsellvolumerate,"
                                   "balancenegotiablepercent,totaltradevolumepercent,netbuyvolumebetween",
                                   usedf=True)[1]
            if downloaded_df.empty:
                print(f"Missing data for {date}, no data downloaded for _upload_missing_data_industry_margin")
                continue

            # Parse the downloaded data and upload it to the database
            df_upload = downloaded_df.rename(
                columns={'totalbalance': '�������',
                         'financingbuybetween': '���ʾ������',
                         'securiesnetsellvolume': '��ȯ��������',
                         'financingbuybetweenrate': '���ʾ������ռ��',
                         'securiesnetsellvolumerate': '��ȯ��������ռ��',
                         'balancenegotiablepercent': '�������ռ��ͨ��ֵ',
                         'totaltradevolumepercent': '���ڽ��׶�ռ�ɽ���ռ��',
                         'netbuyvolumebetween': '���ھ������',
                         })
            df_upload['date'] = date
            df_upload['product_name'] = '������ȯ��ҵ����ͳ��_' + downloaded_df['industryname']
            df_upload.drop("industryname", axis=1, inplace=True)
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value')

            df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def _upload_wide_data_industry_margin(self):
        joined_df = self.get_joined_table_as_dataframe(
            target_table_name='metric_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='metric_static_info_id',
            filter_condition=f"metric_static_info.type_identifier = 'margin_by_industry'"
        )
        selected_df = joined_df[["date", 'product_name', 'field', "value"]]
        # �����ϴ��������ˣ��ҵ�pivot������
        df_upload = selected_df.melt(id_vars=['date', 'product_name'], var_name='field',
                                     value_name='value').sort_values(by="date", ascending=False)

    def logic_north_inflow_by_industry(self):
        """
        1. ���
        :return:
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name', type_identifier='north_inflow')
        if need_update_meta_table:
            for industry in self.today_industries_df['industry'].tolist():
                self.insert_metric_static_info(source_code=f'wind_shscindustryfundflow_{industry}',
                                               chinese_name=f'�����ʽ�_{industry}', english_name='',
                                               type_identifier='north_inflow', unit='')
        # �������data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='north_inflow')
        self._upload_missing_data_north_inflow(missing_dates)

    def _upload_missing_data_north_inflow(self, missing_dates):
        if len(missing_dates) == 0:
            return

        for date in missing_dates:
            print(f'Wind downloading shscindustryfundflow for {date}')
            downloaded_df = w.wset("shscindustryfundflow",
                                   f"industrytype=citic;date={date};"
                                   "field=industry,marketvalue,dailynetinflow,dailyproportionchange", usedf=True)[1]
            if downloaded_df.empty:
                print(f"Missing data for {date}, no data downloaded for _upload_missing_data_industry_margin")
                continue

            # Parse the downloaded data and upload it to the database
            df_upload = downloaded_df.rename(
                columns={'marketvalue': '�ֹ���ֵ',
                         'dailynetinflow': '������',
                         'dailyproportionchange': 'ռ��ҵ����ֵ�ȵı仯',
                         })
            df_upload['date'] = date
            df_upload['product_name'] = '�����ʽ�_' + downloaded_df['industry']
            df_upload.drop("industry", axis=1, inplace=True)
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()

            df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_major_holder(self):
        # �������meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code', type_identifier='major_holder')
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='major_holder')
        if need_update_meta_table:
            # �������data_table
            self._upload_missing_meta_major_holder(missing_dates)
        if missing_dates:
            self._upload_missing_data_major_holder(missing_dates)

    def _upload_missing_meta_major_holder(self, missing_dates):
        if len(missing_dates) == 0:
            return

        for date in missing_dates[-100:-1]:
            print(f'Wind downloading shareplanincreasereduce for {date}')
            # ����meta��������̫��
            downloaded_df = w.wset("shareplanincreasereduce",
                                   f"startdate={date};enddate={date};datetype=firstannouncementdate;type=all;"
                                   "field=windcode,name",
                                   usedf=True)[1]
            if downloaded_df.empty:
                print(f"Missing data for {date}, no data downloaded for _upload_missing_meta_major_holder")
                continue

            # Parse the downloaded data and upload it to the database
            downloaded_df = downloaded_df.rename(
                columns={'windcode': 'code',
                         'name': 'chinese_name',
                         })
            df_meta = downloaded_df.drop_duplicates()
            existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                          ('type_identifier', 'major_holder'),
                                                                          'stk_industry_cs IS NOT NULL')
            df_meta = df_meta[~df_meta['code'].isin(existing_codes)]
            if df_meta.empty:
                continue

            for i, row in df_meta.iterrows():
                code = row['code']
                print(f'Wind downloading industry_citic for {code} on {date}')
                info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
                                      usedf=True)[1]
                if info_df.empty:
                    print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
                    continue
                industry = info_df.iloc[0]['INDUSTRY_CITIC']
                df_meta.loc[i, 'stk_industry_cs'] = industry
            # �ϴ�metadata
            df_meta['source'] = 'wind'
            df_meta['type_identifier'] = 'major_holder'
            df_meta['product_type'] = 'stock'

            self.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
            df_meta.to_sql('product_static_info', self.alch_engine, if_exists='append', index=False)

    def _upload_missing_data_major_holder(self, missing_dates):
        for date in missing_dates[-100:-1]:
            print(f'Wind downloading shareplanincreasereduce for {date}')
            downloaded_df = w.wset("shareplanincreasereduce",
                                   f"startdate={date};enddate={date};datetype=firstannouncementdate;type=all;"
                                   "field=windcode,name,firstpublishdate,latestpublishdate,direction,"
                                   "changemoneyup,changeuppercent,changemoneylimit,changelimitpercent",
                                   usedf=True)[1]
            if downloaded_df.empty:
                print(f"Missing data for {date}, no data downloaded for _upload_missing_meta_major_holder")
                continue

            # Parse the downloaded data and upload it to the database
            downloaded_df = downloaded_df.rename(
                columns={'windcode': 'code',
                         'name': 'product_name',
                         'firstpublishdate': '�״ι�������',
                         'latestpublishdate': '���¹�������',
                         'direction': '�䶯����',
                         'changemoneyup': '��䶯�������',
                         'changeuppercent': '��䶯��������ռ�ܹɱ���',
                         'changemoneylimit': '��䶯�������',
                         'changelimitpercent': '��䶯��������ռ�ܹɱ���',
                         })
            selected_df = downloaded_df[downloaded_df['�״ι�������'] == downloaded_df['���¹�������']]

            for i, row in selected_df.iterrows():
                code = row['code']
                print(f'Wind downloading mkt_cap_ard for {code} on {date}')
                info_df = w.wsd(code, "mkt_cap_ard", f'{date}', f'{date}', "unit=1;industryType=1",
                                      usedf=True)[1]
                if info_df.empty:
                    print(f"Missing data for {code} on {date}, no data downloaded for mkt_cap_ard")
                    continue
                mkt_cap = info_df.iloc[0]['MKT_CAP_ARD']
                selected_df.loc[i, '����ֵ'] = mkt_cap

            # ���������ֽ��
            df_calculated = selected_df.copy()
            df_calculated['item_note_2b_added'] = np.nan
            df_calculated['�����ֽ��'] = np.nan
            df_calculated['����ֽ��'] = np.nan
            for i, row in df_calculated.iterrows():
                change_money_up = row['��䶯�������']
                change_money_limit = row['��䶯�������']
                change_limit_percent_up = row['��䶯��������ռ�ܹɱ���']
                change_limit_percent_limit = row['��䶯��������ռ�ܹɱ���']
                mkt_cap = row['����ֵ']
                direction = row['�䶯����']

                if not pd.isnull(change_money_up) and not pd.isnull(change_money_limit):
                    # ��� '��䶯�������' �� '��䶯�������' ���ǿգ���ȡ��ֵ��Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = (change_money_up + change_money_limit) / 2
                elif not pd.isnull(change_money_up):
                    # ���ֻ�� '��䶯�������' �ǿգ�������Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_money_up
                elif not pd.isnull(change_money_limit):
                    # ���ֻ�� '��䶯�������' �ǿգ�������Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_money_limit
                elif not pd.isnull(change_limit_percent_up) and not pd.isnull(change_limit_percent_limit):
                    # ��� '��䶯��������ռ�ܹɱ���' �� '��䶯��������ռ�ܹɱ���' ���ǿգ���ȡ��ֵ��������ֵ��Ϊ '�����ֽ��'
                    avg_percent = (change_limit_percent_up + change_limit_percent_limit) / 2
                    df_calculated.loc[i, f'��{direction}���'] = avg_percent * mkt_cap
                elif not pd.isnull(change_limit_percent_up):
                    # ���ֻ�� '��䶯��������ռ�ܹɱ���' �ǿգ������������ֵ��Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_limit_percent_up * mkt_cap
                elif not pd.isnull(change_limit_percent_limit):
                    # ���ֻ�� '��䶯��������ռ�ܹɱ���' �ǿգ������������ֵ��Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_limit_percent_limit * mkt_cap
                else:
                    # �����������ж�Ϊ�գ����� '�����ֽ��' �� '����ֽ��' �б�ע 'wind missing data need manually update'
                    df_calculated.loc[i, 'item_note_2b_added'] = 0

            df_upload = df_calculated[['product_name', '�����ֽ��', '����ֽ��', 'item_note_2b_added']].copy(deep=True)
            df_upload['date'] = date
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()
            df_upload_summed = df_upload.groupby(['date', 'product_name', 'field'], as_index=False).sum().dropna()
            df_upload_summed.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def update_reopened_dk_funds(self):
        """
        1. ��ȡfull_name�д���'���ڿ���'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        dk_funds_df = self.select_rows_by_column_strvalue(table_name='product_static_info', column_name='fund_fullname',
                                                          search_value='���ڿ���',
                                                          selected_columns=['code', 'chinese_name', 'fund_fullname'],
                                                          filter_condition="type='fund' AND fund_fullname NOT LIKE '%ծ%'")
        dk_funds_df = dk_funds_df[~dk_funds_df['fund_fullname'].str.contains('ծ')]

        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='����')
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
                upload_date_value = downloaded[['product_name', 'fund_expectedopenday']].melt(id_vars=['product_name'],
                                                                                              var_name='field',
                                                                                              value_name='date_value')
                upload_value = downloaded[['product_name', 'nav_adj', 'fund_fundscale', 'netasset_total']].melt(
                    id_vars=['product_name'], var_name='field', value_name='value')
                upload_date_value['date'] = self.all_dates[-1]
                upload_value['date'] = self.all_dates[-1]
                upload_date_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append',
                                                  index=False)
                upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

        self.upload_joined_products_wide_table(full_name_keyword='���ڿ���')

    def update_reopened_cyq_funds(self):
        """
        1. ��ȡfull_name�д���'������'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        cyq_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='������',
            selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare'],
            filter_condition="type='fund' AND fund_fullname NOT LIKE '%ծ%'")

        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='����')
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
                upload_value = downloaded[['product_name', 'nav_adj', 'fund_fundscale', 'fund_minholdingperiod']].melt(
                    id_vars=['product_name'], var_name='field', value_name='value')
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
            ('type_identifier', 'fund'),
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
            filter_condition="product_static_info.type_identifier = 'fund'"
        )
        filtered_df = markets_daily_long_upload_df[
            ~markets_daily_long_upload_df['product_name'].isin(existing_products)]
        filtered_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)
