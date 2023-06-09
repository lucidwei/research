# coding=gbk
# Time Created: 2023/5/25 9:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from WindPy import w
from sqlalchemy import text


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_all_funds_info()
        # self.logic_reopened_dk_funds()
        # self.logic_reopened_cyq_funds()
        self.logic_etf_lof_funds()
        # self.logic_margin_trade_by_industry()
        # self.logic_north_inflow_by_industry()
        # self.logic_major_holder()
        # self.logic_price_valuation()
        # self.update_MA_processed_data(MA_period=10)

    def logic_margin_trade_by_industry(self):
        """
        1. ���
        :return:
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name',
                                                        type_identifier='margin_by_industry')
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
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # ��ȡ��Ҫ���µ���������
        match type_identifier:
            case 'major_holder' | 'price_valuation' | 'fund':
                filter_condition = f"product_static_info.type_identifier = '{type_identifier}'"
                if additional_filter:
                    filter_condition += f" AND {additional_filter}"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name=table_name,
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'north_inflow' | 'margin_by_industry':
                filter_condition = f"metric_static_info.type_identifier = '{type_identifier}'"
                if additional_filter:
                    filter_condition += f" AND {additional_filter}"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='metric_static_info',
                    target_join_column='internal_id',
                    join_table_name=table_name,
                    join_column='metric_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case _:
                raise Exception(f'type_identifier: {type_identifier} not supported.')

        if len(existing_dates) == 0:
            missing_dates = self.tradedays
        else:
            missing_dates = self.get_missing_dates(all_dates=self.tradedays, existing_dates=existing_dates)

        if not missing_dates:
            print(f"No missing dates for check_data_table, type_identifier={type_identifier}")
            return []
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

            case 'price_valuation':
                # ��ȡ����һ����ҵ��ָ�����������
                print(f'Wind downloading sectorconstituent for {self.tradedays_str[-2]}')
                downloaded_df = w.wset("sectorconstituent",
                                       f"date={self.tradedays_str[-2]};sectorid=a39901012e000000", usedf=True)[1]
                self._price_valuation_required_codes_df = downloaded_df
                required_value = downloaded_df['wind_code'].drop_duplicates().tolist()

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

        for date in missing_dates[:-1]:  # �����µ������ݣ��Է����ص�δ���µ���������
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
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()

            df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def _upload_wide_data_industry_margin(self):
        joined_df = self.read_joined_table_as_dataframe(
            target_table_name='metric_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='metric_static_info_id',
            filter_condition=f"metric_static_info.type_identifier = 'margin_by_industry'"
        )
        selected_df = joined_df[["date", 'product_name', 'field', "value"]]
        # �����ϴ��������ˣ��ҵ�pivot������
        # df_upload = selected_df.melt(id_vars=['date', 'product_name'], var_name='field',
        #                              value_name='value').sort_values(by="date", ascending=False)

    def logic_north_inflow_by_industry(self):
        # �������meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name',
                                                        type_identifier='north_inflow')
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

        for date in missing_dates[:-1]:
            print(f'Wind downloading shareplanincreasereduce for {date}')
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
            for _, row in df_meta.iterrows():
                self.insert_product_static_info(row)
            # df_meta.to_sql('product_static_info', self.alch_engine, if_exists='append', index=False)

    def _upload_missing_data_major_holder(self, missing_dates):
        for date in missing_dates[:-1]:
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
                    avg_percent = (change_limit_percent_up + change_limit_percent_limit) / 200
                    df_calculated.loc[i, f'��{direction}���'] = avg_percent * mkt_cap
                elif not pd.isnull(change_limit_percent_up):
                    # ���ֻ�� '��䶯��������ռ�ܹɱ���' �ǿգ������������ֵ��Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_limit_percent_up * mkt_cap / 100
                elif not pd.isnull(change_limit_percent_limit):
                    # ���ֻ�� '��䶯��������ռ�ܹɱ���' �ǿգ������������ֵ��Ϊ '�����ֽ��'
                    df_calculated.loc[i, f'��{direction}���'] = change_limit_percent_limit * mkt_cap / 100
                else:
                    # �����������ж�Ϊ�գ����� '�����ֽ��' �� '����ֽ��' �б�ע 'wind missing data need manually update'
                    df_calculated.loc[i, 'item_note_2b_added'] = 0

            df_upload = df_calculated[['product_name', '�����ֽ��', '����ֽ��', 'item_note_2b_added']].copy(
                deep=True)
            df_upload['date'] = date
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()
            df_upload_summed = df_upload.groupby(['date', 'product_name', 'field'], as_index=False).sum().dropna()
            df_upload_summed.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_price_valuation(self):
        """
        ������ҵ��ȫA���������ֵ���������ʽ��������Աȡ�
        """
        # �������meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code',
                                                        type_identifier='price_valuation')
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation')
        if need_update_meta_table:
            # �������data_table
            self._upload_missing_meta_price_valuation()
        if missing_dates:
            self._upload_missing_data_price_valuation(missing_dates)

    def _upload_missing_meta_price_valuation(self):
        required_codes_df = self._price_valuation_required_codes_df
        if required_codes_df.empty:
            raise Exception(
                f"Missing data for self._price_valuation_required_codes, no data downloaded for _upload_missing_meta_price_valuation")

        downloaded_df = required_codes_df.rename(
            columns={'wind_code': 'code',
                     'sec_name': 'chinese_name',
                     })
        downloaded_df = downloaded_df.drop("date", axis=1)

        # �������ȫA
        new_row = {'code': '881001.WI', 'chinese_name': '���ȫA'}
        downloaded_df = downloaded_df.append(new_row, ignore_index=True)

        downloaded_df['chinese_name'] = downloaded_df['chinese_name'].str.replace('\(����\)', '')
        downloaded_df['source'] = 'wind'
        downloaded_df['type_identifier'] = 'price_valuation'
        downloaded_df['product_type'] = 'index'
        self.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
        for _, row in downloaded_df.iterrows():
            self.insert_product_static_info(row)

    def _upload_missing_data_price_valuation(self, missing_dates):
        industry_codes = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('type_identifier', 'price_valuation')
        )

        for date in missing_dates[-50:-1]:
            for code in industry_codes:
                print(f'Downloading and uploading close,val_pe_nonnegative,dividendyield2,mkt_cap_ashare for {code} {date}')
                df = w.wsd(code, "close,val_pe_nonnegative,dividendyield2,mkt_cap_ashare", date,
                           date, "unit=1", usedf=True)[1]
                if df.empty:
                    print(f"Missing data for {date} {code}, no data downloaded for _upload_missing_data_price_valuation")
                    continue
                df_upload = df.rename(
                    columns={'WindCodes': 'code',
                             'CLOSE': '���̼�',
                             'VAL_PE_NONNEGATIVE': '��ӯ��PE(TTM,�޳���ֵ)',
                             'DIVIDENDYIELD2': '��Ϣ��(TTM)',
                             'MKT_CAP_ASHARE': 'A����ֵ(�������۹�)',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def update_MA_processed_data(self, MA_period):
        aggregate_inflow_ts = self.get_metabase_results('aggregate_inflow')
        df_ma = aggregate_inflow_ts.copy()
        df_ma[f'��������֮��_MA{MA_period}'] = aggregate_inflow_ts.rolling(window=MA_period).mean()
        # df_ma['date'] = aggregate_inflow_ts['date']
        # df_ma = df_ma.dropna()
        # MA_aggregate_inflow_long = df_ma.rename(columns={'��������֮��': f'��������֮��_MA{MA_period}'})
        MA_aggregate_inflow_long = df_ma.melt(id_vars=['date'], value_vars=[f'��������֮��_MA{MA_period}', '��������֮��'],
                                              var_name='var_name', value_name=f'value').dropna()

        margin_inflow_long = self.get_metabase_results('margin_inflow')
        north_inflow_long = self.get_metabase_results('north_inflow')
        MA_margin_inflow_long = self.get_MA_df_long(margin_inflow_long, '����һ����ҵ', '���ھ�����', MA_period)
        MA_north_inflow_long = self.get_MA_df_long(north_inflow_long, '����һ����ҵ', '��������', MA_period)

        MA_aggregate_inflow_long.to_sql(f'ma{MA_period}_aggregate_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
                         index=False)
        MA_margin_inflow_long.to_sql(f'ma{MA_period}_margin_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
                         index=False)
        MA_north_inflow_long.to_sql(f'ma{MA_period}_north_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
                         index=False)

        # # ���̼� ��ӯ�� ��Ϣ�� ��ֵ������MA��ֻ��Ա߼��ʽ���MA
        # joined_df = self.read_joined_table_as_dataframe(
        #     target_table_name='product_static_info',
        #     target_join_column='internal_id',
        #     join_table_name='markets_daily_long',
        #     join_column='product_static_info_id',
        #     filter_condition=f"product_static_info.type_identifier = 'price_valuation' AND field='���̼�'"
        # )
        # df_upload = self.get_MA_df_upload(joined_df, MA_period=10)
        # df_upload.to_sql('price_valuation_MA10', con=self.alch_engine, schema='processed_data', if_exists='replace', index=False)

    def get_metabase_results(self, task):
        match task:
            case 'aggregate_inflow':
                metabase_query = text(
                    """
                    SELECT "source"."date" AS "date", SUM("source"."��������֮��") AS "sum"
                    FROM (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", "source"."sum" AS "sum", (COALESCE("Question 153"."sum", 0) + COALESCE("source"."sum", 0) + COALESCE("Question 170"."sum", 0)) - COALESCE("Question 167"."�⾻���ֽ��", 0) AS "��������֮��", "Question 153"."date" AS "Question 153__date", "Question 153"."����һ����ҵ" AS "Question 153__����һ����ҵ", "Question 167"."date" AS "Question 167__date", "Question 167"."Product Static Info__stk_industry_cs" AS "Question 167__Product Static Info__stk_industry_cs", "Question 170"."date" AS "Question 170__date", "Question 170"."Product Static Info_2__stk_industry_cs" AS "Question 170__Product Static Info_2__stk_industry_cs", "Question 167"."�⾻���ֽ��" AS "Question 167__�⾻���ֽ��", "Question 153"."sum" AS "Question 153__sum", "Question 170"."sum" AS "Question 170__sum" FROM (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "����һ����ҵ", "Metric Static Info_2"."type_identifier" AS "Metric Static Info_2__type_identifier", "Metric Static Info_2"."internal_id" AS "Metric Static Info_2__internal_id" FROM "public"."markets_daily_long"
                    LEFT JOIN "public"."metric_static_info" AS "Metric Static Info_2" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info_2"."internal_id"
                    WHERE "Metric Static Info_2"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '������'
                    GROUP BY "source"."date", "source"."����һ����ҵ"
                    ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC) AS "source" LEFT JOIN (SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long" LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id" WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '���ھ������' GROUP BY "source"."date", "source"."����һ����ҵ" ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC) AS "Question 153" ON ("source"."date" = "Question 153"."date")
                       AND ("source"."����һ����ҵ" = "Question 153"."����һ����ҵ") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info__stk_industry_cs" AS "Product Static Info__stk_industry_cs", MAX(COALESCE(CASE WHEN "source"."field" = '����ֽ��' THEN "source"."value" END, 0)) - MAX(COALESCE(CASE WHEN "source"."field" = '�����ֽ��' THEN "source"."value" END, 0)) AS "�⾻���ֽ��" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", "public"."markets_daily_long"."product_static_info_id" AS "product_static_info_id", "public"."markets_daily_long"."date_value" AS "date_value", "Product Static Info"."internal_id" AS "Product Static Info__internal_id", "Product Static Info"."code" AS "Product Static Info__code", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", "Product Static Info"."english_name" AS "Product Static Info__english_name", "Product Static Info"."source" AS "Product Static Info__source", "Product Static Info"."type_identifier" AS "Product Static Info__type_identifier", "Product Static Info"."buystartdate" AS "Product Static Info__buystartdate", "Product Static Info"."fundfounddate" AS "Product Static Info__fundfounddate", "Product Static Info"."issueshare" AS "Product Static Info__issueshare", "Product Static Info"."fund_fullname" AS "Product Static Info__fund_fullname", "Product Static Info"."stk_industry_cs" AS "Product Static Info__stk_industry_cs", "Product Static Info"."product_type" AS "Product Static Info__product_type", "Product Static Info"."etf_type" AS "Product Static Info__etf_type" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id" WHERE "Product Static Info"."type_identifier" = 'major_holder') AS "source" WHERE ("source"."Product Static Info__type_identifier" = 'major_holder') AND "source"."date" BETWEEN timestamp with time zone '2022-07-01 00:00:00.000Z' AND timestamp with time zone '2023-05-30 00:00:00.000Z' GROUP BY "source"."date", "source"."Product Static Info__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info__stk_industry_cs" ASC) AS "Question 167" ON ("source"."date" = "Question 167"."date") AND ("source"."����һ����ҵ" = "Question 167"."Product Static Info__stk_industry_cs") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info_2__stk_industry_cs" AS "Product Static Info_2__stk_industry_cs", SUM("source"."value") AS "sum" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "Product Static Info_2"."code" AS "Product Static Info_2__code", "Product Static Info_2"."fund_fullname" AS "Product Static Info_2__fund_fullname", "Product Static Info_2"."stk_industry_cs" AS "Product Static Info_2__stk_industry_cs" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info_2" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info_2"."internal_id" WHERE ("public"."markets_daily_long"."field" = '�������') AND ("Product Static Info_2"."product_type" = 'fund')) AS "source" GROUP BY "source"."date", "source"."Product Static Info_2__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info_2__stk_industry_cs" ASC) AS "Question 170" ON ("source"."date" = "Question 170"."date") AND ("source"."����һ����ҵ" = "Question 170"."Product Static Info_2__stk_industry_cs")) AS "source" GROUP BY "source"."date" ORDER BY "source"."date" ASC
                    """
                )
                aggregate_inflows = self.alch_conn.execute(metabase_query)
                df_result = pd.DataFrame(aggregate_inflows, columns=['date', '��������֮��'])
                return df_result

            case 'margin_inflow':
                metabase_query = text(
                    """
                    SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum"
                    FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long"
                    LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
                    WHERE "Metric Static Info"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '���ھ������'
                    GROUP BY "source"."date", "source"."����һ����ҵ"
                    ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC
                    """
                )
                margin_inflow = self.alch_conn.execute(metabase_query)
                df_result = pd.DataFrame(margin_inflow, columns=['date', '����һ����ҵ', '���ھ�����'])
                return df_result

            case 'north_inflow':
                metabase_query = text(
                    """
                    SELECT "source"."date" AS "date", "source"."����һ����ҵ" AS "����һ����ҵ", SUM("source"."value") AS "sum"
                    FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."����һ����ҵ" AS "����һ����ҵ" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "����һ����ҵ", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
                    LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
                    WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '������'
                    GROUP BY "source"."date", "source"."����һ����ҵ"
                    ORDER BY "source"."date" ASC, "source"."����һ����ҵ" ASC
                    """
                )
                north_inflow = self.alch_conn.execute(metabase_query)
                df_result = pd.DataFrame(north_inflow, columns=['date', '����һ����ҵ', '��������'])
                return df_result


    def logic_reopened_dk_funds(self):
        """
        1. ��ȡfull_name�д���'���ڿ���'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        dk_funds_df = self.select_rows_by_column_strvalue(table_name='product_static_info', column_name='fund_fullname',
                                                          search_value='���ڿ���',
                                                          selected_columns=['code', 'chinese_name'],
                                                          filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%ծ%'")

        # ��Ϊ���������������࣬��˶���Щ��������ͳһ����
        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='����')
        if not today_updated:
            for _, row in dk_funds_df.iterrows():
                downloaded = w.wsd(row['code'],
                                   "NAV_adj,fund_expectedopenday,netasset_total,fund_fundscale,fund_info_name",
                                   self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
                downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'NAV_ADJ': 'nav_adj',
                                                                      'FUND_EXPECTEDOPENDAY': 'fund_expectedopenday',
                                                                      'FUND_FUNDSCALE': 'fund_fundscale',
                                                                      'NETASSET_TOTAL': 'netasset_total'})
                downloaded['product_name'] = row['chinese_name']
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

        # self.upload_joined_products_wide_table(full_name_keyword='���ڿ���')

    def logic_reopened_cyq_funds(self):
        """
        1. ��ȡfull_name�д���'������'��������ծ��ȫ������
        2. ��Ҫ��ȡ�����ݰ��������ο�����������ڡ�����ǰ��ķݶ�䶯���������ڵĻ����ģ
        """
        cyq_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='������',
            selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare'],
            filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%ծ%'")

        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='����')
        if not today_updated:
            for _, row in cyq_funds_df.iterrows():
                downloaded = w.wsd(row['code'],
                                   "NAV_adj,fund_fundscale,fund_info_name,fund_minholdingperiod",
                                   self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
                downloaded['fundfounddate'] = row['fundfounddate']
                downloaded['issueshare'] = row['issueshare']
                downloaded['product_name'] = row['chinese_name']
                downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'NAV_ADJ': 'nav_adj',
                                                                      'FUND_FUNDSCALE': 'fund_fundscale',
                                                                      'FUND_MINHOLDINGPERIOD': 'fund_minholdingperiod'})
                upload_value = downloaded[['product_name', 'nav_adj', 'fund_fundscale', 'fund_minholdingperiod']].melt(
                    id_vars=['product_name'], var_name='field', value_name='value')
                upload_value['date'] = self.all_dates[-1]
                upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

        # self.upload_joined_products_wide_table(full_name_keyword='������')

    def logic_etf_lof_funds(self):
        etf_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='�����Ϳ���ʽ',
            selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare', 'etf_type'],
            filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%ծ%' "
                             "AND fund_fullname NOT LIKE '%����%'")
        # ��һЩ����ʱ��������ҵETF���ݿ���û����¼����excel�ж�ȡ�����µ����ݿ�
        # ��ȡ��ҵ�ʽ��������ģ�ļ�
        file_path = os.path.join(self.base_config.excels_path, '��ҵ�ʽ��������ģ��5.8-5.13).xlsx')
        df_excel = pd.read_excel(file_path, sheet_name='��ҵETF���ܾ�����ͳ��', index_col=None)
        # ɸѡ�� df_excel �д��ڵ� etf_funds_df �в����ڵĴ���
        missing_codes = df_excel['����'][~df_excel['����'].isin(etf_funds_df['code'])]
        for code in missing_codes:
            self._update_specific_funds_meta(code)

        # update ETF������ҵ��ETF����(excel�ļ���ȫ��Ϊ��ҵETF)
        df_excel = df_excel.rename(columns={'����': 'code', '����һ����ҵ': 'stk_industry_cs'})
        df_excel['etf_type'] = '��ҵETF'
        for _, row in df_excel.iterrows():
            self.upload_product_static_info(row, task='etf_industry_and_type')

        # ETF�ֳ�3�ࣺ��ҵETF������ETF��ָ��ETF
        # Ĭ�Ϸ���Ϊָ��ETF
        etf_funds_df.loc[etf_funds_df['etf_type'].isnull(), 'etf_type'] = 'ָ��ETF'
        # ���ݹؼ��ֽ��з���
        theme_keywords = ['����']
        etf_funds_df.loc[etf_funds_df['fund_fullname'].str.contains('|'.join(theme_keywords)), 'etf_type'] = '����ETF'
        # ������ҵ�ʽ��������ģ�ļ����з���
        industry_etf_codes = df_excel['code'].tolist()
        etf_funds_df.loc[etf_funds_df['code'].isin(industry_etf_codes), 'etf_type'] = '��ҵETF'
        etf_funds_df = etf_funds_df.sort_values(by='etf_type')

        # ����markets_daily_long '�������'ʱ������
        for _, row in etf_funds_df.iterrows():
            self._update_etf_inflow(row)

    def _update_etf_inflow(self, etf_info_row):
        missing_dates = self._check_data_table('markets_daily_long', 'fund', additional_filter=f"code='{etf_info_row['code']}'")
        fund_found_date = self.select_column_from_joined_table(
            filter_condition=f"code='{etf_info_row['code']}'",
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column=f'fundfounddate',
        )
        try:
            missing_start_date = max(missing_dates[0], fund_found_date[0])
        except:
            missing_start_date = missing_dates[0]

        # ����������ı䶯���ںͻ���ݶ�-���ּ��ݶ�䶯����һ����Ӧ�þ��Ƿݶ�䶯���Ծ�ֵ
        print(f"_update_etf_inflow Downloading mf_netinflow for {etf_info_row['code']} "
              f"b/t {missing_start_date} and {missing_dates[-2]}")
        downloaded = w.wsd(etf_info_row['code'], "mf_netinflow",
                           missing_start_date, missing_dates[-2], "unit=1", usedf=True)[1]
        downloaded_filtered = downloaded[downloaded['MF_NETINFLOW'] != 0]
        downloaded_filtered = downloaded_filtered.reset_index().rename(
            columns={'index': 'date', 'MF_NETINFLOW': '�������'})
        # ȥ���Ѿ����ڵ�����
        existing_dates = self.select_existing_dates_from_long_table('markets_daily_long',
                                                                    product_name=etf_info_row['chinese_name'],
                                                                    field='�������')
        downloaded_filtered = downloaded_filtered[~downloaded_filtered['date'].isin(existing_dates)]
        downloaded_filtered['product_name'] = etf_info_row['chinese_name']
        upload_value = downloaded_filtered[['product_name', 'date', '�������']].melt(
            id_vars=['product_name', 'date'], var_name='field', value_name='value')

        upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def _update_specific_funds_meta(self, code: str):
        info = w.wsd(code,
                     "fund_fullname,fund_info_name,fund_fullnameen",
                     # ,fund_offnetworkbuystartdate,fund_etflisteddate",
                     self.all_dates_str[-1], self.all_dates_str[-1], "unit=1", usedf=True)[1]
        row = {
            'code': code,
            'fund_fullname': info.iloc[0]['FUND_FULLNAME'],
            'chinese_name': info.iloc[0]['FUND_INFO_NAME'],
            'english_name': info.iloc[0]['FUND_FULLNAMEEN'],
            'product_type': 'fund',
            'source': 'wind',
            'fundfounddate': None,
            'buystartdate': None,
            'issueshare': None
        }
        self.insert_product_static_info(row)

    def _update_funds_name(self):
        code_set = self.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'fund'),
            ('fund_fullname', None)
        )

        for code in code_set:
            print('start download')
            downloaded = w.wsd(code,
                               "fund_fullname,fund_fullnameen",
                               self.all_dates_str[-2], self.all_dates_str[-2], "unit=1", usedf=True)[1]
            # ����������������Ϊһ��, ����������
            downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'FUND_FULLNAME': 'fund_fullname',
                                                                  'FUND_FULLNAMEEN': 'english_name'})
            print(f'Updating {code} name')
            self.upload_product_static_info(downloaded.squeeze(), task='fund_name')

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
            filter_condition="product_static_info.product_type = 'fund'"
        )

        if len(existing_dates) == 0:
            missing_buystartdates = self.all_dates
        else:
            missing_buystartdates = self.get_missing_dates(all_dates=self.all_dates, existing_dates=existing_dates)

        buystartdates_for_missing_fundfounddates = self.select_existing_values_in_target_column('product_static_info',
                                                                                                'buystartdate',
                                                                                                'fundfounddate is NULL')
        # ɸѡ��2�����ڵ�����
        recent_buystartdates_for_missing_fundfounddates = [
            date for date in buystartdates_for_missing_fundfounddates if
            self.all_dates[-1] - relativedelta(months=2) <= date <= self.all_dates[-1]]

        missing_dates = sorted(missing_buystartdates + recent_buystartdates_for_missing_fundfounddates)
        if not missing_dates:
            print("No missing dates for update_all_funds_info")
            return

        # ִ����������
        # ���Ϲ���ʼ����Ϊɸѡ������ѡȡ�����ݸ�����������ǰհ�ԡ�ֻѡȡ�ϸ������ϵ��·�����
        for missing_date in missing_dates[:-1]:
            print(f'Downloading fundissuegeneralview on {missing_date} for update_all_funds_info')
            downloaded_df = w.wset("fundissuegeneralview",
                                   f"startdate={missing_date};enddate={missing_date};datetype=startdate;isvalid=yes;"
                                   f"deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,"
                                   f"openbuystartdate,openrepurchasestartdate",
                                   usedf=True)[1]
            if downloaded_df.empty:
                print(
                    f"Missing fundissuegeneralview data on {missing_date}, but no data downloaded for update_all_funds_info")
                continue

            # �������ص����ݲ��ϴ���product_static_info
            product_metric_upload_df = downloaded_df[
                ['windcode', 'name', 'buystartdate', 'fundfounddate', 'issueshare']].rename(
                columns={'windcode': 'code', 'name': 'chinese_name'})
            # ���source��type�в��ϴ�
            product_metric_upload_df['source'] = 'wind'
            product_metric_upload_df['product_type'] = 'fund'
            for _, row in product_metric_upload_df.iterrows():
                self.insert_product_static_info(row)

            # ���Ǿ�̬�����ϴ���markets_daily_long
            markets_daily_long_upload_df = downloaded_df[
                ['name', 'openbuystartdate', 'openrepurchasestartdate']].rename(
                columns={'name': 'chinese_name',
                         'openbuystartdate': '�����깺��ʼ��',
                         'openrepurchasestartdate': '���������ʼ��'
                         })
            markets_daily_long_upload_df = markets_daily_long_upload_df.melt(id_vars=['chinese_name'], var_name='field',
                                                                             value_name='date_value')
            markets_daily_long_upload_df = markets_daily_long_upload_df.dropna(subset=['date_value']).rename(
                columns={'chinese_name': 'product_name'})
            # ����date�ĺ�������Ϣ��¼��
            markets_daily_long_upload_df['date'] = self.all_dates[-1]

            # �ϴ�ǰҪ�޳��Ѵ��ڵ�product
            existing_products = self.select_column_from_joined_table(
                target_table_name='product_static_info',
                target_join_column='internal_id',
                join_table_name='markets_daily_long',
                join_column='product_static_info_id',
                selected_column='chinese_name',
                filter_condition="product_static_info.product_type = 'fund' "
                                 "AND (markets_daily_long.field = '�����깺��ʼ��'"
                                 "OR markets_daily_long.field = '���������ʼ��')"
            )
            filtered_df = markets_daily_long_upload_df[
                ~markets_daily_long_upload_df['product_name'].isin(existing_products)]
            filtered_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

        self._update_funds_name()
