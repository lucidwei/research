# coding=gbk
# Time Created: 2023/5/25 9:40
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime
import os
import numpy as np
import pandas as pd
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
        self.logic_margin_trade_by_industry()
        self.logic_north_inflow_by_industry()
        self.logic_major_holder()
        self.logic_price_valuation()
        self.update_MA_processed_data(MA_period=10)

    def _check_data_table(self, table_name, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # 获取需要更新的日期区间
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
                required_value = ['融资融券行业交易统计_' + str(value) for value in industry_list]

            case 'north_inflow':
                print(f'Wind downloading shscindustryfundflow industry for _check_meta_table')
                self.today_industries_df = w.wset("shscindustryfundflow",
                                                  f"industrytype=citic;date={self.tradedays_str[-2]};"
                                                  "field=industry", usedf=True)[1]
                industry_list = self.today_industries_df['industry'].tolist()
                required_value = ['北向资金_' + str(value) for value in industry_list]

            case 'major_holder':
                # 检查今日出现的股票是否存在于product_static_info (type_identifier='major_shareholder')
                print(f'Wind downloading shareplanincreasereduce for {self.tradedays_str[-1]}')
                downloaded_df = w.wset("shareplanincreasereduce",
                                       f"startdate={self.tradedays_str[-1]};enddate={self.tradedays_str[-1]};"
                                       f"datetype=firstannouncementdate;type=all;field=windcode", usedf=True)[1]
                required_value = downloaded_df['windcode'].drop_duplicates().tolist()

            case 'price_valuation':
                # 获取中信一级行业的指数代码和名称
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

    def logic_margin_trade_by_industry(self):
        """
        1. 检查
        :return:
        """
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name',
                                                        type_identifier='margin_by_industry')
        if need_update_meta_table:
            for industry in self.today_industries_df['industryname'].tolist():
                self.insert_metric_static_info(source_code=f'wind_tradingstatisticsbyindustry_{industry}',
                                               chinese_name=f'融资融券行业交易统计_{industry}', english_name='',
                                               type_identifier='margin_by_industry', unit='')
        # 检查或更新data_table
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='margin_by_industry')
        self._upload_missing_data_industry_margin(missing_dates)
        # self._upload_wide_data_industry_margin()

    def _upload_missing_data_industry_margin(self, missing_dates):
        if len(missing_dates) == 0:
            return

        for date in missing_dates:  # 不更新当天数据，以防下载到未更新的昨天数据
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
                columns={'totalbalance': '两融余额',
                         'financingbuybetween': '融资净买入额',
                         'securiesnetsellvolume': '融券净卖出额',
                         'financingbuybetweenrate': '融资净买入额占比',
                         'securiesnetsellvolumerate': '融券净卖出额占比',
                         'balancenegotiablepercent': '两融余额占流通市值',
                         'totaltradevolumepercent': '两融交易额占成交额占比',
                         'netbuyvolumebetween': '两融净买入额',
                         })
            df_upload['date'] = date
            df_upload['product_name'] = '融资融券行业交易统计_' + downloaded_df['industryname']
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
        # 不用上传宽数据了，找到pivot方法了
        # df_upload = selected_df.melt(id_vars=['date', 'product_name'], var_name='field',
        #                              value_name='value').sort_values(by="date", ascending=False)

    def logic_north_inflow_by_industry(self):
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('metric_static_info', 'chinese_name',
                                                        type_identifier='north_inflow')
        if need_update_meta_table:
            for industry in self.today_industries_df['industry'].tolist():
                self.insert_metric_static_info(source_code=f'wind_shscindustryfundflow_{industry}',
                                               chinese_name=f'北向资金_{industry}', english_name='',
                                               type_identifier='north_inflow', unit='')
        # 检查或更新data_table
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
                columns={'marketvalue': '持股市值',
                         'dailynetinflow': '净买入',
                         'dailyproportionchange': '占行业总市值比的变化',
                         })
            df_upload['date'] = date
            df_upload['product_name'] = '北向资金_' + downloaded_df['industry']
            df_upload.drop("industry", axis=1, inplace=True)
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()

            df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_major_holder(self):
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code', type_identifier='major_holder')
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='major_holder')
        if need_update_meta_table:
            # 检查或更新data_table
            self._upload_missing_meta_major_holder(missing_dates)
        if missing_dates:
            self._upload_missing_data_major_holder(missing_dates)

    def _upload_missing_meta_major_holder(self, missing_dates):
        if len(missing_dates) == 0:
            return

        for date in missing_dates:
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
            # 上传metadata
            df_meta['source'] = 'wind'
            df_meta['type_identifier'] = 'major_holder'
            df_meta['product_type'] = 'stock'

            self.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
            for _, row in df_meta.iterrows():
                self.insert_product_static_info(row)

    def _upload_missing_data_major_holder(self, missing_dates):
        # TODO: quota充裕时要全部重下一遍，之前误筛掉太多了
        for date in missing_dates:
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
                         'firstpublishdate': '首次公告日期',
                         'latestpublishdate': '最新公告日期',
                         'direction': '变动方向',
                         'changemoneyup': '拟变动金额上限',
                         'changeuppercent': '拟变动数量上限占总股本比',
                         'changemoneylimit': '拟变动金额下限',
                         'changelimitpercent': '拟变动数量下限占总股本比',
                         })
            # 这个避免重复的筛选方法不对
            selected_df = downloaded_df[downloaded_df['首次公告日期'] == downloaded_df['最新公告日期']]
            # 如果把某些公司篩没了，就恢复原来的
            if set(selected_df['code'].tolist()) != set(downloaded_df['code'].tolist()):
                selected_df = downloaded_df.copy()

            for i, row in selected_df.iterrows():
                code = row['code']
                print(f'Wind downloading mkt_cap_ard for {code} on {date}')
                info_df = w.wsd(code, "mkt_cap_ard", f'{date}', f'{date}', "unit=1;industryType=1",
                                usedf=True)[1]
                if info_df.empty:
                    print(f"Missing data for {code} on {date}, no data downloaded for mkt_cap_ard")
                    continue
                mkt_cap = info_df.iloc[0]['MKT_CAP_ARD']
                selected_df.loc[i, '总市值'] = mkt_cap

            # 计算增减持金额
            df_calculated = selected_df.copy()
            df_calculated['item_note_2b_added'] = np.nan
            df_calculated['拟增持金额'] = np.nan
            df_calculated['拟减持金额'] = np.nan
            for i, row in df_calculated.iterrows():
                change_money_up = row['拟变动金额上限']
                change_money_limit = row['拟变动金额下限']
                change_limit_percent_up = row['拟变动数量上限占总股本比']
                change_limit_percent_limit = row['拟变动数量下限占总股本比']
                mkt_cap = row['总市值']
                direction = row['变动方向']

                if not pd.isnull(change_money_up) and not pd.isnull(change_money_limit):
                    # 如果 '拟变动金额上限' 和 '拟变动金额下限' 都非空，则取均值作为 '拟增持金额'
                    df_calculated.loc[i, f'拟{direction}金额'] = (change_money_up + change_money_limit) / 2
                elif not pd.isnull(change_money_up):
                    # 如果只有 '拟变动金额上限' 非空，则将其作为 '拟增持金额'
                    df_calculated.loc[i, f'拟{direction}金额'] = change_money_up
                elif not pd.isnull(change_money_limit):
                    # 如果只有 '拟变动金额下限' 非空，则将其作为 '拟增持金额'
                    df_calculated.loc[i, f'拟{direction}金额'] = change_money_limit
                elif not pd.isnull(change_limit_percent_up) and not pd.isnull(change_limit_percent_limit):
                    # 如果 '拟变动数量上限占总股本比' 和 '拟变动数量下限占总股本比' 都非空，则取均值乘以总市值作为 '拟增持金额'
                    avg_percent = (change_limit_percent_up + change_limit_percent_limit) / 200
                    df_calculated.loc[i, f'拟{direction}金额'] = avg_percent * mkt_cap
                elif not pd.isnull(change_limit_percent_up):
                    # 如果只有 '拟变动数量上限占总股本比' 非空，则将其乘以总市值作为 '拟增持金额'
                    df_calculated.loc[i, f'拟{direction}金额'] = change_limit_percent_up * mkt_cap / 100
                elif not pd.isnull(change_limit_percent_limit):
                    # 如果只有 '拟变动数量下限占总股本比' 非空，则将其乘以总市值作为 '拟增持金额'
                    df_calculated.loc[i, f'拟{direction}金额'] = change_limit_percent_limit * mkt_cap / 100
                else:
                    # 如果所有相关列都为空，则在 '拟增持金额' 和 '拟减持金额' 中标注 'wind missing data need manually update'
                    df_calculated.loc[i, 'item_note_2b_added'] = 0

            df_upload = df_calculated[['product_name', '拟增持金额', '拟减持金额', 'item_note_2b_added']].copy(
                deep=True)
            df_upload['date'] = date
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field', value_name='value').dropna()
            df_upload_summed = df_upload.groupby(['date', 'product_name', 'field'], as_index=False).sum().dropna()
            df_upload_summed.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def logic_price_valuation(self):
        """
        更新行业和全A的行情与估值，用来与资金流向作对比。
        """
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table('product_static_info', 'code',
                                                        type_identifier='price_valuation')
        missing_dates = self._check_data_table(table_name='markets_daily_long',
                                               type_identifier='price_valuation')
        if need_update_meta_table:
            # 检查或更新data_table
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

        # 加入万德全A
        new_row = {'code': '881001.WI', 'chinese_name': '万德全A'}
        downloaded_df = downloaded_df.append(new_row, ignore_index=True)

        downloaded_df['chinese_name'] = downloaded_df['chinese_name'].str.replace('\(中信\)', '')
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

        for date in missing_dates:
            for code in industry_codes:
                print(
                    f'Downloading and uploading close,val_pe_nonnegative,dividendyield2,mkt_cap_ashare for {code} {date}')
                df = w.wsd(code, "close,val_pe_nonnegative,dividendyield2,mkt_cap_ashare", date,
                           date, "unit=1", usedf=True)[1]
                if df.empty:
                    print(
                        f"Missing data for {date} {code}, no data downloaded for _upload_missing_data_price_valuation")
                    continue
                df_upload = df.rename(
                    columns={'WindCodes': 'code',
                             'CLOSE': '收盘价',
                             'VAL_PE_NONNEGATIVE': '市盈率PE(TTM,剔除负值)',
                             'DIVIDENDYIELD2': '股息率(TTM)',
                             'MKT_CAP_ASHARE': 'A股市值(不含限售股)',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                           value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def update_MA_processed_data(self, MA_period):
        # 复盘需要看MA10时需要用到这个函数
        df_full = self.get_metabase_full_df()
        # 收盘价 市盈率 股息率 市值不用求MA，只需对边际资金求MA
        self.margin_inflow_ts = self.get_metabase_results('margin_inflow', df_full)
        self.north_inflow_ts = self.get_metabase_results('north_inflow', df_full)
        self.aggregate_inflow_ts = self.get_metabase_results('aggregate_inflow', df_full)
        self.etf_inflow_ts = self.get_metabase_results('etf_inflow', df_full)
        self.holder_change_ts = self.get_metabase_results('holder_change', df_full)

        # 做VAR分析不需要求MA，MA是用来肉眼观察的
        # MA_margin_inflow_long = self.get_MA_df_long(margin_inflow_long, '中信一级行业', '两融净买入', MA_period)
        # MA_north_inflow_long = self.get_MA_df_long(north_inflow_long, '中信一级行业', '北向净买入', MA_period)
        #
        # MA_aggregate_inflow_long.to_sql(f'ma{MA_period}_aggregate_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
        #                  index=False)
        # MA_margin_inflow_long.to_sql(f'ma{MA_period}_margin_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
        #                  index=False)
        # MA_north_inflow_long.to_sql(f'ma{MA_period}_north_inflow', con=self.alch_engine, schema='processed_data', if_exists='replace',
        #                  index=False)

        # joined_df = self.read_joined_table_as_dataframe(
        #     target_table_name='product_static_info',
        #     target_join_column='internal_id',
        #     join_table_name='markets_daily_long',
        #     join_column='product_static_info_id',
        #     filter_condition=f"product_static_info.type_identifier = 'price_valuation' AND field='收盘价'"
        # )
        # df_upload = self.get_MA_df_upload(joined_df, MA_period=10)
        # df_upload.to_sql('price_valuation_MA10', con=self.alch_engine, schema='processed_data', if_exists='replace', index=False)

    def get_metabase_full_df(self):
        """
        不是metabase question#176 五项资金流动情况(时间序列)的日度变体
        对应不起来了……以后引用metabase question要做好记录
        """
        metabase_query = text(
            """
            SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."北向资金净买入") AS "sum", SUM("source"."ETF净流入") AS "sum_2", SUM("source"."两融净买入") AS "sum_3", SUM("source"."四项流入之和") AS "sum_4", SUM("source"."大股东拟净持仓变动金额") AS "sum_5"
            FROM (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", "source"."sum" AS "sum", COALESCE("source"."sum", 0) AS "北向资金净买入", COALESCE("Question 167"."拟净减持金额", 0) * -1 AS "大股东拟净持仓变动金额", COALESCE("Question 153"."sum", 0) AS "两融净买入", (COALESCE("Question 153"."sum", 0) + COALESCE("source"."sum", 0) + COALESCE("Question 170"."sum", 0)) - COALESCE("Question 167"."拟净减持金额", 0) AS "四项流入之和", COALESCE("Question 170"."sum", 0) AS "ETF净流入", "Question 153"."date" AS "Question 153__date", "Question 153"."中信一级行业" AS "Question 153__中信一级行业", "Question 167"."date" AS "Question 167__date", "Question 167"."Product Static Info__stk_industry_cs" AS "Question 167__Product Static Info__stk_industry_cs", "Question 170"."date" AS "Question 170__date", "Question 170"."Product Static Info_2__stk_industry_cs" AS "Question 170__Product Static Info_2__stk_industry_cs", "Question 167"."拟净减持金额" AS "Question 167__拟净减持金额", "Question 153"."sum" AS "Question 153__sum", "Question 170"."sum" AS "Question 170__sum" FROM (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", substring("public"."markets_daily_long"."product_name" FROM 'CS(.*)') AS "中信一级行业", "Metric Static Info"."type_identifier" AS "Metric Static Info__type_identifier", "Metric Static Info"."internal_id" AS "Metric Static Info__internal_id" FROM "public"."markets_daily_long"
            LEFT JOIN "public"."metric_static_info" AS "Metric Static Info" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info"."internal_id"
            WHERE "Metric Static Info"."type_identifier" = 'north_inflow') AS "source") AS "source" WHERE "source"."field" = '净买入'
            GROUP BY "source"."date", "source"."中信一级行业"
            ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC) AS "source" LEFT JOIN (SELECT "source"."date" AS "date", "source"."中信一级行业" AS "中信一级行业", SUM("source"."value") AS "sum" FROM (SELECT "source"."date" AS "date", "source"."field" AS "field", "source"."value" AS "value", "source"."中信一级行业" AS "中信一级行业" FROM (SELECT "source"."date" AS "date", "source"."product_name" AS "product_name", "source"."field" AS "field", "source"."value" AS "value", substring("source"."product_name" FROM 'CS(.*)') AS "中信一级行业" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value" FROM "public"."markets_daily_long" LEFT JOIN "public"."metric_static_info" AS "Metric Static Info_2" ON "public"."markets_daily_long"."metric_static_info_id" = "Metric Static Info_2"."internal_id" WHERE "Metric Static Info_2"."type_identifier" = 'margin_by_industry') AS "source") AS "source") AS "source" WHERE "source"."field" = '两融净买入额' GROUP BY "source"."date", "source"."中信一级行业" ORDER BY "source"."date" ASC, "source"."中信一级行业" ASC) AS "Question 153" ON ("source"."date" = "Question 153"."date")
               AND ("source"."中信一级行业" = "Question 153"."中信一级行业") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info__stk_industry_cs" AS "Product Static Info__stk_industry_cs", MAX(COALESCE(CASE WHEN "source"."field" = '拟减持金额' THEN "source"."value" END, 0)) - MAX(COALESCE(CASE WHEN "source"."field" = '拟增持金额' THEN "source"."value" END, 0)) AS "拟净减持金额" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "public"."markets_daily_long"."metric_static_info_id" AS "metric_static_info_id", "public"."markets_daily_long"."product_static_info_id" AS "product_static_info_id", "public"."markets_daily_long"."date_value" AS "date_value", "Product Static Info"."internal_id" AS "Product Static Info__internal_id", "Product Static Info"."code" AS "Product Static Info__code", "Product Static Info"."chinese_name" AS "Product Static Info__chinese_name", "Product Static Info"."english_name" AS "Product Static Info__english_name", "Product Static Info"."source" AS "Product Static Info__source", "Product Static Info"."type_identifier" AS "Product Static Info__type_identifier", "Product Static Info"."buystartdate" AS "Product Static Info__buystartdate", "Product Static Info"."fundfounddate" AS "Product Static Info__fundfounddate", "Product Static Info"."issueshare" AS "Product Static Info__issueshare", "Product Static Info"."fund_fullname" AS "Product Static Info__fund_fullname", "Product Static Info"."stk_industry_cs" AS "Product Static Info__stk_industry_cs", "Product Static Info"."product_type" AS "Product Static Info__product_type", "Product Static Info"."etf_type" AS "Product Static Info__etf_type" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info"."internal_id" WHERE "Product Static Info"."type_identifier" = 'major_holder') AS "source" WHERE "source"."Product Static Info__type_identifier" = 'major_holder' GROUP BY "source"."date", "source"."Product Static Info__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info__stk_industry_cs" ASC) AS "Question 167" ON ("source"."date" = "Question 167"."date") AND ("source"."中信一级行业" = "Question 167"."Product Static Info__stk_industry_cs") LEFT JOIN (SELECT "source"."date" AS "date", "source"."Product Static Info_2__stk_industry_cs" AS "Product Static Info_2__stk_industry_cs", SUM("source"."value") AS "sum" FROM (SELECT "public"."markets_daily_long"."date" AS "date", "public"."markets_daily_long"."product_name" AS "product_name", "public"."markets_daily_long"."field" AS "field", "public"."markets_daily_long"."value" AS "value", "Product Static Info_2"."code" AS "Product Static Info_2__code", "Product Static Info_2"."fund_fullname" AS "Product Static Info_2__fund_fullname", "Product Static Info_2"."stk_industry_cs" AS "Product Static Info_2__stk_industry_cs" FROM "public"."markets_daily_long" LEFT JOIN "public"."product_static_info" AS "Product Static Info_2" ON "public"."markets_daily_long"."product_static_info_id" = "Product Static Info_2"."internal_id" WHERE ("public"."markets_daily_long"."field" = '净流入额') AND ("Product Static Info_2"."product_type" = 'fund')) AS "source" GROUP BY "source"."date", "source"."Product Static Info_2__stk_industry_cs" ORDER BY "source"."date" ASC, "source"."Product Static Info_2__stk_industry_cs" ASC) AS "Question 170" ON ("source"."date" = "Question 170"."date") AND ("source"."中信一级行业" = "Question 170"."Product Static Info_2__stk_industry_cs")) AS "source" GROUP BY "source"."date", "source"."中信一级行业" ORDER BY "source"."date" DESC, "source"."中信一级行业" ASC
            """
        )
        full_industry_history = self.alch_conn.execute(metabase_query)
        return pd.DataFrame(full_industry_history,
                            columns=['date', '中信一级行业', '北向资金净买入', 'ETF净流入', '两融净买入',
                                     '四项流入之和', '大股东拟净持仓变动金额'])

    def get_metabase_new_fund_ts(self):
        """
        metabase question 229, 基金-主被动发行规模(日度被引时间序列)
        """
        metabase_query = text(
            """
            SELECT "source"."fundfounddate" AS "fundfounddate", SUM("source"."ETF基金发行份额") AS "sum", SUM("source"."主动类基金发行份额") AS "sum_2", SUM("source"."非债类基金发行份额") AS "sum_3"
            FROM (SELECT "source"."fundfounddate" AS "fundfounddate", "source"."sum" AS "sum", COALESCE("Question 221"."sum", 0) AS "ETF基金发行份额", COALESCE("source"."sum", 0) AS "非债类基金发行份额", COALESCE("source"."sum", 0) - COALESCE("Question 221"."sum", 0) AS "主动类基金发行份额", "Question 221"."fundfounddate" AS "Question 221__fundfounddate", "Question 221"."sum" AS "Question 221__sum", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_2", "Question 221"."fundfounddate" AS "Question 221__fundfounddate_3" FROM (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info"
            WHERE ("public"."product_static_info"."product_type" = 'fund')
               AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%债%')
                OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%存单%') OR ("public"."product_static_info"."fund_fullname" IS NULL))
            GROUP BY "public"."product_static_info"."fundfounddate"
            ORDER BY "public"."product_static_info"."fundfounddate" ASC) AS "source"
            LEFT JOIN (SELECT "public"."product_static_info"."fundfounddate" AS "fundfounddate", SUM("public"."product_static_info"."issueshare") AS "sum" FROM "public"."product_static_info" WHERE ("public"."product_static_info"."product_type" = 'fund') AND (NOT (LOWER("public"."product_static_info"."fund_fullname") LIKE '%债%') OR ("public"."product_static_info"."fund_fullname" IS NULL)) AND (LOWER("public"."product_static_info"."fund_fullname") LIKE '%交易型开放式%') GROUP BY "public"."product_static_info"."fundfounddate" ORDER BY "public"."product_static_info"."fundfounddate" DESC) AS "Question 221" ON "source"."fundfounddate" = "Question 221"."fundfounddate") AS "source" GROUP BY "source"."fundfounddate" ORDER BY "source"."fundfounddate" DESC
            """
        )
        new_fund = self.alch_conn.execute(metabase_query)
        new_fund_ts = pd.DataFrame(new_fund, columns=['date', 'ETF基金发行份额', '主动类基金发行份额', '非债类发行份额']).dropna().set_index('date')
        return new_fund_ts

    def get_metabase_results(self, task, df_full):
        print(f'Calculating for task:{task}')
        match task:
            case 'aggregate_inflow':
                df_inflow_sum = df_full.groupby(['date', '中信一级行业'])['四项流入之和'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_inflow_sum

            case 'margin_inflow':
                df_margin_buy = df_full.groupby(['date', '中信一级行业'])['两融净买入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_margin_buy

            case 'north_inflow':
                df_net_buy = df_full.groupby(['date', '中信一级行业'])['北向资金净买入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_net_buy

            case 'etf_inflow':
                df_etf_flow = df_full.groupby(['date', '中信一级行业'])['ETF净流入'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_etf_flow
            case 'holder_change':
                df_holder_change = df_full.groupby(['date', '中信一级行业'])['大股东拟净持仓变动金额'].apply(
                    lambda x: pd.Series(x.values)).unstack('中信一级行业').reset_index().drop('level_1',
                                                                                              axis=1).set_index('date')
                return df_holder_change
            case _:
                raise Exception(f'task={task} not supported!')

    def logic_reopened_dk_funds(self):
        """
        1. 获取full_name中带有'定期开放'、不包含债的全部基金
        2. 需要获取的数据包括：历次开放申赎的日期、申赎前后的份额变动、最新日期的基金规模
        """
        dk_funds_df = self.select_rows_by_column_strvalue(table_name='product_static_info', column_name='fund_fullname',
                                                          search_value='定期开放',
                                                          selected_columns=['code', 'chinese_name'],
                                                          filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%债%'")

        # 因为定开基金总数不多，因此对这些定开基金统一更新
        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='定开')
        if not today_updated:
            for _, row in dk_funds_df.iterrows():
                downloaded = w.wsd(row['code'],
                                   "NAV_adj,fund_expectedopenday,netasset_total,fund_fundscale,fund_info_name",
                                   self.tradedays_str[-1], self.tradedays_str[-1], "unit=1", usedf=True)[1]
                downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'NAV_ADJ': 'nav_adj',
                                                                      'FUND_EXPECTEDOPENDAY': 'fund_expectedopenday',
                                                                      'FUND_FUNDSCALE': 'fund_fundscale',
                                                                      'NETASSET_TOTAL': 'netasset_total'})
                downloaded['product_name'] = row['chinese_name']
                if downloaded.iloc[0]['code'] == 0:
                    print(f"{row['chinese_name']} {row['code']} 未在w.wsd查询到数据，skipping")
                    continue
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

        # self.upload_joined_products_wide_table(full_name_keyword='定期开放')

    def logic_reopened_cyq_funds(self):
        """
        1. 获取full_name中带有'持有期'、不包含债的全部基金
        2. 需要获取的数据包括：历次开放申赎的日期、申赎前后的份额变动、最新日期的基金规模
        """
        cyq_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='持有期',
            selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare'],
            filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%债%'")

        today_updated = self.is_markets_daily_long_updated_today(field='nav_adj', product_name_key_word='持有')
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

        # self.upload_joined_products_wide_table(full_name_keyword='持有期')

    def logic_etf_lof_funds(self):
        etf_funds_df = self.select_rows_by_column_strvalue(
            table_name='product_static_info', column_name='fund_fullname',
            search_value='交易型开放式',
            selected_columns=['code', 'chinese_name', 'fund_fullname', 'fundfounddate', 'issueshare', 'etf_type'],
            filter_condition="product_type='fund' AND fund_fullname NOT LIKE '%债%' "
                             "AND fund_fullname NOT LIKE '%联接%'")
        # 有一些发行时间较早的行业ETF数据库中没有收录，从excel中读取并更新到数据库
        # 读取行业资金周流入规模文件
        file_path = os.path.join(self.base_config.excels_path, '行业资金周流入规模（5.8-5.13).xlsx')
        df_excel = pd.read_excel(file_path, sheet_name='行业ETF当周净流入统计', index_col=None)
        # 筛选出 df_excel 中存在但 etf_funds_df 中不存在的代码
        missing_codes = df_excel['代码'][~df_excel['代码'].isin(etf_funds_df['code'])]
        for code in missing_codes:
            self._update_specific_funds_meta(code)

        # update ETF所属行业和ETF分类(excel文件里全部为行业ETF)
        df_excel = df_excel.rename(columns={'代码': 'code', '中信一级行业': 'stk_industry_cs'})
        df_excel['etf_type'] = '行业ETF'
        for _, row in df_excel.iterrows():
            self.upload_product_static_info(row, task='etf_industry_and_type')

        # ETF分成3类：行业ETF和主题ETF和指数ETF
        # 默认分类为指数ETF
        etf_funds_df.loc[etf_funds_df['etf_type'].isnull(), 'etf_type'] = '指数ETF'
        # 根据关键字进行分类
        theme_keywords = ['主题']
        etf_funds_df.loc[etf_funds_df['fund_fullname'].str.contains('|'.join(theme_keywords)), 'etf_type'] = '主题ETF'
        # 根据行业资金周流入规模文件进行分类
        industry_etf_codes = df_excel['code'].tolist()
        etf_funds_df.loc[etf_funds_df['code'].isin(industry_etf_codes), 'etf_type'] = '行业ETF'
        etf_funds_df = etf_funds_df.sort_values(by='etf_type')

        # 更新markets_daily_long '净流入额'时间序列
        for _, row in etf_funds_df.iterrows():
            self._update_etf_inflow(row)

    def _update_etf_inflow(self, etf_info_row):
        existing_dates = self.select_column_from_joined_table(
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column=f'date',
            filter_condition=f"code='{etf_info_row['code']}' ORDER BY date ASC"
        )
        fund_found_date = self.select_column_from_joined_table(
            filter_condition=f"code='{etf_info_row['code']}'",
            target_table_name='product_static_info',
            target_join_column='internal_id',
            join_table_name='markets_daily_long',
            join_column='product_static_info_id',
            selected_column=f'fundfounddate',
        )

        # 没有基金成立日，说明基金未发行成功或尚未发行，跳过
        if not fund_found_date:
            return

        # 对于有fund_found_date的情况
        gross_missing_dates = self._check_data_table('markets_daily_long', 'fund',
                                                     additional_filter=f"code='{etf_info_row['code']}'")
        # 检查最早的 existing_date 是否在 fund_found_date 的3个月之内
        earliest_existing_date = min(existing_dates)
        if earliest_existing_date < (fund_found_date[0] + datetime.timedelta(days=100)):
            # 缺失日期设为最近2周的日期序列
            missing_dates = self.tradedays[-10:]
            missing_start_date = missing_dates[0]
        else:
            # 否则说明该etf没有历史数据
            missing_start_date = max(gross_missing_dates[0], fund_found_date[0])
            missing_dates = gross_missing_dates

        # 这个净流入额的变动日期和基金份额-本分级份额变动日期一样，应该就是份额变动乘以净值
        print(f"_update_etf_inflow Downloading mf_netinflow for {etf_info_row['code']} "
              f"b/t {missing_start_date} and {missing_dates[-1]}")
        downloaded = w.wsd(etf_info_row['code'], "mf_netinflow",
                           missing_start_date, missing_dates[-1], "unit=1", usedf=True)[1]
        downloaded_filtered = downloaded[downloaded['MF_NETINFLOW'] != 0]
        downloaded_filtered = downloaded_filtered.reset_index().rename(
            columns={'index': 'date', 'MF_NETINFLOW': '净流入额'})
        # 去除已经存在的日期
        existing_dates = self.select_existing_dates_from_long_table('markets_daily_long',
                                                                    product_name=etf_info_row['chinese_name'],
                                                                    field='净流入额')
        downloaded_filtered = downloaded_filtered[~downloaded_filtered['date'].isin(existing_dates)]
        downloaded_filtered['product_name'] = etf_info_row['chinese_name']
        upload_value = downloaded_filtered[['product_name', 'date', '净流入额']].melt(
            id_vars=['product_name', 'date'], var_name='field', value_name='value')

        upload_value.dropna().to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def _update_specific_funds_meta(self, code: str):
        info = w.wsd(code,
                     "fund_fullname,fund_info_name,fund_fullnameen",
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

    def update_all_funds_info(self):
        """
        1.检查product_static_info中存在的基金信息记录，获取需要更新的日期
            - product_static_info的internal_id列为markets_daily_long表product_static_info_id列的外键，将两个表连接起来，然后通过type=fund筛选出关注的行
            - 针对筛选出的数据行，查询product_static_info表中buystartdate列的最大值和最小值，获取存在数据的区间
            - 如果存在数据的区间为空，missing_dates为self.all_dates
            - self.all_dates[0]到存在数据的区间的下限为第一段missing_dates，存在数据的区间的上限到self.all_dates[-1]为第二段missing_dates
        2.根据missing_dates执行w.wset下载数据，对于空missing_dates跳过执行w.wset
        3.利用下载得到的数据将数据上传至数据库
            - windcode上传至product_static_info的code列
            - name上传至product_static_info的chinese_name列
            - buystartdate,issueshare,fundfounddate因为是固定的，上传至product_static_info的各自列
            - openbuystartdate,openrepurchasestartdate这些列(这些字符串作为field)上传至markets_daily_long表，以长格式数据储存。
        """
        # 获取需要更新的日期区间
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
            missing_buystartdates = self.get_missing_dates(all_dates=self.tradedays, existing_dates=existing_dates)

        self._update_funds_by_buystartdate(missing_buystartdates)
        # 有一些基金没有发行日/认购起始日的记录，wind没有抓取到，因此必要。
        # 因为缺失成立日的基金本来就不多，因此可行，不浪费quota。
        self._update_special_funds_missing_buystartdate(process_historical=True)
        self._update_funds_missing_fundfounddate()

        self._update_funds_name()
        # 取不到数，暂不更新
        # self._update_funds_issueshare()

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
                               self.tradedays_str[-1], self.tradedays_str[-1], "unit=1", usedf=True)[1]
            # 重置索引并将其作为一列, 重命名列名
            downloaded = downloaded.reset_index().rename(columns={'index': 'code', 'FUND_FULLNAME': 'fund_fullname',
                                                                  'FUND_FULLNAMEEN': 'english_name'})
            print(f'Updating {code} name')
            self.upload_product_static_info(downloaded.squeeze(), task='fund_name')

    def _update_funds_issueshare(self):
        code_set = self.select_existing_values_in_target_column(
            'product_static_info',
            ['code', 'chinese_name', 'buystartdate', 'fundfounddate'],
            "fundfounddate is not null and issueshare is null and product_type='fund'"
        )

        for _, row in code_set.iterrows():
            code = row['code']
            print(f'Downloading issueshare for {code}')
            # wsd取不到数，应该是数据缺失。较早的发行规模暂时不更新了
            downloaded = w.wsd(code,
                               "issue_unit",
                               self.tradedays_str[-1], self.tradedays_str[-1], "unit=1", usedf=True)[1]
            if pd.isna(downloaded.iloc[0, 0]):
                continue

            print(f'Uploading {code} issueshare')
            upload_df = downloaded.reset_index().rename(columns={'index': 'code', 'ISSUE_UNIT': 'issueshare'}) / 1e8
            upload_df['chinese_name'] = row['chinese_name']
            upload_df['buystartdate'] = row['buystartdate']
            upload_df['fundfounddate'] = row['fundfounddate']
            upload_df['source'] = 'wind'
            upload_df['product_type'] = 'fund'
            self.insert_product_static_info(upload_df.squeeze())

    def _update_funds_by_buystartdate(self, missing_buystartdates):
        missing_dates = sorted(missing_buystartdates)
        if not missing_dates:
            print("No missing dates for update_all_funds_info")
            return

        # 执行数据下载
        # 以认购起始日作为筛选条件，选取的数据更完整、更有前瞻性。只选取严格意义上的新发基金。
        for missing_date in missing_dates[-30:]:
            print(f'Downloading fundissuegeneralview on {missing_date} for _update_funds_by_buystartdate')
            downloaded_df = w.wset("fundissuegeneralview",
                                   f"startdate={missing_date};enddate={missing_date};datetype=startdate;isvalid=yes;"
                                   f"deltranafter=yes;field=windcode,name,buystartdate,issueshare,fundfounddate,"
                                   f"openbuystartdate,openrepurchasestartdate",
                                   usedf=True)[1]
            if downloaded_df.empty:
                print(
                    f"No fundissuegeneralview data on {missing_date}, no data downloaded for _update_funds_by_buystartdate")
                continue

            # 解析下载的数据并上传至product_static_info
            product_metric_upload_df = downloaded_df[
                ['windcode', 'name', 'buystartdate', 'fundfounddate', 'issueshare']].rename(
                columns={'windcode': 'code', 'name': 'chinese_name'})
            product_metric_upload_df['source'] = 'wind'
            product_metric_upload_df['product_type'] = 'fund'
            for _, row in product_metric_upload_df.iterrows():
                self.insert_product_static_info(row)

            # 将非静态数据上传至markets_daily_long
            markets_daily_long_upload_df = downloaded_df[
                ['name', 'openbuystartdate', 'openrepurchasestartdate']].rename(
                columns={'name': 'chinese_name',
                         'openbuystartdate': '开放申购起始日',
                         'openrepurchasestartdate': '开放赎回起始日'
                         })
            markets_daily_long_upload_df = markets_daily_long_upload_df.melt(id_vars=['chinese_name'], var_name='field',
                                                                             value_name='date_value')
            markets_daily_long_upload_df = markets_daily_long_upload_df.dropna(subset=['date_value']).rename(
                columns={'chinese_name': 'product_name'})
            # 这里date的含义是信息记录日
            markets_daily_long_upload_df['date'] = self.all_dates[-1]

            # 上传前要剔除已存在的product
            existing_products = self.select_column_from_joined_table(
                target_table_name='product_static_info',
                target_join_column='internal_id',
                join_table_name='markets_daily_long',
                join_column='product_static_info_id',
                selected_column='chinese_name',
                filter_condition="product_static_info.product_type = 'fund' "
                                 "AND (markets_daily_long.field = '开放申购起始日'"
                                 "OR markets_daily_long.field = '开放赎回起始日')"
            )
            filtered_df = markets_daily_long_upload_df[
                ~markets_daily_long_upload_df['product_name'].isin(existing_products)]
            filtered_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def _update_special_funds_missing_buystartdate(self, process_historical=False):
        if not process_historical:
            print('Skipping _update_special_funds_missing_buystartdate')
            return

        existing_codes = self.select_existing_values_in_target_column('product_static_info',
                                                                      'code',
                                                                      "product_type='fund'")
        all_funds = pd.read_excel(self.base_config.excels_path + '缺失发行日期的基金.xlsx', header=0,
                                  engine='openpyxl')
        df_cleaned = all_funds[all_funds['发行日期'].isnull()].sort_values(by='证券简称').dropna(
            subset=['证券简称'])
        for code in df_cleaned['证券代码'].tolist():
            if code not in existing_codes:
                print(f'Downloading fund info {code} for _update_special_funds_missing_buystartdate')
                downloaded_df = \
                    w.wsd(code, "issue_date,fund_setupdate,sec_name,fund_fullname,fund_fullnameen,issue_unit",
                          self.tradedays_str[-1], self.tradedays_str[-1], "", usedf=True)[1]
                if downloaded_df.empty:
                    print(
                        f"Empty data downloaded for {code}, in _update_funds_by_buystartdate")
                    continue

                # 解析下载的数据并上传至product_static_info
                upload_df = downloaded_df.reset_index().rename(
                    columns={'index': 'code', 'SEC_NAME': 'chinese_name', 'ISSUE_DATE': 'buystartdate',
                             'FUND_SETUPDATE': 'fundfounddate', 'FUND_FULLNAME': 'fund_fullname',
                             'FUND_FULLNAMEEN': 'english_name', 'ISSUE_UNIT': 'issueshare'})
                upload_df['issueshare'] = upload_df['issueshare'] / 1e8
                upload_df['source'] = 'wind'
                upload_df['product_type'] = 'fund'
                for _, row in upload_df.iterrows():
                    self.insert_product_static_info(row)

    def _update_funds_missing_fundfounddate(self):
        # 用于更新基金成立后的信息
        funds_missing_fundfounddate = self.select_existing_values_in_target_column('product_static_info',
                                                                                   ['code', 'buystartdate',
                                                                                    'chinese_name'],
                                                                                   "fundfounddate is NULL and product_type='fund'")
        # 筛选buystartdate在近3个月之内的行，再老的就是没有发行成功的基金，不必更新
        df_filtered = funds_missing_fundfounddate[
            funds_missing_fundfounddate['buystartdate'] >= self.tradedays[-70]]
        # df_filtered = funds_missing_fundfounddate

        for _, row in df_filtered.iterrows():
            code = row['code']
            print(f'Downloading fund info {code} for _update_funds_missing_fundfounddate')
            downloaded_df = w.wsd(code, "issue_date,fund_setupdate,issue_unit",
                                  self.tradedays_str[-1], self.tradedays_str[-1], "", usedf=True)[1]
            # 解析下载的数据并上传至product_static_info
            upload_df = downloaded_df.reset_index().rename(
                columns={'index': 'code', 'ISSUE_DATE': 'buystartdate', 'FUND_SETUPDATE': 'fundfounddate',
                         'ISSUE_UNIT': 'issueshare'})
            # 尚未发行或发行失败
            if pd.isna(upload_df.iloc[0]['issueshare']):
                continue
            upload_df['issueshare'] = upload_df['issueshare'] / 1e8
            upload_df['source'] = 'wind'
            upload_df['product_type'] = 'fund'
            upload_df['chinese_name'] = row['chinese_name']
            for _, info in upload_df.iterrows():
                self.insert_product_static_info(info)

    def logic_buy_back(self):
        #TODO: 股票回购
        pass