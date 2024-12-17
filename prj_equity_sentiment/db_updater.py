# coding=gbk
# Time Created: 2023/6/21 8:53
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
from utils import has_large_date_gap, match_recent_tradedays


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.industry_data_updater = IndustryDataUpdater(self)
        self.industry_stk_updater = IndustryStkUpdater(self)

    def run_all_updater(self):
        self.industry_data_updater.logic_industry_volume()
        self.industry_data_updater.logic_industry_large_order()
        self.industry_data_updater.logic_industry_order_inflows()
        # self.industry_stk_updater.logic_industry_stk_price_volume()
        # self.logic_analyst()

    def _check_data_table(self, type_identifier, **kwargs):
        # Retrieve the optional filter condition
        additional_filter = kwargs.get('additional_filter')

        # 获取需要更新的日期区间
        match type_identifier:
            case 'industry_volume':
                filter_condition = f"product_static_info.product_type='index' AND markets_daily_long.field='成交额'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'price_volume':
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='stocks_daily_partitioned_close',
                )
            case 'stk_price_volume':
                stock_code = kwargs.get('stock_code')
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='stocks_daily_partitioned_close',
                    product_name=stock_code
                )
            case 'industry_large_order':
                filter_condition = f"markets_daily_long.field='主力净流入额'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
            case 'industry_order_inflows':
                field = kwargs.get('field')
                existing_dates = self.select_existing_dates_from_long_table(
                    table_name='markets_daily_long',
                    field=field,
                    filter_condition=additional_filter
                )
            case 'analyst':
                stock_code = kwargs.get('stock_code')
                filter_condition = f"markets_daily_long.product_name='{stock_code}' AND markets_daily_long.field='券商-研报标题'"
                existing_dates = self.select_column_from_joined_table(
                    target_table_name='product_static_info',
                    target_join_column='internal_id',
                    join_table_name='markets_daily_long',
                    join_column='product_static_info_id',
                    selected_column=f'date',
                    filter_condition=filter_condition
                )
                # 研报数据非连续，因此需要特别处理。
                if len(existing_dates) == 0:
                    missing_dates = self.tradedays
                else:
                    missing_dates = []
                    missing_dates[0] = existing_dates[-1] + datetime.timedelta(days=1)
                    missing_dates[1] = self.tradedays[-1]
                return missing_dates

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

    def logic_analyst(self):
        existing_codes = self.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        for code in existing_codes:
            missing_dates = self._check_data_table(type_identifier='analyst',
                                                   stock_code=code)
            missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]
            print(f"tushare downloading 研报历史 for {code} {missing_dates[0]}-{missing_dates[1]}")
            df = self.pro.report_rc(**{
                "ts_code": code,
                "report_date": "",
                "start_date": missing_dates_str_list[0],
                "end_date": missing_dates_str_list[-1],
                "limit": "",
                "offset": ""
            }, fields=[
                "ts_code",
                "report_date",
                "report_title",
                "report_type",
                "classify",
                "org_name",
                "rating"
            ])
            df = df.rename(columns={
                "ts_code": 'product_name',
                "report_date": 'date',
                "report_title": '报告标题',
                "report_type": '报告类型',
                "classify": '报告分类',
                "org_name": '机构名称',
                "rating": '卖方评级',
            }).drop_duplicates()
            df['券商-研报标题'] = df['机构名称'] + '-' + df['报告标题']
            df_selected = df[['date', 'product_name', '券商-研报标题']].drop_duplicates()
            df_report_counts = df_selected.groupby(['product_name', 'date'])['券商-研报标题'].nunique().reset_index()
            df_report_counts.columns = ['product_name', 'date', '报告数量']

            #TODO: 因为是断续数据，需要剔除已存在的

            # 转换为长格式数据框
            df_long = pd.melt(df_report_counts, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)


class IndustryDataUpdater:
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater
        self.excel_data = None

    def logic_industry_volume(self):
        """
        用wind.py更新行业和全A的成交额，用来构建拥挤度。
        """
        # 不需检查或更新meta_table，因为行业指数和全A已经存在于product_static_info
        # 直接检查或更新data_table
        missing_dates = self.db_updater._check_data_table(type_identifier='industry_volume',
                                               additional_filter=f"field='成交额'")
        missing_dates_filtered = self.db_updater.remove_today_if_trading_day(missing_dates)
        if missing_dates_filtered:
            self._upload_missing_data_industry_volume(missing_dates_filtered)

    def _upload_missing_data_industry_volume(self, missing_dates):
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        if missing_dates[0] == self.db_updater.tradedays[-1]:
            print('Only today is missing, skipping update _upload_missing_data_industry_volume')
            return
        for date in missing_dates[::-1]:
            for code in industry_codes:
                print(
                    f'Wind downloading and upload volume,amt,turn for {code} {date} _upload_missing_data_industry_volume')
                df = w.wsd(code, "volume,amt,turn", date,
                           date, "unit=1", usedf=True)[1]
                if df.empty:
                    print(
                        f"Missing data for {date} {code}, _upload_missing_data_industry_volume")
                    continue
                df_upload = df.rename(
                    columns={'VOLUME': '成交量',
                             'AMT': '成交额',
                             'TURN': '换手率',
                             })
                df_upload['date'] = date
                df_upload['product_name'] = code
                df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                           value_name='value').dropna()

                df_upload.to_sql('markets_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def logic_industry_large_order(self):
        """
        用wind.py更新行业和全A的主力净流入额，用来构建拥挤度。
        """
        # 不需检查或更新meta_table，因为行业指数和全A已经存在于product_static_info
        missing_dates = self.db_updater._check_data_table(type_identifier='industry_large_order',
                                               additional_filter=f"field='主力净流入额'")
        # 每天大概四点多就出了结果，不用等到第二天再更新
        # missing_dates_filtered = self.db_updater.remove_today_if_trading_day(missing_dates)

        if missing_dates:
            self._upload_missing_data_industry_large_order(missing_dates)

    def _upload_missing_data_industry_large_order(self, missing_dates):
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('product_type', 'index')
        )

        # if missing_dates[0] == self.db_updater.tradedays[-1]:
        #     print('Only today is missing, skipping update _upload_missing_data_industry_large_order')
        #     return

        for code in industry_codes:
            print(
                f'Wind downloading and upload mfd_inflow_m for {code} {missing_dates[0]}~{missing_dates[-1]} '
                f'_upload_missing_data_industry_large_order')
            df = w.wsd(code, "mfd_inflow_m", missing_dates[0],
                       missing_dates[-1], "unit=1", usedf=True)[1]
            if df.empty:
                print(
                    f"Missing data for {code} {missing_dates[0]}~{missing_dates[-1]}, "
                    f"_upload_missing_data_industry_large_order")
                continue
            if len(missing_dates) == 1:
                df_upload = df.reset_index().rename(
                    columns={'index': 'product_name',
                             'MFD_INFLOW_M': '主力净流入额',
                             })
                df_upload['date'] = missing_dates[0]
            else:
                df_upload = df.reset_index().rename(
                    columns={'index': 'date',
                             'MFD_INFLOW_M': '主力净流入额',
                             })
                df_upload['product_name'] = code
            df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                       value_name='value').dropna()
            # 去除已经存在的日期
            existing_dates = self.db_updater.select_existing_dates_from_long_table('markets_daily_long',
                                                                        product_name=code,
                                                                        field='主力净流入额')
            downloaded_filtered = df_upload[~df_upload['date'].isin(existing_dates)]

            downloaded_filtered.to_sql('markets_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def logic_industry_order_inflows(self):
        """
        1）挂单额小于4万元，小单；
        2）挂单额4万元到20万元之间，中单；
        3）挂单额20万元至100万元之间，大单；
        4）挂单额大于100万元，超大单。
        流入额：该规模单子买入成交金额
        净买入额：该规模单子买入-卖出金额
        净主动买入额：该规模单子主动去成交而非挂单成交的金额
        large_order为大单和超大单的净买入额之和，非主动
        四种规模的任意口径额，不同规模加和应接近0，才是合理的
        """
        # 获取所有行业代码
        industry_codes = self.db_updater.select_existing_values_in_target_column(
            'product_static_info',
            'code',
            ('internal_id BETWEEN 69000 AND 69030')
        )

        trader_type_dict = {
            1: '机构',
            2: '大户',
            3: '中户',
            4: '散户',
        }
        self.trader_type_dict = trader_type_dict

        # self._preprocess_excel(rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\历史order_inflows.xlsx")

        for code in industry_codes:
            for trader_type in [1, 2, 3, 4]:
                trader_type_name = trader_type_dict.get(trader_type, f"Type{trader_type}")
                field = f'流入额_{trader_type_name}'  # 仅做筛选日期用
                additional_filter = f"product_name='{code}'"

                # 查询当前 code 和 trader_type 的缺失日期
                missing_dates = self.db_updater._check_data_table(
                    type_identifier='industry_order_inflows',
                    field=field,
                    additional_filter=additional_filter
                )

                if missing_dates:
                    print(
                        f"开始上传缺失数据 - 行业代码: {code}, 交易类型: {trader_type_name}, 缺失日期数: {len(missing_dates)}")
                    self._upload_missing_data_industry_order_inflows(code, trader_type, missing_dates)
                else:
                    print(f"No missing dates for 行业代码: {code}, 交易类型: {trader_type_name}")


    def _upload_missing_data_industry_order_inflows(self, code, trader_type, missing_dates):
        if code == 'CI005030.WI' and len(missing_dates)>1: # 该行业缺早期数据
            missing_dates = [date for date in missing_dates if date > datetime.date(2019, 11, 29)]
        # 获取缺失日期的起止时间
        start_date = missing_dates[0]
        end_date = missing_dates[-1]

        # 尝试从Excel中获取数据
        df_upload = self._get_data_from_excel(code, trader_type, start_date, end_date)

        if df_upload is None:
        # 如果Excel中没有数据，则通过w.wsd下载数据
            print(
                f'Wind downloading and upload order_inflows for {code} trader_type {self.trader_type_dict[trader_type]} {start_date}~{end_date} '
                f'_upload_missing_data_industry_order_inflows')
            df = w.wsd(code, "mfd_buyamt_d,mfd_netbuyamt,mfd_netbuyamt_a", start_date,
                       missing_dates[-1], "unit=1", traderType=trader_type, usedf=True)[1]
            if df.empty:
                print(
                    f"Missing data for {code} {start_date}~{end_date}, "
                    f"_upload_missing_data_industry_order_order_inflows")
                return
            if len(missing_dates) == 1:
                df_upload = df.reset_index().rename(
                    columns={'index': 'product_name',
                             'MFD_BUYAMT_D': f'流入额_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT': f'净买入额_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT_A': f'净主动买入额_{self.trader_type_dict[trader_type]}',
                             })
                df_upload['date'] = missing_dates[0]
            else:
                df_upload = df.reset_index().rename(
                    columns={'index': 'date',
                             'MFD_BUYAMT_D': f'流入额_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT': f'净买入额_{self.trader_type_dict[trader_type]}',
                             'MFD_NETBUYAMT_A': f'净主动买入额_{self.trader_type_dict[trader_type]}',
                             })
                df_upload['product_name'] = code

        df_upload = df_upload.melt(id_vars=['date', 'product_name'], var_name='field',
                                   value_name='value').dropna()
        self.db_updater.upsert_dataframe_to_postgresql(df_upload, 'markets_daily_long',
                                            ['date', 'product_name', 'field', 'code'])

    def _preprocess_excel(self, excel_path):
        """
        预处理Excel文件，将其转换为结构化的DataFrame。
        根据用户描述的复杂结构，逐个解析每个数据块。
        """
        try:
            # 读取Excel文件，无表头
            raw_df = pd.read_excel(excel_path, sheet_name='Sheet3', header=None, dtype=str)

            # 列数
            total_cols = raw_df.shape[1]

            # 找到所有数据块的起始列：第0行包含"流入额"
            data_block_cols = raw_df.columns[
                raw_df.iloc[0].astype(str).str.contains('额', na=False)
            ].tolist()

            # 初始化一个列表存储所有数据
            data_records = []

            for start_col in data_block_cols:
                # 获取字段元数据，例如 "流入额 [单位]元 [类型]机构"
                field_metadata = raw_df.iloc[0, start_col]

                if pd.isna(field_metadata):
                    continue

                # 解析字段名，例如 "流入额 [单位]元 [类型]机构" -> "流入额_机构"
                if '[类型]' in field_metadata:
                    parts = field_metadata.split('[类型]')
                    field_name = (
                            parts[0].strip().replace('\n', '').replace('[单位]元', '') +
                            '_' +
                            parts[1].strip(']').strip()
                    )
                else:
                    raise Exception('Not expected in splitting metadata str')

                # 确定数据块的结束列（下一个数据块的起始列之前的列）
                current_index = data_block_cols.index(start_col)
                if current_index < len(data_block_cols) - 1:
                    end_col = data_block_cols[current_index + 1] - 1
                else:
                    end_col = total_cols - 1

                # 获取中文标题（第2行，索引为2）
                chinese_headers = raw_df.iloc[2, start_col:end_col + 1].tolist()

                # 获取英文代码（第3行，索引为3）
                codes = raw_df.iloc[3, start_col:end_col + 1].tolist()

                # 遍历当前数据块中的每个行业列（跳过第一列“日期”）
                date_col = start_col -1
                for col_idx in range(start_col, end_col + 1):
                    code = codes[col_idx - start_col]
                    if pd.isna(code):
                        # 如果行业代码为空，跳过
                        continue

                    # 遍历数据行（从第4行开始，索引为4）
                    for row in range(4, len(raw_df)):
                        date_str = raw_df.iloc[row, date_col]  # 假设日期在流入额前一列
                        if pd.isna(date_str):
                            continue

                        try:
                            date = pd.to_datetime(date_str).date()
                        except:
                            # 如果日期转换失败，跳过该行
                            continue

                        value_str = raw_df.iloc[row, col_idx]
                        if pd.isna(value_str):
                            continue

                        try:
                            value = float(value_str)
                        except ValueError:
                            # 如果值转换失败，跳过
                            continue

                        data_records.append({
                            'date': date,
                            'code': code,
                            'field': field_name,
                            'value': value
                        })

            # 创建DataFrame
            self.excel_data = pd.DataFrame(data_records)

            # 记录预处理成功的信息
            print("Successfully preprocessed Excel data.")

        except Exception as e:
            # 捕获并记录所有异常
            print(f"Error preprocessing Excel file: {e}")
            self.excel_data = None

    def _get_data_from_excel(self, code, trader_type, start_date, end_date):
        """
        从预处理的Excel数据中获取指定的行业代码、交易类型和日期范围的数据。
        如果数据存在，则返回DataFrame；否则，返回None。

        :param code: 行业代码
        :param trader_type: 交易类型（整数，1-4）
        :param start_date: 起始日期（字符串格式）
        :param end_date: 结束日期（字符串格式）
        :return: DataFrame或None
        """
        if self.excel_data is None:
            print("Excel data not loaded or preprocessing failed.")
            return None

        trader_type_dict = {
            1: '机构',
            2: '大户',
            3: '中户',
            4: '散户',
        }
        trader_type_name = trader_type_dict.get(trader_type, f"Type{trader_type}")

        # 转换日期格式
        try:
            start_date_obj = pd.to_datetime(start_date).date()
            end_date_obj = pd.to_datetime(end_date).date()
        except Exception as e:
            print(f"Error converting dates: {e}")
            return None

        # 过滤DataFrame
        filtered_df = self.excel_data[
            (self.excel_data['code'] == code) &
            (self.excel_data['field'].str.contains(trader_type_name)) &
            (self.excel_data['date'] >= start_date_obj) &
            (self.excel_data['date'] <= end_date_obj)
            ]

        if filtered_df.empty:
            print(f"No Excel data found for {code} {trader_type_name} from {start_date} to {end_date}.")
            return None

        # 将数据转换为原始格式，以便上传
        try:
            # Pivot to wide format
            pivot_df = filtered_df.pivot(index='date', columns='field', values='value').reset_index()
            pivot_df['product_name'] = code

            if pivot_df.empty:
                return None

            return pivot_df
        except Exception as e:
            print(f"Error processing Excel data for {code} {trader_type_name}: {e}")
            return None


class IndustryStkUpdater:
    def __init__(self, db_updater: DatabaseUpdater):
        self.db_updater = db_updater

    def _check_meta_table(self):
        # 剔除掉万德全A
        sector_codes = self.db_updater.select_existing_values_in_target_column('product_static_info',
                                                                    ['code', 'chinese_name'],
                                                                    "product_type='index'")
        sector_codes = sector_codes[sector_codes['chinese_name'] != '万德全A']
        for _, row in sector_codes.iterrows():
            industry_name = row['chinese_name']
            # 先判断是否需要更新，以防浪费quota
            sector_update_date = self.db_updater.select_existing_values_in_target_column('product_static_info',
                                                                              'update_date',
                                                                              f"product_type='stock' AND stk_industry_cs='{industry_name}'")
            # 如果现在没有数据，或最近更新的时间超过了30天
            if not sector_update_date:
                print('No sector_update_date, logic_industry_stk_price_volume need update')
                return True
            elif len(sector_update_date) == 1:
                sector_update_datetime = datetime.datetime.strptime(sector_update_date[0], '%Y-%m-%d').date()
                if sector_update_datetime < self.db_updater.tradedays[-1] - pd.Timedelta(days=30):
                    print('sector_update_date last updated over a month ago, '
                          'logic_industry_stk_price_volume need update')
                    return True
                else:
                    print('sector_update_date last updated less than a month ago, '
                          'logic_industry_stk_price_volume not need update')
                    return False
            else:
                raise Exception('len(sector_update_date) should be either 0 or 1')

    def logic_industry_stk_price_volume(self):
        """
        用tushare更新各行业包含的股票（及其量价），用来构建拥挤度。
        """
        # 检查或更新meta_table
        need_update_meta_table = self._check_meta_table()
        # 检查或更新data_table
        missing_dates = self.db_updater._check_data_table(type_identifier='price_volume')
        missing_dates_filtered = self.db_updater.remove_today_if_trading_time(missing_dates)

        if need_update_meta_table:
            self._upload_missing_meta_stk_industry_price_volume()
        if missing_dates_filtered:
            self._upload_whole_market_data_by_missing_dates(missing_dates_filtered)

        self._upload_missing_data_stk_price_volume()

    def _upload_missing_meta_stk_industry_price_volume(self):
        print(f"Wind downloading '全部A股' sectorconstituent for _upload_missing_meta_stk_industry_price_volume")
        today_stocks_df = w.wset("sectorconstituent",
                                          f"date={self.db_updater.tradedays_str[-2]};"
                                          f"sectorid=a001010100000000", usedf=True)[1]
        sector_stk_codes = today_stocks_df['wind_code'].tolist()
        existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                                 f"product_type='stock'")
        existing_codes_missing_industry = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                                 f"product_type='stock' AND (stk_industry_cs='NaN' OR stk_industry_cs is null)")
        new_stk_codes = set(sector_stk_codes) - set(existing_codes)
        stk_needing_update = new_stk_codes.union(set(existing_codes_missing_industry))
        selected_rows = today_stocks_df[today_stocks_df['wind_code'].isin(stk_needing_update)]

        # 构建df，列包含code, chinese_name, source, stk_industry_cs, product_type, update_date
        df_upload = selected_rows[['wind_code', 'sec_name']].rename(
            columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
        df_upload['source'] = 'wind'
        df_upload['product_type'] = 'stock'
        df_upload['update_date'] = self.db_updater.all_dates_str[-1]
        for i, row_stk in df_upload.iterrows():
            code = row_stk['code']
            date = self.db_updater.tradedays[-2]
            print(
                f'Wind downloading industry_citic for {code} on {date} for _upload_missing_meta_stk_industry_price_volume')
            info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
                            usedf=True)[1]
            if info_df.empty:
                print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
                continue
            industry = info_df.iloc[0]['INDUSTRY_CITIC']
            row_stk['stk_industry_cs'] = industry
            self.db_updater.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
            self.db_updater.insert_product_static_info(row_stk)

        # 从行业组成角度更新个股行业，代码略冗余。
        # sector_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', ['code', 'chinese_name'],
        #                                                             "product_type='index'")
        # sector_codes = sector_codes[sector_codes['chinese_name'] != '万德全A']
        #
        # for _, row in sector_codes.iterrows():
        #     sector_code = row['code']
        #     industry_name = row['chinese_name']
        #     print(f'Wind downloading {industry_name} sectorconstituent for _upload_missing_meta_stk_industry_price_volume')
        #     today_industry_stocks_df = w.wset("sectorconstituent",
        #                                       f"date={self.db_updater.tradedays_str[-2]};"
        #                                       f"windcode={sector_code}", usedf=True)[1]
        #     sector_stk_codes = today_industry_stocks_df['wind_code'].tolist()
        #     existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
        #                                                                   f"product_type='stock' AND stk_industry_cs='{industry_name}'")
        #     new_stk_codes = set(sector_stk_codes) - set(existing_codes)
        #     selected_rows = today_industry_stocks_df[today_industry_stocks_df['wind_code'].isin(new_stk_codes)]
        #
        #     # 构建df，列包含code, chinese_name, source, stk_industry_cs, product_type, update_date
        #     df_upload = selected_rows[['wind_code', 'sec_name']].rename(
        #         columns={'wind_code': 'code', 'sec_name': 'chinese_name'})
        #     df_upload['source'] = 'wind and tushare'
        #     df_upload['product_type'] = 'stock'
        #     df_upload['update_date'] = self.db_updater.tradedays_str[-2]
        #     for i, row_stk in df_upload.iterrows():
        #         code = row_stk['code']
        #         date = self.db_updater.tradedays[-2]
        #         print(f'Wind downloading industry_citic for {code} on {date} for _upload_missing_meta_stk_industry_price_volume')
        #         info_df = w.wsd(code, "industry_citic", f'{date}', f'{date}', "unit=1;industryType=1",
        #                         usedf=True)[1]
        #         if info_df.empty:
        #             print(f"Missing data for {code} on {date}, no data downloaded for industry_citic")
        #             continue
        #         industry = info_df.iloc[0]['INDUSTRY_CITIC']
        #         row_stk['stk_industry_cs'] = industry
        #         self.db_updater.adjust_seq_val(seq_name='product_static_info_internal_id_seq')
        #         self.db_updater.insert_product_static_info(row_stk)

    def _upload_whole_market_data_by_missing_dates(self, missing_dates: list):
        for date in missing_dates:
            date_str = date.strftime('%Y%m%d')
            print(f"tushare downloading 全市场行情 for {date}")
            df = self.db_updater.pro.daily(**{
                "ts_code": "",
                "trade_date": date_str,
                "start_date": "",
                "end_date": "",
                "offset": "",
                "limit": ""
            }, fields=[
                "ts_code",
                "open",
                "high",
                "low",
                "close",
                "pct_chg",
                "vol",
                "amount"
            ])
            df = df.rename(columns={
                "ts_code": 'product_name',
                "open": '开盘价',
                "high": '最高价',
                "low": '最低价',
                "close": '收盘价',
                "pct_chg": '当日涨跌幅',
                "vol": '成交量',
                "amount": '成交额'
            })
            df['date'] = date
            # 转换为长格式数据框
            print('_upload_whole_market_data_by_missing_dates上传中……')
            df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

    def _upload_missing_data_stk_price_volume(self):
        """
        更新时间：每日收盘后
        """
        current_time = datetime.datetime.now()
        close_time = datetime.datetime.now().replace(hour=15, minute=10, second=0, microsecond=0)
        if current_time<close_time:
            print("尚未收盘，日度股票数据不做更新")
            return

        existing_codes = self.db_updater.select_existing_values_in_target_column('product_static_info', 'code',
                                                                      f"product_type='stock'")
        empty_request_count = 0
        for code in existing_codes:
            print(f"processing {code}")
            missing_dates = self.db_updater._check_data_table(type_identifier='stk_price_volume',
                                                   stock_code=code)
            missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]

            if code.startswith("900") or code.startswith("200"):
                # B股tushare没有数据
                continue
            if not missing_dates:
                continue

            if 2 <= len(missing_dates) <= 150:
                # 太久远的不更新
                # 保留近半年的missing_dates进行更新
                missing_dates_recent = [date for date in missing_dates if date in self.db_updater.tradedays[-150:]]
                missing_dates_str_list = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates_recent]
            elif has_large_date_gap(missing_dates):
                print(missing_dates) # TODO 出现太频繁需要debug
                print(f'{code} 期间有停牌, 但已经更新过了， skipping update _upload_missing_data_stk_price_volume')
                continue
            elif not has_large_date_gap(missing_dates) and not match_recent_tradedays(missing_dates, self.db_updater.tradedays):
                print(f'{code} 近期停牌, 但曾经更新过了， skipping update _upload_missing_data_stk_price_volume')
                continue
            elif self.db_updater.tradedays[-1] - missing_dates[-6] > datetime.timedelta(days=15):
                print(f'{code}在期间内曾经为新股，且发行后的数据已经更新过了， skipping update _upload_missing_data_stk_price_volume')
                continue

            if not missing_dates_str_list:
                continue
            # 停牌数据会返回空df，因此不必按日更新
            print(f"tushare downloading 行情 for {code} {missing_dates_str_list[0]}~{missing_dates_str_list[-1]}")
            df = self.db_updater.pro.daily(**{
                "ts_code": code,
                "trade_date": "",
                "start_date": missing_dates_str_list[0],
                "end_date": missing_dates_str_list[-1],
                "offset": "",
                "limit": ""
            }, fields=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pct_chg",
                "vol",
                "amount"
            ])
            if df.empty:
                print(f"跳过上传 {code} {missing_dates_str_list[0]}~{missing_dates_str_list[-1]}因为tushare返回空数据(停牌/未上市)")
            df = df.rename(columns={
                "ts_code": 'product_name',
                "trade_date": 'date',
                "open": '开盘价',
                "high": '最高价',
                "low": '最低价',
                "close": '收盘价',
                "pct_chg": '当日涨跌幅',
                "vol": '成交量',
                "amount": '成交额'
            })
            # 转换为长格式数据框
            df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
            df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)

        # 行不通：抱歉，您每天最多访问该接口20000次，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108
        #     for date in missing_dates_str_list:
        #         print(f"tushare downloading 行情 for {code} on {date}")
        #         df = self.db_updater.pro.daily(**{
        #             "ts_code": code,
        #             "trade_date": date,
        #             "start_date": "",
        #             "end_date": "",
        #             "offset": "",
        #             "limit": ""
        #         }, fields=[
        #             "ts_code",
        #             "trade_date",
        #             "open",
        #             "high",
        #             "low",
        #             "close",
        #             "pct_chg",
        #             "vol",
        #             "amount"
        #         ])
        #         if df.empty:
        #             print(f"跳过 {code} on {date} 因为空数据")
        #             empty_request_count += 1
        #             continue
        #         df = df.rename(columns={
        #             "ts_code": 'product_name',
        #             "trade_date": 'date',
        #             "open": '开盘价',
        #             "high": '最高价',
        #             "low": '最低价',
        #             "close": '收盘价',
        #             "pct_chg": '当日涨跌幅',
        #             "vol": '成交量',
        #             "amount": '成交额'
        #         })
        #         # 转换为长格式数据框
        #         df_long = pd.melt(df, id_vars=['date', 'product_name'], var_name='field', value_name='value')
        #         df_long.to_sql('stocks_daily_long', self.db_updater.alch_engine, if_exists='append', index=False)
        # print(f"empty_request_count={empty_request_count}")
