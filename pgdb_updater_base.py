# coding=gbk
# Time Created: 2023/4/26 15:32
# Author  : Lucid
# FileName: pgdb_updater_base.py
# Software: PyCharm
import datetime
from typing import List
import pandas as pd
from WindPy import w
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from utils import check_wind, timeit
from sqlalchemy import text
from sqlalchemy import Table, MetaData
from sqlalchemy.orm import Session
from sqlalchemy import select


class PgDbUpdaterBase(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_dates()

    def set_dates(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str
        self.all_dates = self.base_config.all_dates
        self.all_dates_str = self.base_config.all_dates_str
        self.months_ends = self.base_config.month_ends
        self.months_ends_str = self.base_config.month_ends_str

    @property
    def conversion_dicts(self):
        """
        ���ڵ�ԭ��WSD��EDB��ͬ�����᷵��id����Ӣ������������Ҫת���ֵ䡣
        ��������ID->english�Լ�english->chinese��ת����
        TODO: �����������������Ҫ�ֹ������ͨ��wind API�Զ���ȡ
        """
        # ID��Ӣ������
        id_to_english = {
            '000906.SH': 'csi_g800_index',
            'AU9999.SGE': 'sge_gold_9999',
            'CBA00301.CS': 'cba_total_return_index',
            'S0059749': 'china_t_yield',
            'G0005428': 'us_tips_10y',
            'VIX.GI': 'cboe_vix',
        }

        # Ӣ��������������
        english_to_chinese = {
            'csi_g800_index': '��֤800ָ��',
            'sge_gold_9999': '�����ֻ��ƽ�',
            'cba_total_return_index': '��ծ�ܲƸ�ָ��',
            'china_t_yield': '��ծ��ծ����������:10��',
            'us_tips_10y': '����ͨ��ָ����ծ������:10��',
            'cboe_vix': '֥�Ӹ���Ȩ������������ָ��',
        }

        return {'id_to_english': id_to_english, 'english_to_chinese': english_to_chinese}

    def update_high_freq_by_edb_id(self, code: str):
        """
        TODO: ���ֺ������ܸ��ϴ������ݼ�description�У���������������Ҫ��������ġ���˲��õ���
        ע�⣺
        1. self.get_missing_dates(self.all_dates�����������ֻ�����Ƶ���ݣ���Ƶ���������¶��庯��
        2. �����ԭʼ������������Ӧʹ�ô˷���
        �Ȼ�ȡȱʧ�������б�,��Ҫ���µ����������ǣ�
        - all_dates������һ�쵽���ݿ�������ݵ�����һ��
        - ���ݿ�������ݵ����һ�쵽all_dates�����һ�죬Ҳ���ǽ���
        """
        dates_missing = self.get_missing_dates(self.tradedays, 'high_freq_long',
                                               english_id=self.conversion_dicts['id_to_english'][code])
        if len(dates_missing) == 0:
            return
        old_dates = [date for date in dates_missing if date.year < 2023]
        new_dates = [date for date in dates_missing if date.year >= 2023]
        for dates_update in [old_dates, new_dates]:
            if not dates_update:
                continue
            print(
                f'Wind downloading {code} for table high_freq_long between {str(dates_update[0])} and {str(dates_update[-1])}')
            downloaded_df = w.edb(code, str(dates_update[0]), str(dates_update[-1]), usedf=True)[1]
            # wind���ص�df��ë��������Ϊһ��Ͷ���ĸ�ʽ��һ��
            try:
                # ���Խ�����ת��Ϊ����ʱ�䣬���ʧ��������ë��df
                pd.to_datetime(downloaded_df.index)
            except:
                if dates_update[0] == dates_update[-1]:
                    downloaded_df = downloaded_df.T
                    downloaded_df.index = dates_update
                else:
                    return
            # ��������Ϊ���ݿ�����
            downloaded_df.columns = [self.conversion_dicts['id_to_english'][code]]
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)

            # �����в������ݿ���
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.to_sql('high_freq_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_markets_daily_by_wsd_id_fields(self, code: str, fields: str):
        """
        code��field���ݴ���������CG��ȡ��
        �Ȼ�ȡȱʧ�������б�,��Ҫ���µ����������ǣ�
        - all_dates������һ�쵽���ݿ�������ݵ�����һ��
        - ���ݿ�������ݵ����һ�쵽all_dates�����һ�죬Ҳ���ǽ���
        wind fetchȱʧ���ڵ��ն����ݣ�д��DB��
        """
        fields_list = fields.split(',')

        for field in fields_list:
            dates_missing = self.get_missing_dates(self.tradedays, "markets_daily_long",
                                                   english_id=self.conversion_dicts['id_to_english'][code],
                                                   field=field.lower())
            if len(dates_missing) == 0:
                print(f'No missing data for {code} {field} in markets_daily_long, skipping download')
                return

            print(
                f'Wind downloading {code} {field} for markets_daily_long between {str(dates_missing[0])} and {str(dates_missing[-1])}')
            # ����Days=Weekdays�����ã�Ϊ����product�趨�������е��Ѷȡ�
            downloaded_df = \
            w.wsd(code, field, str(dates_missing[0]), str(dates_missing[-1]), "Days=Weekdays", usedf=True)[1]
            # ת�����ص����ݿ�Ϊ����ʽ
            downloaded_df.index.name = 'date'
            downloaded_df.reset_index(inplace=True)
            downloaded_df.columns = [col.lower() for col in downloaded_df.columns]
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='field', value_name='value')
            downloaded_df.dropna(subset=['value'], inplace=True)

            # �������������
            product_name = self.conversion_dicts['id_to_english'][code]
            source_code = f"wind_{code}"
            chinese_name = self.conversion_dicts['english_to_chinese'][product_name]

            # ȷ��metric_static_info���д��ڶ�Ӧ��source_code��chinese_name
            # ��ȡmetric_static_info���ж�Ӧ��¼��internal_id
            with self.alch_engine.connect() as conn:
                query = f"""
                INSERT INTO metric_static_info (source_code, chinese_name)
                VALUES ('{source_code}', '{chinese_name}')
                ON CONFLICT (source_code) DO UPDATE
                SET chinese_name = EXCLUDED.chinese_name
                RETURNING internal_id;
                """
                internal_id = conn.execute(text(query)).fetchone()[0]

            # �����в������ݿ���, dfҪ�ǿ�
            if downloaded_df.iloc[0, 0] != 0:
                print('Uploading data to database...')
                downloaded_df['product_name'] = product_name
                downloaded_df['metric_static_info_id'] = internal_id
                downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, sheet_name=None, if_rename=False):
        """
        ����excel�ļ��е�metadata��������
        """
        def get_start_month_end(s):
            start_date_str = s.split(':')[0]  # ��ȡ��ʼ�����ַ���
            start_date = pd.to_datetime(start_date_str, format='%Y-%m')  # ת�������ڸ�ʽ
            start_month_end = start_date + pd.offsets.MonthEnd(1)  # ��ȡ��ĩ����
            return start_month_end

        # 'ָ��ID' ���ַ�������С�� 5 ������EDB�м������ֵ����Ϊ��Щָ��Ļ�����������Ҫ����ã��������ת����������Щָ��
        # ����Ҳ���õ�������Щָ��ID�ò�����
        self.metadata, data = self.base_config.process_wind_excel(excel_file, sheet_name)
        # self.metadata = self.metadata[self.metadata['ָ��ID'].apply(lambda x: len(str(x)) >= 5)]

        # excel�п����ж�����С���Select the rows where 'ָ��ID' is in keys
        filtered_metadata = self.metadata[self.metadata['ָ������'].isin(name_mapping.keys())]

        # ���� DataFrame �������ͱ��������Ķ�Ӧ��ϵ
        col_indicator_id = filtered_metadata.loc[:, 'ָ��ID']
        col_indicator_name = filtered_metadata.loc[:, 'ָ������']
        col_unit = filtered_metadata.loc[:, '��λ']
        col_earlist_date = filtered_metadata['ʱ������'].apply(get_start_month_end)
        # maps
        map_id_to_chinese = dict(zip(col_indicator_id, col_indicator_name))
        map_id_to_unit = dict(zip(col_indicator_id, col_unit))
        map_id_to_english = {id: name_mapping[name] for id, name in map_id_to_chinese.items()}
        map_id_to_earlist_date = dict(zip(col_indicator_id, col_earlist_date))
        # renaming�л��õ�,��ʵ��ɾ����excel_file��Ӧ�������У�û�н���ɸѡ������Ϊ������Ҫrename����˲����Ż�
        names_to_delete = list(map_id_to_chinese.values())

        if if_rename:
            self.delete_for_renaming(names_to_delete)

        maps_tuple = (map_id_to_chinese, map_id_to_unit, map_id_to_english)
        # ��������
        for id in col_indicator_id:
            self.update_low_freq_by_excel_indicator_id(map_id_to_earlist_date[id], id, maps_tuple, data)

    def update_low_freq_by_excel_indicator_id(self, earliest_available_date, code, maps: tuple, data):
        map_id_to_name, map_id_to_unit, map_id_to_english = maps
        # ��������
        existing_dates = self.get_existing_dates_from_db('low_freq_long', map_id_to_english[code])
        dates_missing = self.get_missing_months_ends(self.months_ends, earliest_available_date, 'low_freq_long', map_id_to_english[code])
        if len(dates_missing) == 0:
            print(f'No missing data for low_freq_long {map_id_to_name[code]}, skipping download')
            return

        if len(code) <= 5:
            loaded_df = data[code]
            loaded_df.index = pd.to_datetime(loaded_df.index)
            # excel�����0ֵҪȥ��
            loaded_df = loaded_df[loaded_df != 0]
            # ֻ����������
            loaded_df = pd.DataFrame(loaded_df)
            loaded_df = loaded_df.reindex(dates_missing).dropna()

            if loaded_df.empty:
                print(f'No missing data for low_freq_long��{map_id_to_name[code]}, will not upload')
                return
            df_to_upload = loaded_df
        else:
            print(f'Wind downloading for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}')
            downloaded_df = w.edb(code, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
            downloaded_df.columns=[code]

            # ɾ���� existing_dates ��������ͬ����
            existing_dates_set = set(existing_dates)
            downloaded_df = downloaded_df.loc[~downloaded_df.index.isin(existing_dates_set)]

            if downloaded_df.empty:
                print(f'No missing data for low_freq_long��{map_id_to_name[code]}, downloaded but will not upload')
                return

            # wind���ص�df���ֻ��һ�����ݣ�index�ᱻ��Ϊwind_id������Ҫת�������ڡ�
            ## ��һ�����������ֻ������һ������
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing
            ## �ڶ�����������������˶�����ڣ���windֻ����һ�����ڣ�����ֱ�Ӷ���������Ϊ��֪�����ĸ����ڵ�����
            elif downloaded_df.shape[0] == 1:
                print(f'''
                We asked for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}, 
                but wind return only one data point, wind is actively updating this metric. Skipping...''')
                return

            df_to_upload = downloaded_df

        # ��������Ϊ���ݿ�����
        df_to_upload.index.name = 'date'
        df_to_upload.reset_index(inplace=True)
        # downloaded_df.rename(columns={'index': 'date'}, inplace=True)

        # �����������ʼ����
        six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
        # �������������ݵ�������������Ƶ��
        existing_dates_series = pd.Series(existing_dates, dtype='datetime64[D]').dt.date
        combined_dates = pd.to_datetime(pd.concat([existing_dates_series, df_to_upload['date']])).dt.date

        # �������Ƶ��=�����������/�ǿ����ݵ����
        # ѡ��������ǰ֮�������
        recent_dates = combined_dates[combined_dates >= six_months_ago]
        # �������������ݵ�������������Ƶ��
        non_null_data_points = recent_dates.count()
        update_freq = 180 / non_null_data_points

        # ����metric_static_info Ԫ����table
        # �����ͨ��wind����õ�����ֵ����ôԭ����codeû���ã�ʹ��������internal_id
        unit = map_id_to_unit[code]
        source_code = f'wind_{code}' if len(code) >= 5 else None
        chinese_name = map_id_to_name[code]
        english_name = map_id_to_english[code]

        internal_id = self.insert_metric_static_info(source_code, chinese_name, english_name, unit)

        # �������ݲ���low_freq_long��
        df_upload = df_to_upload.rename(columns=map_id_to_english)
        df_upload = df_upload.melt(id_vars=['date'], var_name='metric_name', value_name='value')
        df_upload.dropna(subset=['value'], inplace=True)
        # ��� additional_info:update_freq �� metric_static_info_id ��
        df_upload['update_freq'] = update_freq
        df_upload['metric_static_info_id'] = internal_id
        df_upload.to_sql('low_freq_long', self.alch_engine, if_exists='append', index=False)

    def insert_metric_static_info(self, source_code, chinese_name, english_name, unit):
        # Ensure the corresponding source_code and chinese_name exist in the metric_static_info table
        self.adjust_seq_val()
        with self.alch_engine.connect() as conn:
            if source_code:
                query = text("""
                            INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name)
                            VALUES (:source_code, :chinese_name, :unit, :english_name)
                            ON CONFLICT (source_code) DO UPDATE
                            SET english_name = EXCLUDED.english_name,
                                unit = EXCLUDED.unit
                            RETURNING internal_id;
                            """)
                result = conn.execute(query,
                             {
                                 'source_code': source_code,
                                 'chinese_name': chinese_name,
                                 'unit': unit,
                                 'english_name': english_name
                             })
                internal_id = result.fetchone()[0]
            # ����wind_transformed����
            else:
                # ���Ȳ�ѯ chinese_name �Ƿ��Ѿ�����
                query = text("""
                            SELECT 1
                            FROM metric_static_info
                            WHERE chinese_name = :chinese_name
                            """)
                result = conn.execute(query, {'chinese_name': chinese_name})
                if result.fetchone() is not None:
                    # ��� chinese_name �Ѿ����ڣ���ֱ�ӷ��أ���ִ�в������
                    return

                # �����µļ�¼
                query = text("""
                            INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name)
                            VALUES ('temp_code', :chinese_name, :unit, :english_name)
                            RETURNING internal_id;
                            """)
                result = conn.execute(query,
                                      {
                                          'chinese_name': chinese_name,
                                          'unit': unit,
                                          'english_name': english_name
                                      })
                # Get the internal_id of the corresponding record in the metric_static_info table
                internal_id = result.fetchone()[0]
                source_code = f'wind_transformed_{internal_id}'
                update_query = text("""
                                    UPDATE metric_static_info
                                    SET source_code = :source_code
                                    WHERE internal_id = :internal_id;
                                    """)
                conn.execute(update_query,
                             {
                                 'source_code': source_code,
                                 'internal_id': internal_id
                             })
            conn.commit()
        return internal_id




    def delete_for_renaming(self, names_to_delete):
        with self.alch_engine.connect() as conn:
            # ͨ��metric_static_info��ȡ���п��ܳ�ͻ��metric_name
            conflict_metric_names_set = set()
            for name in names_to_delete:
                query = f"""
                SELECT english_name
                FROM metric_static_info
                WHERE chinese_name = '{name}'
                """
                result = conn.execute(text(query))
                for row in result:
                    conflict_metric_names_set.add(row[0])

            # ɾ�����п��ܳ�ͻ�� metric_name
            for metric_name in conflict_metric_names_set:
                query = f"""
                DELETE FROM low_freq_long
                WHERE metric_name = '{metric_name}'
                """
                conn.execute(text(query))
                conn.commit()

    def execute_pgsql_function(self, function_name, table_name, view_name, chinese_names):
        query = f"""
        SELECT {function_name}(:table_name, :view_name, ARRAY[:chinese_names])
        """

        self.alch_conn.execute(
            text(query),
            {
                'table_name': table_name,
                'view_name': view_name,
                'chinese_names': chinese_names
            }
        )
        self.alch_conn.commit()

    def read_from_high_freq_view(self, code_list: List[str]) -> pd.DataFrame:
        """
        �� high_freq_view (wide format) ��ȡ���ݣ����� code_list��
        :param code_list: Ҫ��ѯ�Ĵ����б����� ['S0059749', 'G0005428']
        :return: ����һ���������������ݵ� DataFrame
        """
        # ��ȡӢ������
        column_names = [self.conversion_dicts['id_to_english'][code] for code in code_list]

        # ���� SQL ��ѯ
        sql_query = f"SELECT date, {', '.join(column_names)} FROM high_freq_wide"

        # �����ݿ��ж�ȡ����
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        return result_df.set_index('date').sort_index(ascending=False)

    def read_from_markets_daily_long(self, code: str, fields: str) -> pd.DataFrame:
        """
        �� markets_daily_long (long format) ��ȡ���ݣ����� code �� fields��
        :param code: Ҫ��ѯ�Ĵ��룬���� 'S0059749'
        :param fields: Ҫ��ѯ���ֶΣ����� "close,volume,pe_ttm"
        :return: ����һ���������������ݵ� DataFrame
        """
        # �����ŷָ��� fields ת��Ϊ�б�
        field_list = fields.split(',')

        # ��ȡ product_name
        product_name = self.conversion_dicts['id_to_english'][code]

        # ���� SQL ��ѯ
        sql_query = f"""SELECT date, field, value
                       FROM markets_daily_long
                       WHERE product_name = '{product_name}'
                       AND field IN ({', '.join([f"'{field}'" for field in field_list])})"""

        # �����ݿ��ж�ȡ����
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        # ת��Ϊ���ʽ
        result_df = result_df.pivot_table(index='date', columns='field', values='value')

        return result_df

    def get_missing_metrics(self, target_table: str, target_column: str, metrics_at_hand: list):
        # ����session
        session = Session(self.alch_engine)

        # ��ȡĿ����Ԫ����
        metadata = MetaData()
        target_table = Table(target_table, metadata, autoload_with=self.alch_engine)

        # ��ѯĿ����е�ֵ
        stmt = select(target_table.c[target_column])
        result = session.execute(stmt)

        existing_values = {row[0] for row in result}

        # �������metrics�б�ת��Ϊ����
        input_set = set(metrics_at_hand)

        # �ҳ������������б��У�����������Ŀ����е�ֵ
        missing_values = input_set - existing_values

        return list(missing_values)

    def calculate_yoy(self, value_str, yoy_str, cn_value_str, cn_yoy_str):
        # Step 1: Select all "*CurrentMonthValue" data from low_freq_long, bypass already-calculated rows
        # Step 1.1: Get the two dataframes
        query_value = f"SELECT * FROM low_freq_long WHERE metric_name LIKE '%{value_str}'"
        df_value = pd.read_sql_query(text(query_value), self.alch_conn)

        query_yoy = f"SELECT * FROM low_freq_long WHERE metric_name LIKE '%{yoy_str}'"
        df_yoy = pd.read_sql_query(text(query_yoy), self.alch_conn)

        # Step 1.2: Create new columns for matching
        df_value['metric_base'] = df_value['metric_name'].str.replace(value_str, '')
        df_yoy['metric_base'] = df_yoy['metric_name'].str.replace(yoy_str, '')

        # Step 1.3: Find the rows in df_value that have a match in df_yoy
        mask = df_value['metric_base'].isin(df_yoy['metric_base']) & df_value['date'].isin(df_yoy['date'])

        # Step 1.4: Remove the matching rows from df_value
        df = df_value[~mask]

        for _, row in df.iterrows():
            metric_name_value = row['metric_name']
            metric_name_yoy = metric_name_value.replace(value_str, yoy_str)

            # Step 2: Find the value from the same period last year
            query = f"""
            SELECT value
            FROM low_freq_long
            WHERE metric_name = '{metric_name_value}'
            AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CAST('{row['date']}' AS DATE)) - 1
            AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CAST('{row['date']}' AS DATE))
            """
            df_last_year = pd.read_sql_query(text(query), self.alch_conn)

            if df_last_year.empty or pd.isnull(df_last_year.loc[0, 'value']):
                # print(f"No data for {metric_name_value} '{row['date']}' from the same period last year.")
                continue

            # Calculate YoY change
            current_value = row['value']
            last_year_value = df_last_year.loc[0, 'value']
            yoy_change = (current_value - last_year_value) / last_year_value * 100

            # Step 3: Insert the calculated YoY data into low_freq_long
            query = f"""
            INSERT INTO low_freq_long (date, metric_name, value)
            VALUES ('{row['date']}', '{metric_name_yoy}', {yoy_change})
            ON CONFLICT (date, metric_name) DO UPDATE SET value = EXCLUDED.value
            """
            self.alch_conn.execute(text(query))

            # Step 4.1: Get the source_code of the corresponding Value variable
            query = f"""
            SELECT source_code, chinese_name
            FROM metric_static_info
            WHERE english_name = '{metric_name_value}'
            """
            df = pd.read_sql_query(text(query), self.alch_conn)
            if df.empty:
                # ���ݿ��д���һЩ�ϾɵĲ���Ҫ�����ݣ�������metric_static_infoû�м�¼
                continue

            source_code_value = df.loc[0, 'source_code']
            chinese_name_value = df.loc[0, 'chinese_name']
            if chinese_name_value not in self.export_chinese_names_for_view:
                # ��������չʾ��metric
                continue

            # Step 4.2: Update source_code in metric_static_info
            self.adjust_seq_val()
            new_source_code = f'calculated from {source_code_value}'
            chinese_name_yoy = chinese_name_value.replace(cn_value_str, cn_yoy_str)
            query = f"""
            INSERT INTO metric_static_info (english_name, source_code, chinese_name, unit)
            VALUES ('{metric_name_yoy}', '{new_source_code}', '{chinese_name_yoy}', '%')
            ON CONFLICT (english_name, source_code) DO UPDATE
            SET source_code = EXCLUDED.source_code, chinese_name = EXCLUDED.chinese_name, unit = EXCLUDED.unit
            """

            self.alch_conn.execute(text(query))
            self.alch_conn.commit()

    def calculate_custom_metric(self, english_name_a, english_name_b, calculation_function, new_english_name,
                                new_chinese_name, new_unit):
        """
        ֻ��һ��һ������
        :param english_name_a:
        :param english_name_b:
        :param calculation_function:
        :param new_english_name:
        :param new_chinese_name:
        :return:
        """
        # Check if calculation_function has a name
        function_name = getattr(calculation_function, '__name__', None)
        if function_name is None:
            raise ValueError("calculation_function must have a name, use regular function definition.")

        # Step 1: Get data for the specified columns
        query_a = f"SELECT date, value as value_a FROM low_freq_long WHERE metric_name = '{english_name_a}'"
        query_b = f"SELECT date, value as value_b FROM low_freq_long WHERE metric_name = '{english_name_b}'"
        df_a = pd.read_sql_query(text(query_a), self.alch_conn)
        df_b = pd.read_sql_query(text(query_b), self.alch_conn)

        # Step 2: Merge data on date
        df = df_a.merge(df_b, on='date')

        # Step 3: Calculate the custom metric using the calculation_function
        df['calculated_value'] = df.apply(lambda row: calculation_function(row['value_a'], row['value_b']), axis=1)

        # Step 4: Insert calculated values into low_freq_long
        for _, row in df.iterrows():
            query = f"""
            INSERT INTO low_freq_long (date, metric_name, value)
            VALUES ('{row['date']}', '{new_english_name}', {row['calculated_value']})
            ON CONFLICT (date, metric_name) DO UPDATE SET value = EXCLUDED.value
            """
            self.alch_conn.execute(text(query))

        # Step 5: Update metric_static_info
        self.adjust_seq_val()
        query = f"""
        INSERT INTO metric_static_info (english_name, source_code, chinese_name, unit)
        VALUES ('{new_english_name}', 'calculated from {english_name_a} and {english_name_b} using {function_name}', '{new_chinese_name}', '{new_unit}')
        ON CONFLICT (english_name, source_code) DO UPDATE
        SET source_code = EXCLUDED.source_code, chinese_name = EXCLUDED.chinese_name, unit = EXCLUDED.unit
        """
        self.alch_conn.execute(text(query))
        self.alch_conn.commit()

    def divide(self, a, b):
        return a/b
