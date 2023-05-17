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
            with self.alch_engine.connect() as conn:
                query = f"""
                INSERT INTO metric_static_info (source_code, chinese_name)
                VALUES ('{source_code}', '{chinese_name}')
                ON CONFLICT (source_code) DO UPDATE
                SET chinese_name = EXCLUDED.chinese_name;
                """
                conn.execute(text(query))

                # ��ȡmetric_static_info���ж�Ӧ��¼��internal_id
                query = f"""
                SELECT internal_id
                FROM metric_static_info
                WHERE source_code = '{source_code}' AND chinese_name = '{chinese_name}';
                """
                internal_id = conn.execute(text(query)).fetchone()[0]

            # �����в������ݿ���, dfҪ�ǿ�
            if downloaded_df.iloc[0, 0] != 0:
                print('Uploading data to database...')
                downloaded_df['product_name'] = product_name
                downloaded_df['metric_static_info_id'] = internal_id
                downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    def update_low_freq_by_edb_id(self, earliest_available_date, code, maps: tuple):
        map_id_to_name, map_id_to_unit, map_id_to_english = maps
        # ��������
        existing_dates = self.get_existing_dates_from_db('low_freq_long', map_id_to_english[code])
        dates_missing = self.get_missing_months_ends(self.months_ends, earliest_available_date, 'low_freq_long', map_id_to_english[code])
        if len(dates_missing) == 0:
            print(f'No missing data for low_freq_long {map_id_to_name[code]}, skipping download')
            return

        print(f'Wind downloading for low_freq_long {map_id_to_name[code]} between {str(dates_missing[0])} and {str(dates_missing[-1])}')
        downloaded_df = w.edb(code, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
        downloaded_df.columns=[code]

        # ɾ���� existing_dates ��������ͬ����
        existing_dates_set = set(existing_dates)
        downloaded_df = downloaded_df.loc[~downloaded_df.index.isin(existing_dates_set)]

        if downloaded_df.empty:
            print(f'No missing data for low_freq_long��{map_id_to_name[code]}, downloaded but will not upload')
            return

        # wind���ص�df������Ϊһ��Ͷ���ĸ�ʽ��һ��
        if dates_missing[0] == dates_missing[-1]:
            downloaded_df = downloaded_df.T
            downloaded_df.index = dates_missing

        # ��������Ϊ���ݿ�����
        downloaded_df.reset_index(inplace=True)
        downloaded_df.rename(columns={'index': 'date'}, inplace=True)

        # �����������ʼ����
        six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
        # �������������ݵ�������������Ƶ��
        existing_dates_series = pd.Series(existing_dates, dtype='datetime64[D]').dt.date
        combined_dates = pd.to_datetime(pd.concat([existing_dates_series, downloaded_df['date']])).dt.date

        # ѡ��������ǰ֮�������
        recent_dates = combined_dates[combined_dates >= six_months_ago]
        # �������������ݵ�������������Ƶ��
        non_null_data_points = recent_dates.count()
        # �������Ƶ��=�����������/�ǿ����ݵ����
        update_freq = 180 / non_null_data_points

        # ����metric_static_info Ԫ����table
        unit = map_id_to_unit[code]
        source_code = f'wind_{code}'
        chinese_name = map_id_to_name[code]
        english_name = map_id_to_english[code]

        # Ensure the corresponding source_code and chinese_name exist in the metric_static_info table
        with self.alch_engine.connect() as conn:
            query = text("""
                        INSERT INTO metric_static_info (source_code, chinese_name, unit, english_name)
                        VALUES (:source_code, :chinese_name, :unit, :english_name)
                        ON CONFLICT (source_code) DO UPDATE
                        SET english_name = EXCLUDED.english_name,
                            unit = EXCLUDED.unit;
                        """)
            conn.execute(query,
                         {
                             'source_code': source_code,
                             'chinese_name': chinese_name,
                             'unit': unit,
                             'english_name': english_name
                         })
            conn.commit()

            # Get the internal_id of the corresponding record in the metric_static_info table
            query = f"""
                    SELECT internal_id
                    FROM metric_static_info
                    WHERE source_code = :source_code AND chinese_name = :chinese_name;
                    """
            internal_id = conn.execute(text(query), {
                'source_code': source_code,
                'chinese_name': chinese_name,
            }).fetchone()[0]

        # �������ݲ������ݿ���
        df_upload = downloaded_df.rename(columns=map_id_to_english)
        df_upload = df_upload.melt(id_vars=['date'], var_name='metric_name', value_name='value')
        df_upload.dropna(subset=['value'], inplace=True)
        # ��� additional_info:update_freq �� metric_static_info_id ��
        df_upload['update_freq'] = update_freq
        df_upload['metric_static_info_id'] = internal_id

        df_upload.to_sql('low_freq_long', self.alch_engine, if_exists='append', index=False)


    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, if_rename=False):
        """
        ����excel�ļ��е�metadata��������
        """
        def get_start_month_end(s):
            start_date_str = s.split(':')[0]  # ��ȡ��ʼ�����ַ���
            start_date = pd.to_datetime(start_date_str, format='%Y-%m')  # ת�������ڸ�ʽ
            start_month_end = start_date + pd.offsets.MonthEnd(1)  # ��ȡ��ĩ����
            return start_month_end

        # �޳� 'ָ��ID' ���ַ�������С�� 5 ���У���Щ��EDB�м������ֵ
        self.metadata = self.base_config.process_wind_metadata(excel_file)
        self.metadata = self.metadata[self.metadata['ָ��ID'].apply(lambda x: len(str(x)) >= 5)]

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
            self.update_low_freq_by_edb_id(map_id_to_earlist_date[id], id, maps_tuple)

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
        SELECT {function_name}(:table_name, :view_name, ARRAY{chinese_names})
        """

        self.alch_conn.execute(
            text(query),
            {
                'table_name': table_name,
                'view_name': view_name,
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