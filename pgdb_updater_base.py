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

    def update_edb_by_id_to_high_freq(self, code: str):
        """
        TODO: ���ֺ������ܸ��ϴ������ݼ�description�У���������������Ҫ��������ġ���˲��õ���
        ע�⣺
        1. self.get_missing_dates(self.all_dates�����������ֻ�����Ƶ���ݣ���Ƶ���������¶��庯��
        2. �����ԭʼ������������Ӧʹ�ô˷���
        �Ȼ�ȡȱʧ�������б�,��Ҫ���µ����������ǣ�
        - all_dates������һ�쵽���ݿ�������ݵ�����һ��
        - ���ݿ�������ݵ����һ�쵽all_dates�����һ�죬Ҳ���ǽ���
        """
        dates_missing = self.get_missing_dates(self.tradedays, 'high_freq_long', column_name=self.conversion_dicts['id_to_english'][code])
        if len(dates_missing) == 0:
            return
        old_dates = [date for date in dates_missing if date.year < 2023]
        new_dates = [date for date in dates_missing if date.year >= 2023]
        for dates_update in [old_dates, new_dates]:
            if not dates_update:
                continue
            print(f'Wind downloading {code} for table high_freq_long between {str(dates_update[0])} and {str(dates_update[-1])}')
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
    def update_wsd(self, code: str, fields: str):
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
                                                   column_name=self.conversion_dicts['id_to_english'][code],
                                                   field=field.lower())

            if len(dates_missing) != 0:
                print(
                    f'Wind downloading {code} {field} for markets_daily_long between {str(dates_missing[0])} and {str(dates_missing[-1])}')
                downloaded_df = w.wsd(code, field, str(dates_missing[0]), str(dates_missing[-1]), "", usedf=True)[1]
                # ת�����ص����ݿ�Ϊ����ʽ
                downloaded_df.index.name = 'date'
                downloaded_df.reset_index(inplace=True)
                downloaded_df.columns = [col.lower() for col in downloaded_df.columns]
                downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='field', value_name='value')
                downloaded_df.dropna(subset=['value'], inplace=True)

                # �������������
                downloaded_df['product_name'] = self.conversion_dicts['id_to_english'][code]
                downloaded_df['source_code'] = f"wind_{code}"
                downloaded_df['chinese_name'] = downloaded_df['product_name'].map(
                    self.conversion_dicts['english_to_chinese'])

                # �����в������ݿ���, dfҪ�ǿ�
                if downloaded_df.iloc[0, 0] != 0:
                    downloaded_df.to_sql('markets_daily_long', self.alch_engine, if_exists='append', index=False)

    @timeit
    def update_low_freq_from_excel_meta(self, excel_file: str, name_mapping: dict, table_name='low_freq_long'):
        """
        �Ȼ�ȡȱʧ����ĩ�����б�,��Ҫ���µ����������ǣ�
        - all_dates������һ�쵽���ݿ�������ݵ�����һ��
        - ���ݿ�������ݵ����һ�쵽all_dates�����һ�죬Ҳ���ǽ���
        wind fetchȱʧ���ڵ�EDB����
        ����ø�ʽ�������ݿ�
        """
        # �޳� 'ָ��ID' ���ַ�������С�� 5 ���У���Щ��EDB�м������ֵ
        self.metadata = self.base_config.process_wind_metadata(excel_file)
        self.metadata = self.metadata[self.metadata['ָ��ID'].apply(lambda x: len(str(x)) >= 5)]
        # ���� DataFrame �������ͱ��������Ķ�Ӧ��ϵ
        indicator_id_col = self.metadata.loc[:, 'ָ��ID']
        indicator_name_col = self.metadata.loc[:, 'ָ������']

        map_id_to_english = {}
        for ind_id, ind_name in zip(indicator_id_col, indicator_name_col):
            # ʹ���ֶ�ӳ����ֵ����滻����
            new_string = name_mapping[ind_name]
            # ��ָ�� ID �����ַ�����ӵ��ֵ���
            map_id_to_english[ind_id] = new_string
        self.db_col = map_id_to_english.values()

        # ��ת map_id_to_english �ֵ䣬��Ӣ������ӳ���ָ������
        map_english_to_id = {v: k for k, v in map_id_to_english.items()}
        map_english_to_name = {v: k for k, v in name_mapping.items()}

        # ��������
        dates_missing = self.get_missing_months_ends(self.months_ends, table_name)
        ids = self.metadata['ָ��ID'].str.cat(sep=',')
        if len(dates_missing) != 0:
            print(f'Wind downloading for {table_name} ����')
            downloaded_df = w.edb(ids, str(dates_missing[0]), str(self.all_dates[-1]), usedf=True)[1]
            # wind���ص�df������Ϊһ��Ͷ���ĸ�ʽ��һ��
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing

            # ��������Ϊ���ݿ�����
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)
            downloaded_df.rename(columns=map_id_to_english, inplace=True)

            # �����������ʼ����
            six_months_ago = datetime.date.today() - datetime.timedelta(days=6 * 30)
            # �������������ݵ������������Ƶ��
            update_freq_dict = {}
            for col in downloaded_df.columns:
                if col != 'date':
                    recent_data = downloaded_df.loc[downloaded_df['date'] >= six_months_ago, col]
                    non_null_data_points = recent_data.count()

                    # �������Ƶ��
                    date_range = 6 * 30  # �����������
                    update_freq = date_range / non_null_data_points
                    update_freq_dict[col] = update_freq

            # �����в������ݿ���
            downloaded_df = downloaded_df.melt(id_vars=['date'], var_name='metric_name', value_name='value')
            downloaded_df.dropna(subset=['value'], inplace=True)
            # ��� chinese_name��update_freq �� source ��
            downloaded_df['chinese_name'] = downloaded_df['metric_name'].map(map_english_to_name)
            downloaded_df['update_freq'] = downloaded_df['metric_name'].map(update_freq_dict)
            downloaded_df['source'] = downloaded_df['metric_name'].map(map_english_to_id).apply(lambda x: f'wind_{x}')

            downloaded_df.to_sql(table_name, self.alch_engine, if_exists='append', index=False)

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
