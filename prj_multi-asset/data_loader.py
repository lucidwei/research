# coding=gbk
# Time Created: 2024/10/30 10:47
# Author  : Lucid
# FileName: data_loader.py
# Software: PyCharm
import pandas as pd
import numpy as np
from pgdb_updater_base import PgDbUpdaterBase
import datetime

class ExcelDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_dict = {}
        self.asset_class_mapping = {
            '����300': 'Equity',
            '��֤500': 'Equity',
            '�����й���ҵָ��': 'Equity',
            '�����Ƽ�': 'Equity',
            '����500': 'Equity',
            '��˹���100': 'Equity',
            '�վ�225': 'Equity',
            '�¹�DAX': 'Equity',
            '����CAC40': 'Equity',
            'Ӣ����ʱ100': 'Equity',
            'ӡ��SENSEX30': 'Equity',
            'Խ��VN30ָ��': 'Equity',
            '��֤1000': 'Equity',
            '��֤2000': 'Equity',
            '��ծ-������ծȯ�ܲƸ�(��ֵ)ָ��': 'Bond',
            '��ծ-ũ����ծȯ��ָ���Ƹ�(��ֵ)ָ��': 'Bond',
            'IBOXX��Ԫծ�ܻر�': 'Bond',
            '��ծ-�ܲƸ�(��ֵ)ָ��': 'Bond',
            'SHFE�ƽ�': 'Commodity',
            '�ϻ��ܻ�ָ��': 'Commodity',
        }
        self.load_data()

    def load_data(self):
        # ��ȡExcel�ļ�
        df = pd.read_excel(self.file_path)

        # ͨ�����ҿ������ָͬ�����ݿ�
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        if not split_indices:
            raise ValueError("�޷��ҵ����ݿ�֮��ķָ���")

        # �����һ�����ݣ����̼�����
        close_data = df.iloc[:, :split_indices[0]]
        self._process_close_data(close_data)

        # ����ڶ������ݣ���������
        edb_data = df.iloc[:, split_indices[0] + 1:split_indices[1]]
        self._process_edb_data(edb_data)

        # ������������ݣ��ۺ�����
        composite_data = df.iloc[:, split_indices[1] + 1:]
        self._process_composite_data(composite_data)

        return self.data_dict

    def _process_close_data(self, df):
        # �����������̼�����
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('����', inplace=True)
        data = data.dropna(how='all')
        # ��������Ƿ����ظ�
        if data.index.duplicated().any():
            raise ValueError("�����������ظ�����������Դ��")

        # �����ݴ洢���ֵ���
        self.data_dict['close_prices'] = data.sort_index()

    def _process_edb_data(self, df):
        # ��������������
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('����', inplace=True)
        data = data.dropna(how='all')
        # ��������Ƿ����ظ�
        if data.index.duplicated().any():
            raise ValueError("�����������ظ�����������Դ��")

        # �����ݴ洢���ֵ���
        self.data_dict['edb_data'] = data.sort_index()

    def _process_composite_data(self, df):
        # ���������ۺ�����
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('����', inplace=True)
        data = data.dropna(how='all')
        # ��������Ƿ����ظ�
        if data.index.duplicated().any():
            raise ValueError("�����������ظ�����������Դ��")

        # ת���ɽ����е���������Ϊfloat
        if 'amt' in data.columns:
            data['amt'] = pd.to_numeric(data['amt'], errors='coerce')

        # ת��PE(TTM)�е���������Ϊfloat
        if 'pe_ttm' in data.columns:
            data['pe_ttm'] = pd.to_numeric(data['pe_ttm'], errors='coerce')

        # �����ݴ洢���ֵ���
        self.data_dict['composite_data'] = data.sort_index()

    def get_data(self):
        """�����Ѽ��ص������ֵ�"""
        if not self.data_dict:
            self.load_data()
        return self.data_dict


class ResultsUploader(PgDbUpdaterBase):
    def __init__(self, strategy_name, strategy, evaluator, base_config):
        # Initialize database connection
        super().__init__(base_config)
        self.strategy_name = strategy_name
        self.strategy = strategy
        self.evaluator = evaluator
        # Collect data to be uploaded
        self.results = []

    def upload_results(self):
        self.collect_net_value()
        self.collect_asset_weights()
        self.collect_performance_metrics()
        # If there are any strategy-specific data (e.g., signals), you can add methods to collect them
        # self.collect_strategy_specific_data()

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)

        # Upload results to the database
        self.upsert_results_to_database(results_df)

    def collect_net_value(self):
        """
        Collect daily net value (portfolio net worth over time).
        """
        net_value_series = self.evaluator.net_value
        for date, value in net_value_series.iteritems():
            self.results.append({
                'date': date.date().isoformat(),
                'strategy_name': self.strategy_name,
                'metric_name': 'net_value',
                'value': value
            })

    def collect_asset_weights(self):
        """
        Collect asset weights at each rebalancing date.
        """
        # Assuming weights are only changed on rebalancing dates
        weights_history = self.evaluator.weights_history
        # ����ÿһ���Ȩ��
        for date in weights_history.index:
            weights = weights_history.loc[date]
            for asset, weight in weights.iteritems():
                self.results.append({
                    'date': date.date().isoformat(),
                    'strategy_name': self.strategy_name,
                    'metric_name': f'weight_{asset}',
                    'value': weight
                })

    def collect_performance_metrics(self):
        """
        Collect performance metrics from the Evaluator.
        """
        performance_metrics = self.evaluator.performance
        metrics_date = datetime.date.today().isoformat()  # Or use the last date in net_value series
        for metric_name, metric_value in performance_metrics.items():
            self.results.append({
                'date': metrics_date,
                'strategy_name': self.strategy_name,
                'metric_name': f'perform_{metric_name}',
                'value': metric_value
            })

    def upsert_results_to_database(self, results_df):
        """
        Upload the results DataFrame to the database in a long format table.
        """
        # Include the 'project' field if necessary
        results_df['project'] = 'multi_asset_strategy'

        # Define the unique keys for upserting data
        unique_keys = ['date', 'project', 'strategy_name', 'metric_name']

        # Assuming the table name is 'results_display_new'
        table_name = 'results_display_new'

        # Use the upsert_dataframe_to_postgresql method from PgDbUpdaterBase
        self.upsert_dataframe_to_postgresql(results_df, table_name, unique_keys)

    # Placeholder for collecting strategy-specific data
    def collect_strategy_specific_data(self):
        """
        Collect any strategy-specific data, such as signals.
        This method can be overridden or extended in subclasses for specific strategies.
        """
        pass