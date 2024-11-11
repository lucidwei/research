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
            '��֤800': 'Equity',
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
    def __init__(self, strategy_name, strategy, evaluator, base_config=None):
        # Initialize database connection
        super().__init__(base_config)
        self.strategy_name = strategy_name
        self.strategy = strategy
        self.evaluator = evaluator
        self.results = []
        # Store rebalancing dates from the evaluator if available
        self.rebalance_dates = getattr(self.strategy, 'rebalance_dates', None)

    def upload_results(self):
        self.collect_net_value()
        self.collect_asset_weights()
        self.collect_performance_metrics()
        self.collect_signals()  # Collect signals if available

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
        Collect asset weights for each date.
        """
        weights_history = self.evaluator.weights_history
        for date in weights_history.index:
            weights = weights_history.loc[date]
            for asset, weight in weights.iteritems():
                # Include asset class in metric_name if available
                if hasattr(self.strategy, 'asset_class_mapping') and asset in self.strategy.asset_class_mapping:
                    asset_class = self.strategy.asset_class_mapping[asset]
                    metric_name = f'weight_{asset_class}_{asset}'
                else:
                    metric_name = f'weight_{asset}'
                self.results.append({
                    'date': date.date().isoformat(),
                    'strategy_name': self.strategy_name,
                    'metric_name': metric_name,
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

    def collect_signals(self):
        """
        Collect signals from the strategy and add to results.
        This method handles both daily signals and monthly rebalancing signals.
        """
        # Check if the strategy has signal attributes
        if self.strategy_name == 'SignalBasedStrategy':
            # Collect and upload signals
            self.collect_daily_signals()
            self.collect_monthly_signals()
        else:
            # Do nothing if the strategy does not have signals
            pass

    def collect_daily_signals(self):
        """
        Collect daily signals from the strategy.
        """
        # Ensure that the strategy has the necessary signal attributes
        signal_names = []
        signals = {}

        if hasattr(self.strategy, 'combined_stock_signal'):
            signals['signal_stock'] = self.strategy.combined_stock_signal
            signal_names.append('signal_stock')
        if hasattr(self.strategy, 'combined_gold_signal'):
            signals['signal_gold'] = self.strategy.combined_gold_signal
            signal_names.append('signal_gold')

        # Collect daily signals
        for signal_name in signal_names:
            signal_series = signals[signal_name]
            for date, value in signal_series.iteritems():
                self.results.append({
                    'date': date.date().isoformat(),
                    'strategy_name': self.strategy_name,
                    'metric_name': signal_name,
                    'value': value
                })

    def collect_monthly_signals(self):
        """
        Calculate and collect monthly signals used for rebalancing.
        """
        if self.rebalance_dates is None:
            print(
                f"Rebalancing dates not available for strategy '{self.strategy_name}'. Cannot collect monthly signals.")
            return

        # Collect daily signals first
        signals = {}
        if hasattr(self.strategy, 'combined_stock_signal'):
            signals['signal_stock'] = self.strategy.combined_stock_signal
        if hasattr(self.strategy, 'combined_gold_signal'):
            signals['signal_gold'] = self.strategy.combined_gold_signal

        if not signals:
            print(f"No signals available to collect for strategy '{self.strategy_name}'.")
            return

        # For each rebalancing period, calculate the mean signal and round
        rebalance_dates = self.rebalance_dates
        rebalance_dates = pd.Series(rebalance_dates, index=rebalance_dates)
        rebalance_dates = rebalance_dates.sort_values()

        date_index = self.evaluator.weights_history.index

        for i, date in enumerate(rebalance_dates):
            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates.iloc[i + 1]
            else:
                next_date = date_index[-1] + pd.Timedelta(days=1)

            # Get date range for the period before the rebalancing date
            signal_dates = date_index[(date_index >= date) & (date_index < next_date)]
            if signal_dates.empty:
                continue  # No signal data for this period

            for signal_name, daily_signal in signals.items():
                # Extract signals for the period
                period_signals = daily_signal.loc[signal_dates]
                if period_signals.empty:
                    continue
                # Calculate mean signal
                mean_signal = period_signals.mean()
                # Round to get the actual signal used for rebalancing
                monthly_signal = int(mean_signal.round())
                # Record the monthly signal
                self.results.append({
                    'date': date.date().isoformat(),
                    'strategy_name': self.strategy_name,
                    'metric_name': f'monthly_{signal_name}',
                    'value': monthly_signal
                })
        print(f"Collected monthly signals for strategy '{self.strategy_name}'.")

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