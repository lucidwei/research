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
    def __init__(self, file_path, exclude_today=False):
        self.file_path = file_path
        self.data_dict = {}
        self.asset_class_mapping = {
            '沪深300': 'Equity',
            '中证500': 'Equity',
            '恒生中国企业指数': 'Equity',
            '恒生科技': 'Equity',
            '标普500': 'Equity',
            '纳斯达克100': 'Equity',
            '日经225': 'Equity',
            '德国DAX': 'Equity',
            '法国CAC40': 'Equity',
            '英国富时100': 'Equity',
            '印度SENSEX30': 'Equity',
            '越南VN30指数': 'Equity',
            '中证1000': 'Equity',
            '中证2000': 'Equity',
            '中债-国开行债券总财富(总值)指数': 'Bond',
            '中债-农发行债券总指数财富(总值)指数': 'Bond',
            'IBOXX美元债总回报': 'Bond',
            '中债-总财富(总值)指数': 'Bond',
            'SHFE黄金': 'Commodity',
            '南华能化指数': 'Commodity',
            '中证800': 'Equity',
            '新交所泛东南亚科技指数': 'Equity',
        }
        self.load_data(exclude_today)

    def load_data(self, exclude_today=False):
        # 读取Excel文件
        df = pd.read_excel(self.file_path)

        # 通过查找空列来分割不同的数据块
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        if not split_indices:
            raise ValueError("无法找到数据块之间的分隔符")

        # 处理第一块数据：收盘价数据
        close_data = df.iloc[:, :split_indices[0]]
        self._process_close_data(close_data, exclude_today)

        # 处理第二块数据：经济数据
        edb_data = df.iloc[:, split_indices[0] + 1:split_indices[1]]
        self._process_edb_data(edb_data, exclude_today)

        # 处理第三块数据：综合数据
        composite_data = df.iloc[:, split_indices[1] + 1:]
        self._process_composite_data(composite_data, exclude_today)

        return self.data_dict

    def _process_data(self, df, exclude_today):
        # 设置索引为日期
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('日期', inplace=True)
        data = data.dropna(how='all')

        # 检查索引是否有重复
        if data.index.duplicated().any():
            raise ValueError("日期索引有重复，请检查数据源。")

        # 如果exclude_today为True，删除今天的数据
        if exclude_today:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            if today in data.index:
                data = data.drop(today)

        return data

    def _process_close_data(self, df, exclude_today):
        data = self._process_data(df, exclude_today)
        self.data_dict['close_prices'] = data.sort_index()

    def _process_edb_data(self, df, exclude_today):
        data = self._process_data(df, exclude_today)
        self.data_dict['edb_data'] = data.sort_index()

    def _process_composite_data(self, df, exclude_today):
        data = self._process_data(df, exclude_today)
        # 转换特定列数据类型
        if 'amt' in data.columns:
            data['amt'] = pd.to_numeric(data['amt'], errors='coerce')
        if 'pe_ttm' in data.columns:
            data['pe_ttm'] = pd.to_numeric(data['pe_ttm'], errors='coerce')
        self.data_dict['composite_data'] = data.sort_index()

    def get_data(self):
        """返回已加载的数据字典"""
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
        if hasattr(self.strategy, 'erp_signal'):
            signals['signal_erp_stock'] = self.strategy.erp_signal
            signal_names.append('signal_erp_stock')
        if hasattr(self.strategy, 'volume_ma_signal'):
            signals['signal_volume_ma_stock'] = self.strategy.volume_ma_signal
            signal_names.append('signal_volume_ma_stock')
        if hasattr(self.strategy, 'us_tips_signal'):
            signals['signal_us_tips_gold'] = self.strategy.us_tips_signal
            signal_names.append('signal_us_tips_gold')
        if hasattr(self.strategy, 'vix_signal'):
            signals['signal_vix_gold'] = self.strategy.vix_signal
            signal_names.append('signal_vix_gold')
        if hasattr(self.strategy, 'gold_momentum_signal'):
            signals['signal_momentum_gold'] = self.strategy.gold_momentum_signal
            signal_names.append('signal_momentum_gold')

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
        if hasattr(self.strategy, 'aligned_stock_signal'):
            signals['monthly_signal_stock'] = self.strategy.aligned_stock_signal
        if hasattr(self.strategy, 'aligned_gold_signal'):
            signals['monthly_signal_gold'] = self.strategy.aligned_gold_signal

        if not signals:
            print(f"No signals available to collect for strategy '{self.strategy_name}'.")
            return

        # Iterate through each rebalance date and collect the corresponding signal
        for date in self.rebalance_dates:
            for signal_name, aligned_signal in signals.items():
                if date in aligned_signal:
                    # Retrieve the signal value for the specific rebalance date
                    signal_value = aligned_signal[date]
                    # Record the monthly signal
                    self.results.append({
                        'date': date.date().isoformat(),
                        'strategy_name': self.strategy_name,
                        'metric_name': f'{signal_name}',
                        'value': signal_value
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