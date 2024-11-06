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
        }
        self.load_data()

    def load_data(self):
        # 读取Excel文件
        df = pd.read_excel(self.file_path)

        # 通过查找空列来分割不同的数据块
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        if not split_indices:
            raise ValueError("无法找到数据块之间的分隔符")

        # 处理第一块数据：收盘价数据
        close_data = df.iloc[:, :split_indices[0]]
        self._process_close_data(close_data)

        # 处理第二块数据：经济数据
        edb_data = df.iloc[:, split_indices[0] + 1:split_indices[1]]
        self._process_edb_data(edb_data)

        # 处理第三块数据：综合数据
        composite_data = df.iloc[:, split_indices[1] + 1:]
        self._process_composite_data(composite_data)

        return self.data_dict

    def _process_close_data(self, df):
        # 清理并处理收盘价数据
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('日期', inplace=True)
        data = data.dropna(how='all')
        # 检查索引是否有重复
        if data.index.duplicated().any():
            raise ValueError("日期索引有重复，请检查数据源。")

        # 将数据存储到字典中
        self.data_dict['close_prices'] = data.sort_index()

    def _process_edb_data(self, df):
        # 清理并处理经济数据
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('日期', inplace=True)
        data = data.dropna(how='all')
        # 检查索引是否有重复
        if data.index.duplicated().any():
            raise ValueError("日期索引有重复，请检查数据源。")

        # 将数据存储到字典中
        self.data_dict['edb_data'] = data.sort_index()

    def _process_composite_data(self, df):
        # 清理并处理综合数据
        data = df.iloc[3:].copy()
        data.columns = df.iloc[1]
        data.set_index('日期', inplace=True)
        data = data.dropna(how='all')
        # 检查索引是否有重复
        if data.index.duplicated().any():
            raise ValueError("日期索引有重复，请检查数据源。")

        # 转换成交额列的数据类型为float
        if 'amt' in data.columns:
            data['amt'] = pd.to_numeric(data['amt'], errors='coerce')

        # 转换PE(TTM)列的数据类型为float
        if 'pe_ttm' in data.columns:
            data['pe_ttm'] = pd.to_numeric(data['pe_ttm'], errors='coerce')

        # 将数据存储到字典中
        self.data_dict['composite_data'] = data.sort_index()

    def get_data(self):
        """返回已加载的数据字典"""
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
        # 遍历每一天的权重
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