# coding=gbk
# Time Created: 2023/4/28 13:33
# Author  : Lucid
# FileName: results_uploader.py
# Software: PyCharm
import pandas as pd
from sqlalchemy import create_engine, text
from pgdb_manager import PgDbManager


class ResultsUploader(PgDbManager):
    def __init__(self, evaluator, base_config):
        super().__init__(base_config)
        self.evaluator = evaluator
        self.data = evaluator.data
        self.asset_weights = evaluator.asset_allocator.adjusted_monthly_weights
        self.signals = evaluator.asset_allocator.signals
        self.portfolio_value = evaluator.portfolio_value
        self.benchmark_value = evaluator.benchmark_value
        self.performance_metrics = evaluator.all_performance_metrics
        self.upload_results()

    def upload_results(self):
        # Prepare a dataframe for the results
        results = []

        # Asset weights
        for asset in self.asset_weights.columns:
            for date, weight in self.asset_weights[asset].iteritems():
                results.append({'date': date.date().isoformat(), 'project': 'risk_parity', 'metric_name': f'asset_weight_{asset}', 'value': weight})

        # Portfolio value and benchmark value
        for date, value in self.portfolio_value.iteritems():
            results.append({'date': date.date().isoformat(), 'project': 'risk_parity', 'metric_name': 'portfolio_value', 'value': value})
        for date, value in self.benchmark_value.iteritems():
            results.append({'date': date.date().isoformat(), 'project': 'risk_parity', 'metric_name': 'benchmark_value', 'value': value})

        # Signals
        for signal_type in ['combined', 'individual']:
            signal_df = self.signals[signal_type]
            for signal_name, signal_series in signal_df.items():
                signal_series.dropna(inplace=True)
                for date, signal_value in signal_series.iteritems():
                    results.append({'date': date.date().isoformat(), 'project': 'risk_parity', 'metric_name': signal_name, 'value': int(signal_value)})

        # Performance metrics
        for evaluated_obj in ['Benchmark', 'Portfolio']:
            for metric_name, metric_value in self.performance_metrics[evaluated_obj].items():
                results.append({'date': self.portfolio_value.index[-1].date().isoformat(), 'project': 'risk_parity', 'metric_name': evaluated_obj+'_'+metric_name, 'value': metric_value})

        results_df = pd.DataFrame(results)

        # Upload results to the database
        results_df.to_sql('results_display_long', self.alch_engine, if_exists='append', index=False)
