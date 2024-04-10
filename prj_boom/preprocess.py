# coding=gbk
# Time Created: 2024/4/3 10:44
# Author  : Lucid
# FileName: preprocess.py
# Software: PyCharm

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from utils import set_index_col_wind  # �� utils.py �е��� set_index_col_wind ����


def transform_cumulative_data(series: pd.Series, data_type: str, period: int = 12) -> pd.Series:
    """
    ���ۼ�ֵ���ۼ�ͬ������ת��Ϊ����ֵ����ͬ������
    :param series: ��ת��������,Series��ʽ
    :param data_type: ��������,'�ۼ�ֵ'��'�ۼ�ͬ��'
    :param period: ͬ������,Ĭ��Ϊ12(�¶�����)
    :return: ת���������,Series��ʽ
    """
    if data_type == '�ۼ�ֵ':
        # ��ʱ�������ز���Ϊ�¶�����
        monthly_series = series.resample('M').last()

        # ���ۼ�ֵת��Ϊ����ֵ
        current_data = monthly_series.copy()
        prev_values = monthly_series.shift(1).ffill()
        current_data = current_data - prev_values
        current_data = current_data.where(current_data > 0, monthly_series)

        # ����һ�·��ۼ�ֵΪ�յ����
        for year, data in current_data.groupby(current_data.index.year):
            if pd.isna(data.iloc[0]) and not pd.isna(data.iloc[1]):
                feb_value = data.iloc[1]
                jan_index = pd.to_datetime(f'{year}-01-01') + pd.offsets.MonthEnd()
                feb_index = pd.to_datetime(f'{year}-02-01') + pd.offsets.MonthEnd()
                current_data.loc[jan_index] = feb_value / 2
                current_data.loc[feb_index] = feb_value / 2

        # ��������²�����ԭʼƵ��
        current_data = current_data.reindex(series.index)

        return current_data

    elif data_type == '�ۼ�ͬ��':
        # ��ʱ�������ز���Ϊ�¶�����
        monthly_series = series.resample('M').last()

        # ���ۼ�ͬ��ת��Ϊ����ͬ��
        cumulative_yoy = monthly_series.copy()
        current_yoy = pd.Series(index=cumulative_yoy.index)

        # ��ʼ������
        start_month_index = None
        covered_months = 0

        # ����ÿ���·�
        for i in range(len(cumulative_yoy)):
            current_month = cumulative_yoy.index[i].month

            # �����ǰ�·�Ϊ1�����ۼ�ͬ�����ݷǿ�,��ʼ�µ�һ��
            if current_month == 1 and not pd.isna(cumulative_yoy.iloc[i]):
                start_month_index = i
                covered_months = 1
            # �����ǰ�·ݲ�Ϊ1�����ۼ�ͬ�����ݷǿ�,����¸����·���
            elif current_month != 1 and not pd.isna(cumulative_yoy.iloc[i]):
                if start_month_index is None:
                    start_month_index = i
                    covered_months = current_month
                else:
                    covered_months += 1
            elif current_month == 1 and pd.isna(cumulative_yoy.iloc[i]):
                start_month_index = i+1
                covered_months = 1
                continue

            # ����Ѿ���ʼ���㵱��ĵ���ͬ��,������������
            if start_month_index is not None:
                if i == start_month_index:
                    current_yoy.iloc[i] = cumulative_yoy.iloc[i]
                else:
                    current_month_yoy = (cumulative_yoy.iloc[i] * covered_months - cumulative_yoy.iloc[i - 1] * (
                                covered_months - 1))
                    current_yoy.iloc[i] = current_month_yoy

        # ����һ�·��ۼ�ֵΪ�յ����
        for year, data in current_yoy.groupby(current_yoy.index.year):
            if pd.isna(data.iloc[0]) and not pd.isna(data.iloc[1]):
                feb_value = data.iloc[1]
                jan_index = pd.to_datetime(f'{year}-01-01') + pd.offsets.MonthEnd()
                current_yoy.loc[jan_index] = feb_value

        # ��������²�����ԭʼƵ��
        current_yoy = current_yoy.reindex(series.index)

        return current_yoy

    else:
        raise ValueError(f"��֧�ֵ���������: {data_type}")


class DataPreprocessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig, freq: str = 'M', info: pd.DataFrame = None):
        """
        ����Ԥ������ĳ�ʼ������
        :param data: ԭʼ����,DataFrame��ʽ
        :param freq: ����Ƶ��,Ĭ��Ϊ'M'(�¶�)
        :param info: ����˵����,����ÿ�����ݵĴ������,DataFrame��ʽ
        """
        super().__init__(base_config)
        self.freq = freq
        self.info = info
        self.k_factors = 1

    def preprocess(self):
        """
        ����Ԥ�����������,����ִ�����²���:
        1. ���⴦��
        2. ���뵽��Ƶ
        3. ƽ���Դ���
        """
        self.read_data_and_info()
        self.special_mannual_treatment()
        self.align_to_month()
        self.fill_internal_missing()
        self.get_stationary()
        self.cap_outliers()
        # return self.data

    def read_data_and_info(self):
        file_path = rf'{self.base_config.excels_path}/����'

        # ��ȡ���ָ������info
        # ���ָ��������Ϊindex(����ָ��ID����Ϊ�������������)
        info = pd.read_excel(rf'{file_path}/indicators_info.xlsx', engine="openpyxl")
        self.id_to_name = dict(zip(info['ָ��ID'], info['ָ������']))
        self.info = info.set_index('ָ��ID')

        # ��ȡ���ָ��
        # ��ָ��������Ϊ�У�����Ϊindex
        # data = pd.read_excel(rf'{file_path}/�к��indicators.xlsx', sheet_name='test')
        data = pd.read_excel(rf'{file_path}/�к��indicators.xlsx')
        data = set_index_col_wind(data, 'ָ��ID')
        # ɸѡ���ڲ�����
        data_filtered = data[data.index >= pd.Timestamp('2010-01-01')].copy(deep=True)
        data_filtered.sort_index(ascending=True, inplace=True)

        # ��ȡ��������
        financials = pd.read_excel(rf"{file_path}/��ҵ��������.xlsx", sheet_name='��������q')
        financials = set_index_col_wind(financials, 'Date')
        financials = financials[financials.index >= pd.Timestamp('2010-01-01')].copy(deep=True)
        financials.sort_index(ascending=True, inplace=True)

        combined_data = pd.merge(data_filtered, financials, left_index=True, right_index=True, how='outer')
        # ����һ���б�,�洢Ҫ�޳�������
        financials_cols = ['roe_ttm2', 'yoyprofit']
        indicators_cols = [col for col in combined_data.columns if col not in financials_cols]

        # �޳�Ӱ������0ֵ
        self.df_indicators = combined_data[indicators_cols].replace(0, np.nan)
        for column in indicators_cols:
            self.df_indicators[column] = pd.to_numeric(self.df_indicators[column], errors='coerce')

        self.df_finalcials = combined_data[financials_cols]

    def special_mannual_treatment(self):
        """
        �����ݽ������⴦��
        - �ϲ�ĳЩ��
        - ���ۼ�ֵת��Ϊ�¶�ֵ
        """
        X = self.df_indicators.copy(deep=True)
        # �������'M5528820'��,������'M0329545'�кϲ�
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # ����info���б��Ϊ�ۼ�ֵ����,����ת��Ϊ�¶�ֵ
        for id in X.columns:
            if self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '�ۼ�ֵ':
                self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
                X.loc[:, id] = transform_cumulative_data(X.loc[:, id], '�ۼ�ֵ')

            elif self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '�ۼ�ͬ��':
                self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
                X.loc[:, id] = transform_cumulative_data(X.loc[:, id], '�ۼ�ͬ��')

        self.data = X

    def align_to_month(self):
        """
        �����ݶ��뵽��Ƶ
        - ����info���е�resample����,��ÿһ�н����ز���
        - ����ĩ����ת��Ϊ�·ݸ�ʽ
        - ���ȱʧֵ
        """
        df = self.data.copy(deep=True)
        month_end_df = pd.DataFrame()
        for id in df.columns:
            if self.info.loc[id, 'resample(��)'] == 'last':
                ts = df.loc[:, id].resample('1M').last()
            elif self.info.loc[id, 'resample(��)'] == 'avg':
                ts = df.loc[:, id].resample('1M').mean()
            elif self.info.loc[id, 'resample(��)'].startswith('rolling_'):
                n = int(self.info.loc[id, 'resample(��)'][8:])
                ts = df.loc[:, id].rolling(n).mean().resample('1M').last()
            elif self.info.loc[id, 'resample(��)'] == 'cumsum':
                ts = df.loc[:, id].cumsum().resample('1M').last()
                self.info.loc[id, 'ָ������'] = 'cumsumed' + self.info.loc[id, 'ָ������']
            else:
                ts = df.loc[:, id].resample('1M').last()
            month_end_df = pd.concat([month_end_df, ts], axis=1)
        month_end_df.index = pd.to_datetime(month_end_df.index)#.to_period('M')
        self.data = month_end_df

    def fill_internal_missing(self):
        """
        ����info���е�fillna����,��ÿһ�����ȱʧֵ
        :return: ���������,DataFrame��ʽ
        """
        for id in self.data.columns:
            if self.info.loc[id, 'fillna'] == 'ffill':
                self.data.loc[:, id].fillna(method='ffill', inplace=True)
            elif self.info.loc[id, 'fillna'] == '0fill':
                self.data.loc[:, id].fillna(value=0, inplace=True)
            elif pd.isnull(self.info.loc[id, 'fillna']):
                pass
            else:
                raise Exception('donno how to fillna')

    def get_stationary(self):
        """
        �����ݽ���ƽ���Դ���
        - ��ÿһ�н���ƽ���Լ���
        - ���ڷ�ƽ�ȵ���,ʹ��STL���зֽ�,��������Ͳв�����Ϊ�µ�����ӵ�������,ͬʱɾ��ԭ��
        """
        df = self.data.copy(deep=True)
        record = {}
        for col_ind, col in df.items():
            # ��鲢����ȱʧֵ
            if df[col_ind].isnull().any():
                col = col.dropna()
            record[col_ind] = self.station_test(col)

        for col_ind, col in df.items():
            if record[col_ind] != 'stationary':
                original_index = df.index
                col_series = pd.Series(col, index=original_index)
                col_series = col_series.dropna()

                if record[col_ind] == 'diff-stationary':
                    diff_series = col_series.diff().dropna()
                    stl = STL(diff_series, period=12)
                    decomposed = stl.fit()
                    decomposed_df = pd.DataFrame({
                        col_ind + '_trend': decomposed.trend.cumsum(),
                        col_ind + '_resid': decomposed.resid
                    }, index=diff_series.index)
                    decomposed_df[col_ind + '_trend'] += col_series.iloc[0]

                elif record[col_ind] == 'trend-stationary':
                    x = np.arange(len(col_series))
                    trend = np.polyfit(x, col_series, 1)[0] * x
                    detrended_series = col_series - trend
                    stl = STL(detrended_series, period=12)
                    decomposed = stl.fit()
                    decomposed_df = pd.DataFrame({
                        col_ind + '_trend': decomposed.trend + trend,
                        col_ind + '_resid': decomposed.resid
                    }, index=col_series.index)

                else:  # 'non-stationary'
                    stl = STL(col_series, period=12)
                    decomposed = stl.fit()
                    decomposed_df = pd.DataFrame({
                        col_ind + '_trend': decomposed.trend,
                        col_ind + '_resid': decomposed.resid
                    }, index=col_series.index)

                decomposed_df = decomposed_df.reindex(original_index)
                df = pd.concat([df, decomposed_df], axis=1)
                df.drop(col_ind, inplace=True, axis=1)

        self.data = df

    @staticmethod
    def station_test(ts):
        """
        ��̬����,���ڼ��鵥��ʱ�����е�ƽ����
        :param ts: �������ʱ������
        :return: ƽ��������,'stationary'��'non-stationary'��'trend-stationary'��'diff-stationary'
        """

        def kpss_test(timeseries):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpsstest = kpss(timeseries, regression="c", nlags="auto")
            kpss_output = pd.Series(
                kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
            )
            return kpss_output[1]

        def adf_test(timeseries):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dftest = adfuller(timeseries, autolag="AIC")
            dfoutput = pd.Series(
                dftest[0:4],
                index=[
                    "Test Statistic",
                    "p-value",
                    "#Lags Used",
                    "Number of Observations Used",
                ], )
            return dfoutput[1]

        p_adf = adf_test(ts)
        p_kpss = kpss_test(ts)
        threshold = 0.05
        if p_adf <= threshold <= p_kpss:
            return 'stationary'
        elif p_kpss < threshold < p_adf:
            return 'non-stationary'
        elif p_adf > threshold and p_kpss > threshold:
            return 'trend-stationary'
        elif p_adf < threshold and p_kpss < threshold:
            return 'diff-stationary'
        else:
            raise Exception('donno stationarity')

    def cap_outliers(self, threshold: float = 3.0):
        """
        ���쳣ֵ�趨Ϊ������׼��λ��
        :param threshold: �쳣ֵ�ж���ֵ,Ĭ��Ϊ3.0(������3����׼��)
        """
        for col in self.data.columns:
            series = self.data[col]
            mean = series.mean()
            std = series.std()
            upper_bound = mean + threshold * std
            lower_bound = mean - threshold * std

            # ���쳣ֵ�趨Ϊ������׼��λ��
            self.data.loc[series > upper_bound, col] = upper_bound
            self.data.loc[series < lower_bound, col] = lower_bound
