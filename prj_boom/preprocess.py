# coding=gbk
# Time Created: 2024/4/3 10:44
# Author  : Lucid
# FileName: preprocess.py
# Software: PyCharm
import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import freq_to_period

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase


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
        self.get_stationary()
        # self.resample_to_monthly()
        # self.fill_internal_missing()
        # self.remove_outliers()
        return self.data

    def read_data_and_info(self):
        file_path = rf'{self.base_config.excels_path}/����'

        def set_index_col_wind(data, col_locator: str) -> pd.DataFrame:
            # �ҵ���һ���������ڵ���
            date_row = data.iloc[:, 0].apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().idxmax()
            # �ҵ�"ָ������"���ڵ���
            indicator_row = data.iloc[:date_row, 0].str.contains(col_locator).fillna(False).idxmax()
            # ����ָ������Ϊ column
            data.columns = data.iloc[indicator_row]
            # ɾ��ָ�������м����Ϸ���������
            data = data.iloc[date_row + 1:]
            # ���õ�һ��Ϊ index,������Ϊ "date"
            data = data.set_index(data.columns[0])
            data.index.name = 'date'
            return data

        # ��ȡ���ָ������info
        # ���ָ��������Ϊindex(����ָ��ID����Ϊ�������������)
        self.info = pd.read_excel(rf'{file_path}/indicators_info.xlsx', index_col=1, engine="openpyxl")

        # ��ȡ���ָ��
        # ��ָ��������Ϊ�У�����Ϊindex
        data = pd.read_excel(rf'{file_path}/�к��indicators.xlsx', sheet_name='test')
        data = set_index_col_wind(data, 'ָ������')
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
                X.loc[:, id] = self.transform_cumulative_data(X.loc[:, id], '�ۼ�ֵ')

            elif self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '�ۼ�ͬ��':
                self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
                X.loc[:, id] = self.transform_cumulative_data(X.loc[:, id], '�ۼ�ͬ��')

        self.data = X

    def transform_cumulative_data(self, series: pd.Series, data_type: str, period: int = 12) -> pd.Series:
        """
        ���ۼ�ֵ���ۼ�ͬ������ת��Ϊ����ֵ����ͬ������
        :param series: ��ת��������,Series��ʽ
        :param data_type: ��������,'�ۼ�ֵ'��'�ۼ�ͬ��'
        :param period: ͬ������,Ĭ��Ϊ12(�¶�����)
        :return: ת���������,Series��ʽ
        """
        if data_type == '�ۼ�ֵ':
            # ���ۼ�ֵת��Ϊ����ֵ
            current_data = series.copy()
            for i in range(1, len(current_data)):
                if np.isfinite(current_data.iloc[i]):
                    try:
                        diff = current_data.iloc[i] - series.iloc[i - 1]
                        current_data.iloc[i] = diff if diff > 0 else current_data.iloc[i]
                    except:
                        pass
            return current_data

        elif data_type == '�ۼ�ͬ��':
            # ���ۼ�ͬ��ת��Ϊ����ͬ��
            cumulative_data = series.copy()

            # �����ۼ�ֵ����
            for i in range(period, len(cumulative_data)):
                cumulative_data.iloc[i] = (cumulative_data.iloc[i] + 1) / (cumulative_data.iloc[i - period] + 1) * \
                                          cumulative_data.iloc[i - period]

            # ���㵱��ֵ����
            current_data = cumulative_data - cumulative_data.shift(1)

            # ���㵱��ͬ������
            current_yoy_data = current_data / current_data.shift(period) - 1

            return current_yoy_data

        else:
            raise ValueError(f"��֧�ֵ���������: {data_type}")

    def align_to_month(self):
        """
        �����ݶ��뵽��Ƶ
        - ����info���е�resample����,��ÿһ�н����ز���
        - ����ĩ����ת��Ϊ�·ݸ�ʽ
        - ���ȱʧֵ
        """
        df = self.data
        df = self.fill_x_na(df)
        month_end_df = pd.DataFrame()
        for id in df.columns:
            if self.info.loc[id, 'resample(��)'] == 'last':
                ts = df.loc[:, id].resample('1M').last()
            elif self.info.loc[id, 'resample(��)'] == 'avg':
                ts = df.loc[:, id].resample('1M').mean()
            elif self.info.loc[id, 'resample(��)'] == 'cumsum':
                ts = df.loc[:, id].cumsum().resample('1M').last()
                self.info.loc[id, 'ָ������'] = 'cumsumed' + self.info.loc[id, 'ָ������']
            else:
                ts = df.loc[:, id].resample('1M').last()
            month_end_df = pd.concat([month_end_df, ts], axis=1)
        month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')
        self.data = self.fill_internal_missing(month_end_df)

    def fill_x_na(self, df):
        """
        ����info���е�fillna����,��ÿһ�����ȱʧֵ
        :param df: ����������,DataFrame��ʽ
        :return: ���������,DataFrame��ʽ
        """
        for id in df.columns:
            if self.info.loc[id, 'fillna'] == 'ffill':
                df.loc[:, id].fillna(method='ffill', inplace=True)
            elif self.info.loc[id, 'fillna'] == '0fill':
                df.loc[:, id].fillna(value=0, inplace=True)
            elif pd.isnull(self.info.loc[id, 'fillna']):
                pass
            else:
                raise Exception('donno how to fillna')
        return df

    def get_stationary(self):
        """
        �����ݽ���ƽ���Դ���
        - ��ÿһ�н���ƽ���Լ���
        - ���ڷ�ƽ�ȵ���,ʹ��STL���зֽ�,��������Ͳв�����Ϊ�µ�����ӵ�������,ͬʱɾ��ԭ��
        """
        df = self.fill_internal_missing(self.data)
        record = {col_ind: self.station_test(col) for col_ind, col in df.iteritems()}
        for col_ind, col in df.iteritems():
            if record[col_ind] != 'stationary':
                stl = STL(col, period=12)
                decomposed = stl.fit()
                df[col_ind + '_trend'] = decomposed.trend
                df[col_ind + '_resid'] = decomposed.resid
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

    def resample_to_monthly(self):
        """
        �������ز�������Ƶ
        """
        if pd.infer_freq(self.data.index) != self.freq:
            self.data = self.data.resample(self.freq).last()

    def fill_internal_missing(self, df):
        """
        ��������ڲ���ȱʧֵ
        :param df: ����������,DataFrame��ʽ
        :return: ���������,DataFrame��ʽ
        """
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df

    def remove_outliers(self, threshold: float = 3.0):
        """
        ȥ���쳣ֵ
        :param threshold: �쳣ֵ�ж���ֵ,Ĭ��Ϊ3.0(������3����׼��)
        """
        for col in self.data.columns:
            series = self.data[col]
            mean = series.mean()
            std = series.std()
            outliers = (series - mean).abs() > threshold * std
            self.data.loc[outliers, col] = np.nan
