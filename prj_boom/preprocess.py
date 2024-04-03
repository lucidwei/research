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


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame, freq: str = 'M', info: pd.DataFrame = None):
        """
        ����Ԥ������ĳ�ʼ������
        :param data: ԭʼ����,DataFrame��ʽ
        :param freq: ����Ƶ��,Ĭ��Ϊ'M'(�¶�)
        :param info: ����˵����,����ÿ�����ݵĴ������,DataFrame��ʽ
        """
        self.data = data
        self.freq = freq
        self.info = info

    def preprocess(self):
        """
        ����Ԥ�����������,����ִ�����²���:
        1. ���⴦��
        2. ���뵽��Ƶ
        3. ƽ���Դ���
        """
        self.special_treatment()
        self.align_to_month()
        self.get_stationary()
        # self.resample_to_monthly()
        # self.fill_internal_missing()
        # self.remove_outliers()
        return self.data

    def special_treatment(self):
        """
        �����ݽ������⴦��
        - �ϲ�ĳЩ��
        - ���ۼ�ֵת��Ϊ�¶�ֵ
        """
        X = self.data
        # �������'M5528820'��,������'M0329545'�кϲ�
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # ����info���б��Ϊ�ۼ�ֵ����,����ת��Ϊ�¶�ֵ
        for id in X.columns:
            if self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '��':
                self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
                sr = X[id]
                sr_ori = sr.copy()
                for date in sr.index:
                    if np.isfinite(sr[date]):
                        try:
                            diff = sr[date] - sr_ori[date - pd.offsets.MonthEnd()]
                            X.loc[date, id] = diff if diff > 0 else sr[date]
                        except:
                            pass
        self.data = X

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


