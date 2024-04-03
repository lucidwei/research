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
        数据预处理类的初始化方法
        :param data: 原始数据,DataFrame格式
        :param freq: 数据频率,默认为'M'(月度)
        :param info: 数据说明表,包含每列数据的处理规则,DataFrame格式
        """
        self.data = data
        self.freq = freq
        self.info = info

    def preprocess(self):
        """
        数据预处理的主流程,依次执行以下步骤:
        1. 特殊处理
        2. 对齐到月频
        3. 平稳性处理
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
        对数据进行特殊处理
        - 合并某些列
        - 将累计值转换为月度值
        """
        X = self.data
        # 如果存在'M5528820'列,则将其与'M0329545'列合并
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # 对于info表中标记为累计值的列,将其转换为月度值
        for id in X.columns:
            if self.info.loc[id, '是否累计值'] == '是':
                self.info.loc[id, '指标名称'] = '(月度化)' + self.info.loc[id, '指标名称']
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
        将数据对齐到月频
        - 根据info表中的resample规则,对每一列进行重采样
        - 将月末日期转换为月份格式
        - 填充缺失值
        """
        df = self.data
        df = self.fill_x_na(df)
        month_end_df = pd.DataFrame()
        for id in df.columns:
            if self.info.loc[id, 'resample(月)'] == 'last':
                ts = df.loc[:, id].resample('1M').last()
            elif self.info.loc[id, 'resample(月)'] == 'avg':
                ts = df.loc[:, id].resample('1M').mean()
            elif self.info.loc[id, 'resample(月)'] == 'cumsum':
                ts = df.loc[:, id].cumsum().resample('1M').last()
                self.info.loc[id, '指标名称'] = 'cumsumed' + self.info.loc[id, '指标名称']
            else:
                ts = df.loc[:, id].resample('1M').last()
            month_end_df = pd.concat([month_end_df, ts], axis=1)
        month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')
        self.data = self.fill_internal_missing(month_end_df)

    def fill_x_na(self, df):
        """
        根据info表中的fillna规则,对每一列填充缺失值
        :param df: 待填充的数据,DataFrame格式
        :return: 填充后的数据,DataFrame格式
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
        对数据进行平稳性处理
        - 对每一列进行平稳性检验
        - 对于非平稳的列,使用STL进行分解,将趋势项和残差项作为新的列添加到数据中,同时删除原列
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
        静态方法,用于检验单个时间序列的平稳性
        :param ts: 待检验的时间序列
        :return: 平稳性类型,'stationary'、'non-stationary'、'trend-stationary'或'diff-stationary'
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
        将数据重采样到月频
        """
        if pd.infer_freq(self.data.index) != self.freq:
            self.data = self.data.resample(self.freq).last()

    def fill_internal_missing(self, df):
        """
        填充数据内部的缺失值
        :param df: 待填充的数据,DataFrame格式
        :return: 填充后的数据,DataFrame格式
        """
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df

    def remove_outliers(self, threshold: float = 3.0):
        """
        去除异常值
        :param threshold: 异常值判断阈值,默认为3.0(即超过3个标准差)
        """
        for col in self.data.columns:
            series = self.data[col]
            mean = series.mean()
            std = series.std()
            outliers = (series - mean).abs() > threshold * std
            self.data.loc[outliers, col] = np.nan


