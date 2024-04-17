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
from utils_prj import *


def transform_cumulative_data(series: pd.Series, data_type: str, period: int = 12) -> pd.Series:
    """
    将累计值或累计同比数据转换为当期值或当期同比数据
    :param series: 待转换的数据,Series格式
    :param data_type: 数据类型,'累计值'或'累计同比'
    :param period: 同比周期,默认为12(月度数据)
    :return: 转换后的数据,Series格式
    """
    if data_type == '累计值':
        # 将时间序列重采样为月度数据
        monthly_series = series.resample('M').last()

        # 将累计值转换为当期值
        current_data = monthly_series.copy()
        prev_values = monthly_series.shift(1).ffill()
        current_data = current_data - prev_values
        current_data = current_data.where(current_data > 0, monthly_series)

        # 处理一月份累计值为空的情况
        for year, data in current_data.groupby(current_data.index.year):
            if pd.isna(data.iloc[0]) and not pd.isna(data.iloc[1]):
                feb_value = data.iloc[1]
                jan_index = pd.to_datetime(f'{year}-01-01') + pd.offsets.MonthEnd()
                feb_index = pd.to_datetime(f'{year}-02-01') + pd.offsets.MonthEnd()
                current_data.loc[jan_index] = feb_value / 2
                current_data.loc[feb_index] = feb_value / 2

        # 将结果重新采样回原始频率
        current_data = current_data.reindex(series.index)

        return current_data

    elif data_type == '累计同比':
        # 将时间序列重采样为月度数据
        monthly_series = series.resample('M').last()

        # 将累计同比转换为当月同比
        cumulative_yoy = monthly_series.copy()
        current_yoy = pd.Series(index=cumulative_yoy.index)

        # 初始化变量
        start_month_index = None
        covered_months = 0

        # 遍历每个月份
        for i in range(len(cumulative_yoy)):
            current_month = cumulative_yoy.index[i].month

            # 如果当前月份为1月且累计同比数据非空,则开始新的一年
            if current_month == 1 and not pd.isna(cumulative_yoy.iloc[i]):
                start_month_index = i
                covered_months = 1
            # 如果当前月份不为1月且累计同比数据非空,则更新覆盖月份数
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

            # 如果已经开始计算当年的当月同比,则进行逆向操作
            if start_month_index is not None:
                if i == start_month_index:
                    current_yoy.iloc[i] = cumulative_yoy.iloc[i]
                else:
                    current_month_yoy = (cumulative_yoy.iloc[i] * covered_months - cumulative_yoy.iloc[i - 1] * (
                                covered_months - 1))
                    current_yoy.iloc[i] = current_month_yoy

        # 处理一月份累计值为空的情况
        for year, data in current_yoy.groupby(current_yoy.index.year):
            if pd.isna(data.iloc[0]) and not pd.isna(data.iloc[1]):
                feb_value = data.iloc[1]
                jan_index = pd.to_datetime(f'{year}-01-01') + pd.offsets.MonthEnd()
                current_yoy.loc[jan_index] = feb_value

        # 将结果重新采样回原始频率
        current_yoy = current_yoy.reindex(series.index)

        return current_yoy

    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


class DataPreprocessor(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig, date_start: str = '2010-01-01', industry: str = None):
        """
        数据预处理类的初始化方法
        :param data: 原始数据,DataFrame格式
        :param freq: 数据频率,默认为'M'(月度)
        :param info: 数据说明表,包含每列数据的处理规则,DataFrame格式
        """
        super().__init__(base_config)
        self.date_start = date_start
        self.industry = industry
        self.k_factors = 1

    def preprocess(self):
        """
        数据预处理的主流程,依次执行以下步骤:
        1. 特殊处理
        2. 对齐到月频
        3. 平稳性处理
        """
        self.read_data_and_info()
        self.special_mannual_treatment()
        self.align_to_month()
        self.fill_internal_missing()
        # self.get_stationary()
        self.cap_outliers()
        # return self.data

    def read_data_and_info(self):
        file_path = rf'{self.base_config.excels_path}/景气'

        # 读取宏观指标手设info
        # 宏观指标名称作为index(不是指标ID，因为不方便人类理解)
        # 使用中文字符串作为 DataFrame 的列名可能会引入一些意外的问题,特别是当列名包含特殊字符或标点符号时。
        # 需要把特殊字符或标点符号全部转换为下划线_
        info = pd.read_excel(rf'{file_path}/indicators_info.xlsx', engine="openpyxl", sheet_name=self.industry)
        self.id_to_name = dict(zip(info['指标ID'], info['指标名称']))
        self.info = info.set_index('指标名称')

        # 读取宏观指标
        # 将指标名称设为列，日期为index
        df = pd.read_excel(rf'{file_path}/行业景气数据库.xlsx', sheet_name=self.industry)
        df_dict = split_dataframe(whole_df=df)

        # 对df_dict中的每个DataFrame进行筛选日期并排序
        df_dict = {
            key: value[value.index >= pd.Timestamp(self.date_start)].sort_index(ascending=True)
            for key, value in df_dict.items()
        }

        # 定义一个列表, 存储要剔除的列名, 挑选只在info中出现的指标进行处理
        financials_cols = ['净资产收益率ROE', '归属母公司股东的净利润同比增长率', '营业收入同比增长率']
        indicators_cols = self.info.index.tolist()
        combined_data = pd.merge(df_dict['基本面'][indicators_cols], df_dict['财务'], left_index=True, right_index=True, how='outer')


        # 剔除影响计算的0值
        self.df_indicators = combined_data[indicators_cols].replace(0, np.nan).astype(float)
        self.df_finalcials = combined_data[financials_cols]

    def special_mannual_treatment(self):
        """
        对数据进行特殊处理
        - 合并某些列
        - 将累计值转换为月度值
        """
        X = self.df_indicators.copy(deep=True)
        # 如果存在'M5528820'列,则将其与'M0329545'列合并
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # 对于info表中标记为累计值的列,将其转换为月度值
        for name in X.columns:
            if not pd.isna(self.info.loc[name, '是否累计值']):
                new_name = '(月度化)' + name
                # 转换指标名
                self.info.loc[new_name] = self.info.loc[name]
                self.info = self.info.drop(name)
                # 累积值转为月度
                X.loc[:, new_name] = transform_cumulative_data(X.loc[:, name], self.info.loc[new_name, '是否累计值'])
                X.drop(name, axis=1, inplace=True)

        self.data = X

    def align_to_month(self):
        """
        将数据对齐到月频
        - 根据info表中的resample规则,对每一列进行重采样
        - 将月末日期转换为月份格式
        - 填充缺失值
        """
        df = self.data.copy(deep=True)
        month_end_df = pd.DataFrame()
        for id in df.columns:
            if self.info.loc[id, 'resample(月)'] == 'last':
                ts = df.loc[:, id].resample('1M').last()
            elif self.info.loc[id, 'resample(月)'] == 'avg':
                ts = df.loc[:, id].resample('1M').mean()
            elif self.info.loc[id, 'resample(月)'].startswith('rolling_'):
                n = int(self.info.loc[id, 'resample(月)'][8:])
                ts = df.loc[:, id].rolling(n).mean().resample('1M').last()
            elif self.info.loc[id, 'resample(月)'] == 'cumsum':
                ts = df.loc[:, id].cumsum().resample('1M').last()
                self.info.loc[id, '指标名称'] = 'cumsumed' + self.info.loc[id, '指标名称']
            else:
                ts = df.loc[:, id].resample('1M').last()
            month_end_df = pd.concat([month_end_df, ts], axis=1)
        month_end_df.index = pd.to_datetime(month_end_df.index)#.to_period('M')
        self.data = month_end_df

    def fill_internal_missing(self):
        """
        根据info表中的fillna规则,对每一列填充缺失值
        :return: 填充后的数据,DataFrame格式
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
        对数据进行平稳性处理
        - 对每一列进行平稳性检验
        - 对于非平稳的列,使用STL进行分解,将趋势项和残差项作为新的列添加到数据中,同时删除原列
        """
        df = self.data.copy(deep=True)
        record = {}
        for col_ind, col in df.items():
            # 检查并处理缺失值
            if df[col_ind].isnull().values.any():
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

    def cap_outliers(self, threshold: float = 3.0):
        """
        将异常值设定为三个标准差位置
        :param threshold: 异常值判断阈值,默认为3.0(即超过3个标准差)
        """
        self.data = cap_outliers(data=self.data, threshold=threshold)
