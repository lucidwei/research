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
                start_month_index = i + 1
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


# def decompose_series(series, stationary_type):
#     if stationary_type == 'diff-stationary':
#         diff_series = series.diff().dropna()
#         stl = STL(diff_series, period=12)
#         decomposed = stl.fit()
#         trend = decomposed.trend.cumsum()
#         trend += series.iloc[0]
#         resid = decomposed.resid
#     elif stationary_type == 'trend-stationary':
#         x = np.arange(len(series))
#         trend = np.polyfit(x, series, 1)[0] * x
#         resid = series - trend
#     else:  # 'non-stationary'
#         series = np.log(series)  # ȡ����,ʹ���нӽ�ƽ��
#         stl = STL(series, period=12)
#         decomposed = stl.fit()
#         trend = decomposed.trend
#         resid = decomposed.resid
#
#     return pd.DataFrame({'trend': trend, 'resid': resid}, index=series.index)
#
#
# def process_dataframe(df, record):
#     original_index = df.index
#     decomposed_dfs = []
#
#     for col_ind, col in df.items():
#         if record[col_ind] != 'stationary':
#             col_series = col.dropna()
#             decomposed_df = decompose_series(col_series, record[col_ind])
#             decomposed_df = decomposed_df.reindex(original_index)
#             decomposed_df.columns = [f'{col_ind}_{c}' for c in decomposed_df.columns]
#             decomposed_dfs.append(decomposed_df)
#
#     if len(decomposed_dfs) > 0:
#         decomposed_df = pd.concat(decomposed_dfs, axis=1)
#         df = pd.concat([df, decomposed_df], axis=1)
#         df.drop(columns=[col for col in record if record[col] != 'stationary'], inplace=True)
#
#     return df


class DataPreprocessor(PgDbUpdaterBase):

    def __init__(self, base_config: BaseConfig, date_start: str = '2010-01-01', industry: str = None,
                 stationary: bool = True):
        """
        ����Ԥ������ĳ�ʼ������
        :param data: ԭʼ����,DataFrame��ʽ
        :param freq: ����Ƶ��,Ĭ��Ϊ'M'(�¶�)
        :param info: ����˵����,����ÿ�����ݵĴ������,DataFrame��ʽ
        """
        super().__init__(base_config)
        self.date_start = date_start
        self.industry = industry
        self.stationary = stationary
        self.excel_file_mapping = {'��ҵ״��': '�������',
                                   '������ָ': '�������',
                                   '����': '�������',
                                   }
        self.additional_data_mapping = {'��ҵ״��': '�������',
                                        '������ָ': '�й�:�������Ʒ�����ܶ�:����ͬ��',
                                        '����': '�й�:���ڽ��:����ͬ��',
                                        # '����': '����:�����ܶ�:����:ͬ��-����:����ܶ�:����:ͬ��:+6��',
                                        }

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
        if self.stationary:
            self.get_stationary()
        self.cap_outliers()

    def read_data_and_info(self):
        file_path = rf'{self.base_config.excels_path}/����'

        # ��ȡ���ָ������info
        # ���ָ��������Ϊindex(����ָ��ID����Ϊ�������������)
        # ʹ�������ַ�����Ϊ DataFrame ���������ܻ�����һЩ���������,�����ظ��С��������ַ��������ȫ��ת��Ϊ�»���_ Ҳû�á�
        # �ҵ�ԭ����combined_data = pd.merge
        info = pd.read_excel(rf'{file_path}/indicators_info.xlsx', engine="openpyxl", sheet_name=self.industry)
        self.id_to_name = dict(zip(info['ָ��ID'], info['ָ������']))
        self.info = info.set_index('ָ������')

        # ��ȡ���ָ��
        # ��ָ��������Ϊ�У�����Ϊindex
        if self.industry in self.excel_file_mapping:
            excel_file_name = self.excel_file_mapping[self.industry]
        else:
            excel_file_name = '��ҵ�������ݿ�'
        df = pd.read_excel(rf'{file_path}/{excel_file_name}.xlsx', sheet_name=self.industry)
        df_dict = split_dataframe(whole_df=df)

        # ��df_dict�е�ÿ��DataFrame����ɸѡ���ڲ�����
        df_dict = {
            key: value[value.index >= pd.Timestamp(self.date_start)].sort_index(ascending=True)
            for key, value in df_dict.items()
        }

        assert df_dict['������'].index.is_unique, "�йۻ��������� �� Index �����ظ�ֵ"

        # ����һ���б�, �洢Ҫ�޳�������, ��ѡֻ��info�г��ֵ�ָ����д���
        financials_cols = ['���ʲ�������ROE', '����ĸ��˾�ɶ��ľ�����ͬ��������', 'Ӫҵ����ͬ��������']
        indicators_cols = self.info.index.tolist()
        indicators_cols.append(
            self.additional_data_mapping[self.industry]) if self.industry in self.excel_file_mapping else None
        combined_data = pd.merge(df_dict['������'][indicators_cols], df_dict['����'], left_index=True, right_index=True,
                                 how='outer')
        # ɾ���ظ�����
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

        # �޳�Ӱ������0ֵ
        self.df_indicators = combined_data[indicators_cols].replace(0, np.nan).astype(float)
        self.df_finalcials = combined_data[financials_cols]

    def special_mannual_treatment(self):
        """
        �����ݽ������⴦��
        - �ϲ�ĳЩ��
        - ���ۼ�ֵת��Ϊ�¶�ֵ
        """
        X = self.df_indicators.copy(deep=True)
        X = X.drop(
            columns=[self.additional_data_mapping[self.industry]]) if self.industry in self.excel_file_mapping else X
        # ɾ��������ȫ��Ϊ NaN ����
        X = X.dropna(how='all', axis=0)
        # �������'M5528820'��,������'M0329545'�кϲ�
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # ����info���б��Ϊ�ۼ�ֵ����,����ת��Ϊ�¶�ֵ
        for name in X.columns:
            if not pd.isna(self.info.loc[name, '�Ƿ��ۼ�ֵ']):
                new_name = '(�¶Ȼ�)' + name
                # ת��ָ����
                self.info.loc[new_name] = self.info.loc[name]
                self.info = self.info.drop(name)
                # �ۻ�ֵתΪ�¶�
                X.loc[:, new_name] = transform_cumulative_data(X.loc[:, name], self.info.loc[new_name, '�Ƿ��ۼ�ֵ'])
                X.drop(name, axis=1, inplace=True)

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
        month_end_df.index = pd.to_datetime(month_end_df.index)  # .to_period('M')
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
            if df[col_ind].isnull().values.any():
                col = col.dropna()
            try:
                record[col_ind] = self.station_test(col)
            except:
                raise Exception

        # self.data = process_dataframe(self.data, record)
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
                    # decomposed_df[col_ind + '_trend'] += col_series.iloc[0]

                elif record[col_ind] == 'trend-stationary':
                    x = np.arange(len(col_series))
                    trend = np.polyfit(x, col_series, 1)[0] * x
                    detrended_series = col_series - trend
                    stl = STL(detrended_series, period=12)
                    decomposed = stl.fit()
                    decomposed_df = pd.DataFrame({
                        # col_ind + '_trend': decomposed.trend + trend,
                        col_ind + '_trend': decomposed.trend,
                        # col_ind + '_resid': decomposed.resid
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
        # for col_ind, col in df.items():
        #     if record[col_ind] != 'stationary':
        #         original_index = df.index
        #         col_series = pd.Series(col, index=original_index)
        #         col_series = col_series.dropna()
        #
        #         if record[col_ind] == 'diff-stationary':
        #             diff_series = col_series.diff().dropna()
        #             stl = STL(diff_series, period=12)
        #             decomposed = stl.fit()
        #             decomposed_df = pd.DataFrame({
        #                 col_ind + '_trend': decomposed.trend.cumsum(),
        #                 col_ind + '_resid': decomposed.resid
        #             }, index=diff_series.index)
        #             decomposed_df[col_ind + '_trend'] += col_series.iloc[0]
        #
        #         elif record[col_ind] == 'trend-stationary':
        #             x = np.arange(len(col_series))
        #             trend = np.polyfit(x, col_series, 1)[0] * x
        #             detrended_series = col_series - trend
        #             stl = STL(detrended_series, period=12)
        #             decomposed = stl.fit()
        #             decomposed_df = pd.DataFrame({
        #                 col_ind + '_trend': trend,  # ʹ���������ƶ���STL�ֽ������
        #                 col_ind + '_resid': decomposed.resid
        #             }, index=col_series.index)
        #
        #         else:  # 'non-stationary'
        #             log_series = np.log(col_series)  # �Է�ƽ������ȡ����
        #             stl = STL(log_series, period=12)
        #             decomposed = stl.fit()
        #             decomposed_df = pd.DataFrame({
        #                 col_ind + '_trend': np.exp(decomposed.trend),  # ������ȡָ����ԭ
        #                 col_ind + '_resid': decomposed.resid
        #             }, index=col_series.index)
        #
        #         decomposed_df = decomposed_df.dropna(axis=1, how='all')
        #         decomposed_df = decomposed_df.reindex(original_index)
        #         df = pd.concat([df, decomposed_df], axis=1)
        #         df.drop(col_ind, inplace=True, axis=1)
        #
        # self.data = df

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
        self.data = cap_outliers(data=self.data, threshold=threshold)
