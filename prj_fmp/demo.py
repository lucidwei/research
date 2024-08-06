# coding=gbk
# Time Created: 2024/7/29 10:57
# Author  : Lucid
# FileName: demo.py
# Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class FMPModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.asset_data = None
        self.macro_data = None
        self.lasso = None
        self.selected_features = None
        self.selected_weights = None
        self.intercept = None
        self.fmp_series = None

    def load_data(self):
        self.asset_data = pd.read_excel(self.file_path, sheet_name='�ʲ�����', header=1)
        self.macro_data = pd.read_excel(self.file_path, sheet_name='�������')

    def preprocess_data(self):
        # �ʲ���������
        self.asset_data.columns = self.asset_data.iloc[0]
        self.asset_data = self.asset_data.drop([0, 1]).reset_index(drop=True)
        self.asset_data['����'] = pd.to_datetime(self.asset_data['����'])
        self.asset_data.set_index('����', inplace=True)
        self.asset_data = self.asset_data.apply(pd.to_numeric, errors='coerce')

        # ���ʲ��۸�����ת��Ϊͬ�ȱ仯��
        self.asset_data = self.asset_data.resample('M').last()
        asset_data_yoy = self.asset_data.pct_change(periods=12) * 100
        # TODO:��ʱ�Դ��� ɸѡ������2016��֮������ݣ�ɾ��֮ǰ������
        asset_data_yoy = asset_data_yoy.loc['2016-01-01':]

        # ɾ��������Ϊ NaN ����
        asset_data_yoy = asset_data_yoy.dropna(how='all')
        # �ҵ��� NaN ֵ����
        cols_with_nan = asset_data_yoy.columns[asset_data_yoy.isna().any()].tolist()
        # ��ȡɾ�������������Ӧ������������ݵ������
        earliest_dates = {}
        for col in cols_with_nan:
            earliest_date = asset_data_yoy[col].first_valid_index()
            earliest_dates[col] = earliest_date
        print("asset_data_yoyɾ�����м�������������ݵ������:")
        for col, date in earliest_dates.items():
            print(f"{col}: {date}")
        # ɾ���� NaN ֵ����
        asset_data_yoy = asset_data_yoy.dropna(axis=1)

        # �������
        self.macro_data.columns = self.macro_data.iloc[0]
        self.macro_data = self.macro_data.drop(range(0, 6)).reset_index(drop=True)
        self.macro_data = self.macro_data.replace(0, np.nan)
        self.macro_data = self.macro_data.rename(columns={'ָ������': '����'})
        self.macro_data['����'] = pd.to_datetime(self.macro_data['����'])
        self.macro_data.set_index('����', inplace=True)

        # �����۾��������ֵ�
        macro_dict = {
            '����': ['�й�:����ҵPMI:12���ƶ�ƽ��:����ƽ��'],
            'ͨ��': ['�й�:PPI:ȫ����ҵƷ:����ͬ��', '�й�:CPI:����ͬ��'],
            '����': ['�й�:������ʹ�ģ:����ֵ:�ۼ�ֵ', '�й�:����Ҵ���:����ֵ:�ۼ�ֵ'],
            '������': ['�й�:M2:ͬ��']
        }

        macro_data_final = pd.DataFrame(index=self.macro_data.index)

        # ���㲨���ʵ�����Ȩƽ��
        for key, columns in macro_dict.items():
            weighted_sum = np.zeros(len(self.macro_data))
            weight_sum = np.zeros(len(self.macro_data))

            for col in columns:
                if col in self.macro_data.columns:
                    std_dev = self.macro_data[col].std()
                    if std_dev != 0:
                        weight = 1 / std_dev
                        weighted_sum += self.macro_data[col] * weight
                        weight_sum += weight

            # �����Ȩƽ��
            if np.any(weight_sum != 0):
                macro_data_final[key] = weighted_sum / weight_sum
            else:
                macro_data_final[key] = np.nan

        self.macro_data = macro_data_final

        return asset_data_yoy

    def fit_model(self, asset_data_yoy, macro_aspect: str):
        self.macro_aspect = macro_aspect
        self.fmp_series = pd.Series(index=asset_data_yoy.index, name='FMP')

        # ʹ�ù������ڽ���ѵ��
        for i in range(36, len(asset_data_yoy)):
            train_start = asset_data_yoy.index[i - 36]
            train_end = asset_data_yoy.index[i - 1]
            test_date = asset_data_yoy.index[i]

            # ����ѵ�����Ͳ��Լ�
            train_data = asset_data_yoy.loc[train_start:train_end].join(
                self.macro_data[macro_aspect].loc[train_start:train_end], how='inner').dropna()
            test_data = asset_data_yoy.loc[test_date:test_date].join(
                self.macro_data[macro_aspect].loc[test_date:test_date], how='inner').dropna()

            if train_data.empty or test_data.empty:
                continue

            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]

            # LassoCV���н�����֤ѡ����ѳͷ�ϵ��
            lasso_cv = LassoCV(cv=10, max_iter=10000).fit(X_train, y_train)
            alpha = lasso_cv.alpha_

            # ʹ����ѳͷ�ϵ������Lasso�ع�
            lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
            selected_features = X_train.columns[(lasso.coef_ != 0)]
            selected_weights = lasso.coef_[lasso.coef_ != 0]
            intercept = lasso.intercept_

            # ��ӡѡ����ʲ�����Ȩ��
            print(f"Training period: {train_start} to {train_end}")
            print("Selected features for FMP and their weights:")
            for feature, weight in zip(selected_features, selected_weights):
                print(f"{feature}: {weight:.3f}")

            # FMP����
            X_test = test_data[selected_features]
            fmp_value = np.dot(X_test, selected_weights) + intercept
            self.fmp_series[test_date] = fmp_value[0]

        self.fmp_series.dropna(inplace=True)

    def plot_results(self):
        # �ϲ�FMP��ԭʼPMI���ݣ�ֻ�������߾������ݵ����ڲ���
        combined_data = pd.concat([self.macro_data[self.macro_aspect], self.fmp_series],
                                  axis=1).dropna()

        # ��ͼ�Ա�
        plt.figure(figsize=(12, 6))
        plt.plot(combined_data.index, combined_data[self.macro_aspect], label=self.macro_aspect, color='b')
        plt.plot(combined_data.index, combined_data['FMP'], label='FMP', color='r')
        plt.xlabel('����')
        plt.ylabel('ֵ')
        plt.title(f'FMP��{self.macro_aspect}ָ��Ա�')
        plt.legend()
        plt.grid(True)
        plt.show()


# ʹ����
file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\FMP\�ʲ�������������.xlsx"
fmp_model = FMPModel(file_path)
fmp_model.load_data()
asset_data_yoy = fmp_model.preprocess_data()
# fmp_model.fit_model(asset_data_yoy, '����')
fmp_model.fit_model(asset_data_yoy, 'ͨ��')
fmp_model.plot_results()