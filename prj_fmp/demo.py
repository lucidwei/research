# coding=gbk
# Time Created: 2024/7/29 10:57
# Author  : Lucid
# FileName: demo.py
# Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
            '����': ['�й�:������ʹ�ģ����:ͬ��', '�й�:M1:ͬ��', '�й�:M2:ͬ��', '�й�:���ڻ���:����������:�����:ͬ��'],
            '������': ['�й�:M2:ͬ��-�й�:������ʹ�ģ����:ͬ��']
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

        # ȷ�� asset_data_yoy �� self.macro_data ������Ϊ DatetimeIndex��������Ƶ��
        self.macro_data = self.macro_data.asfreq('M')

        # �ϲ�����
        data = asset_data_yoy.join(self.macro_data[macro_aspect], how='inner').dropna()
        X = data.iloc[:, :-1]  # �ʲ���������
        y = data.iloc[:, -1]  # �������

        # ��������Ȩ�أ�Խ�����·�����Ȩ��Խ��
        date_index = pd.to_datetime(X.index)
        latest_date = date_index.max()
        months_diff = ((latest_date - date_index) / pd.Timedelta(days=30)).astype(int)
        sample_weights = np.exp(-months_diff / 24)  # ʹ��ָ��˥����һ��İ�˥��

        # LassoCV���н�����֤ѡ����ѳͷ�ϵ��
        lasso_cv = LassoCV(cv=10, max_iter=10000)
        lasso_cv.fit(X, y, sample_weight=sample_weights)

        # ��ȡ��ѵ� alpha ֵ
        alpha = lasso_cv.alpha_

        # ʹ�����alpha������Ȩ�ؽ����������
        self.lasso = Lasso(alpha=alpha, max_iter=10000)
        self.lasso.fit(X, y, sample_weight=sample_weights)
        self.selected_features = X.columns[(self.lasso.coef_ != 0)]
        self.selected_weights = self.lasso.coef_[self.lasso.coef_ != 0]
        self.intercept = self.lasso.intercept_

        # ��ӡѡ����ʲ�����Ȩ��
        weights = {feature: weight for feature, weight in zip(self.selected_features, self.selected_weights)}
        sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        top_3 = sorted_weights[:3]
        bottom_3 = sorted_weights[-3:]

        print("Selected features for FMP and their weights:")
        for feature, weight in zip(self.selected_features, self.selected_weights):
            print(f"{feature}: {weight:.3f}")

        print("\nTop 3 weights:")
        for feature, weight in top_3:
            print(f"  {feature}: {weight:.3f}")

        print("\nBottom 3 weights:")
        for feature, weight in bottom_3:
            print(f"  {feature}: {weight:.3f}\n")

        # FMP����
        fmp = np.dot(data[self.selected_features], self.selected_weights) + self.intercept
        self.fmp_series = pd.Series(fmp, index=data.index, name='FMP')

        # ɾ��û��Ԥ��ֵ������
        self.fmp_series.dropna(inplace=True)

        # �����µ�FMP����Ԥ�������·ݺ������
        last_macro_date = self.macro_data.index[-1]
        print(f"����������һ�ڵ�����: {last_macro_date}")
        self.latest_index = asset_data_yoy.index[-1]
        print(f"Ԥ���ڵ�����: {self.latest_index}")

        latest_data = asset_data_yoy[self.selected_features].iloc[-1].values.reshape(1, -1)
        self.predicted_macro = np.dot(latest_data, self.selected_weights) + self.intercept

        self.asset_data_yoy = asset_data_yoy  # �����ʲ����ݣ����ں�������

        # �洢�ض������FMPϵ�к�ѡ����Ȩ��
        setattr(self, f'fmp_series_{macro_aspect}', self.fmp_series)
        setattr(self, f'selected_weights_{macro_aspect}', dict(zip(self.selected_features, self.selected_weights)))

    def plot_timeseries(self):
        # �ϲ�FMP��ԭʼ������ݣ�ֻ�������߾������ݵ����ڲ���
        combined_data = pd.concat([self.macro_data[self.macro_aspect], self.fmp_series], axis=1).dropna()

        # ��ͼ�Ա�
        plt.figure(figsize=(12, 6))
        plt.plot(combined_data.index, combined_data[self.macro_aspect], label=self.macro_aspect, color='b')
        plt.plot(combined_data.index, combined_data['FMP'], label='FMP', color='r')

        # ƴ��Ԥ��㵽FMP
        combined_data_with_prediction = combined_data.copy()
        combined_data_with_prediction.loc[self.latest_index, self.macro_aspect] = None
        combined_data_with_prediction.loc[self.latest_index, 'FMP'] = self.predicted_macro

        # ����Ԥ��㣬��ɫ��FMP��ͬ����͸���ȸ�һЩ
        plt.plot(combined_data_with_prediction.index, combined_data_with_prediction['FMP'],
                 label='Ԥ��', color='r', alpha=0.5)

        plt.title(f'FMP��{self.macro_aspect}ָ��Ա�')
        plt.legend()
        plt.grid(True)
        plt.show()

    def explain_prediction(self):
        # ��ȡ�ϸ��º͵�ǰ�µ��ʲ�����
        last_month_data = self.asset_data_yoy[self.selected_features].iloc[-2]
        current_month_data = self.asset_data_yoy[self.selected_features].iloc[-1]

        # ��ȡ������Ϣ
        start_date = last_month_data.name.strftime('%Y-%m-%d')
        end_date = current_month_data.name.strftime('%Y-%m-%d')

        # ����ÿ���ʲ��Ĺ���
        contributions = (current_month_data.values - last_month_data.values) * self.selected_weights

        # ����һ��DataFrame���洢���
        explanation_df = pd.DataFrame({
            'Asset': self.selected_features,
            'Weight': self.selected_weights,
            'Last Month Value': last_month_data.values,
            'Current Month Value': current_month_data.values,
            'Change': current_month_data.values - last_month_data.values,
            'Contribution': contributions
        })

        # �����׾���ֵ����
        explanation_df = explanation_df.sort_values('Contribution', key=abs, ascending=False)

        # �����ܱ仯
        total_change = explanation_df['Contribution'].sum()

        return explanation_df, total_change, start_date, end_date

    def plot_explanation(self):
        explanation_df, total_change, start_date, end_date = self.explain_prediction()

        # �����׾���ֵ����
        sorted_df = explanation_df.sort_values('Contribution', ascending=False)

        # ѡȡ����ֵǰ���ͺ���
        top_bottom = pd.concat([sorted_df.head(3), sorted_df.tail(3)])

        print(f"\nԤ��{self.macro_aspect}ָ��仯: {total_change:.2f}")
        print("\n��Ҫ�����ʲ�:")
        print(top_bottom[['Asset', 'Weight', 'Change', 'Contribution']].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"  # ����2λС��
        ))

        def waterfall_plot(labels, values, net):
            plt.figure(figsize=(10, 6))

            # ����һ����������ֵ����������
            indices = np.arange(len(values))

            # �ֱ�����ֵ�͸�ֵ
            pos_mask = values > 0
            neg_mask = values < 0

            # �����ۻ���
            cumulative = np.zeros_like(values)
            cumulative[1:] = np.cumsum(values[:-1])

            # ������ֵ����
            plt.bar(indices[pos_mask], values[pos_mask], bottom=cumulative[pos_mask], color='r', width=0.8)

            # ���Ƹ�ֵ����
            plt.bar(indices[neg_mask], values[neg_mask], bottom=cumulative[neg_mask], color='g', width=0.8)

            # ���Netֵ
            plt.bar(len(values), net, color='b', width=0.8)

            # ����x���ǩ
            all_labels = np.concatenate([labels, ['Net']])
            plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')

            # ����y�᷶Χ
            min_value = min(cumulative.min(), 0)
            max_value = max(cumulative.max(), values.max(), net)
            plt.ylim(min_value, max_value + (max_value - min_value) * 0.1)  # ���10%�Ķ����ռ�

            # ���ˮƽ��
            plt.axhline(0, color='black', linewidth=0.5)

            # ��ӱ���͵�������
            plt.title(f'{self.macro_aspect}ָ��{start_date}��{end_date}�仯����', fontsize=16)
            plt.tight_layout()

            plt.show()

        # ʹ���޸ĺ�� waterfall_plot ����
        waterfall_plot(explanation_df['Asset'].values, explanation_df['Contribution'].values, total_change)

    def plot_results(self):
        self.plot_timeseries()
        self.plot_explanation()

    def predict_asset_direction(self, top_weights_percent=0.3, min_top_weights=10):
        aspects = ['����', 'ͨ��', '������', '����']
        fmp_diff = {}
        latest_macro = {}
        latest_fmp = {}

        for aspect in aspects:
            if not hasattr(self, f'fmp_series_{aspect}'):
                raise ValueError(f"����Ϊ {aspect} ���� fit_model")

            latest_macro[aspect] = self.macro_data[aspect].iloc[-1]
            latest_date = self.macro_data[aspect].index[-1]

            fmp_series = getattr(self, f'fmp_series_{aspect}')
            if latest_date in fmp_series.index:
                latest_fmp[aspect] = fmp_series.loc[latest_date]
            else:
                latest_fmp[aspect] = fmp_series.iloc[-1]

            fmp_diff[aspect] = latest_macro[aspect] - latest_fmp[aspect]

        results = pd.DataFrame({
            'Latest Macro': latest_macro,
            'Latest FMP': latest_fmp,
            'Difference': fmp_diff
        })

        print("��۷���Ԥ������")
        print(results)

        direction = {}
        detailed_direction = {}
        for asset in self.asset_data_yoy.columns:
            asset_direction = 0
            total_weight = 0
            aspect_contributions = {}
            for aspect in aspects:
                weights = getattr(self, f'selected_weights_{aspect}')

                sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
                top_n = max(min_top_weights, int(len(sorted_weights) * top_weights_percent))
                top_weights = dict(sorted_weights[:top_n])

                if asset in top_weights:
                    weight = top_weights[asset]
                    aspect_weight = abs(fmp_diff[aspect])
                    contribution = fmp_diff[aspect] * weight * aspect_weight
                    asset_direction += contribution
                    total_weight += aspect_weight
                    aspect_contributions[aspect] = contribution

            if total_weight > 0:
                direction[asset] = asset_direction / total_weight
                detailed_direction[asset] = {aspect: (contrib / total_weight) for aspect, contrib in
                                             aspect_contributions.items()}
            else:
                pass
                # direction[asset] = 0
                # detailed_direction[asset] = {aspect: 0 for aspect in aspects}

        sorted_direction = sorted(direction.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n���Ԥ���������ʲ��۸�Ӧ�ñ䶯�ķ���(��{latest_date.strftime('%Y-%m-%d')}���̼�Ϊ��׼)��")

        def print_asset_info(asset, dir):
            percentage = round(dir * 100, 1)
            contributions = [f"({aspect}){round(contrib * 100, 1)}%" for aspect, contrib in
                             detailed_direction[asset].items() if contrib != 0]
            print(f"{asset}: {percentage}% = " + " + ".join(contributions))

        # ���ʲ���Ϊ���鲢��ӡ
        non_zhongxin = [(a, d) for a, d in sorted_direction if "����" not in a]
        zhongxin = [(a, d) for a, d in sorted_direction if "����" in a]

        for asset, dir in non_zhongxin:
            print_asset_info(asset, dir)

        if zhongxin:
            print("\n��ҵ�ʲ���")
            for asset, dir in zhongxin:
                print_asset_info(asset, dir)

        return results, direction, detailed_direction

    def pca_analysis(self, asset_data_yoy):
        # ��׼������
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(asset_data_yoy)

        # ִ��PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # ������ͷ����
        explained_variance_ratio = pca.explained_variance_ratio_

        # ����һ������ǰ�������ɷֵ�DataFrame
        pca_df = pd.DataFrame(data=pca_result[:, :3],
                              columns=['PC1', 'PC2', 'PC3'],
                              index=asset_data_yoy.index)

        # ��ӡǰ�������ɷֵĽ��ͷ����
        print("Explained variance ratio of first 3 PCs:")
        for i, ratio in enumerate(explained_variance_ratio[:3], 1):
            print(f"PC{i}: {ratio:.4f}")

        # ����ÿ�����ɷֵĹ���
        component_df = pd.DataFrame(data=pca.components_.T[:, :3],
                                    columns=['PC1', 'PC2', 'PC3'],
                                    index=asset_data_yoy.columns)

        print("\nTop contributors to each principal component:")
        for pc in ['PC1', 'PC2', 'PC3']:
            print(f"\n{pc}:")
            print(component_df[pc].abs().sort_values(ascending=False).head())

        # ����ǰ�������ɷֵ�ʱ������
        plt.figure(figsize=(13, 19))

        # PC1
        plt.subplot(3, 1, 1)
        plt.plot(pca_df.index, pca_df['PC1'])
        plt.title('First Principal Component (PC1)')
        plt.ylabel('Value')
        plt.grid(True)

        # PC2
        plt.subplot(3, 1, 2)
        plt.plot(pca_df.index, pca_df['PC2'])
        plt.title('Second Principal Component (PC2)')
        plt.ylabel('Value')
        plt.grid(True)

        # PC3
        plt.subplot(3, 1, 3)
        plt.plot(pca_df.index, pca_df['PC3'])
        plt.title('Third Principal Component (PC3)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return pca, pca_result, pca_df, component_df




# ʹ����
file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\FMP\�ʲ�������������.xlsx"
fmp_model = FMPModel(file_path)
fmp_model.load_data()
asset_data_yoy = fmp_model.preprocess_data()
# ִ��PCA����
# fmp_model.pca_analysis(asset_data_yoy)

fmp_model.fit_model(asset_data_yoy, '����')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, 'ͨ��')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, '������')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, '����')
fmp_model.plot_results()

# Ԥ���ʲ��۸�䶯����
results = fmp_model.predict_asset_direction()

# def fit_model(self, asset_data_yoy, macro_aspect: str):
#     self.macro_aspect = macro_aspect
#     self.fmp_series = pd.Series(index=asset_data_yoy.index, name='FMP')
#     self.yearly_weights = {}  # ���ڴ洢ÿ���Ȩ��
#
#     # ȷ�� asset_data_yoy �� self.macro_data ������Ϊ DatetimeIndex��������Ƶ��
#     self.macro_data = self.macro_data.asfreq('M')
#
#     # ʹ�ù������ڽ���ѵ����ÿ��ѵ��һ��
#     for year in range(asset_data_yoy.index.year.min() + 3, asset_data_yoy.index.year.max() + 1):  # ȷ�����㹻��ѵ������
#         train_start = pd.Timestamp(f'{year - 3}-01-01')
#         train_end = pd.Timestamp(f'{year - 1}-12-31')
#         test_start = pd.Timestamp(f'{year}-01-01')
#         test_end = pd.Timestamp(f'{year}-12-31')
#
#         # ����ѵ�����Ͳ��Լ�
#         train_data = asset_data_yoy.loc[train_start:train_end].join(
#             self.macro_data[macro_aspect].loc[train_start:train_end], how='inner').dropna()
#         test_data = asset_data_yoy.loc[test_start:test_end].join(
#             self.macro_data[macro_aspect].loc[test_start:test_end], how='inner').dropna()
#
#         if train_data.empty or test_data.empty:
#             continue
#
#         X_train = train_data.iloc[:, :-1]
#         y_train = train_data.iloc[:, -1]
#
#         # LassoCV���н�����֤ѡ����ѳͷ�ϵ��
#         lasso_cv = LassoCV(cv=10, max_iter=10000).fit(X_train, y_train)
#         alpha = lasso_cv.alpha_
#
#         # ʹ����ѳͷ�ϵ������Lasso�ع�
#         lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
#         selected_features = X_train.columns[(lasso.coef_ != 0)]
#         selected_weights = lasso.coef_[lasso.coef_ != 0]
#         intercept = lasso.intercept_
#
#         # �洢ÿ���Ȩ��
#         self.yearly_weights[year] = {feature: weight for feature, weight in
#                                      zip(selected_features, selected_weights)}
#
#         # ��ӡѡ����ʲ�����Ȩ��
#         print(f"Training period: {train_start} to {train_end}")
#         print("Selected features for FMP and their weights:")
#         for feature, weight in zip(selected_features, selected_weights):
#             print(f"{feature}: {weight:.3f}")
#
#         # FMP����
#         for test_date in test_data.index:
#             X_test = test_data.loc[test_date, selected_features].values.reshape(1, -1)
#             fmp_value = np.dot(X_test, selected_weights) + intercept
#             self.fmp_series[test_date] = fmp_value[0]
#
#     # ɾ��û��Ԥ��ֵ������
#     self.fmp_series.dropna(inplace=True)
#
#     # ��ӡÿ���Ȩ�ر仯�����򲢴�ӡǰ���ͺ�����
#     print("Yearly weights (Top 3 and Bottom 3):")
#     for year, weights in self.yearly_weights.items():
#         sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
#         top_3 = sorted_weights[:3]
#         bottom_3 = sorted_weights[-3:]
#         print(f"Year {year}:")
#         print("  Top 3 weights:")
#         for feature, weight in top_3:
#             print(f"    {feature}: {weight:.3f}")
#         print("  Bottom 3 weights:")
#         for feature, weight in bottom_3:
#             print(f"    {feature}: {weight:.3f}")
