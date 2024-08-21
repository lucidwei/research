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
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


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
        self.asset_data = pd.read_excel(self.file_path, sheet_name='资产行情', header=1)
        self.macro_data = pd.read_excel(self.file_path, sheet_name='宏观数据')

    def preprocess_data(self):
        # 资产行情数据
        self.asset_data.columns = self.asset_data.iloc[0]
        self.asset_data = self.asset_data.drop([0, 1]).reset_index(drop=True)
        self.asset_data['日期'] = pd.to_datetime(self.asset_data['日期'])
        self.asset_data.set_index('日期', inplace=True)
        self.asset_data = self.asset_data.apply(pd.to_numeric, errors='coerce')

        # 将资产价格数据转换为同比变化率
        self.asset_data = self.asset_data.resample('M').last()
        asset_data_yoy = self.asset_data.pct_change(periods=12) * 100
        # TODO:临时性代码 筛选日期在2016年之后的数据，删除之前的数据
        asset_data_yoy = asset_data_yoy.loc['2016-01-01':]

        # 删除所有列为 NaN 的行
        asset_data_yoy = asset_data_yoy.dropna(how='all')
        # 找到有 NaN 值的列
        cols_with_nan = asset_data_yoy.columns[asset_data_yoy.isna().any()].tolist()
        # 提取删除的列名及其对应的最早可用数据点的日期
        earliest_dates = {}
        for col in cols_with_nan:
            earliest_date = asset_data_yoy[col].first_valid_index()
            earliest_dates[col] = earliest_date
        print("asset_data_yoy删除的列及其最早可用数据点的日期:")
        for col, date in earliest_dates.items():
            print(f"{col}: {date}")
        # 删除有 NaN 值的列
        asset_data_yoy = asset_data_yoy.dropna(axis=1)

        # 宏观数据
        self.macro_data.columns = self.macro_data.iloc[0]
        self.macro_data = self.macro_data.drop(range(0, 6)).reset_index(drop=True)
        self.macro_data = self.macro_data.replace(0, np.nan)
        self.macro_data = self.macro_data.rename(columns={'指标名称': '日期'})
        self.macro_data['日期'] = pd.to_datetime(self.macro_data['日期'])
        self.macro_data.set_index('日期', inplace=True)

        # 定义宏观经济数据字典
        macro_dict = {
            '增长': ['中国:制造业PMI:12月移动平均:算术平均'],
            '通胀': ['中国:PPI:全部工业品:当月同比', '中国:CPI:当月同比'],
            '信用': ['中国:社会融资规模存量:同比', '中国:M1:同比', '中国:M2:同比', '中国:金融机构:各项贷款余额:人民币:同比'],
            '流动性': ['中国:M2:同比-中国:社会融资规模存量:同比']
        }

        macro_data_final = pd.DataFrame(index=self.macro_data.index)

        # 计算波动率倒数加权平均
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

            # 计算加权平均
            if np.any(weight_sum != 0):
                macro_data_final[key] = weighted_sum / weight_sum
            else:
                macro_data_final[key] = np.nan

        self.macro_data = macro_data_final

        return asset_data_yoy

    def fit_model(self, asset_data_yoy, macro_aspect: str):
        self.macro_aspect = macro_aspect
        self.fmp_series = pd.Series(index=asset_data_yoy.index, name='FMP')

        # 确保 asset_data_yoy 和 self.macro_data 的索引为 DatetimeIndex，并设置频率
        self.macro_data = self.macro_data.asfreq('M')

        # 合并数据
        data = asset_data_yoy.join(self.macro_data[macro_aspect], how='inner').dropna()
        X = data.iloc[:, :-1]  # 资产行情数据
        y = data.iloc[:, -1]  # 宏观数据

        # 计算样本权重：越近的月份数据权重越大
        date_index = pd.to_datetime(X.index)
        latest_date = date_index.max()
        months_diff = ((latest_date - date_index) / pd.Timedelta(days=30)).astype(int)
        sample_weights = np.exp(-months_diff / 24)  # 使用指数衰减，一年的半衰期

        # LassoCV进行交叉验证选择最佳惩罚系数
        lasso_cv = LassoCV(cv=10, max_iter=10000)
        lasso_cv.fit(X, y, sample_weight=sample_weights)

        # 获取最佳的 alpha 值
        alpha = lasso_cv.alpha_

        # 使用最佳alpha和样本权重进行最终拟合
        self.lasso = Lasso(alpha=alpha, max_iter=10000)
        self.lasso.fit(X, y, sample_weight=sample_weights)
        self.selected_features = X.columns[(self.lasso.coef_ != 0)]
        self.selected_weights = self.lasso.coef_[self.lasso.coef_ != 0]
        self.intercept = self.lasso.intercept_

        # 打印选择的资产及其权重
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

        # FMP计算
        fmp = np.dot(data[self.selected_features], self.selected_weights) + self.intercept
        self.fmp_series = pd.Series(fmp, index=data.index, name='FMP')

        # 删除没有预测值的日期
        self.fmp_series.dropna(inplace=True)

        # 用最新的FMP数据预测最新月份宏观数据
        last_macro_date = self.macro_data.index[-1]
        print(f"宏观数据最后一期的日期: {last_macro_date}")
        self.latest_index = asset_data_yoy.index[-1]
        print(f"预测期的日期: {self.latest_index}")

        latest_data = asset_data_yoy[self.selected_features].iloc[-1].values.reshape(1, -1)
        self.predicted_macro = np.dot(latest_data, self.selected_weights) + self.intercept

        self.asset_data_yoy = asset_data_yoy  # 保存资产数据，用于后续分析

        # 存储特定方面的FMP系列和选定的权重
        setattr(self, f'fmp_series_{macro_aspect}', self.fmp_series)
        setattr(self, f'selected_weights_{macro_aspect}', dict(zip(self.selected_features, self.selected_weights)))

    def plot_timeseries(self):
        # 合并FMP和原始宏观数据，只保留两者均有数据的日期部分
        combined_data = pd.concat([self.macro_data[self.macro_aspect], self.fmp_series], axis=1).dropna()

        # 绘图对比
        plt.figure(figsize=(12, 6))
        plt.plot(combined_data.index, combined_data[self.macro_aspect], label=self.macro_aspect, color='b')
        plt.plot(combined_data.index, combined_data['FMP'], label='FMP', color='r')

        # 拼接预测点到FMP
        combined_data_with_prediction = combined_data.copy()
        combined_data_with_prediction.loc[self.latest_index, self.macro_aspect] = None
        combined_data_with_prediction.loc[self.latest_index, 'FMP'] = self.predicted_macro

        # 绘制预测点，颜色与FMP相同，但透明度高一些
        plt.plot(combined_data_with_prediction.index, combined_data_with_prediction['FMP'],
                 label='预测', color='r', alpha=0.5)

        plt.title(f'FMP与{self.macro_aspect}指标对比')
        plt.legend()
        plt.grid(True)
        plt.show()

    def explain_prediction(self):
        # 获取上个月和当前月的资产数据
        last_month_data = self.asset_data_yoy[self.selected_features].iloc[-2]
        current_month_data = self.asset_data_yoy[self.selected_features].iloc[-1]

        # 获取日期信息
        start_date = last_month_data.name.strftime('%Y-%m-%d')
        end_date = current_month_data.name.strftime('%Y-%m-%d')

        # 计算每个资产的贡献
        contributions = (current_month_data.values - last_month_data.values) * self.selected_weights

        # 创建一个DataFrame来存储结果
        explanation_df = pd.DataFrame({
            'Asset': self.selected_features,
            'Weight': self.selected_weights,
            'Last Month Value': last_month_data.values,
            'Current Month Value': current_month_data.values,
            'Change': current_month_data.values - last_month_data.values,
            'Contribution': contributions
        })

        # 按贡献绝对值排序
        explanation_df = explanation_df.sort_values('Contribution', key=abs, ascending=False)

        # 计算总变化
        total_change = explanation_df['Contribution'].sum()

        return explanation_df, total_change, start_date, end_date

    def plot_explanation(self):
        explanation_df, total_change, start_date, end_date = self.explain_prediction()

        # 按贡献绝对值排序
        sorted_df = explanation_df.sort_values('Contribution', ascending=False)

        # 选取贡献值前三和后三
        top_bottom = pd.concat([sorted_df.head(3), sorted_df.tail(3)])

        print(f"\n预测{self.macro_aspect}指标变化: {total_change:.2f}")
        print("\n主要贡献资产:")
        print(top_bottom[['Asset', 'Weight', 'Change', 'Contribution']].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"  # 保留2位小数
        ))

        def waterfall_plot(labels, values, net):
            plt.figure(figsize=(10, 6))

            # 创建一个包含所有值的索引数组
            indices = np.arange(len(values))

            # 分别处理正值和负值
            pos_mask = values > 0
            neg_mask = values < 0

            # 计算累积和
            cumulative = np.zeros_like(values)
            cumulative[1:] = np.cumsum(values[:-1])

            # 绘制正值条形
            plt.bar(indices[pos_mask], values[pos_mask], bottom=cumulative[pos_mask], color='r', width=0.8)

            # 绘制负值条形
            plt.bar(indices[neg_mask], values[neg_mask], bottom=cumulative[neg_mask], color='g', width=0.8)

            # 添加Net值
            plt.bar(len(values), net, color='b', width=0.8)

            # 设置x轴标签
            all_labels = np.concatenate([labels, ['Net']])
            plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')

            # 调整y轴范围
            min_value = min(cumulative.min(), 0)
            max_value = max(cumulative.max(), values.max(), net)
            plt.ylim(min_value, max_value + (max_value - min_value) * 0.1)  # 添加10%的顶部空间

            # 添加水平线
            plt.axhline(0, color='black', linewidth=0.5)

            # 添加标题和调整布局
            plt.title(f'{self.macro_aspect}指标{start_date}至{end_date}变化归因', fontsize=16)
            plt.tight_layout()

            plt.show()

        # 使用修改后的 waterfall_plot 函数
        waterfall_plot(explanation_df['Asset'].values, explanation_df['Contribution'].values, total_change)

    def plot_results(self):
        self.plot_timeseries()
        self.plot_explanation()

    def predict_asset_direction(self, top_weights_percent=0.3, min_top_weights=10):
        aspects = ['增长', '通胀', '流动性', '信用']
        fmp_diff = {}
        latest_macro = {}
        latest_fmp = {}

        for aspect in aspects:
            if not hasattr(self, f'fmp_series_{aspect}'):
                raise ValueError(f"请先为 {aspect} 运行 fit_model")

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

        print("宏观方面预测结果：")
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

        print(f"\n宏观预期修正下资产价格应该变动的方向(以{latest_date.strftime('%Y-%m-%d')}收盘价为基准)：")

        def print_asset_info(asset, dir):
            percentage = round(dir * 100, 1)
            contributions = [f"({aspect}){round(contrib * 100, 1)}%" for aspect, contrib in
                             detailed_direction[asset].items() if contrib != 0]
            print(f"{asset}: {percentage}% = " + " + ".join(contributions))

        # 将资产分为两组并打印
        non_zhongxin = [(a, d) for a, d in sorted_direction if "中信" not in a]
        zhongxin = [(a, d) for a, d in sorted_direction if "中信" in a]

        for asset, dir in non_zhongxin:
            print_asset_info(asset, dir)

        if zhongxin:
            print("\n行业资产：")
            for asset, dir in zhongxin:
                print_asset_info(asset, dir)

        return results, direction, detailed_direction

    def pca_analysis(self, asset_data_yoy):
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(asset_data_yoy)

        # 执行PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # 计算解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_

        # 创建一个包含前三个主成分的DataFrame
        pca_df = pd.DataFrame(data=pca_result[:, :3],
                              columns=['PC1', 'PC2', 'PC3'],
                              index=asset_data_yoy.index)

        # 打印前三个主成分的解释方差比
        print("Explained variance ratio of first 3 PCs:")
        for i, ratio in enumerate(explained_variance_ratio[:3], 1):
            print(f"PC{i}: {ratio:.4f}")

        # 分析每个主成分的构成
        component_df = pd.DataFrame(data=pca.components_.T[:, :3],
                                    columns=['PC1', 'PC2', 'PC3'],
                                    index=asset_data_yoy.columns)

        print("\nTop contributors to each principal component:")
        for pc in ['PC1', 'PC2', 'PC3']:
            print(f"\n{pc}:")
            print(component_df[pc].abs().sort_values(ascending=False).head())

        # 绘制前三个主成分的时间序列
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




# 使用类
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\FMP\资产行情与宏观数据.xlsx"
fmp_model = FMPModel(file_path)
fmp_model.load_data()
asset_data_yoy = fmp_model.preprocess_data()
# 执行PCA分析
# fmp_model.pca_analysis(asset_data_yoy)

fmp_model.fit_model(asset_data_yoy, '增长')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, '通胀')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, '流动性')
# fmp_model.plot_results()
fmp_model.fit_model(asset_data_yoy, '信用')
fmp_model.plot_results()

# 预测资产价格变动方向
results = fmp_model.predict_asset_direction()

# def fit_model(self, asset_data_yoy, macro_aspect: str):
#     self.macro_aspect = macro_aspect
#     self.fmp_series = pd.Series(index=asset_data_yoy.index, name='FMP')
#     self.yearly_weights = {}  # 用于存储每年的权重
#
#     # 确保 asset_data_yoy 和 self.macro_data 的索引为 DatetimeIndex，并设置频率
#     self.macro_data = self.macro_data.asfreq('M')
#
#     # 使用滚动窗口进行训练，每年训练一次
#     for year in range(asset_data_yoy.index.year.min() + 3, asset_data_yoy.index.year.max() + 1):  # 确保有足够的训练数据
#         train_start = pd.Timestamp(f'{year - 3}-01-01')
#         train_end = pd.Timestamp(f'{year - 1}-12-31')
#         test_start = pd.Timestamp(f'{year}-01-01')
#         test_end = pd.Timestamp(f'{year}-12-31')
#
#         # 划分训练集和测试集
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
#         # LassoCV进行交叉验证选择最佳惩罚系数
#         lasso_cv = LassoCV(cv=10, max_iter=10000).fit(X_train, y_train)
#         alpha = lasso_cv.alpha_
#
#         # 使用最佳惩罚系数进行Lasso回归
#         lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
#         selected_features = X_train.columns[(lasso.coef_ != 0)]
#         selected_weights = lasso.coef_[lasso.coef_ != 0]
#         intercept = lasso.intercept_
#
#         # 存储每年的权重
#         self.yearly_weights[year] = {feature: weight for feature, weight in
#                                      zip(selected_features, selected_weights)}
#
#         # 打印选择的资产及其权重
#         print(f"Training period: {train_start} to {train_end}")
#         print("Selected features for FMP and their weights:")
#         for feature, weight in zip(selected_features, selected_weights):
#             print(f"{feature}: {weight:.3f}")
#
#         # FMP计算
#         for test_date in test_data.index:
#             X_test = test_data.loc[test_date, selected_features].values.reshape(1, -1)
#             fmp_value = np.dot(X_test, selected_weights) + intercept
#             self.fmp_series[test_date] = fmp_value[0]
#
#     # 删除没有预测值的日期
#     self.fmp_series.dropna(inplace=True)
#
#     # 打印每年的权重变化（排序并打印前三和后三）
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
