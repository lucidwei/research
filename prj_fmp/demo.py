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
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 读取Excel文件
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\FMP\资产行情与宏观数据.xlsx"  # 你的Excel文件路径
asset_data = pd.read_excel(file_path, sheet_name='资产行情', header=1)
macro_data = pd.read_excel(file_path, sheet_name='宏观数据')

# 数据预处理
# 资产行情数据
asset_data.columns = asset_data.iloc[0]
asset_data = asset_data.drop([0, 1]).reset_index(drop=True)
asset_data['日期'] = pd.to_datetime(asset_data['日期'])
asset_data.set_index('日期', inplace=True)
asset_data = asset_data.apply(pd.to_numeric, errors='coerce')

# 将资产价格数据转换为同比变化率
asset_data = asset_data.resample('M').last()
asset_data_yoy = asset_data.pct_change(periods=12) * 100
# TODO:临时性代码 筛选日期在2016年之后的数据，删除之前的数据
asset_data_yoy = asset_data_yoy.loc['2016-01-01':]

# 先drop所有列为nan的行，然后drop有nan值的列，并把这些列名打印出来
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
macro_data.columns = macro_data.iloc[0]
macro_data = macro_data.drop(range(0, 6)).reset_index(drop=True)
macro_data = macro_data.rename(columns={'指标名称': '日期'})
macro_data['日期'] = pd.to_datetime(macro_data['日期'])
macro_data.set_index('日期', inplace=True)
macro_data = macro_data[['中国:制造业PMI:12月移动平均:算术平均']]
macro_data = macro_data.apply(pd.to_numeric, errors='coerce')


# 合并数据
data = asset_data_yoy.join(macro_data, how='inner').dropna()
X = data.iloc[:, :-1]  # 资产行情数据
y = data.iloc[:, -1]   # PMI数据

# LassoCV进行交叉验证选择最佳惩罚系数
lasso_cv = LassoCV(cv=10, max_iter=10000).fit(X, y)  # 增加最大迭代次数
alpha = lasso_cv.alpha_

# 使用最佳惩罚系数进行Lasso回归
lasso = Lasso(alpha=alpha, max_iter=10000).fit(X, y)  # 增加最大迭代次数
selected_features = X.columns[(lasso.coef_ != 0)]
selected_weights = lasso.coef_[lasso.coef_ != 0]
intercept = lasso.intercept_

# 打印选择的资产及其权重
print("Selected features for FMP and their weights:")
for feature, weight in zip(selected_features, selected_weights):
    print(f"{feature}: {weight:.3f}")

# FMP计算
fmp = np.dot(data[selected_features], selected_weights) + intercept

# 创建包含FMP数据的新DataFrame
fmp_series = pd.Series(fmp, index=data.index, name='FMP')

# 合并FMP和原始PMI数据，只保留两者均有数据的日期部分
combined_data = pd.concat([macro_data['中国:制造业PMI:12月移动平均:算术平均'], fmp_series], axis=1).dropna()

# 绘图对比
plt.figure(figsize=(12, 6))
plt.plot(combined_data.index, combined_data['中国:制造业PMI:12月移动平均:算术平均'], label='原始PMI', color='b')
plt.plot(combined_data.index, combined_data['FMP'], label='FMP', color='r')
plt.xlabel('日期')
plt.ylabel('值')
plt.title('FMP与原始PMI对比')
plt.legend()
plt.grid(True)
plt.show()

# # 绘图对比
# plt.figure(figsize=(12, 6))
# plt.plot(macro_data.index, macro_data['中国:制造业PMI:12月移动平均:算术平均'], label='原始PMI', color='b')
# plt.plot(data.index, fmp, label='FMP', color='r')
# plt.xlabel('日期')
# plt.ylabel('值')
# plt.title('FMP与原始PMI对比')
# plt.legend()
# plt.grid(True)
# plt.show()