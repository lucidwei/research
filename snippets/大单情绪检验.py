# coding=gbk
# Time Created: 2024/6/17 10:57
# Author  : Lucid
# FileName: 大单情绪检验.py
# Software: PyCharm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")


# 读取Excel文件
file_path = r'D:\WPS云盘\WPS云盘\工作-麦高\定期汇报\日报模板整理\各大资金跟踪透视表.xlsx'
sheet_name = '大单情绪'

data = pd.read_excel(file_path, sheet_name=sheet_name)

# 数据预处理
# 将第一行设为列名，并删除前两行
data.columns = data.iloc[1]
data = data.drop([0, 1]).reset_index(drop=True)

# 将日期列转换为日期类型
data['日期'] = pd.to_datetime(data['日期'])
data.set_index('日期', inplace=True)
# 将相关列转换为数值类型
data['万德全A'] = pd.to_numeric(data['万德全A'], errors='coerce')
data['大单情绪'] = pd.to_numeric(data['大单情绪'], errors='coerce')
data['大单情绪_MA5'] = pd.to_numeric(data['大单情绪_MA5'], errors='coerce')

# 删除包含NaN的行
data = data.dropna(subset=['万德全A', '大单情绪', '大单情绪_MA5'])

# 检查数据是否按时间顺序排列
if not data.index.is_monotonic_increasing:
    data = data.sort_index()

# USE_COLUMN = '大单情绪_MA5'
USE_COLUMN = '大单情绪'
# USE_COLUMN = 'Signal'




# # 设定滚动窗口大小
# n = 10
#
# # 计算滚动窗口内的最高值
# data['Max_Sentiment_Rolling'] = data['大单情绪'].rolling(window=n).max()
# data['Min_Sentiment_Rolling'] = data['大单情绪'].rolling(window=n).min()
#
# # 计算新信号列
# data['Signal'] = np.where(data[USE_COLUMN] < (data['Max_Sentiment_Rolling'] - 0.2), -1,
#                         np.where(data[USE_COLUMN] > (data['Min_Sentiment_Rolling'] + 0.2), 1, 0))
#
#
# df = pd.DataFrame(data)
#
# # 画图
# fig, ax1 = plt.subplots(figsize=(14, 7))
#
# # 画万德全A价格
# ax1.plot(df.index, df['万德全A'], color='blue', label='万德全A 价格')
# ax1.set_xlabel('日期')
# ax1.set_ylabel('万德全A 价格', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # 使用不同的颜色和透明度添加阴影表示 Signal
# ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == 1), color='red', alpha=0.3, label='Signal = 1')
# ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == -1), color='green', alpha=0.3, label='Signal = -1')
# # ax1.fill_between(df.index, -1, 6000, where=(df['Signal'] == 0), color='blue', alpha=0.1, label='Signal = 0')
#
# fig.tight_layout()
# fig.suptitle('万德全A 价格与 Signal', y=1.02)
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
# plt.show()




USE_COLUMN = '大单情绪_MA5'
# USE_COLUMN = '大单情绪'
# USE_COLUMN = 'Signal'


# 计算相关性
correlation = data['万德全A'].corr(data[USE_COLUMN])
print(f"Correlation between 万德全A and 大单情绪: {correlation:.4f}")

# 进行1到5阶的Granger因果检验
max_lag = 25
granger_results = grangercausalitytests(data[['万德全A', USE_COLUMN]], max_lag)
# 打印每个滞后期数下的p-value
for lag in range(1, max_lag+1):
    p_value = granger_results[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_value}")

# 构建VAR模型
model = VAR(data[['万德全A', USE_COLUMN]])
results = model.fit(maxlags=25, ic='aic')

# 计算领先性
lead_lag_summary = results.test_causality('万德全A', USE_COLUMN, kind='f')
print(f"Lead-lag relationship (causality test) result:\n {lead_lag_summary}")

# # 计算脉冲响应函数（IRF）
irf = results.irf(10)  # 计算10期的IRF
irf.plot(orth=True)
plt.show()

# 预测误差方差分解
fevd = results.fevd(10)
fevd.plot()

lag_order = results.k_ar
print(f"Optimal lag order: {lag_order}")

# 打印IRF值
# irf_values = irf.irfs
# print("Impulse Response Function values:\n", irf_values)

# 假设data是包含‘万德全A’和‘大单情绪’的DataFrame
model11 = VAR(data[['万德全A', USE_COLUMN]])
results = model11.fit(11)  # 使用最佳滞后期数11

forecast = results.forecast(data[['万德全A', USE_COLUMN]].values[-11:], steps=10)
forecast_df = pd.DataFrame(forecast, columns=['万德全A', USE_COLUMN])
# print(forecast_df)


# 初始化信号列
data['signal'] = 0

# 回测窗口
train_size = 200  # 训练集大小
test_size = 100  # 测试集大小（回测部分）

for t in range(train_size, train_size + test_size):
    # 训练集数据
    train_data = data.iloc[t - train_size:t]

    # 拟合VAR模型
    model = VAR(train_data[['万德全A', USE_COLUMN]])
    results = model.fit(lag_order)

    # 进行一步预测
    forecast = results.forecast(train_data[['万德全A', USE_COLUMN]].values[-lag_order:], steps=1)

    # 获取预测结果
    predicted_value = forecast[0, 0]  # 预测的 '万德全A' 值

    # 决定买卖信号
    if predicted_value > train_data['万德全A'].iloc[-1]:
        data.iloc[t, data.columns.get_loc('signal')] = 1  # 买入信号
    else:
        data.iloc[t, data.columns.get_loc('signal')] = -1  # 卖出信号

# 计算策略收益
data['returns'] = data['万德全A'].pct_change()
data['strategy_returns'] = data['signal'].shift(1) * data['returns']
cumulative_returns = (1 + data['strategy_returns']).cumprod() - 1

print(f"Cumulative returns from strategy: {cumulative_returns.iloc[-1]:.2%}")


# 计算10天滚动相关性
window = 10
data['rolling_corr'] = data['万德全A'].rolling(window).corr(data[USE_COLUMN])

# 绘制万德全A的价格曲线
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['万德全A'], label='万德全A')

# 添加阴影区域
for start_date, end_date in zip(data.index[:-window], data.index[window:]):
    corr_value = data.loc[start_date:end_date, 'rolling_corr'].iloc[-1]
    if corr_value > 0.8:
        plt.axvspan(start_date, end_date, color='red', alpha=0.3)
    elif corr_value < -0.5:
        plt.axvspan(start_date, end_date, color='blue', alpha=0.3)

# 设置图例和标题
plt.legend()
plt.title(f'万德全A价格与{USE_COLUMN}滚动相关性')
plt.xlabel('Date')
plt.ylabel('万德全A')
plt.show()