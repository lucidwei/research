import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 读取数据
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\杂活\指南针要的宏观指标\中美利差\中美利差.xlsx"
data = pd.read_excel(file_path)

# Drop rows with NaN values
data['万得全A_日收益率'] = data['万得全A'].pct_change()
data_cleaned = data.dropna()

# Define the rolling window (approx. 2 years of trading days)
rolling_window = 750

# Calculate rolling percentiles for WanDe QuanA Index and Interest Rate Difference
data_cleaned['万得全A_滚动分位数'] = data_cleaned['万得全A'].rolling(window=rolling_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
data_cleaned['利差_滚动分位数'] = data_cleaned['中国:中债国债到期收益率:10年-美国:国债收益率:10年'].rolling(window=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
data_cleaned['中美利差'] = data_cleaned['中国:中债国债到期收益率:10年-美国:国债收益率:10年'].rolling(window=5).mean()

# Calculate the position indicator and its rolling percentile
data_cleaned['仓位指标'] = data_cleaned['利差_滚动分位数'] - data_cleaned['万得全A_滚动分位数']
data_cleaned['仓位指标_滚动分位数'] = data_cleaned['仓位指标'].rolling(window=250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

# Determine the position (0 to 1 scale)
data_cleaned['仓位'] = data_cleaned['仓位指标_滚动分位数']
# data_cleaned['仓位'] = -data_cleaned['仓位指标']

# Calculate strategy returns
data_cleaned['策略收益'] = data_cleaned['仓位'].shift(1) * data_cleaned['万得全A_日收益率']
data_cleaned['策略累计收益'] = (1 + data_cleaned['策略收益']).cumprod()
data_cleaned['指数累计收益'] = (1 + data_cleaned['万得全A_日收益率']).cumprod()




# 计算中美利差的100日移动平均线
data_cleaned['中美利差100日均线'] = data_cleaned['中国:中债国债到期收益率:10年-美国:国债收益率:10年'].rolling(window=200).mean()

# 实现策略逻辑
data_cleaned['新策略仓位'] = np.where(data_cleaned['中美利差'] > data_cleaned['中美利差100日均线'], 1, 0)

# 计算新策略的净值
data_cleaned['新策略净值'] = (data_cleaned['新策略仓位'].shift(1) * data_cleaned['万得全A'].pct_change() + 1).cumprod()
data_cleaned['新策略净值'].fillna(1, inplace=True)







# Plotting the net value curves
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax1.plot(data_cleaned['指标名称'], data_cleaned['新策略净值'], label='策略累计收益', color='red')
ax1.plot(data_cleaned['指标名称'], data_cleaned['指数累计收益'], label='万得全A累计收益', color='black', alpha=0.5)
ax1.set_xlabel('日期')
ax1.set_ylabel('累计收益', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
# ax2.plot(data_cleaned['指标名称'], data_cleaned['新策略仓位'], label='仓位', color='yellow')
# ax2.plot(data_cleaned['指标名称'], data_cleaned['万得全A_滚动分位数'], label='万得全A_滚动分位数', color='yellow')
# ax2.plot(data_cleaned['指标名称'], data_cleaned['利差_滚动分位数'], label='利差_滚动分位数', color='green')
ax2.plot(data_cleaned['指标名称'], data_cleaned['中美利差'], label='中美利差', color='green')
ax2.plot(data_cleaned['指标名称'], data_cleaned['中美利差100日均线'], label='中美利差100日均线', color='yellow')
ax2.set_ylabel('利差', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')
plt.title('策略与万德全A累计收益比较')
plt.show()