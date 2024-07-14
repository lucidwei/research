# coding=gbk
# Time Created: 2024/7/1 17:15
# Author  : Lucid
# FileName: 全A底部放量上涨.py
# Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 读取Excel文件
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\万德全A.xlsx"  # 替换为你的Excel文件路径
df = pd.read_excel(file_path, header=3)

# 确保Date列是datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 将Date列设置为索引
df.set_index('Date', inplace=True)

# 计算100日滚动分位数的低5%分位
# df['100d_pct'] = df['close'].rolling(window=100).quantile(0.5)
# df['100d_pct'] = df['close'].rolling(window=400).min() * 1.05
df['100d_pct'] = df['close'].rolling(window=100).min() * 1.05
df['400d_pct'] = df['close'].rolling(window=400).min() * 1.10

# 计算20日移动平均成交量
df['20d_avg_amt'] = df['amt'].rolling(window=10).mean()

# 计算收盘价涨幅
df['pct_change'] = df['close'].pct_change() * 100

# 添加条件列
df['cond_close_100d'] = df['close'].shift(1) <= df['100d_pct']
df['cond_close_400d'] = df['close'].shift(1) <= df['400d_pct']
df['cond_amt'] = df['amt'] > df['20d_avg_amt'] * 1.1
df['cond_pct_change'] = df['pct_change'] > 2

# 筛选条件
condition = (
    df['cond_close_100d'] &
    df['cond_close_400d'] &
    df['cond_amt'] &
    df['cond_pct_change']
)


# 筛选满足条件的数据
filtered_df = df[condition]

# # 检查某一天的详细信息
# def check_date_details(date):
#     if date in df.index:
#         details = df.loc[date]
#         print(f"Details for {date}:")
#         print(details)
#         print("\nConditions:")
#         print(f"Close <= 100d_pct: {details['close']} <= {details['100d_pct']} -> {details['close'] <= details['100d_pct']}")
#         print(f"Amt > 20d_avg_amt * 1.2: {details['amt']} > {details['20d_avg_amt'] * 1.2} -> {details['amt'] > details['20d_avg_amt'] * 1.2}")
#         print(f"Pct_change > 1: {details['pct_change']} > 1 -> {details['pct_change'] > 1}")
#     else:
#         print(f"No data available for {date}")
#
# # 例如，检查1995-01-06
# check_date_details(pd.Timestamp('2008-11-10'))

# 输出结果
print(filtered_df)




# 获取历年7月1日的数据
july_1_data = df[(df.index.month == 7) & (df.index.day == 1)]

# 计算涨跌幅
# july_1_data['pct_change'] = july_1_data['close'].pct_change() * 100

# 打印或者保存结果
print(july_1_data[['close', 'pct_change']])




# 绘制收盘价走势
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='收盘价', color='blue')

# 标注筛选出来的日期
plt.scatter(filtered_df.index, filtered_df['close'], color='red', label='筛选日期', marker='o')

# 添加标题和标签
plt.title('收盘价走势及筛选日期')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()

# 如果需要将结果保存回Excel，也可以使用以下代码：
# filtered_df.to_excel('筛选结果.xlsx')