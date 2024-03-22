# coding=gbk
# Time Created: 2024/3/20 22:34
# Author  : Lucid
# FileName: 腾讯历年回购.py
# Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 加载Excel文件
file_path = "D:\Downloads\腾讯控股 0700.HK_股票回购.xlsx"
df = pd.read_excel(file_path)

# 数据清洗：去掉前两行并设定正确的列名
df_cleaned = df.drop([0, 1]).reset_index(drop=True)
df_cleaned.columns = ['日期', '回购数量', '回购金额', '最高价', '最低价', '年初至今回购数']

# 转换数据类型
df_cleaned['日期'] = pd.to_datetime(df_cleaned['日期'])
df_cleaned['回购金额'] = df_cleaned['回购金额'].astype(float)

# 提取年份，并按年份汇总回购金额
df_cleaned['年份'] = df_cleaned['日期'].dt.year
annual_repurchase_amount = df_cleaned.groupby('年份')['回购金额'].sum().reset_index()

# 绘制直方图
plt.figure(figsize=(8, 5))
plt.bar(annual_repurchase_amount['年份'], annual_repurchase_amount['回购金额'], color='skyblue')
plt.xlabel('年份')
plt.ylabel('回购金额 (亿)')
plt.title('腾讯控股历年股票回购金额')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
