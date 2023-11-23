import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Adjust font settings for Chinese characters
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # Specify default font to fix issue with Chinese characters not displaying
mpl.rcParams['axes.unicode_minus'] = False

# Load the Excel file
file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\杂活/反弹以来行业表现及拥挤度.xlsx'
xls = pd.ExcelFile(file_path)

# Preprocessing steps
industry_performance = xls.parse(sheet_name='行业日度涨跌幅热力图')
# 删除第一行和前两列
industry_performance = industry_performance.iloc[1:, 2:]
# 将第一列设置为索引
industry_performance.index = industry_performance.iloc[:, 0]
# 将第一行设置为列名, 删除第一行
industry_performance = industry_performance.rename(columns=industry_performance.iloc[0]).iloc[1:, 1:]
# Reversing the columns order to have the earliest date first
industry_performance = industry_performance.iloc[:, ::-1]

start_date = pd.to_datetime("2023-10-24")
virtual_start_date = start_date - pd.Timedelta(days=1)
industry_performance[virtual_start_date] = 0
industry_performance_filtered = industry_performance.loc[:, virtual_start_date:]
cumulative_performance = industry_performance_filtered.sum(axis=1)
sorted_performance = cumulative_performance.sort_values(ascending=False)

cumulative_performance_ts = industry_performance_filtered.cumsum(axis=1).apply(pd.to_numeric, errors='coerce')
# Selecting the top 5 and bottom 5 industries based on their final cumulative performance
top_5_industries = cumulative_performance_ts.iloc[:, -1].nlargest(5).index
bottom_5_industries = cumulative_performance_ts.iloc[:, -1].nsmallest(5).index

# Filtering the data for these industries
top_5_cumulative_performance = cumulative_performance_ts.loc[top_5_industries]
bottom_5_cumulative_performance = cumulative_performance_ts.loc[bottom_5_industries]


# Load the industry crowdedness data
# Load the Excel file
file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\定期汇报\日报模板整理/行业拥挤度图.xlsx'
xls = pd.ExcelFile(file_path)
industry_crowdedness = xls.parse('交易拥挤度')

# Preprocessing steps
industry_crowdedness = industry_crowdedness.iloc[:-2, 2:].dropna(axis=1)
# 将第一列设置为索引，删除第一列
industry_crowdedness.index = industry_crowdedness.iloc[:, 0]
industry_crowdedness = industry_crowdedness.drop(industry_crowdedness.columns[0], axis=1)
# 将第一行设置为列名, 删除第一行
# industry_crowdedness = industry_crowdedness.rename(columns=industry_crowdedness.iloc[0]).iloc[1:-2, :]

# Reversing the columns order to have the earliest date first
industry_crowdedness = industry_crowdedness.iloc[:, ::-1]
# industry_crowdedness_filtered = industry_crowdedness.loc[:, start_date:]
industry_crowdedness_filtered = industry_crowdedness
# industry_crowdedness_filtered['Average Crowdedness'] = industry_crowdedness_filtered.mean(axis=1)
# sorted_crowdedness = industry_crowdedness_filtered['Average Crowdedness'].sort_values(ascending=False)

# Filtering the data for top 5 and bottom 5 industries
top_5_crowdedness = industry_crowdedness_filtered.loc[top_5_industries]
bottom_5_crowdedness = industry_crowdedness_filtered.loc[bottom_5_industries]


# 为每个行业分配颜色
colors = plt.cm.tab10.colors  # 使用matplotlib的tab10颜色映射
industry_colors = {industry: colors[i % len(colors)] for i, industry in enumerate(industry_crowdedness_filtered.index)}

# Plotting the cumulative performance trends
plt.figure(figsize=(7, 7))

# 使用plt.subplot返回一个ax对象
ax = plt.subplot(1, 1, 1)

# Plot for top 5 industries
for industry in top_5_cumulative_performance.index:
    ax.plot(top_5_cumulative_performance.columns, top_5_cumulative_performance.loc[industry] * 100, label=industry, marker='o', markersize=2, alpha=0.5)
for industry in bottom_5_cumulative_performance.index:
    ax.plot(bottom_5_cumulative_performance.columns, bottom_5_cumulative_performance.loc[industry] * 100, label=industry, marker='o', markersize=2, alpha=0.5)

# 设置标题、坐标轴标签
ax.set_title('Top 5 vs Bottom 5 Industries Cumulative Performance')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Performance (%)')

# 设置x轴标签的旋转角度
plt.xticks(rotation=45)

# 设置y轴标签为百分比格式
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)

# 显示图例
ax.legend()
plt.show()


# Plotting the industry crowdedness for top 5 and bottom 5 industries
plt.figure(figsize=(15, 7))

# Plot for top 5 industries
ax1 = plt.subplot(1, 2, 1)
for industry in top_5_crowdedness.index:
    color = industry_colors[industry]
    # 绘制整条线
    ax1.plot(top_5_crowdedness.columns,
             top_5_crowdedness.loc[industry] * 100,
             label=industry, color=color)
    # 在特定日期之后的部分添加marker
    ax1.plot(top_5_crowdedness.columns[top_5_crowdedness.columns >= start_date],
             top_5_crowdedness.loc[industry, top_5_crowdedness.columns >= start_date] * 100,
             marker='o', markersize=3, alpha=0.7, linestyle='', color=color)

ax1.set_title('Top 5 行业拥挤度')
plt.xticks(rotation=45)
ax1.legend()
ax1.yaxis.set_major_formatter(yticks)

# Plot for bottom 5 industries
ax2 = plt.subplot(1, 2, 2)
for industry in bottom_5_crowdedness.index:
    # ax2.plot(bottom_5_crowdedness.columns, bottom_5_crowdedness.loc[industry] * 100, label=industry, marker='o', markersize=2, alpha=0.5)
    color = industry_colors[industry]
    # 绘制整条线
    ax2.plot(bottom_5_crowdedness.columns,
             bottom_5_crowdedness.loc[industry] * 100,
             label=industry, color=color)
    # 在特定日期之后的部分添加marker
    ax2.plot(bottom_5_crowdedness.columns[bottom_5_crowdedness.columns >= start_date],
             bottom_5_crowdedness.loc[industry, bottom_5_crowdedness.columns >= start_date] * 100,
             marker='o', markersize=3, alpha=0.7, linestyle='', color=color)

ax2.set_title('Bottom 5 行业拥挤度')
plt.xticks(rotation=45)
ax2.legend()
ax2.yaxis.set_major_formatter(yticks)

# Display the plot
plt.tight_layout()
plt.show()



