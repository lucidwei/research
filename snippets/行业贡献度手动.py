# coding=gbk
# Time Created: 2024/3/15 9:50
# Author  : Lucid
# FileName: 行业贡献度手动.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 设置参数
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\定期汇报\日报模板整理\python用"
sheet_name = '日K'
# 用户定义的截断点，可以根据实际情况进行调整
user_break_points分钟 = ['11:00', '13:09', '13:40']
# 用户定义的截断点，对于日度数据，这些是日期
user_break_points日K = ['2024-01-23', '2024-01-26', '2024-02-05', '2024-02-23']

# 读取数据
industry_data = pd.read_excel(rf'{file_path}/指数、行业走势.xlsx', header=3, index_col=0, sheet_name=sheet_name)
weights_df = pd.read_excel(rf'{file_path}/指数行业权重.xlsx')


# 数据预处理
if sheet_name == '日K':
    industry_data = industry_data.iloc[1:]
    user_break_points = user_break_points日K

else:
    user_break_points = user_break_points分钟
    # 暂时不做跨日画图
    today_date = pd.Timestamp.now().normalize()  # 获取当前日期并归一化时间到午夜
    industry_data = industry_data[industry_data.index.date == today_date]
industry_data.index = pd.to_datetime(industry_data.index)
weights_df.set_index('行业名称', inplace=True)
index_data = industry_data.pop("上证指数").dropna().astype(float)
industry_data.columns = industry_data.columns.str.replace("\(中信\)", "", regex=True)
index_data.sort_index(inplace=True)
industry_data.sort_index(inplace=True)

# 计算变化率
index_percentage_changes = index_data.pct_change().fillna(0)
industry_percentage_changes = industry_data.pct_change().fillna(0)
# 因为第一天是没有变化率的，因此后面区间统计时不包含start日期(industry_data.index > start) & (industry_data.index <= end)
if sheet_name == '分钟':
    # 将用户定义的截断点转换为时间戳格式，并添加数据的第一个和最后一个时间点
    break_points = [pd.Timestamp(industry_data.index.min())] + \
                   [pd.Timestamp(industry_data.index[0].date().strftime('%Y-%m-%d') + ' ' + t) for t in user_break_points] + \
                   [pd.Timestamp(industry_data.index.max())]
elif sheet_name == '日K':
    # 转换为pandas的时间戳格式
    break_points = ([pd.Timestamp(industry_data.index.min())] + [pd.Timestamp(bp) for bp in user_break_points] +
                    [pd.Timestamp(industry_data.index.max())])
    # 检查所有的断点日期是否都存在于industry_data的索引中
    assert all(date in industry_data.index for date in
               break_points), "All break points must exist in industry_data index."


# 根据截断点划分区间并分析
results = []
for i in range(len(break_points) - 1):
    start, end = break_points[i], break_points[i + 1]
    mask = (industry_data.index > start) & (industry_data.index <= end)
    segment_contributions = industry_percentage_changes.loc[mask].multiply(weights_df['权重'], axis=1).sum()
    top_contributors = segment_contributions.nlargest(3)
    bottom_contributors = segment_contributions.nsmallest(3)

    # 汇总区间结果
    results.append({
        'Interval Start': start,
        'Interval End': end,
        'Top Contributors': top_contributors.index.tolist(),
        'Top Contributions': top_contributors.values.tolist(),
        'Bottom Contributors': bottom_contributors.index.tolist(),
        'Bottom Contributions': bottom_contributors.values.tolist()
    })

# 转换为DataFrame以便于后续操作和可视化
results_df = pd.DataFrame(results)



# 在绘图之前为每个行业分配颜色
color_map = cm.get_cmap('tab20')  # 使用有20个明确区分的颜色的颜色地图
unique_industries = set(industry for _, result in results_df.iterrows() for industry in
                        result['Top Contributors'] + result['Bottom Contributors'])
industry_colors = {industry: color_map(i) for i, industry in enumerate(unique_industries)}

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(9, 6))

# 绘制index_data
index_range = range(len(index_data))
ax.plot(index_range, index_data.values, label='上证指数', color='black', linewidth=2)


break_points_indices = [np.where(index_data.index == bp)[0][0] for bp in break_points]
if sheet_name == '分钟':
    # 添加隔断点的竖线以及午休和跨天的标记
    for bp_idx in break_points_indices:
        ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)
    time_labels = index_data.index.strftime('%b-%d %H:%M')
    day_changes = index_data.index.normalize().drop_duplicates().tolist()
    lunch_starts = [np.where(index_data.index.time == pd.Timestamp('11:30').time())[0] for _ in day_changes]
    for lunch_start in lunch_starts:
        for start in lunch_start:
            ax.axvline(x=start, color='grey', linestyle='--', linewidth=3)

    # 为分钟数据添加时间标签
    ax.set_xticks(break_points_indices)
    ax.set_xticklabels([index_data.index[i].strftime('%H:%M') for i in break_points_indices], rotation=45, ha='right')
elif sheet_name == '日K':
    for bp_idx in break_points_indices:
        ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)

# 首先绘制所有的行业曲线
for i, result in results_df.iterrows():
    start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[
        results_df.index.get_loc(i) + 1]
    index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
    contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']

    for contrib in contributors:
        industry_series = industry_data[contrib].iloc[start_idx:end_idx+1]
        normalized_start_value = index_data.iloc[start_idx]
        normalized_series = (industry_series / industry_series.iloc[0]) * normalized_start_value
        line_color = industry_colors[contrib]
        ax.plot(range(start_idx, end_idx+1), normalized_series, color=line_color)

# 获取最终的y轴范围，并添加文本标签
y_pos_top = ax.get_ylim()[1]
y_pos_bottom = ax.get_ylim()[0]
y_pos_range = y_pos_top - y_pos_bottom

# 再次循环，添加文本标签
for i, result in results_df.iterrows():
    start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[
        results_df.index.get_loc(i) + 1]
    index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
    contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
    text_y_pos = y_pos_top - 0.08 * y_pos_range if index_change >= 0 else y_pos_bottom + 0.08 * y_pos_range
    mid_point = (end_idx + start_idx) // 2
    vertical_step = y_pos_range * 0.05
    offset = (len(contributors) - 1) * vertical_step / 2

    for j, contrib in enumerate(contributors):
        text_y = text_y_pos + offset - j * vertical_step
        if index_change < 0:
            text_y = text_y_pos - offset + j * vertical_step
        ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=8, ha='center', va='center')

# 设置x轴刻度和标签
if sheet_name == '日K':
    ax.set_xticks(range(len(index_data)))
    # ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in index_data.index], rotation=45, ha='right')
    ax.set_xticklabels([date.strftime('%Y-%m-%d') if i % (len(index_data) // 20) == 0 else '' for i, date in
                        enumerate(index_data.index)], rotation=45, ha='right')

# 设置图例
ax.legend()

# 设置标题和标签
ax.set_title('指数和波段行业贡献')
ax.set_xlabel('时间')
ax.set_ylabel('点位')

plt.tight_layout()
plt.show()






# # 创建图形和轴对象
# fig, ax = plt.subplots(figsize=(9, 6))
#
# # 绘制index_data
# index_range = range(len(index_data))
# ax.plot(index_range, index_data.values, label='上证指数', color='black', linewidth=2)
# # # 用于存储转换后的时间点对应的整数索引
# break_points_indices = [np.where(index_data.index == bp)[0][0] for bp in break_points]
# # 生成时间标签映射和特殊时间点（午休开始和结束，以及每天的开始）
# time_labels = index_data.index.strftime('%b-%d %H:%M')
# day_changes = index_data.index.normalize().drop_duplicates().tolist()
# lunch_starts = [np.where(index_data.index.time == pd.Timestamp('11:30').time())[0] for _ in day_changes]
# lunch_ends = [np.where(index_data.index.time == pd.Timestamp('13:00').time())[0] for _ in day_changes[:-1]]
#
# # 添加隔断点的竖线
# for bp_idx in break_points_indices:
#     ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)
# # 标记午休和跨天
# for lunch_start in lunch_starts:
#     for start in lunch_start:
#         ax.axvline(x=start, color='grey', linestyle='--', linewidth=3)
# for day_change in day_changes[1:]:  # 跳过第一天
#     day_start_index = index_data.index.get_loc(day_change, method='nearest')
#     ax.axvline(x=day_start_index, color='blue', linestyle=':', linewidth=3)
#
#
# # 在绘图之前为每个行业分配颜色
# color_map = cm.get_cmap('tab20')  # 'tab10'有10个明确区分的颜色 https://matplotlib.org/stable/users/explain/colors/colormaps.html
# unique_industries = set(industry for _, result in results_df.iterrows() for industry in result['Top Contributors'] + result['Bottom Contributors'])
# industry_colors = {industry: color_map(i) for i, industry in enumerate(unique_industries)}
#
# # 首先绘制所有的行业曲线
# for i, result in results_df.iterrows():
#     start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[results_df.index.get_loc(i) + 1]
#     index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
#
#     # 选择行业
#     contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
#
#     for contrib in contributors:
#         industry_series = industry_data[contrib].iloc[start_idx:end_idx]
#         # 将行业数据点的起点调整至index_data的起始点，并按照跌涨幅比例调整y轴位置
#         normalized_start_value = index_data.iloc[start_idx]
#         normalized_series = (industry_series / industry_series.iloc[0]) * normalized_start_value
#         # 使用预先分配的颜色绘制行业数据
#         line_color = industry_colors[contrib]
#         ax.plot(range(start_idx, end_idx), normalized_series, color=line_color)
#
# # 现在我们可以获取最终的y轴范围，并添加文本标签
# y_pos_top = ax.get_ylim()[1]  # 获取y轴的最大值
# y_pos_bottom = ax.get_ylim()[0]  # 获取y轴的最小值
# y_pos_range = y_pos_top - y_pos_bottom
#
# # 再次循环，这次是添加文本标签
# for i, result in results_df.iterrows():
#     start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[results_df.index.get_loc(i) + 1]
#     index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
#     # 选择行业
#     contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
#     # 确定文本的垂直位置
#     text_y_pos = y_pos_top - 0.08*y_pos_range if index_change >= 0 else y_pos_bottom + 0.08*y_pos_range
#
#     # 计算每个区间的中间点
#     mid_point = (end_idx + start_idx) // 2
#     # 确定行业名称不会重叠的位置
#     vertical_step = (y_pos_top - y_pos_bottom) * 0.05  # 计算垂直步长，这里取图表高度的5%
#     offset = (len(contributors) - 1) * vertical_step / 2  # 计算第一个文本标签的垂直偏移量
#     # 添加文本
#     for j, contrib in enumerate(contributors):
#         line_color = industry_colors[contrib]
#         text_y = text_y_pos + offset - j * vertical_step
#         if index_change < 0:  # 如果是下跌的情况，需要将文本向下移动
#             text_y = text_y_pos - offset + j * vertical_step
#         # 使用预先分配的颜色添加文本标签
#         ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=8, ha='center', va='center')
#
#
# # 设置x轴刻度和标签
# ax.set_xticks(break_points_indices + [i for i in range(0, len(index_data), len(index_data)//10)]) # 保留用户指定的隔断时间并添加额外的x轴标记
# # 首先，确保我们正确地计算了额外的刻度位置和对应的标签
# additional_xticks = [i for i in range(0, len(index_data), len(index_data)//10)]
# additional_xticklabels = [index_data.index[i].strftime('%b-%d %H:%M') for i in additional_xticks]
#
# # 计算用户指定的隔断时间的标签
# user_defined_xticklabels = [index_data.index[i].strftime('%b-%d %H:%M') if i in break_points_indices else '' for i in break_points_indices]
#
# # 合并刻度位置和标签
# all_xticks = break_points_indices + additional_xticks
# all_xticklabels = user_defined_xticklabels + additional_xticklabels
#
# # 应用刻度位置和标签
# ax.set_xticks(all_xticks)
# ax.set_xticklabels(all_xticklabels, rotation=45, ha='right')
#
# # 设置图例
# ax.legend()
#
# # 设置标题和标签
# ax.set_title('指数和波段行业贡献')
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
#
# plt.tight_layout()
# plt.show()










