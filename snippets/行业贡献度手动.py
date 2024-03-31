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
from datetime import datetime
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 设置参数
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\定期汇报\日报模板整理\python用"
sheet_name = '分钟'
draw_both_sides = True
draw_lunch_break = False
latest_n_days = 5  # >0,最近n个交易日; =0,今天; <0, 今天往前推几天
# 用户定义的截断点，可以根据实际情况进行调整
# user_break_points分钟 = ['13:05', '14:35']
user_break_points分钟 = ['25-11:18', '26-13:36', '26-14:41', '28-09:44', '28-13:03', '28-14:34']
# 用户定义的截断点，对于日度数据，这些是日期
user_break_points日K = ['2024-01-23', '2024-01-26', '2024-02-05', '2024-02-23',  '2024-03-18']

# 读取数据
industry_data = pd.read_excel(rf'{file_path}/指数、行业走势.xlsx', header=3, index_col=0, sheet_name=sheet_name)
weights_df = pd.read_excel(rf'{file_path}/指数行业权重.xlsx')


# 数据预处理
if sheet_name == '日K':
    industry_data = industry_data.iloc[1:]
    user_break_points = user_break_points日K
    industry_data.index = pd.to_datetime(industry_data.index)
else:
    user_break_points = user_break_points分钟
    industry_data.index = pd.to_datetime(industry_data.index)
    today_date = pd.Timestamp.now().normalize()  # 获取当前日期并归一化时间到午夜
    if latest_n_days > 0:
        # 选取最近N个交易日的数据
        unique_dates = industry_data.index.normalize().unique()
        target_dates = unique_dates[-latest_n_days:]  # 获取最后N个唯一日期
        target_date = today_date
        industry_data = industry_data[industry_data.index.normalize().isin(target_dates)]
    else:
        if latest_n_days == 0:
            target_date = today_date  # 使用今天的数据
        else:
            target_date = today_date - pd.Timedelta(days=-latest_n_days)  # 使用昨天的单日日内数据
        assert target_date.date() in industry_data.index.date, f"数据未更新，目标日期 {target_date.date()} 不在industry_data中"
        industry_data = industry_data[industry_data.index.date == target_date]

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
    break_points = []
    for t in user_break_points分钟:
        if '-' in t:  # 检测是否指定了日期
            day, time = t.split('-')
            break_point = pd.Timestamp(datetime(target_date.year, target_date.month, int(day), int(time.split(':')[0]), int(time.split(':')[1])))
        else:  # 没有指定日期，使用当天日期
            break_point = pd.Timestamp(datetime(target_date.year, target_date.month, target_date.day, int(t.split(':')[0]), int(t.split(':')[1])))
        break_points.append(break_point)
    # 添加数据的第一个和最后一个时间点
    break_points = [industry_data.index.min()] + break_points + [industry_data.index.max()]

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


def draw_chart(draw_both_sides):
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

    # 绘制隔断点（和午休）分割竖线
    break_points_indices = [np.where(index_data.index == bp)[0][0] for bp in break_points]
    if sheet_name == '分钟':
        # 添加隔断点的竖线以及午休和跨天的标记
        for bp_idx in break_points_indices:
            ax.axvline(x=bp_idx, color='grey', linestyle='-', linewidth=1)
        day_changes = index_data.index.normalize().drop_duplicates().tolist()
        if draw_lunch_break:
            lunch_starts = [np.where(index_data.index.time == pd.Timestamp('11:30').time())[0] for _ in day_changes]
            for lunch_start in lunch_starts:
                for start in lunch_start:
                    ax.axvline(x=start, color='grey', linestyle='--', linewidth=2)
        # 为分钟数据添加时间标签
        ax.set_xticks(break_points_indices)
        ax.set_xticklabels([index_data.index[i].strftime('%d-%H:%M') for i in break_points_indices], rotation=45, ha='right')
    elif sheet_name == '日K':
        for bp_idx in break_points_indices:
            ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)

    # 首先绘制所有的行业曲线
    for i, result in results_df.iterrows():
        start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[
            results_df.index.get_loc(i) + 1]
        index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()

        # 根据draw_both_sides变量绘制单边或双边
        if draw_both_sides:
            # 如果是下跌区间，仅画出Top Contributors中贡献为正值的行业曲线
            if index_change < 0:
                positive_contributors = [contrib for contrib, contrib_value in
                                         zip(result['Top Contributors'], result['Top Contributions']) if
                                         contrib_value > 0]
                contributors = positive_contributors + result['Bottom Contributors']
            else:
                negative_contributors = [contrib for contrib, contrib_value in
                                         zip(result['Bottom Contributors'], result['Bottom Contributions']) if
                                         contrib_value < 0]
                contributors = negative_contributors + result['Top Contributors']

        else:
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
        mid_point = (end_idx + start_idx) // 2
        vertical_step = y_pos_range * 0.05

        # 根据draw_both_sides变量绘制单边或双边
        if draw_both_sides:
            # 如果是下跌区间，仅为Top Contributors中贡献为正值的行业添加文本标注
            if index_change < 0:
                positive_contributors = [contrib for contrib, contrib_value in
                                         zip(result['Top Contributors'], result['Top Contributions']) if
                                         contrib_value > 0]
                top_contributors = positive_contributors
                bottom_contributors = result['Bottom Contributors']
            else:
                negative_contributors = [contrib for contrib, contrib_value in
                                         zip(result['Bottom Contributors'], result['Bottom Contributions']) if
                                         contrib_value < 0]
                top_contributors = result['Top Contributors']
                bottom_contributors = negative_contributors

            # 为每个区间设置顶部和底部标签
            # 设置顶部标签（上涨行业）
            for j, contrib in enumerate(top_contributors):
                text_y = y_pos_top - 0.08 * y_pos_range - vertical_step + j * vertical_step
                ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=11, ha='center',
                        va='center')
            # 设置底部标签（下跌行业）
            for j, contrib in enumerate(bottom_contributors):
                text_y = y_pos_bottom + 0.08 * y_pos_range + vertical_step - j * vertical_step
                ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=11, ha='center',
                        va='center')

        else:
            contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
            text_y_pos = y_pos_top - 0.08 * y_pos_range if index_change >= 0 else y_pos_bottom + 0.08 * y_pos_range
            offset = (len(contributors) - 1) * vertical_step / 2

            for j, contrib in enumerate(contributors):
                text_y = text_y_pos + offset - j * vertical_step
                if index_change < 0:
                    text_y = text_y_pos - offset + j * vertical_step
                ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=11, ha='center', va='center')

    # 设置x轴刻度和标签
    if sheet_name == '日K':
        ax.set_xticks(range(len(index_data)))
        ax.set_xticklabels([date.strftime('%Y-%m-%d') if i % (len(index_data) // 20) == 0 else '' for i, date in
                            enumerate(index_data.index)], rotation=45, ha='right')
    # 绘制固定的x轴刻度
    # if sheet_name == '分钟':
    #     # 确保x轴刻度的数量合适，这里假设我们也想在x轴上大约显示20个标签
    #     tick_spacing = max(len(index_data) // 8, 1)  # 避免除以0
    #     ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # 用MaxNLocator来确保不超过20个刻度
    #     tick_labels = [index_data.index[i].strftime('%H:%M') if i % tick_spacing == 0 else '' for i in
    #                    range(len(index_data))]
    #     ax.set_xticks(range(len(index_data))[::tick_spacing])  # 设置刻度
    #     ax.set_xticklabels(tick_labels[::tick_spacing], rotation=45, ha='right')  # 设置标签

    # 设置图例
    ax.legend()

    # 设置标题和标签
    ax.set_title('指数和波段行业贡献')
    # ax.set_xlabel('时间')
    ax.set_ylabel('点位')

    plt.tight_layout()
    plt.show()


draw_chart(draw_both_sides=True)









