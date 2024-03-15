# coding=gbk
# Time Created: 2024/3/15 9:50
# Author  : Lucid
# FileName: ��ҵ���׶��ֶ�.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

# ���ò���
file_path = rf"D:\WPS����\WPS����\����-���\���ڻ㱨\�ձ�ģ������\python��"
sheet_name = '��K'
# �û�����Ľضϵ㣬���Ը���ʵ��������е���
user_break_points���� = ['11:00', '13:09', '13:40']
# �û�����Ľضϵ㣬�����ն����ݣ���Щ������
user_break_points��K = ['2024-01-23', '2024-01-26', '2024-02-05', '2024-02-23']

# ��ȡ����
industry_data = pd.read_excel(rf'{file_path}/ָ������ҵ����.xlsx', header=3, index_col=0, sheet_name=sheet_name)
weights_df = pd.read_excel(rf'{file_path}/ָ����ҵȨ��.xlsx')


# ����Ԥ����
if sheet_name == '��K':
    industry_data = industry_data.iloc[1:]
    user_break_points = user_break_points��K

else:
    user_break_points = user_break_points����
    # ��ʱ�������ջ�ͼ
    today_date = pd.Timestamp.now().normalize()  # ��ȡ��ǰ���ڲ���һ��ʱ�䵽��ҹ
    industry_data = industry_data[industry_data.index.date == today_date]
industry_data.index = pd.to_datetime(industry_data.index)
weights_df.set_index('��ҵ����', inplace=True)
index_data = industry_data.pop("��ָ֤��").dropna().astype(float)
industry_data.columns = industry_data.columns.str.replace("\(����\)", "", regex=True)
index_data.sort_index(inplace=True)
industry_data.sort_index(inplace=True)

# ����仯��
index_percentage_changes = index_data.pct_change().fillna(0)
industry_percentage_changes = industry_data.pct_change().fillna(0)
# ��Ϊ��һ����û�б仯�ʵģ���˺�������ͳ��ʱ������start����(industry_data.index > start) & (industry_data.index <= end)
if sheet_name == '����':
    # ���û�����Ľضϵ�ת��Ϊʱ�����ʽ����������ݵĵ�һ�������һ��ʱ���
    break_points = [pd.Timestamp(industry_data.index.min())] + \
                   [pd.Timestamp(industry_data.index[0].date().strftime('%Y-%m-%d') + ' ' + t) for t in user_break_points] + \
                   [pd.Timestamp(industry_data.index.max())]
elif sheet_name == '��K':
    # ת��Ϊpandas��ʱ�����ʽ
    break_points = ([pd.Timestamp(industry_data.index.min())] + [pd.Timestamp(bp) for bp in user_break_points] +
                    [pd.Timestamp(industry_data.index.max())])
    # ������еĶϵ������Ƿ񶼴�����industry_data��������
    assert all(date in industry_data.index for date in
               break_points), "All break points must exist in industry_data index."


# ���ݽضϵ㻮�����䲢����
results = []
for i in range(len(break_points) - 1):
    start, end = break_points[i], break_points[i + 1]
    mask = (industry_data.index > start) & (industry_data.index <= end)
    segment_contributions = industry_percentage_changes.loc[mask].multiply(weights_df['Ȩ��'], axis=1).sum()
    top_contributors = segment_contributions.nlargest(3)
    bottom_contributors = segment_contributions.nsmallest(3)

    # ����������
    results.append({
        'Interval Start': start,
        'Interval End': end,
        'Top Contributors': top_contributors.index.tolist(),
        'Top Contributions': top_contributors.values.tolist(),
        'Bottom Contributors': bottom_contributors.index.tolist(),
        'Bottom Contributions': bottom_contributors.values.tolist()
    })

# ת��ΪDataFrame�Ա��ں��������Ϳ��ӻ�
results_df = pd.DataFrame(results)



# �ڻ�ͼ֮ǰΪÿ����ҵ������ɫ
color_map = cm.get_cmap('tab20')  # ʹ����20����ȷ���ֵ���ɫ����ɫ��ͼ
unique_industries = set(industry for _, result in results_df.iterrows() for industry in
                        result['Top Contributors'] + result['Bottom Contributors'])
industry_colors = {industry: color_map(i) for i, industry in enumerate(unique_industries)}

# ����ͼ�κ������
fig, ax = plt.subplots(figsize=(9, 6))

# ����index_data
index_range = range(len(index_data))
ax.plot(index_range, index_data.values, label='��ָ֤��', color='black', linewidth=2)


break_points_indices = [np.where(index_data.index == bp)[0][0] for bp in break_points]
if sheet_name == '����':
    # ��Ӹ��ϵ�������Լ����ݺͿ���ı��
    for bp_idx in break_points_indices:
        ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)
    time_labels = index_data.index.strftime('%b-%d %H:%M')
    day_changes = index_data.index.normalize().drop_duplicates().tolist()
    lunch_starts = [np.where(index_data.index.time == pd.Timestamp('11:30').time())[0] for _ in day_changes]
    for lunch_start in lunch_starts:
        for start in lunch_start:
            ax.axvline(x=start, color='grey', linestyle='--', linewidth=3)

    # Ϊ�����������ʱ���ǩ
    ax.set_xticks(break_points_indices)
    ax.set_xticklabels([index_data.index[i].strftime('%H:%M') for i in break_points_indices], rotation=45, ha='right')
elif sheet_name == '��K':
    for bp_idx in break_points_indices:
        ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)

# ���Ȼ������е���ҵ����
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

# ��ȡ���յ�y�᷶Χ��������ı���ǩ
y_pos_top = ax.get_ylim()[1]
y_pos_bottom = ax.get_ylim()[0]
y_pos_range = y_pos_top - y_pos_bottom

# �ٴ�ѭ��������ı���ǩ
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

# ����x��̶Ⱥͱ�ǩ
if sheet_name == '��K':
    ax.set_xticks(range(len(index_data)))
    # ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in index_data.index], rotation=45, ha='right')
    ax.set_xticklabels([date.strftime('%Y-%m-%d') if i % (len(index_data) // 20) == 0 else '' for i, date in
                        enumerate(index_data.index)], rotation=45, ha='right')

# ����ͼ��
ax.legend()

# ���ñ���ͱ�ǩ
ax.set_title('ָ���Ͳ�����ҵ����')
ax.set_xlabel('ʱ��')
ax.set_ylabel('��λ')

plt.tight_layout()
plt.show()






# # ����ͼ�κ������
# fig, ax = plt.subplots(figsize=(9, 6))
#
# # ����index_data
# index_range = range(len(index_data))
# ax.plot(index_range, index_data.values, label='��ָ֤��', color='black', linewidth=2)
# # # ���ڴ洢ת�����ʱ����Ӧ����������
# break_points_indices = [np.where(index_data.index == bp)[0][0] for bp in break_points]
# # ����ʱ���ǩӳ�������ʱ��㣨���ݿ�ʼ�ͽ������Լ�ÿ��Ŀ�ʼ��
# time_labels = index_data.index.strftime('%b-%d %H:%M')
# day_changes = index_data.index.normalize().drop_duplicates().tolist()
# lunch_starts = [np.where(index_data.index.time == pd.Timestamp('11:30').time())[0] for _ in day_changes]
# lunch_ends = [np.where(index_data.index.time == pd.Timestamp('13:00').time())[0] for _ in day_changes[:-1]]
#
# # ��Ӹ��ϵ������
# for bp_idx in break_points_indices:
#     ax.axvline(x=bp_idx, color='grey', linestyle='--', linewidth=1)
# # ������ݺͿ���
# for lunch_start in lunch_starts:
#     for start in lunch_start:
#         ax.axvline(x=start, color='grey', linestyle='--', linewidth=3)
# for day_change in day_changes[1:]:  # ������һ��
#     day_start_index = index_data.index.get_loc(day_change, method='nearest')
#     ax.axvline(x=day_start_index, color='blue', linestyle=':', linewidth=3)
#
#
# # �ڻ�ͼ֮ǰΪÿ����ҵ������ɫ
# color_map = cm.get_cmap('tab20')  # 'tab10'��10����ȷ���ֵ���ɫ https://matplotlib.org/stable/users/explain/colors/colormaps.html
# unique_industries = set(industry for _, result in results_df.iterrows() for industry in result['Top Contributors'] + result['Bottom Contributors'])
# industry_colors = {industry: color_map(i) for i, industry in enumerate(unique_industries)}
#
# # ���Ȼ������е���ҵ����
# for i, result in results_df.iterrows():
#     start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[results_df.index.get_loc(i) + 1]
#     index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
#
#     # ѡ����ҵ
#     contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
#
#     for contrib in contributors:
#         industry_series = industry_data[contrib].iloc[start_idx:end_idx]
#         # ����ҵ���ݵ����������index_data����ʼ�㣬�����յ��Ƿ���������y��λ��
#         normalized_start_value = index_data.iloc[start_idx]
#         normalized_series = (industry_series / industry_series.iloc[0]) * normalized_start_value
#         # ʹ��Ԥ�ȷ������ɫ������ҵ����
#         line_color = industry_colors[contrib]
#         ax.plot(range(start_idx, end_idx), normalized_series, color=line_color)
#
# # �������ǿ��Ի�ȡ���յ�y�᷶Χ��������ı���ǩ
# y_pos_top = ax.get_ylim()[1]  # ��ȡy������ֵ
# y_pos_bottom = ax.get_ylim()[0]  # ��ȡy�����Сֵ
# y_pos_range = y_pos_top - y_pos_bottom
#
# # �ٴ�ѭ�������������ı���ǩ
# for i, result in results_df.iterrows():
#     start_idx, end_idx = break_points_indices[results_df.index.get_loc(i)], break_points_indices[results_df.index.get_loc(i) + 1]
#     index_change = index_data.iloc[start_idx:end_idx].pct_change().sum()
#     # ѡ����ҵ
#     contributors = result['Bottom Contributors'] if index_change < 0 else result['Top Contributors']
#     # ȷ���ı��Ĵ�ֱλ��
#     text_y_pos = y_pos_top - 0.08*y_pos_range if index_change >= 0 else y_pos_bottom + 0.08*y_pos_range
#
#     # ����ÿ��������м��
#     mid_point = (end_idx + start_idx) // 2
#     # ȷ����ҵ���Ʋ����ص���λ��
#     vertical_step = (y_pos_top - y_pos_bottom) * 0.05  # ���㴹ֱ����������ȡͼ��߶ȵ�5%
#     offset = (len(contributors) - 1) * vertical_step / 2  # �����һ���ı���ǩ�Ĵ�ֱƫ����
#     # ����ı�
#     for j, contrib in enumerate(contributors):
#         line_color = industry_colors[contrib]
#         text_y = text_y_pos + offset - j * vertical_step
#         if index_change < 0:  # ������µ����������Ҫ���ı������ƶ�
#             text_y = text_y_pos - offset + j * vertical_step
#         # ʹ��Ԥ�ȷ������ɫ����ı���ǩ
#         ax.text(mid_point, text_y, contrib, color=industry_colors[contrib], fontsize=8, ha='center', va='center')
#
#
# # ����x��̶Ⱥͱ�ǩ
# ax.set_xticks(break_points_indices + [i for i in range(0, len(index_data), len(index_data)//10)]) # �����û�ָ���ĸ���ʱ�䲢��Ӷ����x����
# # ���ȣ�ȷ��������ȷ�ؼ����˶���Ŀ̶�λ�úͶ�Ӧ�ı�ǩ
# additional_xticks = [i for i in range(0, len(index_data), len(index_data)//10)]
# additional_xticklabels = [index_data.index[i].strftime('%b-%d %H:%M') for i in additional_xticks]
#
# # �����û�ָ���ĸ���ʱ��ı�ǩ
# user_defined_xticklabels = [index_data.index[i].strftime('%b-%d %H:%M') if i in break_points_indices else '' for i in break_points_indices]
#
# # �ϲ��̶�λ�úͱ�ǩ
# all_xticks = break_points_indices + additional_xticks
# all_xticklabels = user_defined_xticklabels + additional_xticklabels
#
# # Ӧ�ÿ̶�λ�úͱ�ǩ
# ax.set_xticks(all_xticks)
# ax.set_xticklabels(all_xticklabels, rotation=45, ha='right')
#
# # ����ͼ��
# ax.legend()
#
# # ���ñ���ͱ�ǩ
# ax.set_title('ָ���Ͳ�����ҵ����')
# ax.set_xlabel('Time')
# ax.set_ylabel('Value')
#
# plt.tight_layout()
# plt.show()










