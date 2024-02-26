import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\杂活\万得全A开盘收盘价.xlsx"
# df = pd.read_excel(file_path)
#
# # 数据预处理
# df.columns = ['Date', 'Opening Price', 'Closing Price']
# df = df.drop(index=[0, 1])
# df['Date'] = pd.to_datetime(df['Date'])
# df['Opening Price'] = pd.to_numeric(df['Opening Price'], errors='coerce')
# df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
# df = df.iloc[::-1].reset_index(drop=True)
# df['Increase'] = df['Closing Price'].diff() > 0
# df['Streak'] = df['Increase']*(df['Increase'].groupby((df['Increase'] != df['Increase'].shift()).cumsum()).cumcount() + 1)
# df['Streak End'] = df['Streak'].shift(-1) < df['Streak']
#
# streak_results = []
# for streak_length in range(7, df['Streak'].max() + 1):
#     exact_streaks = df[df['Streak'] == streak_length]
#     exact_streaks['Streak Start Date'] = exact_streaks['Date'] - pd.to_timedelta(streak_length - 1, unit='D')
#     for _, row in exact_streaks.iterrows():
#         streak_results.append({
#             'Streak Length': streak_length,
#             'Start Date': row['Streak Start Date'].strftime('%Y-%m-%d'),
#             'End Date': row['Date'].strftime('%Y-%m-%d')
#         })
#
# streak_results_df = pd.DataFrame(streak_results)
# print(streak_results_df.groupby('Streak Length').apply(lambda x: x.drop('Streak Length', axis=1).to_dict('records')))

df = pd.read_excel(file_path)

# 数据预处理
df.columns = ['Date', 'Opening Price', 'Closing Price']
df = df.drop(index=[0, 1])
df['Date'] = pd.to_datetime(df['Date'])
df['Opening Price'] = pd.to_numeric(df['Opening Price'], errors='coerce')
df['Closing Price'] = pd.to_numeric(df['Closing Price'], errors='coerce')
df = df.iloc[::-1].reset_index(drop=True)
df['Increase'] = df['Closing Price'].diff() > 0
df['Streak ID'] = (df['Increase'].diff() != 0).cumsum()
df['Streak Length'] = df.groupby('Streak ID')['Increase'].transform('sum').fillna(0)

# Identify and process streaks
increasing_streaks = df[df['Increase'] & (df['Streak Length'] >= 7)]
increasing_streaks['Streak Start Date'] = increasing_streaks.groupby('Streak ID')['Date'].transform('min')
increasing_streaks['Streak End Date'] = increasing_streaks.groupby('Streak ID')['Date'].transform('max')
unique_streaks = increasing_streaks.drop_duplicates(subset=['Streak ID'], keep='first')
unique_streaks_sorted = unique_streaks.sort_values(by=['Streak Length', 'Streak Start Date'], ascending=[False, True])

# Filter for final streaks without overlaps
final_streaks = pd.DataFrame()
used_dates = set()
for _, row in unique_streaks_sorted.iterrows():
    streak_dates = pd.date_range(start=row['Streak Start Date'], end=row['Streak End Date'])
    if not set(streak_dates).intersection(used_dates):
        final_streaks = final_streaks.append(row)
        used_dates.update(streak_dates)

# Group and display final streaks
final_streaks_grouped = final_streaks.iloc[:, -3:]
excel_file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\杂活\连阳统计.xlsx'
# final_streaks.iloc[:, -3:].to_excel(excel_file_path, index=False)

print(final_streaks_grouped)


# extended_normalized_streaks = []
# # Calculate the new ranges for each streak
# for _, streak in final_streaks.iterrows():
#     # Calculate the new start and end dates
#     extended_start_date = streak['Streak Start Date'] - pd.Timedelta(days=7)
#     extended_end_date = streak['Streak End Date'] + pd.Timedelta(days=15)
#
#     # Filter the data for the extended range
#     extended_streak_data = df[(df['Date'] >= extended_start_date) & (df['Date'] <= extended_end_date)]
#
#     # Normalize the closing prices using the price on the original start date (t+0)
#     normalization_factor = df[df['Date'] == streak['Streak Start Date']]['Closing Price'].values[0]
#     normalized_prices = extended_streak_data['Closing Price'] / normalization_factor
#
#     # Append the normalized prices to the list, resetting the index to start from -7 (7 days before the streak start)
#     extended_normalized_streaks.append(normalized_prices.reset_index(drop=True).rename(lambda x: x - 7))
#
# # Prepare the plot for the extended ranges
# plt.figure(figsize=(15, 10))
# for streak in extended_normalized_streaks:
#     plt.plot(streak.index, streak, alpha=0.7)
#
# # Customizing the plot
# plt.title('Normalized Closing Prices Around Consecutive Rising Streaks')
# plt.xlabel('Days Relative to Streak Start (t+0)')
# plt.ylabel('Normalized Closing Price')
# plt.axvline(x=0, color='red', linestyle='--', label='Streak Start (t+0)')
# plt.legend()
# plt.grid(True)
# plt.show()


# Plot the historical closing prices of 万德全A with arrows indicating the consecutive rising streaks
# Improved approach to dynamically adjust the vertical position of each annotation to avoid overlap


plt.figure(figsize=(20, 12))
plt.plot(df['Date'], df['Closing Price'], label='Closing Price', color='lightgray', alpha=0.5)

# 初始化一些变量用于动态文本定位
last_position = 0
direction = 1  # 初始方向向上
max_price = df['Closing Price'].max()
min_price = df['Closing Price'].min()
price_range = max_price - min_price
spacing_factor = price_range / 100  # 调整因子

# 对于 final_streaks 中的每个连涨区间进行迭代
for index, streak in final_streaks.iterrows():
    end_date = streak['Streak End Date']
    streak_length = streak['Streak Length']
    end_price = df[df['Date'] == end_date]['Closing Price'].values[0]

    # 动态调整注释文本的垂直位置
    if last_position == 0:  # 第一次迭代
        text_y_position = end_price
    else:
        if direction > 0:
            text_y_position = last_position + spacing_factor * 3  # 向上移动
        else:
            text_y_position = last_position - spacing_factor * 3  # 向下移动

    # 如果需要，更改下一个注释的方向
    if text_y_position > max_price or text_y_position < min_price:
        direction *= -1  # 反转方向
        text_y_position = end_price  # 重置位置到当前结束价格

    last_position = text_y_position  # 更新下一次迭代的上一个位置

    # 添加箭头和注释
    plt.annotate(
        f'{int(streak_length)} days, ends {end_date.date()}',
        xy=(end_date, end_price),
        xytext=(end_date, text_y_position),
        arrowprops=dict(facecolor='red', arrowstyle='->', connectionstyle='arc3'),
        horizontalalignment='right',
        verticalalignment='bottom'
    )

plt.title('万德全A 大于7天连涨统计')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



