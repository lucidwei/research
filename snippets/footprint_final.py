import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.transforms as transforms
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# Parameters
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\footprint.xlsx"
sheet_name = '上证'
n_y_ticks = 15  # Number of y-ticks to display
# buy_color = 'red'
# sell_color = 'green'
base_color_buy = (1, 0, 0)  # 红色
base_color_sell = (0, 1, 0)  # 绿色
middle_space = 0.08  # Proportion of the figure width to leave in the middle

# Read and preprocess the data
data = pd.read_excel(file_path, skiprows=7, sheet_name=sheet_name)
data = data.rename(columns={'时间': 'time', '收盘价': 'close', '成交额': 'volume'})[['time', 'close', 'volume']]
data['time'] = pd.to_datetime(data['time'], errors='coerce')
# 使用七天前的日期进行筛选
seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
data = data[data['time'] > seven_days_ago]

data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['volume'] = pd.to_numeric(data['volume'], errors='coerce') / 1e8
data.dropna(subset=['time', 'close'], inplace=True)
latest_close_price = data['close'].iloc[-1]
time_start = data['time'].iloc[0].strftime('%Y-%m-%d %H:%M')
time_end = data['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')
data['price_change'] = data['close'].diff()
data['active_buy'] = data['volume'].where(data['price_change'] > 0, 0)
data['active_sell'] = data['volume'].where(data['price_change'] < 0, 0)
data.drop(columns=['price_change'], inplace=True)
data['month_day'] = data['time'].dt.strftime('%m-%d')
# 为每个不同的日期生成一个颜色映射
unique_days = data['month_day'].unique()
alpha_values = np.linspace(0.2, 1, len(unique_days))
color_map_buy = {day: (*base_color_buy, alpha) for day, alpha in zip(unique_days, alpha_values)}
color_map_sell = {day: (*base_color_sell, alpha) for day, alpha in zip(unique_days, alpha_values)}


# Aggregate data
price_grouped_data = data.groupby(['month_day', 'close']).agg({'active_buy': 'sum', 'active_sell': 'sum'}).reset_index()
# price_grouped_data = data.groupby('close').agg({'active_buy': 'sum', 'active_sell': 'sum'}).reset_index()
price_min, price_max = price_grouped_data['close'].min(), price_grouped_data['close'].max()

bins = np.linspace(price_min, price_max, n_y_ticks * 2 + 1)
price_grouped_data['group'] = pd.cut(price_grouped_data['close'], bins, include_lowest=True)
binned_data = price_grouped_data.groupby(['month_day','group']).agg({'active_buy': 'sum', 'active_sell': 'sum'}).reset_index()
binned_data['close'] = binned_data['group'].apply(lambda x: x.mid)
binned_data.drop('group', axis=1, inplace=True)

price_binned_data = price_grouped_data.groupby(['group']).agg({'active_buy': 'sum', 'active_sell': 'sum'}).reset_index()
price_binned_data['close'] = price_binned_data['group'].apply(lambda x: x.mid)
price_binned_data.drop('group', axis=1, inplace=True)

# Calculate specific parameters
max_net_inflow = price_binned_data.loc[price_binned_data['active_buy'] - price_binned_data['active_sell'] == (price_binned_data['active_buy'] - price_binned_data['active_sell']).max(), 'close'].values[0]
max_net_outflow = price_binned_data.loc[price_binned_data['active_sell'] - price_binned_data['active_buy'] == (price_binned_data['active_sell'] - price_binned_data['active_buy']).max(), 'close'].values[0]
price_binned_data['volume'] = price_binned_data['active_sell'] + price_binned_data['active_buy']
max_volume_price = price_binned_data.loc[price_binned_data['volume'] == price_binned_data['volume'].max(), 'close'].values[0]
max_value = max(price_binned_data['active_buy'].max(), price_binned_data['active_sell'].max())
evenly_spaced_prices = np.linspace(price_min, price_max, n_y_ticks)

# Create the figure with custom axes
fig = plt.figure(facecolor='white', edgecolor='none')
left_width, right_start = (1 - middle_space) / 2, (1 - middle_space) / 2 + middle_space
ax_buy = fig.add_axes([0.05, 0.1, left_width - 0.05, 0.8])
ax_sell = fig.add_axes([right_start, 0.1, left_width - 0.05, 0.8])
ax_buy.set_xlim(0, 1.3*max_value)
ax_sell.set_xlim(0, 1.3*max_value)

# Plotting
cumulative_buy = dict.fromkeys(binned_data['close'].unique(), 0)
cumulative_sell = dict.fromkeys(binned_data['close'].unique(), 0)

# 绘制每个唯一日期的条形图
for day in unique_days[::-1]:
    day_data = binned_data[binned_data['month_day'] == day]
    # 为买入量条形图添加堆叠效果
    ax_buy.barh(day_data['close'], day_data['active_buy'], left=[cumulative_buy[price] for price in day_data['close']], color=color_map_buy[day], label=day)
    # 更新累积买入量
    for price, volume in zip(day_data['close'], day_data['active_buy']):
        cumulative_buy[price] += volume
    # 为卖出量条形图添加堆叠效果
    ax_sell.barh(day_data['close'], day_data['active_sell'], left=[cumulative_sell[price] for price in day_data['close']], color=color_map_sell[day], label=day)
    # 更新累积卖出量
    for price, volume in zip(day_data['close'], day_data['active_sell']):
        cumulative_sell[price] += volume

# ax_buy.barh(binned_data['close'], binned_data['active_buy'], color=buy_color)
# ax_sell.barh(binned_data['close'], binned_data['active_sell'], color=sell_color)
ax_buy.invert_xaxis()
for ax in (ax_buy, ax_sell):
    for loc, spine in ax.spines.items():
        if loc in ['top', 'bottom']:
            spine.set_visible(False)
ax_sell.spines['right'].set_visible(False)
ax_buy.spines['left'].set_visible(False)
ax_sell.yaxis.set_visible(False)
ax_buy.yaxis.set_visible(False)

# Set the y-ticks to be the stock price and align them to the center using a custom transformation
transform = transforms.blended_transform_factory(
    fig.transFigure, ax_buy.transData
)
# Add the evenly spaced price labels in the center of the figure
for price in evenly_spaced_prices:
    fig.text(0.5, price, f'{price:.0f}', transform=transform, ha='center', va='center')

# Annotations
text_offset = max_value * 0.08
for price, label in zip([latest_close_price, max_net_inflow, max_net_outflow, max_volume_price],
                        ['收盘价', '最大净买入', '最大净卖出', '最大成交']):
    # Create annotation with an arrow for buy volume
    if label == '最大净买入':
        volume_buy = binned_data.loc[binned_data['close'] == price, 'active_buy'].sum()
        ax_buy.annotate(label, xy=(volume_buy-text_offset, price), xytext=(volume_buy+text_offset, price),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')

    if label == '最大净卖出':
        volume_sell = binned_data.loc[binned_data['close'] == price, 'active_sell'].sum()
        ax_sell.annotate(label, xy=(volume_sell-text_offset, price), xytext=(volume_sell+3*text_offset, price),
                         arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                         ha='right', va='center')

    if label == '收盘价':
        ax_buy.annotate(label, xy=(0, price), xytext=(text_offset, price),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')

    if label == '最大成交':
        ax_sell.annotate(label, xy=(0, price), xytext=(5*text_offset, price+1),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')

# Title and labels
ax_buy.legend(title='Trade Date', loc='upper left', fancybox=True)
ax_buy.set_title('主动买额(亿)')
ax_sell.set_title('主动卖额(亿)')
fig.suptitle(f'{sheet_name} Volume by price 从{time_start} 至 {time_end}', fontsize=10)

# Layout adjustment and show figure
fig.tight_layout()
plt.show()
