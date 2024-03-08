import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.transforms as transforms
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# Parameters
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\footprint.xlsx"

# Read the data from the Excel file, skipping rows that do not contain the relevant data
data = pd.read_excel(file_path, skiprows=7)

# Rename the columns for clarity
data.rename(columns={'时间': 'time', '收盘价': 'close', '成交额': 'volume'}, inplace=True)

# Keep only the necessary columns
data = data[['time', 'close', 'volume']]

# Convert 'time' to datetime and 'close' to float
data['time'] = pd.to_datetime(data['time'], errors='coerce')
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['volume'] = pd.to_numeric(data['volume'], errors='coerce') / 1e8

# Drop rows where 'time' or 'close' could not be converted
data.dropna(subset=['time', 'close'], inplace=True)
latest_close_price = data['close'].iloc[-1]

# Calculate the difference in 'close' price compared to the previous minute
data['price_change'] = data['close'].diff()

# Categorize volume as 'active_buy' or 'active_sell' based on the price change
data['active_buy'] = data['volume'].where(data['price_change'] > 0, 0)
data['active_sell'] = data['volume'].where(data['price_change'] < 0, 0)

# Drop the 'price_change' column as it is no longer needed
data.drop(columns=['price_change'], inplace=True)

# Group by the 'close' price to aggregate the volumes for each price level
grouped_data = data.groupby('close').agg({
    'active_buy': 'sum',
    'active_sell': 'sum'
}).reset_index()


# Set the number of groups based on n_y_ticks
n_y_ticks = 10
num_groups = n_y_ticks * 2  # Twice the number of desired y-ticks

# Define the price range for the groups
price_min = grouped_data['close'].min()
price_max = grouped_data['close'].max()

# Create the bins for grouping
bins = np.linspace(price_min, price_max, num_groups + 1)

# Group the data by these bins and sum the volumes
grouped_data['group'] = pd.cut(grouped_data['close'], bins, include_lowest=True)
aggregated_data = grouped_data.groupby('group').agg({
    'active_buy': 'sum',
    'active_sell': 'sum'
}).reset_index()

# Calculate the mid-point price for each bin to use as the representative price for the group
aggregated_data['close'] = aggregated_data['group'].apply(lambda x: x.mid)

# Drop the 'group' column as we no longer need it after obtaining 'mid_price'
aggregated_data.drop('group', axis=1, inplace=True)

grouped_data_copy = grouped_data.copy(deep=True)

grouped_data = aggregated_data

# 参数计算
max_net_inflow = grouped_data.loc[grouped_data['active_buy'] - grouped_data['active_sell'] == (grouped_data['active_buy'] - grouped_data['active_sell']).max(), 'close'].values[0]
max_net_outflow = grouped_data.loc[grouped_data['active_sell'] - grouped_data['active_buy'] == (grouped_data['active_sell'] - grouped_data['active_buy']).max(), 'close'].values[0]
grouped_data['volume'] = grouped_data['active_sell'] + grouped_data['active_buy']
max_volume_price = grouped_data.loc[grouped_data['volume'] == grouped_data['volume'].max(), 'close'].values[0]

max_buy = grouped_data['active_buy'].max()
max_sell = grouped_data['active_sell'].max()
# 取两个最大值中的较大者
max_value = max(max_buy, max_sell)

# Determine the range of the 'close' prices
price_min = grouped_data['close'].min()
price_max = grouped_data['close'].max()


# 参数设置
buy_color = 'red'
sell_color = 'green'

# Generate evenly spaced y-ticks within the range of 'close' prices
n_y_ticks = 10
evenly_spaced_prices = np.linspace(price_min, price_max, n_y_ticks)

# Define the space you want to leave in the middle as a proportion of the figure width
middle_space = 0.1  # This can be adjusted as needed

# Calculate positions for the left and right axes based on the middle_space
left_width = (1 - middle_space) / 2
right_start = left_width + middle_space



# Create the figure and the left and right axes
fig = plt.figure(facecolor='white', edgecolor='none')
ax_buy = fig.add_axes([0.05, 0.1, left_width - 0.05, 0.8])
ax_sell = fig.add_axes([right_start, 0.1, left_width - 0.05, 0.8])
ax_buy.set_xlim(0, max_value)
ax_sell.set_xlim(0, max_value)


# Plot the active buys on the left axis (ax_buy) and the active sells on the right axis (ax_sell)
ax_buy.barh(grouped_data['close'], grouped_data['active_buy'], color=buy_color, height=0.7)
ax_sell.barh(grouped_data['close'], grouped_data['active_sell'], color=sell_color, height=0.7)

# Invert x-axis for the buys to have the bars grow towards center
ax_buy.invert_xaxis()

# Turn off the axes spines except on the y-axis inside
for ax in (ax_buy, ax_sell):
    for loc, spine in ax.spines.items():
        if loc in ['top', 'bottom']:
            spine.set_visible(False)

# Remove x-ticks and labels for a cleaner look
# ax_sell.set_yticks([])  # Add this line to remove the y-axis ticks on the right axis
ax_sell.yaxis.set_visible(False)  # Add this line to hide the entire y-axis on the right axis
# ax_buy.set_yticks([])  # Add this line to remove the y-axis ticks on the right axis
ax_buy.yaxis.set_visible(False)  # Add this line to hide the entire y-axis on the right axis
# Hide the right spine of the buy axis
ax_sell.spines['right'].set_visible(False)
# Hide the left spine of the sell axis
ax_buy.spines['left'].set_visible(False)

# Set the y-ticks to be the stock price and align them to the center using a custom transformation
transform = transforms.blended_transform_factory(
    fig.transFigure, ax_buy.transData
)

# Add the evenly spaced price labels in the center of the figure
for price in evenly_spaced_prices:
    fig.text(0.5, price, f'{price:.2f}', transform=transform, ha='center', va='center')


# # Highlight specific price points with annotations in the center
# for price, label in zip([latest_close_price, max_net_inflow, max_net_outflow, max_volume_price],
#                         ['Latest Close', 'Max Net Inflow', 'Max Net Outflow', 'Max Volume']):
#     fig.text(0.5, price, f'{label}\n{price:.2f}', transform=transform, ha='center', va='center', color='black')

# Define an offset for the annotation text relative to the bar ends
text_offset = max_value * 0.08  # 2% of the max value for some padding
for price, label in zip([latest_close_price, max_net_inflow, max_net_outflow, max_volume_price],
                        ['Latest Close', 'Max Net Inflow', 'Max Net Outflow', 'Max Volume']):
    # Create annotation with an arrow for buy volume
    if label == 'Max Net Inflow':
        volume_buy = grouped_data.loc[grouped_data['close'] == price, 'active_buy'].values[0]
        ax_buy.annotate(label, xy=(volume_buy-text_offset, price), xytext=(volume_buy+text_offset, price),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')

    if label == 'Max Net Outflow':
        volume_sell = grouped_data.loc[grouped_data['close'] == price, 'active_sell'].values[0]
        ax_sell.annotate(label, xy=(volume_sell-text_offset, price), xytext=(volume_sell+text_offset, price),
                         arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                         ha='right', va='center')

    if label == 'Latest Close':
        ax_buy.annotate(label, xy=(0, price), xytext=(text_offset, price),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')

    if label == 'Max Volume':
        ax_sell.annotate(label, xy=(0, price), xytext=(5*text_offset, price),
                        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),
                        ha='right', va='center')


# Set title and labels
ax_buy.set_title('Active Buy Volume')
ax_sell.set_title('Active Sell Volume')

# Adjust layout
fig.tight_layout()

# Display the figure
plt.show()


# # Plot the 'active_buy' volumes as a bar chart
# ax.barh(grouped_data['close'], grouped_data['active_buy'], color=buy_color, label='Active Buy')
#
# # Plot the 'active_sell' volumes as a bar chart
# ax.barh(grouped_data['close'], -grouped_data['active_sell'], color=sell_color, label='Active Sell')
#
# # Add labels and a title to the plot
# ax.set_xlabel('成交额')
# ax.set_ylabel('Price')
# ax.set_title('Volume by Price')
# ax.annotate('最新收盘价', xy=(0, latest_close_price), xytext=(10, 0), textcoords='offset points')
# ax.annotate('最大净流入', xy=(grouped_data['active_buy'].max(), max_net_inflow), xytext=(10, 0), textcoords='offset points')
# ax.annotate('最大净流出', xy=(-grouped_data['active_sell'].max(), max_net_outflow), xytext=(10, 0), textcoords='offset points')
# ax.annotate('最大成交', xy=(0, max_volume_price), xytext=(20, 0), textcoords='offset points')


# # Plot the 'active_buy' volumes as a bar chart
# ax.barh(grouped_data['close'], grouped_data['active_buy'], color=buy_color, label='Active Buy', zorder=3)
#
# # Plot the 'active_sell' volumes as a bar chart
# ax.barh(grouped_data['close'], -grouped_data['active_sell'], color=sell_color, label='Active Sell', zorder=3)
#
# # Add a vertical line for the latest closing price
# ax.axhline(y=latest_close_price, color='blue', linestyle='--', label='Latest Close Price', zorder=2)
#
# # Text annotations for maximums and latest close
# ax.text(grouped_data['active_buy'].max()/2, latest_close_price, '最新收盘价', verticalalignment='bottom', horizontalalignment='center', color='blue', fontsize=8, zorder=5)
# ax.text(grouped_data['active_buy'].max(), max_net_inflow, '最大净流入', verticalalignment='center', horizontalalignment='right', color='green', fontsize=8, zorder=5)
# ax.text(-grouped_data['active_sell'].max(), max_net_outflow, '最大净流出', verticalalignment='center', horizontalalignment='left', color='red', fontsize=8, zorder=5)
# ax.text(grouped_data['active_buy'].max()/2, max_volume_price, '最大成交', verticalalignment='top', horizontalalignment='center', color='black', fontsize=8, zorder=5)
#
#
# # Add a legend
# ax.legend()
#
# # Show the plot with a tight layout
# plt.tight_layout()
# plt.show()
