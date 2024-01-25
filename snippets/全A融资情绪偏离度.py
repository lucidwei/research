
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


# Load the data
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\杂活\暴跌融资变动统计.xlsx"
data = pd.read_excel(file_path, header=2).iloc[8:,]
data.dropna(how='all', axis=0, inplace=True)
data.dropna(how='all', axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

# Renaming and formatting columns
data.columns = ['Date', 'Wande_QuanA', 'Wande_QuanA_Change', 'Financing_Balance', 'Financing_Balance_Increase', 'Financing_Balance_Change']
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Wande_QuanA'] = pd.to_numeric(data['Wande_QuanA'], errors='coerce')
data['Financing_Balance'] = pd.to_numeric(data['Financing_Balance'], errors='coerce')
data.dropna(subset=['Date', 'Wande_QuanA', 'Financing_Balance'], inplace=True)
data.sort_values(by=['Date'], ascending=True, inplace=True)

# Filtering data for dates after 2017
data = data[data['Date'] >= '2017-01-01']

# Calculating 100-day rolling percentiles and emotional deviation
data['Wande_QuanA_100d_Percentile'] = data['Wande_QuanA'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
data['Financing_Balance_100d_Percentile'] = data['Financing_Balance'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
data['Emotional_Deviation'] = data['Wande_QuanA_100d_Percentile'] - data['Financing_Balance_100d_Percentile']

# Plotting
plot_data = data.dropna(subset=['Emotional_Deviation'])
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('万德全A', color='tab:red')
ax1.plot(plot_data['Date'], plot_data['Wande_QuanA'], color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('偏离度', color='tab:blue')
ax2.plot(plot_data['Date'], plot_data['Emotional_Deviation'], color='tab:blue', alpha=0.8)
ax2.plot(plot_data['Date'], plot_data['Wande_QuanA_100d_Percentile'], color='tab:red', alpha=0.3, label='Wande QuanA 100D Percentile')
ax2.plot(plot_data['Date'], plot_data['Financing_Balance_100d_Percentile'], color='tab:green', alpha=0.3, label='Financing Balance 100D Percentile')
ax2.tick_params(axis='y', labelcolor='tab:blue')
plt.title('(全A-融资)情绪偏离度')
plt.show()
