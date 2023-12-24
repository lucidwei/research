import pandas as pd
import itertools
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 读取数据
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\杂活\指南针要的宏观指标\剩余流动性\剩余流动性数据.xlsx"
data = pd.read_excel(file_path)

# 数据清洗
# 删除更新时间的行
data = data.drop(index=0)

# 将日期列转换为datetime格式
data['指标名称'] = pd.to_datetime(data['指标名称'])

# 将相关列转换为数值类型
columns_to_convert = ['中国:M2:同比', '中国:社会融资规模存量:同比', 'M2:同比:-社会融资规模存量:同比', '万得全A']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 计算月度数据
# 按月汇总数据并计算平均值
data['万得全A_ori'] = data['万得全A']
data['万得全A'] = data['万得全A'].shift(-12)
monthly_data = data.set_index('指标名称').resample('M').last()

# 删除NaN值
# monthly_data = monthly_data.dropna()
monthly_data.fillna(method='ffill', inplace=True)


# 计算变化方向
monthly_data['M2_change_dir'] = monthly_data['中国:M2:同比'].diff().apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'NaN'))
monthly_data['Social_Financing_change_dir'] = monthly_data['中国:社会融资规模存量:同比'].diff().apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'NaN'))
monthly_data['Remaining_Liquidity_change_dir'] = monthly_data['M2:同比:-社会融资规模存量:同比'].diff().apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'NaN'))


# 计算每个月的观测值数量
monthly_data['Observations_Count'] = monthly_data['M2:同比:-社会融资规模存量:同比'].expanding().apply(lambda x: len(x.dropna()))

# 计算滚动平均
monthly_data['Rolling_Avg_12M'] = monthly_data['M2:同比:-社会融资规模存量:同比'].rolling(window=12, min_periods=12).mean()

monthly_data['Remaining_Liquidity_level'] = monthly_data.apply(lambda row: 'high' if row['M2:同比:-社会融资规模存量:同比'] > row['Rolling_Avg_12M'] else 'low', axis=1)

monthly_data = monthly_data.dropna(subset=['Rolling_Avg_12M'])
# 计算万得全A的月度收益率
# monthly_data['万得全A_return'] = monthly_data['万得全A'].pct_change().shift(-2) * 100
monthly_data['万得全A_return'] = monthly_data['万得全A'].pct_change().shift(-1) * 100

# 生成所有可能的组合
m2_states = ['up', 'down', 'NaN']
social_financing_states = ['up', 'down', 'NaN']
remaining_liquidity_change_states = ['up', 'down', 'NaN']
remaining_liquidity_level_states = ['high', 'low', 'NaN']

all_combinations = list(itertools.product(m2_states, social_financing_states, remaining_liquidity_change_states, remaining_liquidity_level_states))
combinations_df = pd.DataFrame(all_combinations, columns=['M2_change_dir', 'Social_Financing_change_dir', 'Remaining_Liquidity_change_dir', 'Remaining_Liquidity_level'])

# 寻找每个组合的对应月份
def find_months_for_combination(row, data):
    matching_data = data[(data['M2_change_dir'] == row['M2_change_dir']) &
                         (data['Social_Financing_change_dir'] == row['Social_Financing_change_dir']) &
                         (data['Remaining_Liquidity_change_dir'] == row['Remaining_Liquidity_change_dir']) &
                         (data['Remaining_Liquidity_level'] == row['Remaining_Liquidity_level'])]
    return matching_data.index.strftime('%Y-%m').tolist() if not matching_data.empty else 'Not Observed'

combinations_df['Observed_Months'] = combinations_df.apply(lambda row: find_months_for_combination(row, monthly_data), axis=1)

# 计算每个组合的平均收益率和中位数收益率
def calculate_avg_median_returns(combination, data):
    matching_data = data[(data['M2_change_dir'] == combination['M2_change_dir']) &
                         (data['Social_Financing_change_dir'] == combination['Social_Financing_change_dir']) &
                         (data['Remaining_Liquidity_change_dir'] == combination['Remaining_Liquidity_change_dir']) &
                         (data['Remaining_Liquidity_level'] == combination['Remaining_Liquidity_level'])]
    return matching_data['万得全A_return'].mean(), matching_data['万得全A_return'].median()

# 筛选出有数据的组合
observed_combinations = combinations_df[combinations_df['Observed_Months'] != 'Not Observed']
observed_combinations[['Average_Return', 'Median_Return']] = observed_combinations.apply(lambda row: calculate_avg_median_returns(row, monthly_data), axis=1, result_type='expand')

# 显示结果
observed_combinations







# 生成所有可能的组合，只保留 M2 和 剩余流动性维度
m2_states = ['up', 'down']
remaining_liquidity_change_states = ['up', 'down']

all_combinations = list(itertools.product(m2_states, remaining_liquidity_change_states))
combinations_df = pd.DataFrame(all_combinations, columns=['M2_change_dir', 'Remaining_Liquidity_change_dir'])

# 修改寻找每个组合的对应月份的函数，只考虑 M2 和 剩余流动性维度
def find_months_for_combination(row, data):
    matching_data = data[(data['M2_change_dir'] == row['M2_change_dir']) &
                         (data['Remaining_Liquidity_change_dir'] == row['Remaining_Liquidity_change_dir'])]
    return matching_data.index.strftime('%Y-%m').tolist() if not matching_data.empty else 'Not Observed'

combinations_df['Observed_Months'] = combinations_df.apply(lambda row: find_months_for_combination(row, monthly_data), axis=1)

# 修改计算每个组合的平均收益率和中位数收益率的函数，只考虑 M2 和 剩余流动性维度
def calculate_avg_median_returns(combination, data):
    matching_data = data[(data['M2_change_dir'] == combination['M2_change_dir']) &
                         (data['Remaining_Liquidity_change_dir'] == combination['Remaining_Liquidity_change_dir'])]
    return matching_data['万得全A_return'].mean(), matching_data['万得全A_return'].median()

# 应用新的逻辑到 observed_combinations
observed_combinations = combinations_df[combinations_df['Observed_Months'] != 'Not Observed']
observed_combinations[['Average_Return', 'Median_Return']] = observed_combinations.apply(lambda row: calculate_avg_median_returns(row, monthly_data), axis=1, result_type='expand')

# 显示结果
observed_combinations












# 创建图形和双坐标轴
fig, ax1 = plt.subplots(figsize=(15, 7))

# 绘制剩余流动性和剩余流动性移动平均值
ax1.plot(monthly_data.index, monthly_data['M2:同比:-社会融资规模存量:同比'], label='剩余流动性', color='blue')
ax1.plot(monthly_data.index, monthly_data['Rolling_Avg_12M'], label='剩余流动性移动平均值', color='orange')
ax1.set_xlabel('时间')
ax1.set_ylabel('剩余流动性值')
ax1.legend(loc='upper left')

# 创建第二个坐标轴
ax2 = ax1.twinx()
ax2.bar(monthly_data.index, monthly_data['万得全A_return'], label='万德全A涨跌幅', color='green', width=10, alpha=0.6)
ax2.set_ylabel('万德全A涨跌幅 (%)')
ax2.set_ylim([-8, 8])  # 设置涨跌幅的上下限
ax2.legend(loc='upper right')

# 设置图表标题
plt.title('剩余流动性、剩余流动性移动平均值与万德全A涨跌幅')

# 显示图形
plt.show()