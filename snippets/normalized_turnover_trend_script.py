
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# Load the data
# file_path = "E:\Downloads\成交金额_万得全A.xlsx"
file_path = "E:\Downloads\万得全A.xlsx"
data_actual = pd.read_excel(file_path, skiprows=8, names=['Date', 'Turnover'])

# Clean the data
data_actual = data_actual.dropna(subset=['Date'])  # Drop rows with NaN in the 'Date' column
data_actual = data_actual[data_actual['Date'] != '数据来源：Wind']  # Remove the row with "数据来源：Wind"
data_actual['Date'] = pd.to_datetime(data_actual['Date'], errors='coerce')  # Convert to datetime format
data_actual = data_actual.dropna(subset=['Date'])  # Drop any rows that could not be converted to datetime
data_actual = data_actual.sort_values('Date', ascending=True)  # Sort by date

# Initialize the plot
plt.figure(figsize=(15, 7))

# Prepare an empty list to store all the normalized turnover series for mean calculation
all_normalized_turnover = []

# Loop through each year, skipping the years 1994 and 2023
for idx, year in enumerate(sorted(data_actual['Date'].dt.year.unique())):
    if year in [1994, 2023]:
        continue

    # Extract data for the specific year
    yearly_data = data_actual[data_actual['Date'].dt.year == year]
    
    # Get 15 trading days before and after October 1st
    before_data = yearly_data[yearly_data['Date'] < f'{year}-10-01'].tail(15)
    after_data = yearly_data[yearly_data['Date'] >= f'{year}-10-01'].head(15)
    
    # Normalize the turnover values based on the last trading day of September
    normalized_turnover_before = before_data['Turnover'] / before_data.iloc[-1]['Turnover']
    normalized_turnover_after = after_data['Turnover'] / before_data.iloc[-1]['Turnover']
    
    # Combine the normalized turnover values
    normalized_turnover = pd.concat([normalized_turnover_before.reset_index(drop=True), normalized_turnover_after.reset_index(drop=True)])
    
    # Store the normalized turnover for mean calculation
    all_normalized_turnover.append(normalized_turnover)
    
    # Plot with increasing transparency for older years
    plt.plot(range(-14, 16), normalized_turnover, color='blue', alpha=(idx + 1) / len(sorted(data_actual['Date'].dt.year.unique())))

# Calculate the mean normalized turnover
mean_normalized_turnover = pd.concat(all_normalized_turnover, axis=1).mean(axis=1)

# Plot the mean normalized turnover
plt.plot(range(-14, 16), mean_normalized_turnover, color='red', linewidth=4, label="Mean")

# Add a vertical line for the last trading day of September
plt.axvline(0, color='red', linestyle='--', label="Last trading day of September")

# Add title and labels
# plt.title("历年国庆节前后全市场成交量趋势(节前节后15天)")
# plt.xlabel("Trading Days (0为9月最后一天)")
# plt.ylabel("标准化后成交量(9月最后一天为标准)")
plt.title("历年国庆节前后全A走势(节前节后15天)")
plt.xlabel("Trading Days (0为9月最后一天)")
plt.ylabel("标准化后全A(9月最后一天为标准)")

# Add legend for the mean and the vertical line
plt.legend(loc="upper left")

# Add grid
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
