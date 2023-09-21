# coding=gbk
# Time Created: 2023/9/8 15:34
# Author  : Lucid
# FileName: big_decline_plot.py
# Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the provided Excel file
data = pd.read_excel("E:\Downloads\北向_当日资金净流入.xlsx", skiprows=7)

# Rename the columns for easier reference
data.columns = ['Date', 'Net_Inflow', 'Net_Inflow_MA20']
# Remove rows where 'Date' column is not a valid date
data = data[data['Date'] != '数据来源：Wind']
# Convert the columns to appropriate data types
data['Date'] = pd.to_datetime(data['Date'])
data['Net_Inflow'] = pd.to_numeric(data['Net_Inflow'])
data['Net_Inflow_MA20'] = pd.to_numeric(data['Net_Inflow_MA20'])

# Identify local maxima for the MA20 column
data['Is_Max'] = (data['Net_Inflow_MA20'] > data['Net_Inflow_MA20'].shift(1)) & (
            data['Net_Inflow_MA20'] > data['Net_Inflow_MA20'].shift(-1))

# # Extract the segments of data where there's a local maximum and the difference in subsequent days is more than 50, without overlapping
# segments = []
# processed_dates = set()
#
# for index, row in data[data['Is_Max']].iterrows():
#     if row['Date'] in processed_dates:
#         continue
#     subsequent_data = data.loc[index:].copy()
#     subsequent_data['Cumulative_Drop'] = subsequent_data['Net_Inflow_MA20'] - row['Net_Inflow_MA20']
#     end_date_row = subsequent_data[subsequent_data['Cumulative_Drop'] <= -50].head(1)
#     if not end_date_row.empty:
#         start_date = row['Date']
#         end_date = end_date_row['Date'].values[0]
#         processed_dates.update(pd.date_range(start=start_date, end=end_date).tolist())
#         segments.append((start_date, end_date))

# Extract the segments of data where there's a local maximum and the difference in subsequent days is more than 50, without overlapping
corrected_segments = []
last_end_index = -1

for index, row in data[data['Is_Max']].iterrows():
    # Skip if the date is within a previously found segment
    if index <= last_end_index:
        continue

    subsequent_data = data.loc[index:].copy()
    subsequent_data['Cumulative_Drop'] = subsequent_data['Net_Inflow_MA20'] - row['Net_Inflow_MA20']
    end_date_row = subsequent_data[subsequent_data['Cumulative_Drop'] <= -50].head(1)

    if not end_date_row.empty:
        start_date = row['Date']
        end_date = end_date_row['Date'].values[0]
        last_end_index = end_date_row.index[0]
        corrected_segments.append((start_date, end_date))

# Filter data for the date range 2016 to the latest
filtered_data = data[data['Date'] >= '2016-01-01']

# Plot the trended outflow segments using the accurate date range and without overlap
plt.figure(figsize=(15, 10))

for end_date, start_date in corrected_segments:
    segment_data = filtered_data[(filtered_data['Date'] >= start_date) & (filtered_data['Date'] <= end_date)].copy()
    segment_data['Cumulative'] = segment_data['Net_Inflow'].cumsum()

    # Convert numpy's datetime64 objects to Python's datetime objects
    start_date_py = pd.Timestamp(start_date).to_pydatetime()
    end_date_py = pd.Timestamp(end_date).to_pydatetime()

    plt.plot(segment_data['Date'], segment_data['Cumulative'],
             label=f'Start: {start_date_py.strftime("%Y-%m-%d")}, End: {end_date_py.strftime("%Y-%m-%d")}')

plt.gca().invert_yaxis()  # Invert the y-axis for a downward trend
plt.xlabel('Date')
plt.ylabel('Cumulative Outflow (Billions)')
plt.title('Trended Outflows of Northbound Capital Based on MA20')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
