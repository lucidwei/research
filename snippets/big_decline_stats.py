# coding=gbk
# Time Created: 2023/8/28 21:09
# Author  : Lucid
# FileName: big_decline_stats.py
# Software: PyCharm
import pandas as pd

# Load the Excel file
data = pd.read_excel(r"D:\WPS_cloud\WPSDrive\13368898\WPS云盘\工作-麦高\杂活\万得全A开盘收盘价.xlsx")
# Rename columns for easier referencing
data.columns = ["Date", "Open", "Close"]

# Remove non-numeric rows
data = data[2:].reset_index(drop=True)

# Convert data types
data["Date"] = pd.to_datetime(data["Date"])
data["Open"] = pd.to_numeric(data["Open"], errors='coerce')
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Calculate the daily percentage change
data["Change"] = ((data["Close"] - data["Open"]) / data["Open"]) * 100

# Filter rows where the change is less than -4%
declined_dates = data[data["Change"] < -4]

# Show the declined dates and their changes
declined_dates[["Date", "Change"]].head()

# Get the next 10 trading days' data for each declined date
next_10_days_list = []

for date in declined_dates["Date"]:
    mask = (data["Date"] > date)
    next_10_days_data = data[mask].head(10)
    next_10_days_list.append(next_10_days_data)

# Concatenate all the next 10 days' data
next_10_days_df = pd.concat(next_10_days_list, axis=0).reset_index(drop=True)

# Calculate 5 days and 10 days return
next_10_days_df["5_Day_Return"] = next_10_days_df.groupby(next_10_days_df.index // 10)["Close"].transform(lambda x: (x.iloc[4] - x.iloc[0]) / x.iloc[0] * 100 if len(x) > 4 else None)
next_10_days_df["10_Day_Return"] = next_10_days_df.groupby(next_10_days_df.index // 10)["Close"].transform(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)

# Drop duplicates and keep only the first row for each declined date
result_df = next_10_days_df.drop_duplicates(subset=["5_Day_Return", "10_Day_Return"], keep='first')[["Date", "5_Day_Return", "10_Day_Return"]]

# Save the result to Excel
output_path = r"D:\WPS_cloud\WPSDrive\13368898\WPS云盘\工作-麦高\杂活\declined_dates_follow_up.xlsx"
result_df.to_excel(output_path, index=False)

output_path
