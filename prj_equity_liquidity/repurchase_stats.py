# coding=gbk
# Time Created: 2023/8/18 16:16
# Author  : Lucid
# FileName: repurchase_stats.py
# Software: PyCharm
import pandas as pd

def process_excel(input_path, output_path):
    # Load the excel file
    data = pd.read_excel(input_path)

    # Set the correct column names
    data.columns = data.iloc[0]
    data = data.drop(0)

    # Extract the necessary columns and convert to appropriate data types
    subset_data = data[["最新公告日期", "已回购金额(元)", " 预计回购金额(万元)"]]
    subset_data["已回购金额(万元)"] = pd.to_numeric(subset_data["已回购金额(元)"], errors='coerce')/1e4
    subset_data["预计回购金额(万元)"] = pd.to_numeric(subset_data[" 预计回购金额(万元)"], errors='coerce')

    # Group by date and calculate the sum
    grouped_data = subset_data.groupby("最新公告日期").sum().reset_index().sort_values(by='最新公告日期', ascending=True)

    # Calculate 20-day moving average for both columns
    grouped_data["已回购金额(万元) 20日均值"] = grouped_data["已回购金额(万元)"].rolling(window=20).mean()
    grouped_data["预计回购金额(万元) 20日均值"] = grouped_data["预计回购金额(万元)"].rolling(window=20).mean()

    # Save the updated data to the output Excel file
    grouped_data.to_excel(output_path, index=False)

if __name__ == "__main__":
    # input_file = input("Please enter the path to the input Excel file: ")
    # output_file = input("Please enter the path where you want to save the processed Excel file: ")
    input_file = "E:\Downloads\股票回购明细.xlsx"
    output_file = "E:\Downloads\股票回购明细result.xlsx"
    process_excel(input_file, output_file)
    print(f"Processed data saved to: {output_file}")
