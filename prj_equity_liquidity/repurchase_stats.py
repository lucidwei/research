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
    subset_data = data[["���¹�������", "�ѻع����(Ԫ)", " Ԥ�ƻع����(��Ԫ)"]]
    subset_data["�ѻع����(��Ԫ)"] = pd.to_numeric(subset_data["�ѻع����(Ԫ)"], errors='coerce')/1e4
    subset_data["Ԥ�ƻع����(��Ԫ)"] = pd.to_numeric(subset_data[" Ԥ�ƻع����(��Ԫ)"], errors='coerce')

    # Group by date and calculate the sum
    grouped_data = subset_data.groupby("���¹�������").sum().reset_index().sort_values(by='���¹�������', ascending=True)

    # Calculate 20-day moving average for both columns
    grouped_data["�ѻع����(��Ԫ) 20�վ�ֵ"] = grouped_data["�ѻع����(��Ԫ)"].rolling(window=20).mean()
    grouped_data["Ԥ�ƻع����(��Ԫ) 20�վ�ֵ"] = grouped_data["Ԥ�ƻع����(��Ԫ)"].rolling(window=20).mean()

    # Save the updated data to the output Excel file
    grouped_data.to_excel(output_path, index=False)

if __name__ == "__main__":
    # input_file = input("Please enter the path to the input Excel file: ")
    # output_file = input("Please enter the path where you want to save the processed Excel file: ")
    input_file = "E:\Downloads\��Ʊ�ع���ϸ.xlsx"
    output_file = "E:\Downloads\��Ʊ�ع���ϸresult.xlsx"
    process_excel(input_file, output_file)
    print(f"Processed data saved to: {output_file}")
