import pandas as pd

# Load the Excel file
file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\杂活\固收加基金阶段性总结\长期业绩.xlsx'  # Update with your local file path
data = pd.read_excel(file_path)

# Renaming the columns for easier analysis
columns_rename = {
    "证券代码": "Fund Code",
    "证券简称": "Fund Name",
    data.columns[2]: "Return 2023",  # Assuming the first return column is for 2022
    data.columns[3]: "Return 2022",
    data.columns[4]: "Return 2021",
    data.columns[5]: "Return 2020",
    data.columns[6]: "Return 2019",
    data.columns[7]: "Return 2018",
    data.columns[8]: "Return 2017"
}
data_renamed = data.rename(columns=columns_rename)

# Keeping the top 203 rows as requested
data_top_203 = data_renamed.head(203)

# Identify top 5 funds for each year
years = ["Return 2023", "Return 2022", "Return 2021", "Return 2020", "Return 2019", "Return 2018", "Return 2017"]
top_5_funds_per_year = {year: data_top_203.nlargest(5, year)['Fund Code'] for year in years}

# Compute the percentile rank of top 5 funds in all other years
yearly_percentile_ranks_df = {year: pd.DataFrame(columns=years) for year in years}

for year in years:
    for fund in top_5_funds_per_year[year]:
        fund_row = data_top_203[data_top_203['Fund Code'] == fund]
        fund_name = fund_row['Fund Name'].iloc[0]
        index_name = f"{fund}-{fund_name}"

        for other_year in years:
            if year != other_year:
                rank_percentile = data_top_203[other_year].rank(pct=True, method='min')[data_top_203['Fund Code'] == fund].iloc[0]
                yearly_percentile_ranks_df[year].loc[index_name, other_year] = rank_percentile



# Calculate the average percentile rank for the top 5 funds in other years

# Convert the dictionary of DataFrames into a single DataFrame
combined_df = pd.concat(yearly_percentile_ranks_df, axis=0)

# Calculate the average percentile rank for each fund across years
average_ranks_df = combined_df.mean(axis=1).sort_values()

# Creating the summary table with years as both rows and columns
summary_table_yearly = pd.DataFrame(index=years, columns=years)
for year in years:
    for other_year in years:
        if year != other_year:
            summary_table_yearly.at[year, other_year] = combined_df.loc[combined_df.index.get_level_values(0) == year, other_year].mean()
        else:
            summary_table_yearly.at[year, other_year] = None  # No value for the same year

# Saving the summary table to an Excel file
output_file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\杂活\固收加基金阶段性总结\average_percentile_ranks_summary_by_year.xlsx'
# summary_table_yearly.to_excel(output_file_path)

print("Summary table saved to:", output_file_path)


# Calculate cumulative returns
years_last_five = ["Return 2023", "Return 2022", "Return 2021", "Return 2020", "Return 2019",]
data_top_203['Cumulative Return'] = data_top_203[years_last_five].sum(axis=1)

# Step 2: Select top 10 funds based on cumulative returns
top_10_funds = data_top_203.nlargest(10, 'Cumulative Return')

# Step 3: Calculate the percentile rank for these funds in each year
funds_percentile_rank = pd.DataFrame(index=top_10_funds['Fund Name'], columns=years_last_five)
for year in years_last_five:
    for index, row in top_10_funds.iterrows():
        fund_name = row['Fund Name']
        rank_percentile = data_top_203[year].rank(pct=True, method='min')[index]
        funds_percentile_rank.at[fund_name, year] = rank_percentile

output_file_path = rf'D:\WPS云盘\WPS云盘\工作-麦高\杂活\固收加基金阶段性总结\top_funds_percentile_rank.xlsx'
funds_percentile_rank.to_excel(output_file_path)