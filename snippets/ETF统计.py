import pandas as pd

# 加载Excel文件
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\增量资金历史梳理\ETF基金市场规模变化(至2023末).xlsx"  # 替换为您的文件路径
sheet_all_a = pd.read_excel(file_path, '全A与流入汇总')

# 提取日期、万得全A指数和资金流入MA30列
date_column = sheet_all_a.columns[0]
wande_all_a_column = sheet_all_a.columns[1]
fund_inflow_ma30_column = sheet_all_a.columns[2]

# 转换日期列和数值列的数据类型
sheet_all_a[date_column] = pd.to_datetime(sheet_all_a[date_column])
sheet_all_a[wande_all_a_column] = pd.to_numeric(sheet_all_a[wande_all_a_column], errors='coerce')
sheet_all_a[fund_inflow_ma30_column] = pd.to_numeric(sheet_all_a[fund_inflow_ma30_column], errors='coerce')


# 寻找显著的ETF流入时间段
def find_significant_periods(sheet, threshold, start_year, end_year):
    periods = []
    start_date = None
    for index, row in sheet.iterrows():
        if row[date_column].year >= start_year and row[date_column].year < end_year:
            if row[fund_inflow_ma30_column] > threshold:
                if start_date is None:
                    start_date = row[date_column]
            else:
                if start_date is not None:
                    periods.append((start_date, row[date_column] - pd.Timedelta(days=1)))
                    start_date = None
        else:
            if start_date is not None:
                periods.append((start_date, row[date_column] - pd.Timedelta(days=1)))
                start_date = None
    if start_date is not None:
        periods.append((start_date, sheet[date_column].iloc[-1]))
    return periods


significant_periods_pre_2019 = find_significant_periods(sheet_all_a, 10, 2005, 2019)
significant_periods_post_2019 = find_significant_periods(sheet_all_a, 20, 2019, 2024)
significant_periods = significant_periods_pre_2019 + significant_periods_post_2019


# 计算百分比变化和平均资金流入
def calculate_metrics(start_date, end_date):
    period_data = sheet_all_a[(sheet_all_a[date_column] >= start_date) & (sheet_all_a[date_column] <= end_date)]
    if period_data.empty:
        return 'NaN', 'NaN'
    avg_fund_inflow = period_data[fund_inflow_ma30_column].mean()
    start_value = period_data[wande_all_a_column].iloc[-1]
    end_value = period_data[wande_all_a_column].iloc[0]
    percentage_change = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 'NaN'
    return avg_fund_inflow, percentage_change


# 重新计算结果
results = []
for end_date, start_date in significant_periods:
    avg_inflow, change_during_period = calculate_metrics(start_date, end_date)
    _, change_after_period = calculate_metrics(end_date, end_date + pd.DateOffset(months=1))

    results.append({
        "起始日期": start_date,
        "截止日期": end_date,
        "MA30平均净流入": avg_inflow,
        "区间内涨跌幅": change_during_period,
        "区间后一月涨跌幅": change_after_period
    })

results_df = pd.DataFrame(results).sort_values(by='起始日期')
output_file_path = 'D:\WPS云盘\WPS云盘\工作-麦高\专题研究\增量资金历史梳理\ETF统计结果1.xlsx'
results_df.to_excel(output_file_path, index=False)

