# coding=gbk
# Time Created: 2024/4/9 14:00
# Author  : Lucid
# FileName: 萝卜景气分析.py
# Software: PyCharm
import os
import pandas as pd
from datetime import datetime


def read_data(directory):
    # 获取目录中所有以"景气列表数据_"开头、以".xlsx"结尾的文件名
    files = [f for f in os.listdir(directory) if f.startswith('景气列表数据_') and f.endswith('.xlsx')]

    # 用于存储不同日期的DataFrame的字典
    dfs = {}

    for file in files:
        # 提取文件名中的日期部分
        date = file.split('_')[1].split('.')[0]

        # 读取Excel文件到DataFrame
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path)
        df = df.set_index("行业")

        # 将DataFrame存储到字典中,以日期作为键
        dfs[date] = df

    return dfs


def read_history_data(file_path):
    # 读取参考数据的Excel文件,并将所有sheet存储在一个字典中
    history_data = pd.read_excel(file_path, sheet_name=None, index_col=0)

    # 对每个sheet的数据进行处理
    for sheet_name, sheet_data in history_data.items():
        sheet_data.index = pd.to_datetime(sheet_data.index).strftime('%Y%m%d')
        history_data[sheet_name] = sheet_data

    return history_data


def static_analysis(dfs, date=None):
    if date is None:
        # 如果没有指定日期,使用最新一期的数据
        date = max(dfs.keys())

    df = dfs[date]

    # 计算行业拥挤度和行业趋势强度的90%分位数
    crowdedness_90 = df['行业拥挤度'].quantile(0.7)
    trend_strength_90 = df['行业趋势强度'].quantile(0.7)

    # 根据筛选条件创建布尔掩码
    mask_high_prosperity = df['最新行业景气度'] > 0
    mask_low_valuation_PE = df['当前PE历史百分位'] < 0.5
    mask_low_valuation_PB = df['当前PB历史百分位'] < 0.5
    mask_low_crowdedness = df['行业拥挤度'] <= crowdedness_90
    mask_low_trend_strength = df['行业趋势强度'] <= trend_strength_90

    # 将布尔掩码组合起来,选择满足所有条件的行
    selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness & mask_low_trend_strength]

    # 按照'最新行业景气度'从大到小排序
    selected_rows_sorted = selected_rows.sort_values(by='当前景气度历史百分位', ascending=False)

    return selected_rows_sorted


def dynamic_analysis(dfs, history_data, start_date=None, end_date=None):
    if end_date is None:
        # 如果没有指定结束日期,使用最新一期的数据
        end_date = max(dfs.keys())

    if start_date is None:
        # 如果没有指定起始日期,使用结束日期7天前的日期
        start_date = pd.to_datetime(end_date) - pd.Timedelta(days=7)
        start_date = start_date.strftime('%Y%m%d')

    def get_df(dfs, history_data, date):
        if date in dfs.keys():
            return dfs[date]
        else:
            df = pd.DataFrame(index=history_data['行业景气度'].columns)
            for sheet_name, sheet_data in history_data.items():
                if date in sheet_data.index:
                    df[sheet_name] = sheet_data.loc[date]
            return df

    # 获取起始日期和结束日期对应的df
    df_start = get_df(dfs, history_data, start_date)
    df_end = get_df(dfs, history_data, end_date)

    # 定义列名映射关系, 重命名列名
    column_mapping = {
        '最新行业景气度': '行业景气度变化',
        '行业景气度': '行业景气度变化',
        '行业平均PE(TTM)': '行业PE变化',
        '行业PE': '行业PE变化',
        '行业平均PB(MRQ)': '行业PB变化',
        '行业PB': '行业PB变化',
        '行业趋势强度': '行业趋势强度变化',
        '行业拥挤度': '行业拥挤度变化',
        '当前景气度历史百分位': '景气度历史百分位变化',
        '当前PE历史百分位': 'PE历史百分位变化',
        '当前PB历史百分位': 'PB历史百分位变化',
    }
    df_start = df_start.rename(columns=column_mapping)
    df_end = df_end.rename(columns=column_mapping)

    # 定义完整的numeric_columns列表
    full_numeric_columns = ['行业景气度变化', '景气度历史百分位变化', '行业PE变化', 'PE历史百分位变化',
                            '行业PB变化', 'PB历史百分位变化', '行业趋势强度变化', '行业拥挤度变化']
    # 如果利用了历史数据，那么会少一些列
    # 检查df_start和df_end的列是否与完整的numeric_columns列表相同
    if set(df_start.columns) == set(full_numeric_columns) and set(df_end.columns) == set(full_numeric_columns):
        numeric_columns = full_numeric_columns
    else:
        numeric_columns = ['行业景气度变化', '行业PE变化', '行业PB变化', '行业趋势强度变化', '行业拥挤度变化']

    # 计算指标变化
    change_df = df_end[numeric_columns] - df_start[numeric_columns]
    change_df = change_df.sort_values(by='行业景气度变化', ascending=False)

    return change_df, start_date, end_date


def save_to_excel(selected_industries, change_df, directory, start_date, end_date):
    # 获取当前时间戳,精确到年月日时
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # 生成输出文件名
    output_file = os.path.join(directory, 'output', f'output_{end_date}-{start_date}_{timestamp}.xlsx')

    # 创建一个新的 ExcelWriter 对象
    with pd.ExcelWriter(output_file) as writer:
        # 将 selected_industries 写入名为 "静态分析结果" 的工作表
        selected_industries.to_excel(writer, sheet_name='静态分析结果', index=True)

        # 将 change_df 写入名为 "动态分析结果" 的工作表
        change_df.to_excel(writer, sheet_name='动态分析结果', index=True)

    return output_file


# 设置Excel文件所在的目录
directory = rf"D:\WPS云盘\WPS云盘\工作-麦高\数据库相关\景气\萝卜"

# 读取数据
dfs = read_data(directory)
history_data = read_history_data(rf"{directory}\Industry Prosperity_data.xls")

# 进行静态分析
selected_industries = static_analysis(dfs)
print("高景气、低估值、低拥挤的行业:")
print(selected_industries)

# 进行动态分析
change_df, start_date, end_date = dynamic_analysis(dfs, history_data, start_date='20240517', end_date='20240524')
print("指标变化:")
print(change_df)

# 将结果保存到 Excel 文件
output_file = save_to_excel(selected_industries, change_df, directory, start_date, end_date)
print(f"结果已保存到 {output_file}")
