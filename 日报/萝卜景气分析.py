# coding=gbk
# Time Created: 2024/4/9 14:00
# Author  : Lucid
# FileName: 萝卜景气分析.py
# Software: PyCharm
import os
import pandas as pd

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

def static_analysis(dfs, date=None):
    if date is None:
        # 如果没有指定日期,使用最新一期的数据
        date = max(dfs.keys())

    df = dfs[date]

    # 根据筛选条件创建布尔掩码
    mask_high_prosperity = df['最新行业景气度'] > 0
    mask_low_valuation_PE = df['当前PE历史百分位'] < 0.5
    mask_low_valuation_PB = df['当前PB历史百分位'] < 0.5
    mask_low_crowdedness = df['行业拥挤度'] < 0

    # 将布尔掩码组合起来,选择满足所有条件的行
    selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness]

    return selected_rows

def dynamic_analysis(dfs, start_date=None, end_date=None):
    if end_date is None:
        # 如果没有指定结束日期,使用最新一期的数据
        end_date = max(dfs.keys())

    if start_date is None:
        # 如果没有指定起始日期,使用结束日期7天前的日期
        start_date = pd.to_datetime(end_date) - pd.Timedelta(days=7)
        start_date = start_date.strftime('%Y%m%d')

    df_start = dfs[start_date]
    df_end = dfs[end_date]

    # 计算指标变化
    numeric_columns = ['最新行业景气度', '当前景气度历史百分位', '行业平均PE(TTM)', '当前PE历史百分位',
                       '行业平均PB(MRQ)', '当前PB历史百分位', '行业趋势强度', '行业拥挤度']

    # 计算指标变化
    change_df = df_end[numeric_columns] - df_start[numeric_columns]

    return change_df

# 设置Excel文件所在的目录
directory = rf"D:\WPS云盘\WPS云盘\工作-麦高\数据库相关\景气\萝卜"

# 读取数据
dfs = read_data(directory)

# 进行静态分析
selected_industries = static_analysis(dfs)
print("高景气、低估值、低拥挤的行业:")
print(selected_industries)

# 进行动态分析
change_df = dynamic_analysis(dfs, start_date='20240407', end_date='20240409')
print("指标变化:")
print(change_df)






# import os
# import pandas as pd
#
# # 设置Excel文件所在的目录
# directory = rf"D:\WPS云盘\WPS云盘\工作-麦高\数据库相关\景气\萝卜"
#
# # 获取目录中所有以"景气列表数据_"开头、以".xlsx"结尾的文件名
# files = [f for f in os.listdir(directory) if f.startswith('景气列表数据_') and f.endswith('.xlsx')]
#
# # 用于存储不同日期的DataFrame的字典
# dfs = {}
#
# for file in files:
#     # 提取文件名中的日期部分
#     date = file.split('_')[1].split('.')[0]
#
#     # 读取Excel文件到DataFrame
#     file_path = os.path.join(directory, file)
#     df = pd.read_excel(file_path)
#     df = df.set_index("行业")
#
#     # 将DataFrame存储到字典中,以日期作为键
#     dfs[date] = df
#
# # 打印每个日期对应的DataFrame的信息
# for date, df in dfs.items():
#     # 根据筛选条件创建布尔掩码
#     mask_high_prosperity = df['最新行业景气度'] > 0
#     mask_low_valuation_PE = df['当前PE历史百分位'] < 0.5
#     mask_low_valuation_PB = df['当前PB历史百分位'] < 0.5
#     mask_low_crowdedness = df['行业拥挤度'] < 0
#
#     # 将布尔掩码组合起来,选择满足所有条件的行
#     selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness]