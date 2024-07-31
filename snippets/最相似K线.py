# coding=gbk
# Time Created: 2024/7/31 16:42
# Author  : Lucid
# FileName: 最相似K线.py
# Software: PyCharm
import pandas as pd
import numpy as np

# 读取Excel文件
file_path = 'D:\\Downloads\\000852.SH.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')
WINDOW_SIZE = 7

# 提取相关列和重命名列名
data = data[['日期', '开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '成交额(百万)']]
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# 转换日期列为日期格式
data['Date'] = pd.to_datetime(data['Date'])

# 计算涨跌幅
data['Pct_Change'] = data['Close'].pct_change()
data.dropna(inplace=True)

# 计算均值和标准差
means = data.tail(300)[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
std_devs = data.tail(300)[['Open', 'High', 'Low', 'Close', 'Volume']].std()
volatility = std_devs / means
vol_scaler = volatility['Volume'] / volatility['Close']

# 提取最近五天的数据
recent_data = data.tail(WINDOW_SIZE)

# 标准化数据
def standardize(df):
    standardized = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        first_value = df[col].iloc[0]
        standardized[col] = df[col] / first_value
    return standardized
recent_standardized = standardize(recent_data)


# 滑动窗口提取历史五天的数据
def get_windows(df, window_size=WINDOW_SIZE):
    windows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size].copy()
        window = standardize(window)
        windows.append(window)
    return windows
windows = get_windows(data)


# 计算相似度
def calculate_similarity(window, target):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Pct_Change']
    # window_values = window[features].dropna().values.flatten()
    # target_values = target[features].dropna().values.flatten()

    # 对数据进行缩放
    window_scaled = window[features].copy()
    target_scaled = target[features].copy()

    window_scaled['Volume'] = window['Volume'] / vol_scaler
    target_scaled['Volume'] = target['Volume'] / vol_scaler

    # 展平数据
    window_values = window_scaled.dropna().values.flatten()
    target_values = target_scaled.dropna().values.flatten()

    # 如果窗口内的数据不足，返回一个非常大的值
    if len(window_values) != len(target_values):
        return np.inf

    # 计算欧氏距离
    distance = np.linalg.norm(window_values - target_values)
    return distance


# 计算相似度并排序
similarities = [calculate_similarity(window, recent_standardized) for window in windows]
sorted_indices = np.argsort(similarities)[:5]
most_similar_windows = [(windows[i], similarities[i]) for i in sorted_indices]

# 打印最相似的三个窗口及其相似度
for idx, (window, similarity) in enumerate(most_similar_windows):
    print(f"第{idx}个最相似的窗口起始日期：{window.iloc[0]['Date']}")
    # print(f"相似度：{similarity:.3f}")
    # print(window)
    # print()