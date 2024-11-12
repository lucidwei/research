# coding=gbk
# Time Created: 2024/11/5 14:38
# Author  : Lucid
# FileName: 相似K线.py
# Software: PyCharm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

class KLineSimilarityFinder:
    def __init__(self, file_path, window_size=5, algorithm='euclidean'):
        self.file_path = file_path
        self.window_size = window_size
        self.algorithm = algorithm
        self.data = self.load_data()
        self.recent_standardized = self.standardize(self.data.tail(window_size))

    def load_data(self):
        data = pd.read_excel(self.file_path, engine='openpyxl')
        data = data[['日期', '开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '成交额(百万)']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data['Pct_Change'] = data['Close'].pct_change()
        return data.dropna()

    def standardize(self, df):
        standardized = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            first_value = df[col].iloc[0]
            standardized[col] = df[col] / first_value
        return standardized

    def get_windows(self):
        windows = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data.iloc[i:i + self.window_size]
            window = self.standardize(window)
            windows.append(window)
        return windows

    def calculate_similarity(self, window, target):
        if self.algorithm == 'euclidean':
            return self.euclidean_distance(window, target)
        elif self.algorithm == 'pearson':
            return self.pearson_correlation(window, target)
        else:
            raise ValueError("Unsupported algorithm. Choose 'euclidean' or 'pearson'.")

    def euclidean_distance(self, window, target):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        window_values = window[features].values.flatten()
        target_values = target[features].values.flatten()
        return np.linalg.norm(window_values - target_values)

    def pearson_correlation(self, window, target):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        correlations = []
        for feature in features:
            correlation, _ = pearsonr(window[feature], target[feature])
            correlations.append(correlation)  # 使用原始相关系数
        # 计算平均相关系数，然后转换为距离度量
        average_correlation = np.mean(correlations)
        return 1 - average_correlation  # 1 - 平均相关系数，这样正相关会得到较小的距离，负相关得到较大的距离

    def find_similar_k_lines(self):
        windows = self.get_windows()
        total_windows = len(windows)
        print(f"开始计算相似度，共 {total_windows} 个窗口。")
        similarities = []
        for i, window in enumerate(windows):
            similarity = self.calculate_similarity(window, self.recent_standardized)
            similarities.append(similarity)
            # 每处理大约20%的窗口时打印一次进度
            if (i + 1) % (total_windows // 5) == 0 or i == total_windows - 1:
                print(f"已处理 {i + 1} / {total_windows} 窗口 ({(i + 1) / total_windows * 100:.2f}%)")
        sorted_indices = np.argsort(similarities)[:5]
        most_similar_windows = [(windows[i], similarities[i]) for i in sorted_indices]
        return most_similar_windows

    def display_similar_windows(self):
        similar_windows = self.find_similar_k_lines()
        for idx, (window, similarity) in enumerate(similar_windows):
            print(f"第{idx}个最相似的窗口起始日期：{window.iloc[0]['Date']}")
            print(f"相似度：{similarity:.3f}")
            # print(window)
            # print()
        print(f"window size: {self.window_size}")

# 使用示例
finder = KLineSimilarityFinder('D:\\Downloads\\000001.SH.xlsx', window_size=5,  algorithm='pearson')
finder.display_similar_windows()