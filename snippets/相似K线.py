# coding=gbk
# Time Created: 2024/11/5 14:38
# Author  : Lucid
# FileName: ����K��.py
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
        data = data[['����', '���̼�(Ԫ)', '��߼�(Ԫ)', '��ͼ�(Ԫ)', '���̼�(Ԫ)', '�ɽ���(����)']]
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
            correlations.append(correlation)  # ʹ��ԭʼ���ϵ��
        # ����ƽ�����ϵ����Ȼ��ת��Ϊ�������
        average_correlation = np.mean(correlations)
        return 1 - average_correlation  # 1 - ƽ�����ϵ������������ػ�õ���С�ľ��룬����صõ��ϴ�ľ���

    def find_similar_k_lines(self):
        windows = self.get_windows()
        total_windows = len(windows)
        print(f"��ʼ�������ƶȣ��� {total_windows} �����ڡ�")
        similarities = []
        for i, window in enumerate(windows):
            similarity = self.calculate_similarity(window, self.recent_standardized)
            similarities.append(similarity)
            # ÿ�����Լ20%�Ĵ���ʱ��ӡһ�ν���
            if (i + 1) % (total_windows // 5) == 0 or i == total_windows - 1:
                print(f"�Ѵ��� {i + 1} / {total_windows} ���� ({(i + 1) / total_windows * 100:.2f}%)")
        sorted_indices = np.argsort(similarities)[:5]
        most_similar_windows = [(windows[i], similarities[i]) for i in sorted_indices]
        return most_similar_windows

    def display_similar_windows(self):
        similar_windows = self.find_similar_k_lines()
        for idx, (window, similarity) in enumerate(similar_windows):
            print(f"��{idx}�������ƵĴ�����ʼ���ڣ�{window.iloc[0]['Date']}")
            print(f"���ƶȣ�{similarity:.3f}")
            # print(window)
            # print()
        print(f"window size: {self.window_size}")

# ʹ��ʾ��
finder = KLineSimilarityFinder('D:\\Downloads\\000001.SH.xlsx', window_size=5,  algorithm='pearson')
finder.display_similar_windows()