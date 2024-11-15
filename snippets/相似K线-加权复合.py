# coding=gbk
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import mplfinance as mpf
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

class KLineSimilarityFinder:
    def __init__(self, file_path, window_sizes=[5, 20, 200], weights=[0.2, 0.4, 0.4],
                 top_n=5, algorithm='euclidean'):
        """
        ��ʼ�������Բ�������

        :param file_path: �����ļ�·����
        :param window_sizes: Ҫʹ�õĴ��ڴ�С�б�
        :param weights: ÿ�����ڴ�С��Ӧ��Ȩ���б�
        :param top_n: ����ѡ���ǰ���������ƵĴ���������
        :param algorithm: �����Լ����㷨��֧�� 'euclidean' �� 'pearson'��
        """
        if len(window_sizes) != len(weights):
            raise ValueError("window_sizes �� weights ���������ͬ�ĳ��ȡ�")

        self.file_path = file_path
        self.window_sizes = window_sizes
        self.weights = weights
        self.top_n = top_n
        self.algorithm = algorithm
        self.data = self.load_data()
        self.target_windows_standardized = self.prepare_target_windows()

    def load_data(self):
        data = pd.read_excel(self.file_path, engine='openpyxl')
        data = data[['����', '���̼�(Ԫ)', '��߼�(Ԫ)', '��ͼ�(Ԫ)', '���̼�(Ԫ)', '�ɽ���(����)']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)  # ȷ������������
        data['Pct_Change'] = data['Close'].pct_change()
        return data.dropna().reset_index(drop=True)

    def standardize(self, df):
        standardized = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            first_value = df[col].iloc[0]
            standardized[col] = df[col] / first_value if first_value != 0 else df[col]
        return standardized

    def prepare_target_windows(self):
        """
        ׼�����µĸ������ڴ�С�ı�׼��Ŀ�괰�ڡ�
        """
        target_windows = {}
        for window_size in self.window_sizes:
            if len(self.data) < window_size:
                raise ValueError(f"���ݳ��Ȳ�����ʹ�ô��ڴ�С {window_size}")
            target_window = self.standardize(self.data.iloc[-window_size:]).reset_index(drop=True)
            target_windows[window_size] = target_window
        return target_windows

    def calculate_similarity(self, window, target):
        """
        ����ָ�����㷨�����������ڵ������ԡ�

        :param window: ��ʷ���ڡ�
        :param target: Ŀ�괰�ڡ�
        :return: �����Է�������ֵԽСԽ���ƣ���
        """
        if self.algorithm == 'euclidean':
            return self.euclidean_distance(window, target)
        elif self.algorithm == 'pearson':
            return self.pearson_correlation_distance(window, target)
        else:
            raise ValueError("Unsupported algorithm. Choose 'euclidean' or 'pearson'.")

    def euclidean_distance(self, window, target):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        window_values = window[features].values.flatten()
        target_values = target[features].values.flatten()
        return np.linalg.norm(window_values - target_values)

    def pearson_correlation_distance(self, window, target):
        """
        ����Ƥ��ѷ���ϵ�����룺1 - ƽ�����ϵ����

        :param window: ��ʷ���ڡ�
        :param target: Ŀ�괰�ڡ�
        :return: 1 - ƽ�����ϵ����
        """
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        correlations = []
        for feature in features:
            if window[feature].std() == 0 or target[feature].std() == 0:
                correlation = 0  # �����׼��Ϊ0�����ϵ��δ���壬��Ϊ0
            else:
                correlation, _ = pearsonr(window[feature], target[feature])
            correlations.append(correlation)
        average_correlation = np.mean(correlations)
        return 1 - average_correlation

    def find_similar_days(self):
        """
        ����������ʷ��������Ŀ�괰�ڵ������ԣ�������Ȩ�ؼ����ۺ������ԡ�

        :return: �����������ƽ������б��������ں��ۺ������Է�����
        """
        all_scores = []
        total_days = len(self.data)
        window_max = max(self.window_sizes)
        print("��ʼ����������...")

        # ȷ�����㹻�����ݽ��бȽϣ��Ҳ���Ŀ�괰���ص�
        for i in range(window_max, total_days - window_max):
            similarities = []
            for idx, window_size in enumerate(self.window_sizes):
                weight = self.weights[idx]
                window_start = i - window_size
                window_end = i
                window_data = self.data.iloc[window_start:window_end]
                standardized_window = self.standardize(window_data).reset_index(drop=True)
                target_window = self.target_windows_standardized[window_size]
                similarity = self.calculate_similarity(standardized_window, target_window)
                weighted_similarity = similarity * weight
                similarities.append(weighted_similarity)

            total_similarity = sum(similarities)
            all_scores.append({
                'Date': self.data.iloc[i]['Date'],
                'Similarity_Score': total_similarity
            })

            # ��ӡ����
            if (i - window_max + 1) % max(1,
                                          (total_days - 2 * window_max) // 5) == 0 or i == total_days - window_max - 1:
                progress = (i - window_max + 1) / (total_days - 2 * window_max) * 100
                print(f"�Ѵ��� {i - window_max + 1} / {total_days - 2 * window_max} �� ({progress:.2f}%)")

        print("�����Լ�����ɣ����ڽ����ۺ�����...")
        scores_df = pd.DataFrame(all_scores)
        sorted_scores = scores_df.sort_values('Similarity_Score').reset_index(drop=True)
        top_similar_days = sorted_scores.head(self.top_n)
        return top_similar_days

    def display_similar_days(self):
        top_similar_days = self.find_similar_days()
        print(f"\n�ۺ�����ǰ {self.top_n} �������ƵĽ����գ�")
        for idx, row in top_similar_days.iterrows():
            date = row['Date']
            score = row['Similarity_Score']
            print(f"�� {idx + 1} ��:")
            print(f"  ������: {date.strftime('%Y-%m-%d')}")
            print(f"  �ۺ����ƶ�: {score:.4f}")
            print("-" * 50)
        print(f"ʹ�õĴ��ڴ�С��Ȩ��: {list(zip(self.window_sizes, self.weights))}")

        # ����ͼ��
        self.plot_similar_days(top_similar_days)

    def plot_similar_days(self, top_similar_days):
        """
        ����ǰ N �����ƽ����յ�K��ͼ�����´��ڵĶԱȡ�

        :param top_similar_days: �����ƵĽ����� DataFrame��
        """
        target_end_date = self.data.iloc[-1]['Date']
        target_window_size = max(self.window_sizes)
        target_window = self.data.iloc[-target_window_size:].copy().set_index('Date')
        target_window_mpf = target_window.copy()
        target_window_mpf.index.name = 'Date'

        for idx, row in top_similar_days.iterrows():
            similar_date = row['Date']
            window_end_idx = self.data[self.data['Date'] == similar_date].index[0]
            window_start_idx = window_end_idx - target_window_size
            if window_start_idx < 0:
                print(f"����: {similar_date.strftime('%Y-%m-%d')} ��ǰ {target_window_size} �����ݲ��㣬������ͼ��")
                continue
            similar_window = self.data.iloc[window_start_idx:window_end_idx].copy().set_index('Date')
            similar_window_mpf = similar_window.copy()
            similar_window_mpf.index.name = 'Date'

            # ������ͼ
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # ����������ʷK��
            mpf.plot(similar_window_mpf, type='candle', ax=axes[0], style='classic', show_nontrading=False)
            axes[0].set_title(f"�� {idx + 1} �� ���ƽ�����: {similar_date.strftime('%Y-%m-%d')}")
            axes[0].set_ylabel('��ʷK��')

            # ��������K��
            mpf.plot(target_window_mpf, type='candle', ax=axes[1], style='classic', show_nontrading=False)
            axes[1].set_title(f"���½�����: {target_end_date.strftime('%Y-%m-%d')}")
            axes[1].set_ylabel('����K��')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # �������� Excel �ļ�·��Ϊ 'kline_data.xlsx'
    finder = KLineSimilarityFinder(
        file_path='D:\\Downloads\\000001.SH.xlsx',
        window_sizes=[5, 20, 200],
        weights=[0.4, 0.3, 0.3],
        top_n=6,
        algorithm='pearson'  # �� 'euclidean'
        # algorithm='euclidean'
    )
    finder.display_similar_days()