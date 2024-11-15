# coding=gbk
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import mplfinance as mpf
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

class KLineSimilarityFinder:
    def __init__(self, file_path, window_sizes=[5, 20, 200], weights=[0.2, 0.4, 0.4],
                 top_n=5, algorithm='euclidean'):
        """
        初始化相似性查找器。

        :param file_path: 数据文件路径。
        :param window_sizes: 要使用的窗口大小列表。
        :param weights: 每个窗口大小对应的权重列表。
        :param top_n: 最终选择的前几个最相似的窗口数量。
        :param algorithm: 相似性计算算法，支持 'euclidean' 和 'pearson'。
        """
        if len(window_sizes) != len(weights):
            raise ValueError("window_sizes 和 weights 必须具有相同的长度。")

        self.file_path = file_path
        self.window_sizes = window_sizes
        self.weights = weights
        self.top_n = top_n
        self.algorithm = algorithm
        self.data = self.load_data()
        self.target_windows_standardized = self.prepare_target_windows()

    def load_data(self):
        data = pd.read_excel(self.file_path, engine='openpyxl')
        data = data[['日期', '开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '成交额(百万)']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)  # 确保按日期排序
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
        准备最新的各个窗口大小的标准化目标窗口。
        """
        target_windows = {}
        for window_size in self.window_sizes:
            if len(self.data) < window_size:
                raise ValueError(f"数据长度不足以使用窗口大小 {window_size}")
            target_window = self.standardize(self.data.iloc[-window_size:]).reset_index(drop=True)
            target_windows[window_size] = target_window
        return target_windows

    def calculate_similarity(self, window, target):
        """
        根据指定的算法计算两个窗口的相似性。

        :param window: 历史窗口。
        :param target: 目标窗口。
        :return: 相似性分数（数值越小越相似）。
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
        计算皮尔逊相关系数距离：1 - 平均相关系数。

        :param window: 历史窗口。
        :param target: 目标窗口。
        :return: 1 - 平均相关系数。
        """
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        correlations = []
        for feature in features:
            if window[feature].std() == 0 or target[feature].std() == 0:
                correlation = 0  # 如果标准差为0，相关系数未定义，设为0
            else:
                correlation, _ = pearsonr(window[feature], target[feature])
            correlations.append(correlation)
        average_correlation = np.mean(correlations)
        return 1 - average_correlation

    def find_similar_days(self):
        """
        查找所有历史交易日与目标窗口的相似性，并根据权重计算综合相似性。

        :return: 排序后的最相似交易日列表，包含日期和综合相似性分数。
        """
        all_scores = []
        total_days = len(self.data)
        window_max = max(self.window_sizes)
        print("开始计算相似性...")

        # 确保有足够的数据进行比较，且不与目标窗口重叠
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

            # 打印进度
            if (i - window_max + 1) % max(1,
                                          (total_days - 2 * window_max) // 5) == 0 or i == total_days - window_max - 1:
                progress = (i - window_max + 1) / (total_days - 2 * window_max) * 100
                print(f"已处理 {i - window_max + 1} / {total_days - 2 * window_max} 天 ({progress:.2f}%)")

        print("相似性计算完成，正在进行综合排序...")
        scores_df = pd.DataFrame(all_scores)
        sorted_scores = scores_df.sort_values('Similarity_Score').reset_index(drop=True)
        top_similar_days = sorted_scores.head(self.top_n)
        return top_similar_days

    def display_similar_days(self):
        top_similar_days = self.find_similar_days()
        print(f"\n综合排名前 {self.top_n} 个最相似的交易日：")
        for idx, row in top_similar_days.iterrows():
            date = row['Date']
            score = row['Similarity_Score']
            print(f"第 {idx + 1} 名:")
            print(f"  交易日: {date.strftime('%Y-%m-%d')}")
            print(f"  综合相似度: {score:.4f}")
            print("-" * 50)
        print(f"使用的窗口大小及权重: {list(zip(self.window_sizes, self.weights))}")

        # 绘制图表
        self.plot_similar_days(top_similar_days)

    def plot_similar_days(self, top_similar_days):
        """
        绘制前 N 名相似交易日的K线图与最新窗口的对比。

        :param top_similar_days: 最相似的交易日 DataFrame。
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
                print(f"警告: {similar_date.strftime('%Y-%m-%d')} 的前 {target_window_size} 天数据不足，跳过绘图。")
                continue
            similar_window = self.data.iloc[window_start_idx:window_end_idx].copy().set_index('Date')
            similar_window_mpf = similar_window.copy()
            similar_window_mpf.index.name = 'Date'

            # 创建新图
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # 绘制相似历史K线
            mpf.plot(similar_window_mpf, type='candle', ax=axes[0], style='classic', show_nontrading=False)
            axes[0].set_title(f"第 {idx + 1} 名 相似交易日: {similar_date.strftime('%Y-%m-%d')}")
            axes[0].set_ylabel('历史K线')

            # 绘制最新K线
            mpf.plot(target_window_mpf, type='candle', ax=axes[1], style='classic', show_nontrading=False)
            axes[1].set_title(f"最新交易日: {target_end_date.strftime('%Y-%m-%d')}")
            axes[1].set_ylabel('最新K线')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # 假设您的 Excel 文件路径为 'kline_data.xlsx'
    finder = KLineSimilarityFinder(
        file_path='D:\\Downloads\\000001.SH.xlsx',
        window_sizes=[5, 20, 200],
        weights=[0.4, 0.3, 0.3],
        top_n=6,
        algorithm='pearson'  # 或 'euclidean'
        # algorithm='euclidean'
    )
    finder.display_similar_days()