# coding=gbk
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, percentileofscore
import matplotlib.pyplot as plt
import mplfinance as mpf
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

class KLineSimilarityFinder:
    def __init__(self, file_path, window_sizes=None, weights=None,
                 top_n=5, algorithm='euclidean', dates_after=None):
        """
        初始化相似性查找器。

        :param file_path: 数据文件路径。
        :param window_sizes: 要使用的窗口大小列表。
        :param weights: 每个窗口大小对应的权重列表。
        :param top_n: 最终选择的前几个最相似的窗口数量。
        :param algorithm: 相似性计算算法，支持 'euclidean' 和 'pearson'。
        """
        if weights is None:
            weights = [0.2, 0.4, 0.4]
        if window_sizes is None:
            window_sizes = [5, 20, 200]
        if len(window_sizes) != len(weights):
            raise ValueError("window_sizes 和 weights 必须具有相同的长度。")

        self.file_path = file_path
        self.window_sizes = window_sizes
        self.weights = weights
        self.top_n = top_n
        self.algorithm = algorithm
        self.dates_after = pd.to_datetime(dates_after) if dates_after else None
        self.data = self.load_data()
        self.target_windows_standardized = self.prepare_target_windows()

    def load_data(self):
        data = pd.read_excel(self.file_path, engine='openpyxl')
        data = data[['日期', '开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '成交额(百万)']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        # 如果指定了dates_after，则筛选日期
        if self.dates_after:
            data = data[data['Date'] >= self.dates_after]

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



class KLinePatternAnalyzer(KLineSimilarityFinder):
    def __init__(self, file_path, dates_after=None):
        super().__init__(file_path=file_path, dates_after=dates_after)

    def filter_pattern(self, open_threshold=0.02, pullback_ratio=0.5):
        """
        Filters the data based on:
        - Open > previous close by `open_threshold` (e.g., 2%)
        - Close is a pullback of at least `pullback_ratio` of the day's rise (e.g., 50%)

        Parameters:
        - open_threshold: float, e.g., 0.02 for 2%
        - pullback_ratio: float, e.g., 0.5 for 50%

        Returns:
        - filtered_dates: DataFrame containing the filtered rows
        """
        df = self.data.copy()

        # Calculate the percentage change from previous close to current open
        df['Prev_Close'] = df['Close'].shift(1)
        df['Open_Pct_Change'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # Calculate the day's price movement
        df['Day_Move'] = (df['Close'] - df['Open']) / df['Open']

        # Calculate the pullback: (Close - Open) / Day_Move
        df['Pullback_Ratio'] = -df['Day_Move'] / df['Open_Pct_Change']

        # Apply the filters
        filtered = df[
            (df['Open_Pct_Change'] > open_threshold) &
            (df['Day_Move'] < 0) &  # Assuming "回吐" means price decreased from open to close
            (df['Pullback_Ratio'] >= pullback_ratio)
            ]

        # 计算百分位数
        periods = [50, 250, 1000]
        for period in periods:
            percentile_col = f'Pct_{period}'
            # 使用滚动窗口计算百分位数
            df_sorted = self.data['Close'].rolling(window=period, min_periods=1)
            # 计算每一天的百分位数
            filtered[percentile_col] = filtered.apply(
                lambda row: percentileofscore(
                    self.data.loc[:row.name, 'Close'].iloc[-period - 1:-1],
                    row['Close'])
                if row.name >= period else np.nan,
                axis=1
            )

        # 处理百分位数列，先保留两位小数后转换为整数表示百分比
        filtered['Pct_50'] = filtered['Pct_50'].round(0).astype('Int64')
        filtered['Pct_250'] = filtered['Pct_250'].round(0).astype('Int64')
        filtered['Pct_1000'] = filtered['Pct_1000'].round(0).astype('Int64')

        # 处理其他数值列，保留0位小数后转换为整数
        filtered['Pullback_Ratio'] = (filtered['Pullback_Ratio'] * 100).round(0).astype('Int64')  # 转换为百分比

        self.filtered_dates = filtered
        print("筛选出的符合条件的日期及相关信息：")
        print(filtered[['Date', 'Close', 'Open_Pct_Change', 'Day_Move', 'Pullback_Ratio', 'Pct_50', 'Pct_250', 'Pct_1000']])
        return filtered

    def compute_statistics(self, filtered_df, periods=[5, 25]):
        """
        计算筛选日期后指定天数的涨跌统计，同时统计收盘价的相对位置。

        参数:
        - filtered_df: DataFrame，符合条件的筛选日期。
        - periods: list of int，指定的统计天数（如[5, 25]）。

        返回:
        - stats: dict，每个统计周期的统计结果。
        """
        stats = {}
        percentile_columns = ['Pct_50', 'Pct_250', 'Pct_1000']
        percentile_labels = {
            'Pct_50': '50天百分位数',
            'Pct_250': '250天百分位数',
            'Pct_1000': '1000天百分位数'
        }
        for period in periods:
            pct_changes = []
            win_count = 0
            total = 0
            for idx in filtered_df.index:
                end_idx = idx + period
                if end_idx < len(self.data):
                    pct_change = (self.data.loc[end_idx, 'Close'] - self.data.loc[idx, 'Close']) / self.data.loc[
                        idx, 'Close']
                    pct_changes.append(pct_change)
                    if pct_change > 0:
                        win_count += 1
                    total += 1
            pct_changes = np.array(pct_changes)
            average = np.mean(pct_changes) * 100
            median = np.median(pct_changes) * 100
            win_rate = (win_count / total) * 100 if total > 0 else 0
            stats[period] = {
                'average_pct_change': average,
                'median_pct_change': median,
                'win_rate': win_rate
            }
            print(f"\n接下来 {period} 个交易日的统计信息：")
            print(f"平均涨跌幅: {average:.2f}%")
            print(f"中位数涨跌幅: {median:.2f}%")
            print(f"胜率: {win_rate:.2f}%")

        # 统计收盘价的相对位置
        # 低位和高位比例统计：
        # 低位比例：计算有多少筛选日期的收盘价百分位数低于33 %（即处于低位）。
        # 高位比例：计算有多少筛选日期的收盘价百分位数高于66 %（即处于高位）。
        # 这样可以了解在筛选出的特定K线模式出现时，收盘价整体处于何种相对水平。
        print("\n筛选日期的收盘价相对位置统计：")
        for pct_col in percentile_columns:
            label = percentile_labels[pct_col]
            # 定义低位和高位的阈值
            low_threshold = 33
            high_threshold = 66
            low_count = filtered_df[pct_col] < low_threshold
            high_count = filtered_df[pct_col] > high_threshold
            total_count = filtered_df[pct_col].notna().sum()
            low_ratio = (low_count.sum() / total_count) * 100 if total_count > 0 else 0
            high_ratio = (high_count.sum() / total_count) * 100 if total_count > 0 else 0
            print(f"{label}:")
            print(f"  处于低位 (<{low_threshold}%) 的比例: {low_ratio:.2f}%")
            print(f"  处于高位 (>{high_threshold}%) 的比例: {high_ratio:.2f}%")
        self.stats = stats
        return stats

    def plot_patterns_and_trends(self, filtered_df, periods=[5, 25]):
        """
        绘制筛选出的K线模式及其后续走势。

        参数:
        - filtered_df: DataFrame，符合条件的筛选日期。
        - periods: list of int，指定的统计天数（如[5, 25]）。
        """
        for period in periods:
            plt.figure(figsize=(10, 6))
            all_trends = []

            for idx in filtered_df.index:
                end_idx = idx + period
                if end_idx >= len(self.data):
                    continue  # 跳过数据不足的情况
                trend = self.data.loc[idx:end_idx, 'Close'].values
                if len(trend) != period + 1:
                    continue  # 确保趋势长度一致

                # 归一化趋势，以第0天的收盘价为基准
                normalized_trend = trend / trend[0]
                all_trends.append(normalized_trend)

                # 绘制单个趋势
                plt.plot(range(period + 1), normalized_trend, color='blue', alpha=0.3)

            if not all_trends:
                print(f"没有足够的数据来绘制 {period} 天的趋势图。")
                continue

            all_trends = np.array(all_trends)
            # 计算平均趋势
            average_trend = all_trends.mean(axis=0)
            plt.plot(range(period + 1), average_trend, color='red', linewidth=2, label='平均趋势')

            plt.title(f'{period} 天后走势趋势图')
            plt.xlabel('天数')
            plt.ylabel('归一化收盘价')
            plt.legend()
            plt.grid(True)
            plt.xticks(range(0, period + 1, max(1, period // 5)))  # 动态设置x轴刻度
            plt.ylim(all_trends.min() * 0.95, all_trends.max() * 1.05)  # 动态设置y轴范围
            plt.show()

    def analyze(self, open_threshold=0.02, pullback_ratio=0.5, periods=[5, 25]):
        """
        Executes the full analysis: filtering, statistics, and plotting.

        Parameters:
        - open_threshold: float for open price increase threshold
        - pullback_ratio: float for pullback ratio
        - periods: list of integers for subsequent days analysis
        """
        filtered_df = self.filter_pattern(open_threshold, pullback_ratio)
        if filtered_df.empty:
            print("No matching patterns found.")
            return
        self.compute_statistics(filtered_df, periods)
        self.plot_patterns_and_trends(filtered_df, periods)



if __name__ == "__main__":
    # 假设您的 Excel 文件路径为 'kline_data.xlsx'
    file_path = rf"D:\Downloads\a000001.xlsx"

    # finder = KLineSimilarityFinder(
    #     file_path=file_path,
    #     window_sizes=[5, 20, 200],
    #     weights=[0.7, 0.2, 0.1],
    #     top_n=6,
    #     algorithm='pearson'  # 或 'euclidean'
    #     # algorithm='euclidean'
    # )
    # finder.display_similar_days()

    analyzer = KLinePatternAnalyzer(file_path=file_path, dates_after='1993-01-01')
    analyzer.analyze(open_threshold=0.025, pullback_ratio=0.5, periods=[5, 25])