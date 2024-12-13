# coding=gbk
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, percentileofscore
import matplotlib.pyplot as plt
import mplfinance as mpf
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������

class KLineSimilarityFinder:
    def __init__(self, file_path, window_sizes=None, weights=None,
                 top_n=5, algorithm='euclidean', dates_after=None):
        """
        ��ʼ�������Բ�������

        :param file_path: �����ļ�·����
        :param window_sizes: Ҫʹ�õĴ��ڴ�С�б�
        :param weights: ÿ�����ڴ�С��Ӧ��Ȩ���б�
        :param top_n: ����ѡ���ǰ���������ƵĴ���������
        :param algorithm: �����Լ����㷨��֧�� 'euclidean' �� 'pearson'��
        """
        if weights is None:
            weights = [0.2, 0.4, 0.4]
        if window_sizes is None:
            window_sizes = [5, 20, 200]
        if len(window_sizes) != len(weights):
            raise ValueError("window_sizes �� weights ���������ͬ�ĳ��ȡ�")

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
        data = data[['����', '���̼�(Ԫ)', '��߼�(Ԫ)', '��ͼ�(Ԫ)', '���̼�(Ԫ)', '�ɽ���(����)']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        # ���ָ����dates_after����ɸѡ����
        if self.dates_after:
            data = data[data['Date'] >= self.dates_after]

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
            (df['Day_Move'] < 0) &  # Assuming "����" means price decreased from open to close
            (df['Pullback_Ratio'] >= pullback_ratio)
            ]

        # ����ٷ�λ��
        periods = [50, 250, 1000]
        for period in periods:
            percentile_col = f'Pct_{period}'
            # ʹ�ù������ڼ���ٷ�λ��
            df_sorted = self.data['Close'].rolling(window=period, min_periods=1)
            # ����ÿһ��İٷ�λ��
            filtered[percentile_col] = filtered.apply(
                lambda row: percentileofscore(
                    self.data.loc[:row.name, 'Close'].iloc[-period - 1:-1],
                    row['Close'])
                if row.name >= period else np.nan,
                axis=1
            )

        # ����ٷ�λ���У��ȱ�����λС����ת��Ϊ������ʾ�ٷֱ�
        filtered['Pct_50'] = filtered['Pct_50'].round(0).astype('Int64')
        filtered['Pct_250'] = filtered['Pct_250'].round(0).astype('Int64')
        filtered['Pct_1000'] = filtered['Pct_1000'].round(0).astype('Int64')

        # ����������ֵ�У�����0λС����ת��Ϊ����
        filtered['Pullback_Ratio'] = (filtered['Pullback_Ratio'] * 100).round(0).astype('Int64')  # ת��Ϊ�ٷֱ�

        self.filtered_dates = filtered
        print("ɸѡ���ķ������������ڼ������Ϣ��")
        print(filtered[['Date', 'Close', 'Open_Pct_Change', 'Day_Move', 'Pullback_Ratio', 'Pct_50', 'Pct_250', 'Pct_1000']])
        return filtered

    def compute_statistics(self, filtered_df, periods=[5, 25]):
        """
        ����ɸѡ���ں�ָ���������ǵ�ͳ�ƣ�ͬʱͳ�����̼۵����λ�á�

        ����:
        - filtered_df: DataFrame������������ɸѡ���ڡ�
        - periods: list of int��ָ����ͳ����������[5, 25]����

        ����:
        - stats: dict��ÿ��ͳ�����ڵ�ͳ�ƽ����
        """
        stats = {}
        percentile_columns = ['Pct_50', 'Pct_250', 'Pct_1000']
        percentile_labels = {
            'Pct_50': '50��ٷ�λ��',
            'Pct_250': '250��ٷ�λ��',
            'Pct_1000': '1000��ٷ�λ��'
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
            print(f"\n������ {period} �������յ�ͳ����Ϣ��")
            print(f"ƽ���ǵ���: {average:.2f}%")
            print(f"��λ���ǵ���: {median:.2f}%")
            print(f"ʤ��: {win_rate:.2f}%")

        # ͳ�����̼۵����λ��
        # ��λ�͸�λ����ͳ�ƣ�
        # ��λ�����������ж���ɸѡ���ڵ����̼۰ٷ�λ������33 %�������ڵ�λ����
        # ��λ�����������ж���ɸѡ���ڵ����̼۰ٷ�λ������66 %�������ڸ�λ����
        # ���������˽���ɸѡ�����ض�K��ģʽ����ʱ�����̼����崦�ں������ˮƽ��
        print("\nɸѡ���ڵ����̼����λ��ͳ�ƣ�")
        for pct_col in percentile_columns:
            label = percentile_labels[pct_col]
            # �����λ�͸�λ����ֵ
            low_threshold = 33
            high_threshold = 66
            low_count = filtered_df[pct_col] < low_threshold
            high_count = filtered_df[pct_col] > high_threshold
            total_count = filtered_df[pct_col].notna().sum()
            low_ratio = (low_count.sum() / total_count) * 100 if total_count > 0 else 0
            high_ratio = (high_count.sum() / total_count) * 100 if total_count > 0 else 0
            print(f"{label}:")
            print(f"  ���ڵ�λ (<{low_threshold}%) �ı���: {low_ratio:.2f}%")
            print(f"  ���ڸ�λ (>{high_threshold}%) �ı���: {high_ratio:.2f}%")
        self.stats = stats
        return stats

    def plot_patterns_and_trends(self, filtered_df, periods=[5, 25]):
        """
        ����ɸѡ����K��ģʽ����������ơ�

        ����:
        - filtered_df: DataFrame������������ɸѡ���ڡ�
        - periods: list of int��ָ����ͳ����������[5, 25]����
        """
        for period in periods:
            plt.figure(figsize=(10, 6))
            all_trends = []

            for idx in filtered_df.index:
                end_idx = idx + period
                if end_idx >= len(self.data):
                    continue  # �������ݲ�������
                trend = self.data.loc[idx:end_idx, 'Close'].values
                if len(trend) != period + 1:
                    continue  # ȷ�����Ƴ���һ��

                # ��һ�����ƣ��Ե�0������̼�Ϊ��׼
                normalized_trend = trend / trend[0]
                all_trends.append(normalized_trend)

                # ���Ƶ�������
                plt.plot(range(period + 1), normalized_trend, color='blue', alpha=0.3)

            if not all_trends:
                print(f"û���㹻������������ {period} �������ͼ��")
                continue

            all_trends = np.array(all_trends)
            # ����ƽ������
            average_trend = all_trends.mean(axis=0)
            plt.plot(range(period + 1), average_trend, color='red', linewidth=2, label='ƽ������')

            plt.title(f'{period} �����������ͼ')
            plt.xlabel('����')
            plt.ylabel('��һ�����̼�')
            plt.legend()
            plt.grid(True)
            plt.xticks(range(0, period + 1, max(1, period // 5)))  # ��̬����x��̶�
            plt.ylim(all_trends.min() * 0.95, all_trends.max() * 1.05)  # ��̬����y�᷶Χ
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
    # �������� Excel �ļ�·��Ϊ 'kline_data.xlsx'
    file_path = rf"D:\Downloads\a000001.xlsx"

    # finder = KLineSimilarityFinder(
    #     file_path=file_path,
    #     window_sizes=[5, 20, 200],
    #     weights=[0.7, 0.2, 0.1],
    #     top_n=6,
    #     algorithm='pearson'  # �� 'euclidean'
    #     # algorithm='euclidean'
    # )
    # finder.display_similar_days()

    analyzer = KLinePatternAnalyzer(file_path=file_path, dates_after='1993-01-01')
    analyzer.analyze(open_threshold=0.025, pullback_ratio=0.5, periods=[5, 25])