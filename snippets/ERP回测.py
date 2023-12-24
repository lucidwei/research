import pandas as pd
import matplotlib.pyplot as plt


class PortfolioManager:
    def __init__(self, file_path, window_size=500):
        self.file_path = file_path
        self.window_size = window_size
        self.erp_data = None
        self.load_data()

    def load_data(self):
        self.erp_data = pd.read_excel(self.file_path)
        self.erp_data = self.erp_data.dropna(subset=['ERP(左轴)'])
        self.erp_data = self.erp_data.sort_values(by='Unnamed: 0')
        self.erp_data['wan_de_returns'] = self.erp_data['万得全A'].pct_change()
        self.erp_data['ERP(左轴)'] = self.erp_data['ERP(左轴)'].shift(50)
        self.erp_data['ERP(左轴)'] = self.erp_data['ERP(左轴)'].rolling(window=100, min_periods=1).mean()

    def calculate_normalized_erp(self):
        rolling_min = self.erp_data['ERP(左轴)'].rolling(window=self.window_size, min_periods=1).min()
        rolling_max = self.erp_data['ERP(左轴)'].rolling(window=self.window_size, min_periods=1).max()
        self.erp_data['erp_normalized_rolling'] = (self.erp_data['ERP(左轴)'] - rolling_min) / (rolling_max - rolling_min) * 100

    def basic_investment_strategy(self):
        self.calculate_normalized_erp()
        portfolio_returns = self.erp_data['erp_normalized_rolling'].shift(1) / 100 * self.erp_data['wan_de_returns']
        self.erp_data['position'] = self.erp_data['erp_normalized_rolling']
        self.erp_data['portfolio_net_value'] = (1 + portfolio_returns).cumprod()

    def threshold_investment_strategy(self, lower_threshold=10, upper_threshold=90):
        self.calculate_normalized_erp()
        self.erp_data['position'] = self.erp_data['erp_normalized_rolling'].apply(lambda x: 1 if x > upper_threshold else (0 if x < lower_threshold else 0.5))
        portfolio_returns = self.erp_data['position'].shift(1) * self.erp_data['wan_de_returns']
        self.erp_data['portfolio_net_value'] = (1 + portfolio_returns).cumprod()

    def plot_net_value(self, strategy='basic'):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        dates = self.erp_data['Unnamed: 0']

        # 计算归一化净值
        portfolio_net_value_normalized = self.erp_data['portfolio_net_value']
        wan_de_net_value_normalized = self.erp_data['万得全A'] / self.erp_data['万得全A'].iloc[2]

        # 绘制ERP-Based Portfolio和WanDe QuanA在左侧y轴上
        ax1.plot(dates, portfolio_net_value_normalized, label=f'{strategy.title()} Strategy Portfolio', color='green')
        ax1.plot(dates, wan_de_net_value_normalized, label='WanDe QuanA', color='red')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Cumulative Returns (Net Value)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # 在右侧y轴上创建ERP归一化滚动的第二y轴
        ax2 = ax1.twinx()
        ax2.plot(dates, self.erp_data['position'], label='position', color='yellow', alpha=0.5)
        # ax2.plot(dates, self.erp_data['ERP(左轴)'], label='ERP', color='yellow', alpha=0.5)
        ax2.set_ylabel('ERP Normalized (0-100%)', color='yellow')
        ax2.tick_params(axis='y', labelcolor='yellow')
        ax2.legend(loc='upper right')

        plt.title(f'Normalized {strategy.title()} Strategy Portfolio vs WanDe QuanA Net Value with ERP Rolling')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Usage:
manager = PortfolioManager("D:\WPS云盘\WPS云盘\工作-麦高\杂活\指南针要的宏观指标\ERP\ERP截面.xlsx")
manager.basic_investment_strategy()
manager.plot_net_value(strategy='basic')

manager.threshold_investment_strategy()
manager.plot_net_value(strategy='threshold')













