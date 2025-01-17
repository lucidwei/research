# coding=gbk
# Time Created: 2025/1/16 16:24
# Author  : Lucid
# FileName: 凯利仓位示例.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 结论：使用凯利仓位会优化净值曲线形状/夏普比率，但在正ev下，终值仍然是高仓位的更高。


def calculate_zero_cost_trials(win_prob, position_ratio, win_odds, loss_odds, total_trials, simulation_times):
    # 存储每次模拟的结果
    simulation_results = []
    for _ in range(simulation_times):
        initial_asset = 1
        trial_count = 0
        df = pd.DataFrame({})
        # 初始化资产和胜负记录
        win_or_lose_list = ['-'] + [True] * total_trials
        assets_list = [initial_asset] * (total_trials + 1)
        df.insert(0, 'win_or_lose', win_or_lose_list)
        df.insert(1, 'assets', assets_list)

        while trial_count < total_trials:
            random_result = np.random.uniform(0, 1)
            trial_count += 1
            if random_result < win_prob:  # 根据胜率判断是否获胜
                df.loc[trial_count, 'assets'] = df.loc[trial_count - 1, 'assets'] * (1 + position_ratio * win_odds)
                df.loc[trial_count, 'win_or_lose'] = True
            else:
                df.loc[trial_count, 'assets'] = df.loc[trial_count - 1, 'assets'] * (1 - position_ratio * loss_odds)
                df.loc[trial_count, 'win_or_lose'] = False

        simulation_results.append(df)
    return simulation_results


# 参数设置
win_odds = 3        # 获胜时的赔率
loss_odds = 1       # 失败时的赔率
win_prob = 0.3     # 胜率（可以调整）
total_trials = 100  # 总试验次数
simulation_times = 200  # 模拟次数
# 仓位比例
# position_ratio = 0.25
position_ratio = win_prob - (1 - win_prob) * (loss_odds / win_odds)
print(f'position_ratio={position_ratio}')
ev = win_prob * win_odds + (1 - win_prob) * (-loss_odds)
print(f'ev={ev}')

simulation_dfs = calculate_zero_cost_trials(win_prob, position_ratio, win_odds, loss_odds, total_trials, simulation_times)


# 计算每日资产的均值、最小值和最大值，并挑选终值最大和最小的路径
def calculate_statistics(simulation_dfs):
    # 将每次模拟的资产列提取出来并组合成一个新的 DataFrame，设置唯一列名
    combined_assets = pd.concat([df['assets'].rename(i) for i, df in enumerate(simulation_dfs)], axis=1)
    # 计算每日资产的均值
    mean_assets = combined_assets.mean(axis=1)
    # 计算每日资产的最小值和最大值
    min_assets = combined_assets.min(axis=1)
    max_assets = combined_assets.max(axis=1)

    # 计算每条路径的最终资产值
    final_assets = combined_assets.iloc[-1]
    # 找到终值最小和最大的路径索引
    min_path_idx = final_assets.idxmin()  # 终值最小路径的列索引
    max_path_idx = final_assets.idxmax()  # 终值最大路径的列索引

    # 返回均值、最小值、最大值，以及终值最小和最大的路径
    return mean_assets, min_assets, max_assets, combined_assets[min_path_idx], combined_assets[max_path_idx]


# 计算统计值
mean_assets, min_assets, max_assets, min_path, max_path = calculate_statistics(simulation_dfs)

# 绘制结果
def plot_results(mean_assets, min_assets, max_assets, min_path, max_path):
    plt.figure(figsize=(10, 6))
    # 绘制均值曲线
    plt.plot(mean_assets.index, mean_assets, label='Mean Asset', color='blue')
    # 绘制最小值和最大值范围
    # plt.fill_between(mean_assets.index, min_assets, max_assets, color='gray', alpha=0.3, label='Min-Max Range')
    # # 绘制终值最小和最大的路径
    # plt.plot(min_path.index, min_path, label='Min Path (Final Value)', color='red', linestyle='--')
    # plt.plot(max_path.index, max_path, label='Max Path (Final Value)', color='green', linestyle='--')
    plt.xlabel('Trial Number')
    plt.ylabel('Asset Value')
    plt.title('Daily Assets Curve with Min-Max Range and Extreme Paths')
    plt.legend()
    plt.show()

# 绘制图表
plot_results(mean_assets, min_assets, max_assets, min_path, max_path)
