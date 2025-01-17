# coding=gbk
# Time Created: 2025/1/16 16:24
# Author  : Lucid
# FileName: ������λʾ��.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ���ۣ�ʹ�ÿ�����λ���Ż���ֵ������״/���ձ��ʣ�������ev�£���ֵ��Ȼ�Ǹ߲�λ�ĸ��ߡ�


def calculate_zero_cost_trials(win_prob, position_ratio, win_odds, loss_odds, total_trials, simulation_times):
    # �洢ÿ��ģ��Ľ��
    simulation_results = []
    for _ in range(simulation_times):
        initial_asset = 1
        trial_count = 0
        df = pd.DataFrame({})
        # ��ʼ���ʲ���ʤ����¼
        win_or_lose_list = ['-'] + [True] * total_trials
        assets_list = [initial_asset] * (total_trials + 1)
        df.insert(0, 'win_or_lose', win_or_lose_list)
        df.insert(1, 'assets', assets_list)

        while trial_count < total_trials:
            random_result = np.random.uniform(0, 1)
            trial_count += 1
            if random_result < win_prob:  # ����ʤ���ж��Ƿ��ʤ
                df.loc[trial_count, 'assets'] = df.loc[trial_count - 1, 'assets'] * (1 + position_ratio * win_odds)
                df.loc[trial_count, 'win_or_lose'] = True
            else:
                df.loc[trial_count, 'assets'] = df.loc[trial_count - 1, 'assets'] * (1 - position_ratio * loss_odds)
                df.loc[trial_count, 'win_or_lose'] = False

        simulation_results.append(df)
    return simulation_results


# ��������
win_odds = 3        # ��ʤʱ������
loss_odds = 1       # ʧ��ʱ������
win_prob = 0.3     # ʤ�ʣ����Ե�����
total_trials = 100  # ���������
simulation_times = 200  # ģ�����
# ��λ����
# position_ratio = 0.25
position_ratio = win_prob - (1 - win_prob) * (loss_odds / win_odds)
print(f'position_ratio={position_ratio}')
ev = win_prob * win_odds + (1 - win_prob) * (-loss_odds)
print(f'ev={ev}')

simulation_dfs = calculate_zero_cost_trials(win_prob, position_ratio, win_odds, loss_odds, total_trials, simulation_times)


# ����ÿ���ʲ��ľ�ֵ����Сֵ�����ֵ������ѡ��ֵ������С��·��
def calculate_statistics(simulation_dfs):
    # ��ÿ��ģ����ʲ�����ȡ��������ϳ�һ���µ� DataFrame������Ψһ����
    combined_assets = pd.concat([df['assets'].rename(i) for i, df in enumerate(simulation_dfs)], axis=1)
    # ����ÿ���ʲ��ľ�ֵ
    mean_assets = combined_assets.mean(axis=1)
    # ����ÿ���ʲ�����Сֵ�����ֵ
    min_assets = combined_assets.min(axis=1)
    max_assets = combined_assets.max(axis=1)

    # ����ÿ��·���������ʲ�ֵ
    final_assets = combined_assets.iloc[-1]
    # �ҵ���ֵ��С������·������
    min_path_idx = final_assets.idxmin()  # ��ֵ��С·����������
    max_path_idx = final_assets.idxmax()  # ��ֵ���·����������

    # ���ؾ�ֵ����Сֵ�����ֵ���Լ���ֵ��С������·��
    return mean_assets, min_assets, max_assets, combined_assets[min_path_idx], combined_assets[max_path_idx]


# ����ͳ��ֵ
mean_assets, min_assets, max_assets, min_path, max_path = calculate_statistics(simulation_dfs)

# ���ƽ��
def plot_results(mean_assets, min_assets, max_assets, min_path, max_path):
    plt.figure(figsize=(10, 6))
    # ���ƾ�ֵ����
    plt.plot(mean_assets.index, mean_assets, label='Mean Asset', color='blue')
    # ������Сֵ�����ֵ��Χ
    # plt.fill_between(mean_assets.index, min_assets, max_assets, color='gray', alpha=0.3, label='Min-Max Range')
    # # ������ֵ��С������·��
    # plt.plot(min_path.index, min_path, label='Min Path (Final Value)', color='red', linestyle='--')
    # plt.plot(max_path.index, max_path, label='Max Path (Final Value)', color='green', linestyle='--')
    plt.xlabel('Trial Number')
    plt.ylabel('Asset Value')
    plt.title('Daily Assets Curve with Min-Max Range and Extreme Paths')
    plt.legend()
    plt.show()

# ����ͼ��
plot_results(mean_assets, min_assets, max_assets, min_path, max_path)
