import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# 设置随机种子以便结果可复现
np.random.seed(42)

# 生成示例数据
# 假设我们有60天的数据
days = 60
# 基金的初始行业持仓占比
initial_positions = np.array([0.4, 0.3, 0.3]) # 科技、金融、消费品
# 生成各行业的日度收益率
industry_returns = np.random.normal(0, 0.01, (days, 3)) # 科技、金融、消费品
# 生成基金的日度收益率，通过行业收益率和持仓比例加权得到，加入一些噪声模拟实际情况
fund_returns = industry_returns.dot(initial_positions)# + np.random.normal(0, 0.002, days)

# 创建DataFrame
dates = pd.date_range(start="2023-01-01", periods=days)
data = pd.DataFrame(industry_returns, columns=['Tech', 'Finance', 'Consumer'], index=dates)
data['Fund'] = fund_returns

# 使用Lasso回归估算日度基金的行业仓位
X = data[['Tech', 'Finance', 'Consumer']] # 特征矩阵
y = data['Fund'] # 目标变量

# 实例化Lasso回归模型，alpha值决定了模型的复杂程度（L1正则化的强度）
lasso = Lasso(alpha=3e-6)
lasso.fit(X, y)

# 打印Lasso回归得到的系数，这些系数代表基金对各个行业的依赖程度
lasso_coefficients = lasso.coef_
lasso_coefficients_dict = dict(zip(X.columns, lasso_coefficients))

lasso_coefficients_dict


# 假设状态变量（行业仓位）和观测变量（基金与行业的日度收益率）
initial_state_mean = [0.4, 0.3, 0.3]  # 基金的初始行业持仓占比
# observation_matrix = np.column_stack([data['Tech'], data['Finance'], data['Consumer']])  # 各行业的日度收益率作为观测矩阵
# observation_matrices = np.column_stack([data['Tech'], data['Finance'], data['Consumer']])


def run_kalman_filter(data):
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(dim_x=3, dim_z=1)

    # 定义初始状态 (行业持仓比例)
    kf.x = np.array([0.4, 0.3, 0.3])  # initial_state_mean

    # 定义状态转移矩阵
    kf.F = np.eye(3)  # transition_matrices

    # 定义观测矩阵，假设我们直接观测到的是基金的日度收益率
    kf.H = np.array([[0.4, 0.3, 0.3]])  # observation_matrices

    # 定义状态协方差
    kf.P *= 0.1

    # 定义测量噪声
    kf.R = np.array([[0.1]])  # observation_covariance

    # 定义过程噪声
    kf.Q = Q_discrete_white_noise(dim=3, dt=1, var=0.01)

    # 准备观测数据
    measurements = data['Fund'].values  # 基金的日度收益率

    # 运行卡尔曼滤波
    state_estimates = []

    for z in measurements:
        kf.predict()
        kf.update(z)
        state_estimates.append(kf.x.copy())

    return state_estimates


def post_constraint_kf(data):
    kf = KalmanFilter(dim_x=3, dim_z=1) # 初始化卡尔曼滤波器

    # 定义初始状态 (行业持仓比例)
    kf.x = np.array([0.4, 0.3, 0.3])  # initial_state_mean

    # 定义状态转移矩阵
    kf.F = np.eye(3)  # transition_matrices

    # # 定义观测矩阵，假设我们直接观测到的是基金的日度收益率
    # kf.H = np.array([[0.4, 0.3, 0.3]])  # observation_matrices
    #
    # # 定义状态协方差
    # kf.P *= 0.1
    #
    # # 定义测量噪声
    # kf.R = np.array([[0.1]])  # observation_covariance
    #
    # # 定义过程噪声
    # kf.Q = Q_discrete_white_noise(dim=3, dt=1, var=0.01)

    # 观测矩阵为每日各行业收益率
    kf.H = np.zeros((1, 3))  # 初始化观测矩阵，稍后更新

    # 定义状态协方差
    kf.P *= 0.1

    # 由于我们假设没有噪声，这里将测量噪声和过程噪声设置得很小
    kf.R = np.array([[1e-5]])  # 观测噪声协方差
    kf.Q = Q_discrete_white_noise(dim=3, dt=1, var=1e-5)

    # 准备观测数据
    measurements = data['Fund'].values  # 基金的日度收益率

    state_estimates = []
    for z in measurements:
        kf.predict()
        kf.update(z)

        # 应用约束：非负和总和为1
        constrained_state = np.maximum(kf.x, 0)  # 保持非负
        # 如果所有的持仓比例都非常小，则进行归一化处理
        if np.sum(constrained_state) > 1e-4:  # 设置一个阈值，例如1e-4
            constrained_state = constrained_state / np.sum(constrained_state)
        else:
            # 如果所有的持仓比例都接近于零，则可能是估计出现了问题
            # 在这种情况下，我们可以选择维持原始的持仓比例，或者使用其他方法处理
            constrained_state = np.array([0.4, 0.3, 0.3])  # 例如使用初始持仓比例

        state_estimates.append(constrained_state.copy())

    return state_estimates



# 假设`data`是之前创建的DataFrame，包含'Tech', 'Finance', 'Consumer', 和'Fund'列
state_estimates = run_kalman_filter(data)
state_estimates_post = post_constraint_kf(data)
print("Estimated final state (industry positions):", state_estimates[-1])



# 假设我们有以下数据
# 每日行业表现，这里我们使用固定的行业收益率
industry_performance = np.array([0.1, 0.05, 0.03])
# 基金的行业持仓，这里我们使用固定的持仓比例
fund_position = np.array([0.4, 0.3, 0.3])

# 计算基金的日度收益率，这里假设没有噪声，收益率完全由行业表现加权得到
fund_daily_return = np.dot(industry_performance, fund_position)

# 初始化卡尔曼滤波器
kf = KalmanFilter(dim_x=3, dim_z=1)
kf.x = np.array([0.4, 0.3, 0.3])  # 初始状态
kf.F = np.eye(3)  # 状态转移矩阵
kf.H = np.array([industry_performance])  # 观测矩阵
kf.P *= 1.0  # 初始状态协方差
kf.R = 0  # 观测噪声协方差
kf.Q = 0  # 过程噪声协方差

# 使用卡尔曼滤波器估计基金的行业持仓
state_estimates = []
for _ in range(10):  # 假设有10天的数据
    kf.predict()
    kf.update(fund_daily_return)
    state_estimates.append(kf.x.copy())

# 输出估计的状态
for day, estimate in enumerate(state_estimates, 1):
    print(f"Day {day}: {estimate}")