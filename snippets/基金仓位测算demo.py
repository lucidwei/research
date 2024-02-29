import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import Lasso

# 路径 - 根据你的文件路径进行调整
# file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\单基金仓位测算 - 副本.xlsx"
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\单基金仓位测算22q4 - 副本.xlsx"

# 读取Excel文件的三个sheet
中报持仓 = pd.read_excel(file_path, sheet_name='中报持仓')
基金净值 = pd.read_excel(file_path, sheet_name='基金净值')
行业收盘价 = pd.read_excel(file_path, sheet_name='行业收盘价')

# 处理“中报持仓”sheet
初始持仓日期 = 中报持仓.iloc[0, 1]
初始持仓日期 = pd.to_datetime(初始持仓日期)
中报持仓_useful = 中报持仓.iloc[4:].rename(columns={
    "Unnamed: 7": "占净值比(%)",
    "Unnamed: 10": "中信行业"
})
中报持仓_useful["占净值比(%)"] = 中报持仓_useful["占净值比(%)"].astype(float)
行业持仓占比 = 中报持仓_useful.groupby("中信行业")["占净值比(%)"].sum() / 100
总占比 = 行业持仓占比.sum()
现金持仓占比 = 1 - 总占比 if 总占比 < 1 else 0
行业持仓占比.loc['现金'] = 现金持仓占比
行业持仓占比 = 行业持仓占比.drop(0)

# 处理“基金净值”sheet
基金净值_useful = 基金净值.iloc[:, 1:].dropna()
基金净值_useful.columns = ['日期', '单位净值']
基金净值_useful['日度收益率'] = 基金净值_useful['单位净值'].pct_change()

# 处理“行业收盘价”sheet
行业收盘价.columns = 行业收盘价.iloc[1].str.replace("\(中信\)", "", regex=True)
行业收盘价 = 行业收盘价.drop(index=[0, 1]).reset_index(drop=True)
行业收盘价 = 行业收盘价.rename(columns={行业收盘价.columns[1]: "日期"})
行业收盘价 = 行业收盘价.dropna(axis=1).set_index('日期')
行业收盘价 = 行业收盘价.apply(pd.to_numeric, errors='coerce')
行业日度收益率 = 行业收盘价.pct_change()
行业日度收益率['现金'] = 0

# 打印结果示例
print(行业持仓占比.head())  # 打印行业持仓占比的前几行
print(f"现金持仓占比: {现金持仓占比*100:.2f}%")
print(基金净值_useful.head())


# 确保索引为DatetimeIndex
行业日度收益率.index = pd.to_datetime(行业日度收益率.index)
基金净值_useful.index = pd.to_datetime(基金净值_useful['日期'])
行业日度收益率_filtered = 行业日度收益率.loc[行业日度收益率.index > 初始持仓日期]
# 截取基金净值_useful中索引大于等于2022-12-31的部分
基金净值_useful_filtered = 基金净值_useful.loc[基金净值_useful.index > 初始持仓日期]


def post_constraint_kf(initial_holdings_ratio, industry_daily_return, fund_daily_return):
    industry_amount = len(industry_daily_return.columns)

    aligned_holdings_ratio = initial_holdings_ratio.reindex(industry_daily_return.columns, fill_value=0)

    kf = KalmanFilter(dim_x=industry_amount, dim_z=1) # 初始化卡尔曼滤波器

    # 定义初始状态 (行业持仓比例)
    kf.x = aligned_holdings_ratio.to_numpy()

    # 定义状态转移矩阵
    kf.F = np.eye(industry_amount)  # transition_matrices

    # 定义状态协方差
    # kf.P *= 0.1
    # 由于我们假设没有噪声，这里将测量噪声和过程噪声设置得很小
    kf.R = np.array([[1e-3]])  # 观测噪声协方差
    kf.Q = np.eye(industry_amount) * 1e-5

    # 准备观测数据
    measurements = fund_daily_return['日度收益率'].dropna().to_numpy()

    state_estimates = []
    for measurement, returns in zip(measurements, industry_daily_return.dropna().iterrows()):
        _date, return_ = returns

        # 更新观测矩阵为当日各行业收益率
        kf.H = return_.values.reshape(1, -1)

        kf.predict()
        kf.update(measurement)

        # 应用约束：非负和总和为1
        constrained_state = np.maximum(kf.x, 0)  # 保持非负
        # 如果所有的持仓比例都非常小，则进行归一化处理
        if np.sum(constrained_state) > 1e-4:  # 设置一个阈值，例如1e-4
            constrained_state = constrained_state / np.sum(constrained_state)
        else:
            # 如果所有的持仓比例都接近于零，则可能是估计出现了问题
            # 在这种情况下，我们可以选择维持原始的持仓比例，或者使用其他方法处理
            constrained_state = aligned_holdings_ratio.to_numpy()

        state_estimates.append(constrained_state.copy())

    dates = fund_daily_return.dropna().日期.values
    state_estimates_df = pd.DataFrame(state_estimates, index=dates, columns=industry_daily_return.columns)

    return state_estimates_df

state_estimates_post = post_constraint_kf(行业持仓占比, 行业日度收益率_filtered, 基金净值_useful_filtered)


def dynamic_lasso_estimate_positions(fund_daily_return, industry_daily_return, initial_period=10, alpha=3e-6):
    # 确保数据按日期对齐并去除缺失值
    data = pd.concat([fund_daily_return['日度收益率'], industry_daily_return], axis=1).dropna()

    # 初始化存储每日估算持仓占比的DataFrame
    daily_positions = pd.DataFrame(index=data.index, columns=industry_daily_return.columns)

    # 遍历每一天，动态增加样本点进行Lasso回归
    for i in range(initial_period, len(data)):
        # 使用当前窗口内的数据
        Y = data.iloc[:i, 0].values  # 基金的日度收益率
        X = data.iloc[:i, 1:].values  # 各行业的日度收益率

        # 初始化Lasso回归模型
        lasso = Lasso(alpha=alpha, max_iter=10000)

        # 拟合模型
        lasso.fit(X, Y)

        # 获取系数（即每个行业对基金收益率的影响）
        coefficients = lasso.coef_

        # 更新当天的持仓占比估算结果
        daily_positions.iloc[i] = coefficients

    # 处理持仓占比数据，确保所有系数非负且和为1
    daily_positions = daily_positions.apply(lambda x: np.maximum(x, 0), axis=1)  # 确保非负
    daily_positions = daily_positions.div(daily_positions.sum(axis=1), axis=0)  # 归一化

    return daily_positions


# 示例调用
daily_positions_lasso = dynamic_lasso_estimate_positions(基金净值_useful_filtered, 行业日度收益率_filtered,
                                                         initial_period=10, alpha=3e-6)

file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\单基金仓位测算自22q4-结果对比.xlsx"
with pd.ExcelWriter(file_path) as writer:
    # 将 df1 保存到第一个 sheet 中
    state_estimates_post.to_excel(writer, sheet_name='Kf', index=True)

    # 将 df2 保存到第二个 sheet 中
    daily_positions_lasso.to_excel(writer, sheet_name='Lasso', index=True)
    行业持仓占比.to_excel(writer, sheet_name='初始行业持仓占比', index=True)