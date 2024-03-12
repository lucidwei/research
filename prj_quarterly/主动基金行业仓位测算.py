import glob, os, re
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr

from WindPy import w

from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase


class CalcFundPosition(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig, initial_q_str: str, calibrate: bool):
        super().__init__(base_config)
        self.initial_q_str = initial_q_str
        self.calibrate = calibrate
        self.config_quarterly_dates()
        self.load_asset_positions()
        self.load_industry_return()
        self.load_fund_return()
        self.load_industry_positions()
        self.prepare_data()

    def config_quarterly_dates(self):
        self.quarterly_dates = w.tdays('2018-11-20', '2024-03-01', "Period=Q").Data[0]
        self.quarterly_dates_str = [str(x) for x in self.quarterly_dates]

        # 四个季度的末尾月份，用于确定季度
        date_dict = {}
        quarter_end_months = {3: 'q1', 6: 'q2', 9: 'q3', 12: 'q4'}

        # 遍历日期列表，并构建字典
        for date in self.quarterly_dates:
            # 获取年份的后两位和月份
            year = date.year % 100  # 获取年份的后两位数字
            month = date.month
            # 确定季度
            quarter = quarter_end_months.get(month, '')
            # 构建字典的键
            key = f"{year}{quarter}"
            # 添加到字典中
            date_dict[key] = date

        self.quarterly_dates_dict = date_dict
        self.初始持仓日期 = date_dict[self.initial_q_str]

    def load_asset_positions(self):
        """
        数据源：wind-基金-资产配置-资产配置(汇总)
        要把左下角调整为（普通股票、偏股混合、灵活配置）
        """
        pattern = rf"{self.base_config.excels_path}基金仓位测算\资产配置(汇总)*.xlsx"

        # 使用glob.glob找到所有匹配的文件
        files = glob.glob(pattern)

        # 创建一个空字典来存储DataFrame
        asset_position_dfs = {}
        quarterly_positions = {}

        # 遍历文件列表，读取每个文件到DataFrame，并以文件名作为字典的键
        for file in files:
            # 使用os.path.basename获取文件的基本名称（包括扩展名）
            basename = os.path.basename(file)
            # 提取文件名（无扩展名）作为字典键
            key = os.path.splitext(basename)[0]
            # 使用正则表达式提取季度字符串
            match = re.search(r'汇总\)(.*?)\.xlsx', basename)
            matched_string = match.group(1)  # 这将是 '22q4', '23q1', 等等
            # 读取文件到DataFrame
            df = pd.read_excel(file)
            # 预处理
            df = df.iloc[:-2, 1:]
            # 设置第0行为列名
            df.columns = df.iloc[0]
            # 删除原来的第0行
            df = df.drop(df.index[0])
            # 设置第0列为索引
            df = df.set_index('资产科目')
            asset_position_dfs[key] = df  # 更新字典中的DataFrame

            positions = {
                'A股': df.loc['  其中：A股', '占净值比(%)'] / 100,
                '港股': df.loc['股票', '占净值比(%)'] / 100 - df.loc['  其中：A股', '占净值比(%)'] / 100,
                '债券': df.loc['债券', '占净值比(%)'] / 100,
                '资产净值(亿元)': df.loc['资产净值', '市值(万元)'] / 1e4,
            }
            positions['现金'] = 1 - positions['A股'] - positions['港股'] - positions['债券']
            quarterly_positions[matched_string] = positions

        # 打印出所有的DataFrame的键（即文件名），确认已成功加载
        print(asset_position_dfs.keys())
        self.quarterly_positions = quarterly_positions

    def load_industry_positions(self):
        """
        数据源：wind-基金-资产配置-行业分布(汇总 第三方)
        要把左下角调整为（普通股票、偏股混合、灵活配置）
        """
        pattern = rf"{self.base_config.excels_path}基金仓位测算\行业分布(汇总 第三方)*.xlsx"

        # 使用glob.glob找到所有匹配的文件
        files = glob.glob(pattern)

        # 创建一个空字典来存储DataFrame
        industry_position_series = {}

        # 遍历文件列表，读取每个文件到DataFrame，并以文件名作为字典的键
        for file in files:
            # 使用os.path.basename获取文件的基本名称（包括扩展名）
            basename = os.path.basename(file)

            # 使用正则表达式提取季度字符串
            match = re.search(r'第三方\)(.*?)\.xlsx', basename)
            matched_string = match.group(1)  # 这将是 '22q4', '23q1', 等等

            # 读取文件到DataFrame
            df = pd.read_excel(file)
            # 预处理
            df = df.iloc[:-2, 2:]
            # 设置第0行为列名
            df.columns = df.iloc[0]
            # 删除原来的第0行
            df = df.drop(df.index[0])
            industry_position_series[matched_string] = pd.Series(df['占股票投资市值比(%)'].values, index=df['行业名称'])

        # 利用季度资产配置比例校准
        for quarter in industry_position_series.keys():
            industry_position_series[quarter] *= 0.01 * self.quarterly_positions[quarter]['A股']
            industry_position_series[quarter]['债券'] = self.quarterly_positions[quarter]['债券']
            # industry_position_series[quarter]['现金'] = self.quarterly_positions[quarter]['现金']
            industry_position_series[quarter]['港股'] = self.quarterly_positions[quarter]['港股']
            industry_position_series[quarter]['现金'] = 1 - industry_position_series[quarter].sum()

        self.industry_position_series = industry_position_series

    def load_industry_return(self):
        file_path = rf"{self.base_config.excels_path}基金仓位测算\基金净值和指数.xlsx"
        行业收盘价 = pd.read_excel(file_path, sheet_name='行业收盘价')
        # 处理“行业收盘价”sheet
        行业收盘价.columns = 行业收盘价.iloc[1].str.replace("\(中信\)", "", regex=True)
        行业收盘价 = 行业收盘价.drop(index=[0, 1]).iloc[:, 1:].reset_index(drop=True)
        行业收盘价 = 行业收盘价.rename(columns={
            行业收盘价.columns[0]: "日期",
            '恒生科技': "港股",
            '中债-总财富(总值)指数': "债券",
        })
        行业收盘价 = 行业收盘价.dropna(axis=0).set_index('日期').sort_index()
        行业收盘价 = 行业收盘价.apply(pd.to_numeric, errors='coerce')
        self.industry_index = 行业收盘价.copy(deep=True)
        # self.industry_index.columns = [f"{col}指数" for col in self.industry_index.columns]

        行业日度收益率 = 行业收盘价.pct_change()
        行业日度收益率['现金'] = 0
        行业日度收益率.index = pd.to_datetime(行业日度收益率.index)
        self.行业日度收益率 = 行业日度收益率

    def load_fund_return(self):
        file_path = rf"{self.base_config.excels_path}基金仓位测算\基金净值和指数.xlsx"

        index_data = pd.read_excel(file_path, header=1, sheet_name='基金指数').dropna(axis=0).set_index(
            '日期').sort_index()
        weight_data = pd.read_excel(file_path, header=0, sheet_name='权重')

        # 排序权重数据，确保日期顺序正确
        weight_data.sort_values('权重(基金专题-市场概况-基金市场概况-占比)', inplace=True)

        # 计算指数复合列
        index_data['指数复合'] = None  # 初始化指数复合列

        # 对于指数数据中的每个日期，寻找合适的权重
        for date, row in index_data.iterrows():
            # 找到小于当前日期的最大权重日期
            weight_row = weight_data[weight_data['权重(基金专题-市场概况-基金市场概况-占比)'] <= date].iloc[-1]

            # 计算指数复合
            index_data.at[date, '指数复合'] = (
                                                      row['普通股票型基金指数'] * weight_row['普通股票型基金指数'] +
                                                      row['偏股混合型基金指数'] * weight_row['偏股混合型基金指数'] +
                                                      row['灵活配置型基金指数'] * weight_row['灵活配置型基金指数']
                                              ) / (weight_row['普通股票型基金指数'] + weight_row['偏股混合型基金指数'] +
                                                   weight_row['灵活配置型基金指数'])

        index_data['日度收益率'] = index_data['指数复合'].pct_change()
        self.total_return = index_data

    def prepare_data(self):
        # 对齐日期
        self.industry_return = self.行业日度收益率.loc[self.行业日度收益率.index > self.初始持仓日期]
        self.total_return = self.total_return.loc[self.total_return.index > self.初始持仓日期]

    def estimate_R(self, fund_daily_return):
        # 计算基金日度收益率的方差
        R = np.var(fund_daily_return)
        return np.array([[R]])  # 返回一个形状为1x1的矩阵

    def adjust_Q_for_low_initial_holdings(self, Q, initial_holdings_ratio, threshold=0.01, adjustment_factor=100):
        """
        调整过程噪声协方差Q以增加初始持仓占比较少的行业的变化空间。

        :param Q: 原始的过程噪声协方差矩阵。
        :param initial_holdings_ratio: 初始持仓比例，一个字典或类似结构，键为行业名称，值为比例。
        :param threshold: 用于判断持仓占比是否“较少”的阈值。
        :param adjustment_factor: 调整因子，增加的Q值将乘以此因子。
        :return: 调整后的Q矩阵。
        """
        for i in range(len(Q)):
            if initial_holdings_ratio[i] < threshold:
                Q[i, i] *= adjustment_factor
        return Q

    def estimate_Q(self, industry_daily_return, initial_holdings_ratio):
        """
        根据行业日度收益率计算过程噪声协方差矩阵，并调整初始持仓占比较少的行业。

        :param industry_daily_return: 行业日度收益率的DataFrame。
        :param initial_holdings_ratio: 初始持仓比例，一个字典或DataFrame列，键/索引为行业名称，值为比例。
        :return: 调整后的过程噪声协方差矩阵Q。
        """
        # 计算行业日度收益率的协方差矩阵
        cov_matrix = industry_daily_return.cov()
        # 取对角线元素形成对角矩阵，并乘以小系数
        Q = np.diag(np.diag(cov_matrix)) * 0.01
        # 调整Q以反映初始持仓占比较少的行业
        Q = self.adjust_Q_for_low_initial_holdings(Q, initial_holdings_ratio)
        return Q

    def post_constraint_kf(self, industry_daily_return, fund_daily_return):
        industry_amount = len(industry_daily_return.columns)

        self.initial_holdings_ratio = self.industry_position_series[self.initial_q_str].reindex(
            industry_daily_return.columns, fill_value=0)
        for key, series in self.industry_position_series.items():
            self.industry_position_series[key] = series.reindex(industry_daily_return.columns, fill_value=0)

        kf = KalmanFilter(dim_x=industry_amount, dim_z=1)  # 初始化卡尔曼滤波器

        # 定义初始状态 (行业持仓比例)
        kf.x = self.initial_holdings_ratio.to_numpy()
        cash_index = self.initial_holdings_ratio.index.get_loc("现金")
        bond_index = self.initial_holdings_ratio.index.get_loc("债券")

        # 定义状态转移矩阵
        kf.F = np.eye(industry_amount)  # transition_matrices

        # # 定义状态协方差
        # kf.P *= 1e-2
        # # 由于我们假设没有噪声，这里将测量噪声和过程噪声设置得很小
        # kf.R = np.array([[1e-4]])  # 观测噪声协方差
        # kf.Q = np.eye(industry_amount) * 1e-6

        # 定义状态协方差
        kf.P *= 1e-2
        kf.R = self.estimate_R(fund_daily_return['日度收益率'])  # 根据基金日度收益率波动性动态估计观测噪声
        kf.Q = self.estimate_Q(industry_daily_return, self.initial_holdings_ratio)  # 根据行业持仓比例变化的历史波动性动态估计过程噪声

        # 准备观测数据
        measurements = fund_daily_return['日度收益率'].dropna().to_numpy()

        state_estimates = []
        return_errors = []
        self.pre_calibration_positions = {}
        self.post_calibration_positions = {}
        full_calibrate_dict = {k: v for k, v in self.quarterly_dates_dict.items() if "q2" in k or "q4" in k}
        full_calibrate_dict.pop("23q4", None)  # 23q4年报还没出
        swapped_dict = {value: key for key, value in self.quarterly_dates_dict.items()}

        alpha = 0.9  # 平滑系数，用于调整当前估计与前一天估计的权重
        previous_state = kf.x.copy()
        for measurement, returns in zip(measurements, industry_daily_return.dropna().iterrows()):
            date, return_ = returns

            # 更新观测矩阵为当日各行业收益率
            kf.H = return_.values.reshape(1, -1)

            kf.predict()
            kf.update(measurement)

            smoothed_state = alpha * kf.x + (1 - alpha) * previous_state
            previous_state = smoothed_state.copy()

            # 应用约束：非负和总和为1
            # if (kf.x < 0).any():
            #     print(f'{_date}出现负数, sum{sum(kf.x)}')
            constrained_state = np.maximum(smoothed_state, 0.0001)  # 设置最小持仓比例
            constrained_state /= np.sum(constrained_state)

            # 定期校准
            if self.calibrate:
                if date.to_pydatetime() in self.quarterly_dates_dict.values():
                    q_str = swapped_dict[date.to_pydatetime()]
                    self.pre_calibration_positions[q_str] = pd.Series(constrained_state.copy(),
                                                                      index=self.initial_holdings_ratio.index)
                    # Q2和Q4的完全校准
                    if date.to_pydatetime() in full_calibrate_dict.values():
                        constrained_state = self.industry_position_series[q_str].to_numpy()
                    # Q1和Q3的部分校准，只更新现金和债券
                    else:
                        scaling_factor = ((1 - self.quarterly_positions[q_str]['现金'] -
                                           self.quarterly_positions[q_str]['债券'])
                                          / (1 - constrained_state[cash_index] - constrained_state[bond_index]))
                        for i in range(len(constrained_state)):
                            if i not in [cash_index, bond_index]:
                                constrained_state[i] *= scaling_factor
                        constrained_state[cash_index] = self.quarterly_positions[q_str]['现金']
                        constrained_state[bond_index] = self.quarterly_positions[q_str]['债券']
                    self.post_calibration_positions[q_str] = pd.Series(constrained_state.copy(),
                                                                       index=self.initial_holdings_ratio.index)
                    kf.x = constrained_state
            # kf.x = constrained_state

            return_error = (constrained_state * return_).sum() - measurement
            # print(f"return_error: {100*return_error}%")

            state_estimates.append(constrained_state.copy())
            return_errors.append(100 * round(return_error, 4))

        dates = fund_daily_return.index
        state_estimates_df = pd.DataFrame(state_estimates, index=dates, columns=industry_daily_return.columns)
        return_errors_df = pd.Series(return_errors, index=dates, name='日度收益率误差')

        return state_estimates_df, return_errors_df

    def generate_noisy_holdings(self, initial_holdings_ratio, industry_daily_return, fund_daily_return,
                                noise_scale=1e-3):
        """
        在初始持仓比例基础上添加随机噪声来模拟持仓比例的波动。

        :param initial_holdings_ratio: DataFrame, 各行业的初始持仓占比
        :param industry_daily_return: DataFrame, 各行业的日度收益率
        :param noise_scale: float, 噪声的标准差
        :return: 含噪声的每日行业持仓比例
        """
        # 使用DataFrame以保持与行业日度收益率的维度一致
        noisy_holdings = pd.DataFrame(0, index=industry_daily_return.index, columns=industry_daily_return.columns)
        # 遍历每个行业，为初始持仓比例添加随机噪声
        for column in industry_daily_return.columns:
            # 生成噪声
            noise = np.random.normal(loc=0, scale=noise_scale, size=len(industry_daily_return))
            # 应用噪声到初始持仓比例，并确保比例非负
            noisy_holdings[column] = initial_holdings_ratio[column] + noise
            noisy_holdings[column] = noisy_holdings[column].clip(lower=0)

        # 归一化每一行的持仓比例和为1
        noisy_holdings_sum = noisy_holdings.sum(axis=1)
        noisy_holdings = noisy_holdings.div(noisy_holdings_sum, axis=0)

        estimated_returns = (noisy_holdings * industry_daily_return).sum(axis=1)
        return_errors = fund_daily_return['日度收益率'] - estimated_returns
        return_errors_df = pd.Series(return_errors * 100, index=fund_daily_return.index, name='日度收益率误差')

        return noisy_holdings, return_errors_df

    def dynamic_lasso_estimate_positions(self, industry_daily_return, fund_daily_return, initial_period=10,
                                         window_size=30, alpha=3e-6):
        # 确保数据按日期对齐并去除缺失值
        data = pd.concat([fund_daily_return['日度收益率'], industry_daily_return], axis=1).dropna()

        # 初始化存储每日估算持仓占比的DataFrame
        daily_positions = pd.DataFrame(index=data.index, columns=industry_daily_return.columns)

        # 遍历每一天，动态增加样本点进行Lasso回归
        for i in range(initial_period, len(data)):
            # 确定滚动窗口的起始和结束位置
            start_idx = max(0, i - window_size)
            end_idx = i

            # 使用当前窗口内的数据
            Y = data.iloc[start_idx:end_idx, 0].values  # 基金的日度收益率
            X = data.iloc[start_idx:end_idx, 1:].values  # 各行业的日度收益率

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

        estimated_returns = (daily_positions * industry_daily_return).sum(axis=1)
        return_errors = fund_daily_return['日度收益率'] - estimated_returns
        return_errors_df = pd.Series(return_errors * 100, index=fund_daily_return.index, name='日度收益率误差')

        return daily_positions, return_errors_df

    def calculate_active_adjustment(self, state_estimates_df, industry_daily_return):
        # 初始化主动调仓的DataFrame
        active_adjustments = pd.DataFrame(0, index=state_estimates_df.index, columns=state_estimates_df.columns)

        # 遍历每一天，计算主动调仓
        for i in range(0, len(state_estimates_df)):
            # 获取前一天的行业持仓占比
            if i == 0:
                previous_holdings = self.initial_holdings_ratio
            else:
                previous_holdings = state_estimates_df.iloc[i - 1]
            # 获取当天的行业收益率
            current_returns = industry_daily_return.iloc[i]
            # 计算市场波动导致的持仓变动
            market_movements = previous_holdings * (1 + current_returns)
            market_movements /= market_movements.sum()  # 归一化

            # 获取当天的实际行业持仓占比
            current_holdings = state_estimates_df.iloc[i]

            # 主动调仓是实际持仓占比与市场波动后持仓占比的差异
            active_adjustment = current_holdings - market_movements
            active_adjustments.iloc[i] = active_adjustment

        self.active_adjustments = active_adjustments
        return active_adjustments

    def calculate_active_adjustment_amount(self, active_adjustments):
        data = [(self.quarterly_dates_dict[q], v['资产净值(亿元)']) for q, v in self.quarterly_positions.items()]
        asset_total = pd.DataFrame(data, columns=['date', 'assets'])
        asset_total = asset_total.set_index('date')
        asset_total_origin = asset_total.copy(deep=True)
        # 重新索引 asset_df 以匹配 active_adjustments 的索引
        asset_total = asset_total.reindex(active_adjustments.index, method='ffill')
        # 两个 DataFrame 相乘
        active_adjustments_amount = active_adjustments.multiply(asset_total['assets'], axis='index')

        self.active_adjustments_amount = active_adjustments_amount
        self.asset_total_base = asset_total_origin
        return active_adjustments_amount

    def evaluate_model(self, pre_calibration_positions, post_calibration_positions):
        pre_calibration_positions = pre_calibration_positions.astype(float)
        post_calibration_positions = post_calibration_positions.astype(float)
        # 初始化存储评估结果的字典
        evaluation_results = {
            'overlap_count': [],  # 重合的行业数
            'spearman_corr': [],  # 斯皮尔曼相关性系数
            'change_spearman_corr': [np.nan]  # 加仓行业排名变化的斯皮尔曼相关性系数
        }

        # 遍历所有季度
        for quarter in pre_calibration_positions.columns:
            # 获取预估和实际的前十仓位行业
            estimated_top10 = pre_calibration_positions[quarter].nlargest(10)
            actual_top10 = post_calibration_positions[quarter].nlargest(10)

            # 计算重合的行业数
            overlap_count = len(set(estimated_top10.index) & set(actual_top10.index))
            evaluation_results['overlap_count'].append(overlap_count)

            # 计算趋势一致性 - 斯皮尔曼相关性系数
            # 获取预估和实际的所有行业排名
            estimated_rankings = pre_calibration_positions[quarter].rank(ascending=False)
            actual_rankings = post_calibration_positions[quarter].rank(ascending=False)

            # 计算斯皮尔曼相关性系数
            corr, _ = spearmanr(estimated_rankings, actual_rankings)
            evaluation_results['spearman_corr'].append(corr)

        # 遍历除了第一个季度之外的所有季度
        quarters = pre_calibration_positions.columns
        for i in range(1, len(quarters)):
            prev_quarter = quarters[i - 1]
            current_quarter = quarters[i]

            # 计算实际和估计的仓位变化
            actual_changes = post_calibration_positions[current_quarter] - post_calibration_positions[prev_quarter]
            estimated_changes = pre_calibration_positions[current_quarter] - post_calibration_positions[prev_quarter]

            # 获取变化的排名
            actual_rankings = actual_changes.rank(ascending=False)
            estimated_rankings = estimated_changes.rank(ascending=False)

            # 计算斯皮尔曼相关性系数
            corr, _ = spearmanr(estimated_rankings, actual_rankings)
            evaluation_results['change_spearman_corr'].append(corr)

        # 将评估结果转换为DataFrame
        evaluation_df = pd.DataFrame(evaluation_results, index=pre_calibration_positions.columns)


        return evaluation_df


# 定义列的顺序
columns = [
    '有色金属', '石油石化', '煤炭', '基础化工', '钢铁', '建材', '建筑', '电力设备及新能源', '机械', '国防军工',
    '电力及公用事业', '轻工制造', '医药', '食品饮料', '家电', '消费者服务', '汽车', '交通运输', '商贸零售', '农林牧渔',
    '纺织服装', '电子', '计算机', '传媒', '通信', '房地产', '银行', '非银行金融', '综合金融', '综合', '港股', '债券',
    '现金'
]

if __name__ == "__main__":
    base_config = BaseConfig('quarterly')
    obj = CalcFundPosition(base_config, initial_q_str='21q4', calibrate=True)

    state_estimates_post, return_errors = obj.post_constraint_kf(obj.industry_return, obj.total_return)
    return_errors_abs_mean = sum(abs(x) for x in return_errors.tolist()) / len(return_errors)
    print(f'return_errors_abs_mean:{return_errors_abs_mean}')

    active_adjustments = obj.calculate_active_adjustment(state_estimates_post, obj.industry_return)
    active_adjustments_cumsum = active_adjustments.cumsum()
    active_adjustments_amount = obj.calculate_active_adjustment_amount(active_adjustments)

    state_estimates_noise, return_errors_noise = obj.generate_noisy_holdings(
        obj.industry_position_series[obj.initial_q_str], obj.industry_return, obj.total_return)
    return_errors_noise_abs_mean = sum(abs(x) for x in return_errors_noise.tolist()) / len(return_errors_noise)
    print(f'return_errors_noise_abs_mean:{return_errors_noise_abs_mean}')

    state_estimates_lasso, return_errors_lasso = obj.dynamic_lasso_estimate_positions(obj.industry_return,
                                                                                      obj.total_return)
    return_errors_lasso_abs_mean = sum(abs(x) for x in return_errors_lasso.tolist()) / len(return_errors_lasso)
    print(f'return_errors_lasso_abs_mean:{return_errors_lasso_abs_mean}')

    return_abs_mean = 100 * sum(abs(x) for x in obj.total_return['日度收益率'].tolist()) / len(
        obj.total_return['日度收益率'])
    print(f'return_abs_mean:{return_abs_mean}')

    # res_start = obj.industry_position_series['22q4']
    # res_estimate = state_estimates_post.loc[pd.Timestamp('2023-06-30 00:00:00')]
    # res_real = obj.industry_position_series['23q2']

    # real_change = res_real - res_start
    # estimate_change = res_estimate - res_start
    # real_change = real_change.abs().mean()
    # estimate_change = estimate_change.abs().mean()
    #
    # error = res_estimate - res_real
    # error_abs = error.abs().mean()

    post_calibration_positions = pd.DataFrame.from_dict(obj.post_calibration_positions)
    pre_calibration_positions = pd.DataFrame.from_dict(obj.pre_calibration_positions)
    position_error = pre_calibration_positions - post_calibration_positions
    evaluation_results = obj.evaluate_model(pre_calibration_positions, post_calibration_positions)

    # 按指定顺序排列列
    state_estimates_post = state_estimates_post.reindex(columns=columns)
    active_adjustments_cumsum = active_adjustments_cumsum.reindex(columns=columns)
    active_adjustments_amount = active_adjustments_amount.reindex(columns=columns)
    obj.industry_index = obj.industry_index.reindex(columns=columns)
    obj.industry_index.columns = [f"{col}指数" for col in obj.industry_index.columns]
    post_calibration_positions = post_calibration_positions.reindex(index=columns)
    pre_calibration_positions = pre_calibration_positions.reindex(index=columns)
    position_error = position_error.reindex(index=columns)

    file_path = rf"{obj.base_config.excels_path}基金仓位测算\全基金仓位测算自21q4(展示).xlsx"
    with pd.ExcelWriter(file_path) as writer:
        state_estimates_post.to_excel(writer, sheet_name='卡尔曼滤波结果', index=True)
        # active_adjustments.to_excel(writer, sheet_name='主动调仓(全部持仓占比)', index=True)
        active_adjustments_cumsum.to_excel(writer, sheet_name='主动调仓累积(全部持仓占比)', index=True)
        active_adjustments_cumsum.rolling(window=5).mean().to_excel(writer, sheet_name='主动调仓累积(ma5)', index=True)
        active_adjustments_amount.to_excel(writer, sheet_name='主动调仓资金流(亿元)', index=True)
        obj.industry_index.to_excel(writer, sheet_name='行业指数', index=True)
        obj.asset_total_base.to_excel(writer, sheet_name='基期资产市值', index=True)
        # 使用 pd.concat 横向合并这些 Series
        # res_start_sorted = res_start.sort_values(ascending=False).reset_index(name='22q4实际仓位')
        # res_real_sorted = res_real.sort_values(ascending=False).reset_index(name='23q2实际仓位')
        # res_estimate_sorted = res_estimate.sort_values(ascending=False).reset_index(name='23q2测算仓位')
        # combined_df = pd.concat([res_start_sorted, res_real_sorted, res_estimate_sorted], axis=1)
        # combined_df.to_excel(writer, sheet_name='仓位截面(排序后)', index=False)

        pre_calibration_positions.to_excel(writer, sheet_name='校准前仓位', index=True)
        post_calibration_positions.to_excel(writer, sheet_name='校准后仓位', index=True)
        position_error.to_excel(writer, sheet_name='校准前后仓位误差', index=True)

        return_errors.to_excel(writer, sheet_name='日度收益误差', index=True)
        return_errors_noise.to_excel(writer, sheet_name='日度收益误差noise', index=True)
        # state_estimates_noise.to_excel(writer, sheet_name='噪声结果', index=True)
        # state_estimates_lasso.to_excel(writer, sheet_name='lasso结果', index=True)
