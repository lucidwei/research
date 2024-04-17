# coding=gbk
# Time Created: 2024/4/10 15:12
# Author  : Lucid
# FileName: data_visualizer.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_prj import set_index_col_wind, split_dataframe
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


class DataVisualizer:
    def __init__(self, file_path, sheet_name):
        """
        初始化 DataVisualizer 类。

        参数:
        - file_path: Excel 文件的路径。
        - sheet_name: 要读取的工作表名称。
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.df_dict = split_dataframe(self.df)

    def plot_data(self, df1_key, df2_key, df1_col, df2_col, start_date=None, end_date=None, marker1=None, marker2=None):
        """
        绘制两个 DataFrame 中指定列的数据。

        参数:
        - df1_key: 第一个 DataFrame 的键。
        - df2_key: 第二个 DataFrame 的键。
        - df1_col: 第一个 DataFrame 中要绘制的列名。
        - df2_col: 第二个 DataFrame 中要绘制的列名。
        - start_date: 可选的起始日期,用于过滤数据。
        - end_date: 可选的结束日期,用于过滤数据。
        """
        df1 = self.df_dict[df1_key]
        df2 = self.df_dict[df2_key]

        # 检查指定的列名是否存在于对应的 DataFrame 中
        if df1_col not in df1.columns:
            raise ValueError(f"Column '{df1_col}' does not exist in DataFrame '{df1_key}'.")
        if df2_col not in df2.columns:
            raise ValueError(f"Column '{df2_col}' does not exist in DataFrame '{df2_key}'.")

        # 将两个 DataFrame 按日期索引合并,并填充缺失值
        merged_df = pd.merge(df1[[df1_col]], df2[[df2_col]], left_index=True, right_index=True, how='outer').dropna(how='all')
        merged_df.fillna(method='ffill', inplace=True)
        # merged_df.fillna(method='bfill', inplace=True)

        if start_date:
            merged_df = merged_df.loc[start_date:]
        if end_date:
            merged_df = merged_df.loc[:end_date]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # 如果是 '财务' 和 '行情' 数据的组合,调用 add_shading 方法添加阴影
        if df1_key == '财务' and df2_key == '行情':
            mask = (df1.index > pd.to_datetime('2023-01-01')) & (df1.isnull().all(axis=1))
            first_missing_financial_date = df1.loc[mask].index.min()
            self.add_shading(merged_df[df1_col], merged_df[df2_col], ax1, first_missing_financial_date)
        if df1_key == '财务' and df2_key == '基本面':
            mask = (df1.index > pd.to_datetime('2023-01-01')) & (df1.isnull().all(axis=1))
            first_missing_financial_date = df1.loc[mask].index.min()
            # 获取财务数据的最早日期，避免画出过于久远的基本面数据
            earliest_date_finance = df1.index.min() - pd.Timedelta(days=5)
            # 截取基本面数据
            merged_df = merged_df.loc[earliest_date_finance:]

            self.add_shading(merged_df[df1_col], merged_df[df2_col], ax1, first_missing_financial_date)
        if df1_key == '行情' and df2_key == '基本面':
            # 获取财务数据的最早日期
            earliest_date_finance = df1.index.min()
            # 计算基本面数据的截取日期(财务数据最早日期 - 5 日)
            cut_off_date = earliest_date_finance - pd.Timedelta(days=5)
            # 截取基本面数据
            merged_df = merged_df.loc[cut_off_date:]

            self.add_shading_quote_fundamental(merged_df[df1_col], merged_df[df2_col], ax1)

        ax1.plot(merged_df[df1_col], label=df1_col, linestyle='-', marker=marker1)
        ax2.plot(merged_df[df2_col], label=df2_col, linestyle='-', color='red', marker=marker2)

        # ax1.set_ylabel(df1_col)
        # ax2.set_ylabel(df2_col)
        ax1.set_title(self.sheet_name)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()

    def add_shading(self, finance_data, quote_data, ax, first_missing_financial_date=None):
        """
        根据财务数据和行情数据的变化趋势,在图中添加阴影。

        参数:
        - finance_data: 财务数据的 DataFrame。
        - quote_data: 行情数据的 DataFrame。
        - ax: 要添加阴影的 Axes 对象。
        """
        # 获取财务数据的季度末日期
        quarter_ends = finance_data.dropna().resample('Q').last().index
        if first_missing_financial_date is not None:
            quarter_ends = quarter_ends[quarter_ends < first_missing_financial_date]
        # # 找到第一个缺失的季末日期
        # missing_index = quarter_ends.difference(finance_data.index)
        # if not missing_index.empty:
        #     first_missing_date = missing_index[0]
        #     # 删除第一个缺失日期及其之后的所有季末日期
        #     quarter_ends = quarter_ends[quarter_ends < first_missing_date]

        # 统计红色和蓝色季度的个数
        red_count = 0
        blue_count = 0
        # 遍历每个季度
        for idx in range(len(quarter_ends)):
            if idx < len(quarter_ends) - 1:
                start_date = quarter_ends[idx]
                end_date = quarter_ends[idx + 1]

                # 计算财务数据在该季度的变化
                finance_change = finance_data.loc[end_date] - finance_data.loc[start_date]

                # 计算行情数据在该时间段内的变化
                quote_change = quote_data.loc[end_date] - quote_data.loc[start_date]

                # 判断两个数据的变化方向是否一致
                if (finance_change * quote_change) > 0:
                    # 财务数据上升且行情上涨,添加红色阴影
                    ax.axvspan(start_date, end_date, alpha=0.2, color='red')
                    red_count += 1
                elif (finance_change * quote_change) < 0:
                    # 财务数据下降且行情下跌,添加蓝色阴影
                    ax.axvspan(start_date, end_date, alpha=0.2, color='blue')
                    blue_count += 1
        # 计算红色季度的占比
        total_count = red_count + blue_count
        if total_count > 0:
            red_ratio = red_count / total_count
        else:
            red_ratio = 0

        # 打印统计结果
        print(f"红色季度数: {red_count}")
        print(f"蓝色季度数: {blue_count}")
        print(f"红色季度占比: {red_ratio:.1%}")

    def add_shading_quote_fundamental(self, quote_data, fundamental_data, ax):
        """
        根据行情数据和基本面数据的变化趋势,在图中添加阴影。

        参数:
        - quote_data: 行情数据的 DataFrame。
        - fundamental_data: 基本面数据的 DataFrame。
        - ax: 要添加阴影的 Axes 对象。
        """
        # 获取行情数据和基本面数据的交集索引
        common_index = quote_data.index.intersection(fundamental_data.index)

        # 根据交集索引对行情数据和基本面数据进行重采样
        quote_data_resampled = quote_data.loc[common_index].resample('Q').last()
        fundamental_data_resampled = fundamental_data.loc[common_index].resample('Q').last()

        # 获取重采样后的季度末日期
        quarter_ends = quote_data_resampled.index

        # 统计红色和蓝色季度的个数
        red_count = 0
        blue_count = 0

        # 遍历每个季度
        for idx in range(len(quarter_ends)):
            if idx < len(quarter_ends) - 1:
                start_date = quarter_ends[idx]
                end_date = quarter_ends[idx + 1]

                # 获取行情数据和基本面数据在该季度的起始和结束值
                quote_start = quote_data_resampled.iloc[idx]
                quote_end = quote_data_resampled.iloc[idx + 1]
                fundamental_start = fundamental_data_resampled.iloc[idx]
                fundamental_end = fundamental_data_resampled.iloc[idx + 1]

                # 检查数据是否为 NaN,如果是,则尝试获取临近的非空数值
                if pd.isnull(quote_start):
                    quote_start_ts = quote_data_resampled.iloc[:idx].last_valid_index()
                    if quote_start_ts is not None:
                        quote_start = quote_data_resampled.loc[quote_start_ts]
                if pd.isnull(quote_end):
                    quote_end_ts = quote_data_resampled.iloc[idx + 1:].first_valid_index()
                    if quote_end_ts is not None:
                        quote_end = quote_data_resampled.loc[quote_end_ts]
                if pd.isnull(fundamental_start):
                    fundamental_start_ts = fundamental_data_resampled.iloc[:idx].last_valid_index()
                    if fundamental_start_ts is not None:
                        fundamental_start = fundamental_data_resampled.loc[fundamental_start_ts]
                if pd.isnull(fundamental_end):
                    fundamental_end_ts = fundamental_data_resampled.iloc[idx + 1:].first_valid_index()
                    if fundamental_end_ts is not None:
                        fundamental_end = fundamental_data_resampled.loc[fundamental_end_ts]

                # 计算行情数据和基本面数据在该季度的变化
                if quote_start is not None and quote_end is not None and fundamental_start is not None and fundamental_end is not None:
                    quote_change = quote_end - quote_start
                    fundamental_change = fundamental_end - fundamental_start

                    # 判断两个数据的变化方向是否一致
                    if (quote_change * fundamental_change) > 0:
                        # 行情数据上升且基本面数据上升,添加红色阴影
                        ax.axvspan(start_date, end_date, alpha=0.2, color='red')
                        red_count += 1
                    elif (quote_change * fundamental_change) < 0:
                        # 行情数据下降且基本面数据下降,添加蓝色阴影
                        ax.axvspan(start_date, end_date, alpha=0.2, color='blue')
                        blue_count += 1

        # 计算红色季度的占比
        total_count = red_count + blue_count
        if total_count > 0:
            red_ratio = red_count / total_count
        else:
            red_ratio = 0

        # 打印统计结果
        print(f"红色季度数: {red_count}")
        print(f"蓝色季度数: {blue_count}")
        print(f"红色季度占比: {red_ratio:.2%}")


def analyze_industry(visualizer, commodity_price_col, start_date=None, end_date=None):
    # 财务数据与基本面数据的关系
    # visualizer.plot_data('财务', '基本面', '营业收入同比增长率', commodity_price_col)
    # visualizer.plot_data('财务', '基本面', '归属母公司股东的净利润同比增长率', commodity_price_col)
    # visualizer.plot_data('财务', '基本面', '净资产收益率ROE', commodity_price_col)

    # 财务数据与行情数据的关系
    visualizer.plot_data('财务', '行情', '营业收入同比增长率', '收盘价', marker1=None, marker2=None, start_date=start_date)
    # visualizer.plot_data('财务', '行情', '归属母公司股东的净利润同比增长率', '收盘价', marker1='o', marker2=None, start_date=start_date)
    visualizer.plot_data('财务', '行情', '净资产收益率ROE', '收盘价', marker1=None, marker2=None, start_date=start_date)

    # 行情数据与基本面数据的关系
    # visualizer.plot_data('行情', '基本面', '收盘价', commodity_price_col, start_date=start_date, end_date=end_date)

file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\景气研究\行业景气数据库与展示.xlsx"

# 石油石化行业分析
visualizer = DataVisualizer(file_path, '石油石化')
analyze_industry(visualizer, '现货价:原油:英国布伦特Dtd')
# visualizer = DataVisualizer(file_path, '石油石化')
# analyze_industry(visualizer, '现货价:原油:英国布伦特Dtd', start_date='2020-01-02')
# visualizer = DataVisualizer(file_path, '石油石化')
# analyze_industry(visualizer, '现货价:原油:英国布伦特Dtd', start_date='2014-10-02', end_date='2020-03-02')

visualizer = DataVisualizer(file_path, '煤炭')
analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)', start_date='2020-01-02')
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)', start_date='2014-10-02', end_date='2020-03-02')

# visualizer = DataVisualizer(file_path, '有色金属')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')

visualizer = DataVisualizer(file_path, '基础化工')
analyze_industry(visualizer, '中国化工产品价格指数')

visualizer = DataVisualizer(file_path, '钢铁')
analyze_industry(visualizer, '中国:价格:螺纹钢(HRB400,20mm)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
#
# visualizer = DataVisualizer(file_path, '煤炭')
# analyze_industry(visualizer, '秦皇岛港:平仓价:动力煤(Q5000K)')
