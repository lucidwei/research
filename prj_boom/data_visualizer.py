# coding=gbk
# Time Created: 2024/4/10 15:12
# Author  : Lucid
# FileName: data_visualizer.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_index_col_wind
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
        self.df_dict = self._split_dataframe()

    def _split_dataframe(self):
        """
        根据空列将 DataFrame 分割成多个子 DataFrame,并设置日期索引和列名。

        返回:
        - df_dict: 包含分割后的子 DataFrame 的字典,键为 '财务'、'行情' 和 '基本面'。
        """
        df_dict = {}
        df_names = ['财务', '行情', '基本面']
        col_locators = ['日期', '日期', '指标名称']

        start_idx = 0
        for idx, col in enumerate(self.df.columns):
            if pd.isna(self.df[col]).all() or idx == len(self.df.columns) - 1:
                if start_idx <= idx:
                    sub_df = self.df.iloc[:, start_idx:idx + 1]
                    df_name = df_names.pop(0)
                    col_locator = col_locators.pop(0)
                    sub_df = set_index_col_wind(sub_df, col_locator)
                    # 删除全部为NaN的列
                    sub_df = sub_df.dropna(axis=1, how='all')

                    if df_name == '基本面':
                        sub_df = sub_df.replace(0, np.nan)

                    df_dict[df_name] = sub_df.astype(float)
                start_idx = idx + 1

        return df_dict

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
        # merged_df.fillna(method='ffill', inplace=True)
        # merged_df.fillna(method='bfill', inplace=True)

        if start_date:
            merged_df = merged_df.loc[start_date:]
        if end_date:
            merged_df = merged_df.loc[:end_date]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(merged_df[df1_col], label=df1_col, linestyle='-', marker=marker1)
        ax2.plot(merged_df[df2_col], label=df2_col, linestyle='-', color='red', marker=marker2)

        ax1.set_xlabel('Date')
        ax1.set_ylabel(df1_col)
        ax2.set_ylabel(df2_col)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()


# 使用示例
visualizer = DataVisualizer(rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\景气研究\行业景气数据库与展示.xlsx", '石油石化')
# visualizer.plot_data('财务', '基本面', '营业收入(同比增长率)', '中国:规模以上工业增加值:石油和天然气开采业:当月同比')#, start_date='2022-01-01', end_date='2023-12-31')
# visualizer.plot_data('财务', '行情', '营业收入(同比增长率)', '收盘价')#, start_date='2022-01-01', end_date='2023-12-31')
visualizer.plot_data('财务', '行情', '营业收入(同比增长率)', '收盘价', marker1='o', marker2=None)
