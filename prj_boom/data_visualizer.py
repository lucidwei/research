# coding=gbk
# Time Created: 2024/4/10 15:12
# Author  : Lucid
# FileName: data_visualizer.py
# Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_index_col_wind


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
        self.df_list = self._split_dataframe()

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

                    df_dict[df_name] = sub_df
                start_idx = idx + 1

        return df_dict

    def plot_data(self, df1_idx, df2_idx, df1_col, df2_col, start_date=None, end_date=None):
        """
        绘制两个 DataFrame 中指定列的数据。

        参数:
        - df1_idx: 第一个 DataFrame 在 df_list 中的索引。
        - df2_idx: 第二个 DataFrame 在 df_list 中的索引。
        - df1_col: 第一个 DataFrame 中要绘制的列名。
        - df2_col: 第二个 DataFrame 中要绘制的列名。
        - start_date: 可选的起始日期,用于过滤数据。
        - end_date: 可选的结束日期,用于过滤数据。
        """
        df1 = self.df_list[df1_idx]
        df2 = self.df_list[df2_idx]

        if 'Date' in df1.columns:
            df1.set_index('Date', inplace=True)
        elif '时间区间' in df1.columns:
            df1.set_index('时间区间', inplace=True)

        if 'Date' in df2.columns:
            df2.set_index('Date', inplace=True)
        elif '时间区间' in df2.columns:
            df2.set_index('时间区间', inplace=True)

        if start_date:
            df1 = df1.loc[start_date:]
            df2 = df2.loc[start_date:]
        if end_date:
            df1 = df1.loc[:end_date]
            df2 = df2.loc[:end_date]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(df1[df1_col], label=df1_col)
        ax2.plot(df2[df2_col], label=df2_col, color='red')

        ax1.set_xlabel('Date')
        ax1.set_ylabel(df1_col)
        ax2.set_ylabel(df2_col)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()


# 使用示例
visualizer = DataVisualizer(rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\景气研究\行业景气数据库与展示.xlsx", '石油石化')
visualizer.plot_data(0, 2, '营业收入(同比增长率)', '中国:规模以上工业增加值:石油和天然气开采业:当月同比', start_date='2022-01-01', end_date='2023-12-31')