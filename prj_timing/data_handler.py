# coding=gbk
# Time Created: 2025/1/14 14:07
# Author  : Lucid
# FileName: data_handler.py
# Software: PyCharm
import pandas as pd
from utils import process_wind_excel


class DataHandler:
    def __init__(self, file_path, frequency='M'):
        """
        Initializes the DataHandler.

        Parameters:
            file_path (str): Path to the Excel file.
            frequency (str): Resampling frequency ('D' for daily, 'M' for monthly, etc.).
        """
        self.file_path = file_path
        self.frequency = frequency
        self.macro_data, self.indices_data_dict = self.load_data()

    def load_data(self):
        """
        Loads and preprocesses the Excel data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        # Read the Excel file
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='指标名称')

        # Convert all columns to numeric
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')

        # Sort by date
        df_macro.sort_index(inplace=True)

        # 读取 Sheet2 中的所有数据
        df_sheet2 = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # 动态识别所有指数（假设非 'Unnamed' 列为指数名称）
        index_columns = [col for col in df_sheet2.columns if not col.startswith('Unnamed')]

        indices_data = {}
        for index_col in index_columns:
            # 假设每个指数的数据占用4列：'日期', '收盘价', '成交额', '市净率PB(LF,内地)'
            start_loc = df_sheet2.columns.get_loc(index_col) - 1
            df_index = df_sheet2.iloc[:, start_loc:start_loc + 4].copy()
            df_index.columns = ['日期', '收盘价', '成交额', '市净率PB(LF,内地)']

            # # Drop first three rows and rows with NaN dates
            df_index = df_index[3:].dropna(subset=['日期'])

            # 将 '日期' 转换为 datetime 类型并设置为索引
            df_index['日期'] = pd.to_datetime(df_index['日期'])
            df_index.set_index('日期', inplace=True)
            df_index.sort_index(inplace=True)

            # 将所有列转换为数值类型（忽略错误）
            df_index = df_index.apply(pd.to_numeric, errors='coerce')

            # 按设定频率重采样数据
            resampled_df = self.resample_data(df_index)

            # 计算所需指标
            final_df = self.compute_indicators(resampled_df)

            # 存储到字典
            indices_data[index_col] = final_df

        return df_macro, indices_data

    def resample_data(self, df):
        """
        Resamples the data to the desired frequency.

        Parameters:
            df (pd.DataFrame): Original dataframe.

        Returns:
            pd.DataFrame: Resampled dataframe.
        """
        if self.frequency == 'M':
            monthly = df.resample('M').agg({
                '收盘价': 'last',
                '成交额': 'sum',
                '市净率PB(LF,内地)': 'last'
            })
            return monthly
        elif self.frequency == 'D':
            # For daily frequency, ensure data is already daily
            return df
        else:
            raise ValueError("Unsupported frequency. Use 'D' for daily or 'M' for monthly.")

    def compute_indicators(self, df):
        """
        Computes additional indicators.

        Parameters:
            df (pd.DataFrame): Resampled dataframe.

        Returns:
            pd.DataFrame: Dataframe with additional indicators.
        """
        df['指数:最后一条'] = df['收盘价']
        df['指数:最后一条:同比'] = df['收盘价'].pct_change(252 if self.frequency == 'D' else 12)
        df['指数:最后一条:环比'] = df['收盘价'].pct_change(1)
        df['指数:成交金额:合计值'] = df['成交额']
        df['指数:成交金额:合计值:同比'] = df['成交额'].pct_change(252 if self.frequency == 'D' else 12)
        df['指数:成交金额:合计值:环比'] = df['成交额'].pct_change(1)
        df['市净率:指数'] = df['市净率PB(LF,内地)']

        # 选择并重新排列需要的列
        final_df = df[[
            '市净率:指数',
            '指数:最后一条',
            '指数:最后一条:同比',
            '指数:最后一条:环比',
            '指数:成交金额:合计值',
            '指数:成交金额:合计值:同比',
            '指数:成交金额:合计值:环比'
        ]]

        return final_df

    def get_macro_data(self):
        """
        返回预处理后的宏观数据。

        Returns:
            pd.DataFrame: 宏观数据。
        """
        return self.macro_data

    def get_indices_data(self, index_name=None):
        """
        返回预处理后的指数数据。

        Parameters:
            index_name (str, optional): 指定的指数名称。如果为 None，则返回所有数据。

        Returns:
            pd.DataFrame or dict: 指定指数的数据框或所有指数的数据字典。
        """
        if index_name:
            return self.indices_data_dict.get(index_name)
        return self.indices_data_dict