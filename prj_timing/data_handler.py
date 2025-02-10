# coding=gbk
# Time Created: 2025/1/14 14:07
# Author  : Lucid
# FileName: data_handler.py
# Software: PyCharm
import pandas as pd
from utils import process_wind_excel


class DataHandler:
    def __init__(self, file_path):
        """
        Initializes the DataHandler.

        Parameters:
            file_path (str): Path to the Excel file.
        """
        self.file_path = file_path
        # 加载后同时生成宏观数据、日频指数数据与月频指数数据
        self.macro_data, self.daily_indices_data, self.monthly_indices_data = self.load_data()

    def load_data(self):
        """
        Loads and preprocesses the Excel data.

        Returns:
            tuple: (macro_data, daily_indices_data, monthly_indices_data)
                macro_data (pd.DataFrame): 预处理后的宏观数据
                daily_indices_data (dict): 日频指数数据字典，键为指数名称，值为处理后的 DataFrame
                monthly_indices_data (dict): 月频指数数据字典，键为指数名称，值为处理后的 DataFrame
        """
        # 读取宏观数据所在的 Sheet1
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='指标名称')

        # 将宏观数据所有列转换为数值类型
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        df_macro.sort_index(inplace=True)

        # 读取 Sheet2 中的所有数据（指数数据）
        df_sheet2 = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # 动态识别所有指数（假设非 'Unnamed' 列为指数名称）
        index_columns = [col for col in df_sheet2.columns if not col.startswith('Unnamed')]

        daily_indices_data = {}
        monthly_indices_data = {}

        for index_col in index_columns:
            # 假设每个指数的数据占用4列：'日期', '收盘价', '成交额', '市净率PB(LF,内地)'
            start_loc = df_sheet2.columns.get_loc(index_col) - 1
            df_index = df_sheet2.iloc[:, start_loc:start_loc + 4].copy()
            df_index.columns = ['日期', '收盘价', '成交额', '市净率PB(LF,内地)']

            # 删除前3行及 '日期' 缺失的行
            df_index = df_index[3:].dropna(subset=['日期'])

            # 将 '日期' 转换为 datetime 类型，并设置为索引
            df_index['日期'] = pd.to_datetime(df_index['日期'])
            df_index.set_index('日期', inplace=True)
            df_index.sort_index(inplace=True)

            # 将所有列转换为数值类型（忽略错误）
            df_index = df_index.apply(pd.to_numeric, errors='coerce')

            # 日频数据：原始数据不做重采样，直接计算指标（pct_change参数按日计算）
            daily_df = df_index.copy()
            daily_df = self.compute_indicators(daily_df, freq='D')
            daily_indices_data[index_col] = daily_df

            # 月频数据：先重采样，再计算指标（pct_change参数按月计算）
            monthly_df = self.resample_data(df_index, freq='M')
            monthly_df = self.compute_indicators(monthly_df, freq='M')
            monthly_indices_data[index_col] = monthly_df

        return df_macro, daily_indices_data, monthly_indices_data

    def resample_data(self, df, freq):
        """
        Resamples the data to the desired frequency.

        Parameters:
            df (pd.DataFrame): Original dataframe.
            freq (str): Desired frequency ('D' for daily, 'M' for monthly).

        Returns:
            pd.DataFrame: Resampled dataframe.
        """
        if freq == 'M':
            monthly = df.resample('M').agg({
                '收盘价': 'last',
                '成交额': 'sum',
                '市净率PB(LF,内地)': 'last'
            })
            return monthly
        elif freq == 'D':
            # 对于日频数据，假设数据已经是每日数据
            return df
        else:
            raise ValueError("Unsupported frequency. Use 'D' for daily or 'M' for monthly.")

    def compute_indicators(self, df, freq):
        """
        Computes additional indicators.

        Parameters:
            df (pd.DataFrame): Resampled (或原始) dataframe.
            freq (str): Frequency of the data ('D' for daily, 'M' for monthly).

        Returns:
            pd.DataFrame: DataFrame with additional indicators.
        """
        # 根据频率确定同比计算的周期：日频用252，月频用12
        if freq == 'D':
            period = 252
        elif freq == 'M':
            period = 12
        else:
            period = 1

        df['指数:最后一条'] = df['收盘价']
        df['指数:最后一条:同比'] = df['收盘价'].pct_change(period)
        df['指数:最后一条:环比'] = df['收盘价'].pct_change(1)
        df['指数:成交金额:合计值'] = df['成交额']
        df['指数:成交金额:合计值:同比'] = df['成交额'].pct_change(period)
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

    def get_indices_data(self, index_name=None, freq='D'):
        """
        返回预处理后的指数数据。

        Parameters:
            index_name (str, optional): 指定的指数名称。如果为 None，则返回所有数据。
            freq (str): 数据频率，'D' 日频或 'M' 月频。

        Returns:
            pd.DataFrame or dict: 指定指数的数据框或所有指数的数据字典。
        """
        if freq == 'D':
            if index_name:
                return self.daily_indices_data.get(index_name)
            return self.daily_indices_data
        elif freq == 'M':
            if index_name:
                return self.monthly_indices_data.get(index_name)
            return self.monthly_indices_data
        else:
            raise ValueError("Unsupported frequency. Use 'D' for daily or 'M' for monthly.")