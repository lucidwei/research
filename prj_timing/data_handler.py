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
        self.df = self.load_data()

    def load_data(self):
        """
        Loads and preprocesses the Excel data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        # Read the Excel file
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='ָ������')

        # Convert all columns to numeric
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')

        # Sort by date
        df_macro.sort_index(inplace=True)

        # Read Sheet2 data
        df = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # Split data based on empty columns
        empty_cols = df.columns[df.isna().all()]
        split_indices = [df.columns.get_loc(col) for col in empty_cols]

        if split_indices:
            first_split = split_indices[0]
            z_index_df = df.iloc[:, :first_split].copy()
        else:
            z_index_df = df.copy()

        # Rename columns
        z_index_df.columns = ['����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)']

        # Drop first three rows and rows with NaN dates
        z_index_df = z_index_df[3:].dropna(subset=['����'])

        # Convert '����' to datetime and set as index
        z_index_df['����'] = pd.to_datetime(z_index_df['����'])
        z_index_df.set_index('����', inplace=True)
        z_index_df.sort_index(inplace=True)

        # Convert all columns to numeric
        z_index_df = z_index_df.apply(pd.to_numeric, errors='coerce')

        # Resample data based on desired frequency
        resampled_df = self.resample_data(z_index_df)

        # Compute additional indicators
        final_df = self.compute_indicators(resampled_df)

        # Ensure df_macro and final_df have aligned indices
        if not df_macro.index.equals(final_df.index):
            raise ValueError("df_macro �� final_df ������������ƥ�䣬�޷��ϲ���")

        # Merge dataframes
        merged_df = pd.concat([df_macro, final_df], axis=1)

        return merged_df

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
                '���̼�': 'last',
                '�ɽ���': 'sum',
                '�о���PB(LF,�ڵ�)': 'last'
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
        df['��֤�ۺ�ָ��:��:���һ��'] = df['���̼�']
        df['��֤�ۺ�ָ��:��:���һ��:ͬ��'] = df['���̼�'].pct_change(12)
        df['��֤�ۺ�ָ��:��:���һ��:����'] = df['���̼�'].pct_change(1)
        df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ'] = df['�ɽ���']
        df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��'] = df['�ɽ���'].pct_change(12)
        df['��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'] = df['�ɽ���'].pct_change(1)
        df['�о���:��ָ֤��:��:���һ��'] = df['�о���PB(LF,�ڵ�)']

        # Select and reorder columns
        final_df = df[[
            '�о���:��ָ֤��:��:���һ��',
            '��֤�ۺ�ָ��:��:���һ��',
            '��֤�ۺ�ָ��:��:���һ��:ͬ��',
            '��֤�ۺ�ָ��:��:���һ��:����',
            '��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:ͬ��',
            '��֤�ۺ�ָ��:�ɽ����:��:�ϼ�ֵ:����'
        ]]

        return final_df

    def get_data(self):
        """
        Returns the preprocessed dataframe.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        return self.df