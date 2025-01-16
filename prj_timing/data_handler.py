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
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='ָ������')

        # Convert all columns to numeric
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')

        # Sort by date
        df_macro.sort_index(inplace=True)

        # ��ȡ Sheet2 �е���������
        df_sheet2 = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # ��̬ʶ������ָ��������� 'Unnamed' ��Ϊָ�����ƣ�
        index_columns = [col for col in df_sheet2.columns if not col.startswith('Unnamed')]

        indices_data = {}
        for index_col in index_columns:
            # ����ÿ��ָ��������ռ��4�У�'����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)'
            start_loc = df_sheet2.columns.get_loc(index_col) - 1
            df_index = df_sheet2.iloc[:, start_loc:start_loc + 4].copy()
            df_index.columns = ['����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)']

            # # Drop first three rows and rows with NaN dates
            df_index = df_index[3:].dropna(subset=['����'])

            # �� '����' ת��Ϊ datetime ���Ͳ�����Ϊ����
            df_index['����'] = pd.to_datetime(df_index['����'])
            df_index.set_index('����', inplace=True)
            df_index.sort_index(inplace=True)

            # ��������ת��Ϊ��ֵ���ͣ����Դ���
            df_index = df_index.apply(pd.to_numeric, errors='coerce')

            # ���趨Ƶ���ز�������
            resampled_df = self.resample_data(df_index)

            # ��������ָ��
            final_df = self.compute_indicators(resampled_df)

            # �洢���ֵ�
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
        df['ָ��:���һ��'] = df['���̼�']
        df['ָ��:���һ��:ͬ��'] = df['���̼�'].pct_change(252 if self.frequency == 'D' else 12)
        df['ָ��:���һ��:����'] = df['���̼�'].pct_change(1)
        df['ָ��:�ɽ����:�ϼ�ֵ'] = df['�ɽ���']
        df['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] = df['�ɽ���'].pct_change(252 if self.frequency == 'D' else 12)
        df['ָ��:�ɽ����:�ϼ�ֵ:����'] = df['�ɽ���'].pct_change(1)
        df['�о���:ָ��'] = df['�о���PB(LF,�ڵ�)']

        # ѡ������������Ҫ����
        final_df = df[[
            '�о���:ָ��',
            'ָ��:���һ��',
            'ָ��:���һ��:ͬ��',
            'ָ��:���һ��:����',
            'ָ��:�ɽ����:�ϼ�ֵ',
            'ָ��:�ɽ����:�ϼ�ֵ:ͬ��',
            'ָ��:�ɽ����:�ϼ�ֵ:����'
        ]]

        return final_df

    def get_macro_data(self):
        """
        ����Ԥ�����ĺ�����ݡ�

        Returns:
            pd.DataFrame: ������ݡ�
        """
        return self.macro_data

    def get_indices_data(self, index_name=None):
        """
        ����Ԥ������ָ�����ݡ�

        Parameters:
            index_name (str, optional): ָ����ָ�����ơ����Ϊ None���򷵻��������ݡ�

        Returns:
            pd.DataFrame or dict: ָ��ָ�������ݿ������ָ���������ֵ䡣
        """
        if index_name:
            return self.indices_data_dict.get(index_name)
        return self.indices_data_dict