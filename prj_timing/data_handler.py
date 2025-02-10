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
        # ���غ�ͬʱ���ɺ�����ݡ���Ƶָ����������Ƶָ������
        self.macro_data, self.daily_indices_data, self.monthly_indices_data = self.load_data()

    def load_data(self):
        """
        Loads and preprocesses the Excel data.

        Returns:
            tuple: (macro_data, daily_indices_data, monthly_indices_data)
                macro_data (pd.DataFrame): Ԥ�����ĺ������
                daily_indices_data (dict): ��Ƶָ�������ֵ䣬��Ϊָ�����ƣ�ֵΪ������ DataFrame
                monthly_indices_data (dict): ��Ƶָ�������ֵ䣬��Ϊָ�����ƣ�ֵΪ������ DataFrame
        """
        # ��ȡ����������ڵ� Sheet1
        metadata, df_macro = process_wind_excel(self.file_path, sheet_name='Sheet1', column_name='ָ������')

        # ���������������ת��Ϊ��ֵ����
        for col in df_macro.columns:
            df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        df_macro.sort_index(inplace=True)

        # ��ȡ Sheet2 �е��������ݣ�ָ�����ݣ�
        df_sheet2 = pd.read_excel(self.file_path, sheet_name='Sheet2', header=0)

        # ��̬ʶ������ָ��������� 'Unnamed' ��Ϊָ�����ƣ�
        index_columns = [col for col in df_sheet2.columns if not col.startswith('Unnamed')]

        daily_indices_data = {}
        monthly_indices_data = {}

        for index_col in index_columns:
            # ����ÿ��ָ��������ռ��4�У�'����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)'
            start_loc = df_sheet2.columns.get_loc(index_col) - 1
            df_index = df_sheet2.iloc[:, start_loc:start_loc + 4].copy()
            df_index.columns = ['����', '���̼�', '�ɽ���', '�о���PB(LF,�ڵ�)']

            # ɾ��ǰ3�м� '����' ȱʧ����
            df_index = df_index[3:].dropna(subset=['����'])

            # �� '����' ת��Ϊ datetime ���ͣ�������Ϊ����
            df_index['����'] = pd.to_datetime(df_index['����'])
            df_index.set_index('����', inplace=True)
            df_index.sort_index(inplace=True)

            # ��������ת��Ϊ��ֵ���ͣ����Դ���
            df_index = df_index.apply(pd.to_numeric, errors='coerce')

            # ��Ƶ���ݣ�ԭʼ���ݲ����ز�����ֱ�Ӽ���ָ�꣨pct_change�������ռ��㣩
            daily_df = df_index.copy()
            daily_df = self.compute_indicators(daily_df, freq='D')
            daily_indices_data[index_col] = daily_df

            # ��Ƶ���ݣ����ز������ټ���ָ�꣨pct_change�������¼��㣩
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
                '���̼�': 'last',
                '�ɽ���': 'sum',
                '�о���PB(LF,�ڵ�)': 'last'
            })
            return monthly
        elif freq == 'D':
            # ������Ƶ���ݣ����������Ѿ���ÿ������
            return df
        else:
            raise ValueError("Unsupported frequency. Use 'D' for daily or 'M' for monthly.")

    def compute_indicators(self, df, freq):
        """
        Computes additional indicators.

        Parameters:
            df (pd.DataFrame): Resampled (��ԭʼ) dataframe.
            freq (str): Frequency of the data ('D' for daily, 'M' for monthly).

        Returns:
            pd.DataFrame: DataFrame with additional indicators.
        """
        # ����Ƶ��ȷ��ͬ�ȼ�������ڣ���Ƶ��252����Ƶ��12
        if freq == 'D':
            period = 252
        elif freq == 'M':
            period = 12
        else:
            period = 1

        df['ָ��:���һ��'] = df['���̼�']
        df['ָ��:���һ��:ͬ��'] = df['���̼�'].pct_change(period)
        df['ָ��:���һ��:����'] = df['���̼�'].pct_change(1)
        df['ָ��:�ɽ����:�ϼ�ֵ'] = df['�ɽ���']
        df['ָ��:�ɽ����:�ϼ�ֵ:ͬ��'] = df['�ɽ���'].pct_change(period)
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

    def get_indices_data(self, index_name=None, freq='D'):
        """
        ����Ԥ������ָ�����ݡ�

        Parameters:
            index_name (str, optional): ָ����ָ�����ơ����Ϊ None���򷵻��������ݡ�
            freq (str): ����Ƶ�ʣ�'D' ��Ƶ�� 'M' ��Ƶ��

        Returns:
            pd.DataFrame or dict: ָ��ָ�������ݿ������ָ���������ֵ䡣
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