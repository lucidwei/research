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
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


class DataVisualizer:
    def __init__(self, file_path, sheet_name):
        """
        ��ʼ�� DataVisualizer �ࡣ

        ����:
        - file_path: Excel �ļ���·����
        - sheet_name: Ҫ��ȡ�Ĺ��������ơ�
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.df_dict = split_dataframe(self.df)

    def plot_data(self, df1_key, df2_key, df1_col, df2_col, start_date=None, end_date=None, marker1=None, marker2=None):
        """
        �������� DataFrame ��ָ���е����ݡ�

        ����:
        - df1_key: ��һ�� DataFrame �ļ���
        - df2_key: �ڶ��� DataFrame �ļ���
        - df1_col: ��һ�� DataFrame ��Ҫ���Ƶ�������
        - df2_col: �ڶ��� DataFrame ��Ҫ���Ƶ�������
        - start_date: ��ѡ����ʼ����,���ڹ������ݡ�
        - end_date: ��ѡ�Ľ�������,���ڹ������ݡ�
        """
        df1 = self.df_dict[df1_key]
        df2 = self.df_dict[df2_key]

        # ���ָ���������Ƿ�����ڶ�Ӧ�� DataFrame ��
        if df1_col not in df1.columns:
            raise ValueError(f"Column '{df1_col}' does not exist in DataFrame '{df1_key}'.")
        if df2_col not in df2.columns:
            raise ValueError(f"Column '{df2_col}' does not exist in DataFrame '{df2_key}'.")

        # ������ DataFrame �����������ϲ�,�����ȱʧֵ
        merged_df = pd.merge(df1[[df1_col]], df2[[df2_col]], left_index=True, right_index=True, how='outer').dropna(how='all')
        merged_df.fillna(method='ffill', inplace=True)
        # merged_df.fillna(method='bfill', inplace=True)

        if start_date:
            merged_df = merged_df.loc[start_date:]
        if end_date:
            merged_df = merged_df.loc[:end_date]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # ����� '����' �� '����' ���ݵ����,���� add_shading ���������Ӱ
        if df1_key == '����' and df2_key == '����':
            mask = (df1.index > pd.to_datetime('2023-01-01')) & (df1.isnull().all(axis=1))
            first_missing_financial_date = df1.loc[mask].index.min()
            self.add_shading(merged_df[df1_col], merged_df[df2_col], ax1, first_missing_financial_date)
        if df1_key == '����' and df2_key == '������':
            mask = (df1.index > pd.to_datetime('2023-01-01')) & (df1.isnull().all(axis=1))
            first_missing_financial_date = df1.loc[mask].index.min()
            # ��ȡ�������ݵ��������ڣ����⻭�����ھ�Զ�Ļ���������
            earliest_date_finance = df1.index.min() - pd.Timedelta(days=5)
            # ��ȡ����������
            merged_df = merged_df.loc[earliest_date_finance:]

            self.add_shading(merged_df[df1_col], merged_df[df2_col], ax1, first_missing_financial_date)
        if df1_key == '����' and df2_key == '������':
            # ��ȡ�������ݵ���������
            earliest_date_finance = df1.index.min()
            # ������������ݵĽ�ȡ����(���������������� - 5 ��)
            cut_off_date = earliest_date_finance - pd.Timedelta(days=5)
            # ��ȡ����������
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
        ���ݲ������ݺ��������ݵı仯����,��ͼ�������Ӱ��

        ����:
        - finance_data: �������ݵ� DataFrame��
        - quote_data: �������ݵ� DataFrame��
        - ax: Ҫ�����Ӱ�� Axes ����
        """
        # ��ȡ�������ݵļ���ĩ����
        quarter_ends = finance_data.dropna().resample('Q').last().index
        if first_missing_financial_date is not None:
            quarter_ends = quarter_ends[quarter_ends < first_missing_financial_date]
        # # �ҵ���һ��ȱʧ�ļ�ĩ����
        # missing_index = quarter_ends.difference(finance_data.index)
        # if not missing_index.empty:
        #     first_missing_date = missing_index[0]
        #     # ɾ����һ��ȱʧ���ڼ���֮������м�ĩ����
        #     quarter_ends = quarter_ends[quarter_ends < first_missing_date]

        # ͳ�ƺ�ɫ����ɫ���ȵĸ���
        red_count = 0
        blue_count = 0
        # ����ÿ������
        for idx in range(len(quarter_ends)):
            if idx < len(quarter_ends) - 1:
                start_date = quarter_ends[idx]
                end_date = quarter_ends[idx + 1]

                # ������������ڸü��ȵı仯
                finance_change = finance_data.loc[end_date] - finance_data.loc[start_date]

                # �������������ڸ�ʱ����ڵı仯
                quote_change = quote_data.loc[end_date] - quote_data.loc[start_date]

                # �ж��������ݵı仯�����Ƿ�һ��
                if (finance_change * quote_change) > 0:
                    # ����������������������,��Ӻ�ɫ��Ӱ
                    ax.axvspan(start_date, end_date, alpha=0.2, color='red')
                    red_count += 1
                elif (finance_change * quote_change) < 0:
                    # ���������½��������µ�,�����ɫ��Ӱ
                    ax.axvspan(start_date, end_date, alpha=0.2, color='blue')
                    blue_count += 1
        # �����ɫ���ȵ�ռ��
        total_count = red_count + blue_count
        if total_count > 0:
            red_ratio = red_count / total_count
        else:
            red_ratio = 0

        # ��ӡͳ�ƽ��
        print(f"��ɫ������: {red_count}")
        print(f"��ɫ������: {blue_count}")
        print(f"��ɫ����ռ��: {red_ratio:.1%}")

    def add_shading_quote_fundamental(self, quote_data, fundamental_data, ax):
        """
        �����������ݺͻ��������ݵı仯����,��ͼ�������Ӱ��

        ����:
        - quote_data: �������ݵ� DataFrame��
        - fundamental_data: ���������ݵ� DataFrame��
        - ax: Ҫ�����Ӱ�� Axes ����
        """
        # ��ȡ�������ݺͻ��������ݵĽ�������
        common_index = quote_data.index.intersection(fundamental_data.index)

        # ���ݽ����������������ݺͻ��������ݽ����ز���
        quote_data_resampled = quote_data.loc[common_index].resample('Q').last()
        fundamental_data_resampled = fundamental_data.loc[common_index].resample('Q').last()

        # ��ȡ�ز�����ļ���ĩ����
        quarter_ends = quote_data_resampled.index

        # ͳ�ƺ�ɫ����ɫ���ȵĸ���
        red_count = 0
        blue_count = 0

        # ����ÿ������
        for idx in range(len(quarter_ends)):
            if idx < len(quarter_ends) - 1:
                start_date = quarter_ends[idx]
                end_date = quarter_ends[idx + 1]

                # ��ȡ�������ݺͻ����������ڸü��ȵ���ʼ�ͽ���ֵ
                quote_start = quote_data_resampled.iloc[idx]
                quote_end = quote_data_resampled.iloc[idx + 1]
                fundamental_start = fundamental_data_resampled.iloc[idx]
                fundamental_end = fundamental_data_resampled.iloc[idx + 1]

                # ��������Ƿ�Ϊ NaN,�����,���Ի�ȡ�ٽ��ķǿ���ֵ
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

                # �����������ݺͻ����������ڸü��ȵı仯
                if quote_start is not None and quote_end is not None and fundamental_start is not None and fundamental_end is not None:
                    quote_change = quote_end - quote_start
                    fundamental_change = fundamental_end - fundamental_start

                    # �ж��������ݵı仯�����Ƿ�һ��
                    if (quote_change * fundamental_change) > 0:
                        # �������������һ�������������,��Ӻ�ɫ��Ӱ
                        ax.axvspan(start_date, end_date, alpha=0.2, color='red')
                        red_count += 1
                    elif (quote_change * fundamental_change) < 0:
                        # ���������½��һ����������½�,�����ɫ��Ӱ
                        ax.axvspan(start_date, end_date, alpha=0.2, color='blue')
                        blue_count += 1

        # �����ɫ���ȵ�ռ��
        total_count = red_count + blue_count
        if total_count > 0:
            red_ratio = red_count / total_count
        else:
            red_ratio = 0

        # ��ӡͳ�ƽ��
        print(f"��ɫ������: {red_count}")
        print(f"��ɫ������: {blue_count}")
        print(f"��ɫ����ռ��: {red_ratio:.2%}")


def analyze_industry(visualizer, commodity_price_col, start_date=None, end_date=None):
    # ������������������ݵĹ�ϵ
    # visualizer.plot_data('����', '������', 'Ӫҵ����ͬ��������', commodity_price_col)
    # visualizer.plot_data('����', '������', '����ĸ��˾�ɶ��ľ�����ͬ��������', commodity_price_col)
    # visualizer.plot_data('����', '������', '���ʲ�������ROE', commodity_price_col)

    # �����������������ݵĹ�ϵ
    visualizer.plot_data('����', '����', 'Ӫҵ����ͬ��������', '���̼�', marker1=None, marker2=None, start_date=start_date)
    # visualizer.plot_data('����', '����', '����ĸ��˾�ɶ��ľ�����ͬ��������', '���̼�', marker1='o', marker2=None, start_date=start_date)
    visualizer.plot_data('����', '����', '���ʲ�������ROE', '���̼�', marker1=None, marker2=None, start_date=start_date)

    # ������������������ݵĹ�ϵ
    # visualizer.plot_data('����', '������', '���̼�', commodity_price_col, start_date=start_date, end_date=end_date)

file_path = rf"D:\WPS����\WPS����\����-���\ר���о�\�����о�\��ҵ�������ݿ���չʾ.xlsx"

# ʯ��ʯ����ҵ����
visualizer = DataVisualizer(file_path, 'ʯ��ʯ��')
analyze_industry(visualizer, '�ֻ���:ԭ��:Ӣ��������Dtd')
# visualizer = DataVisualizer(file_path, 'ʯ��ʯ��')
# analyze_industry(visualizer, '�ֻ���:ԭ��:Ӣ��������Dtd', start_date='2020-01-02')
# visualizer = DataVisualizer(file_path, 'ʯ��ʯ��')
# analyze_industry(visualizer, '�ֻ���:ԭ��:Ӣ��������Dtd', start_date='2014-10-02', end_date='2020-03-02')

visualizer = DataVisualizer(file_path, 'ú̿')
analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)', start_date='2020-01-02')
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)', start_date='2014-10-02', end_date='2020-03-02')

# visualizer = DataVisualizer(file_path, '��ɫ����')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')

visualizer = DataVisualizer(file_path, '��������')
analyze_industry(visualizer, '�й�������Ʒ�۸�ָ��')

visualizer = DataVisualizer(file_path, '����')
analyze_industry(visualizer, '�й�:�۸�:���Ƹ�(HRB400,20mm)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
#
# visualizer = DataVisualizer(file_path, 'ú̿')
# analyze_industry(visualizer, '�ػʵ���:ƽ�ּ�:����ú(Q5000K)')
