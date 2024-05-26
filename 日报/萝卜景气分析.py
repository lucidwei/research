# coding=gbk
# Time Created: 2024/4/9 14:00
# Author  : Lucid
# FileName: �ܲ���������.py
# Software: PyCharm
import os
import pandas as pd
from datetime import datetime


def read_data(directory):
    # ��ȡĿ¼��������"�����б�����_"��ͷ����".xlsx"��β���ļ���
    files = [f for f in os.listdir(directory) if f.startswith('�����б�����_') and f.endswith('.xlsx')]

    # ���ڴ洢��ͬ���ڵ�DataFrame���ֵ�
    dfs = {}

    for file in files:
        # ��ȡ�ļ����е����ڲ���
        date = file.split('_')[1].split('.')[0]

        # ��ȡExcel�ļ���DataFrame
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path)
        df = df.set_index("��ҵ")

        # ��DataFrame�洢���ֵ���,��������Ϊ��
        dfs[date] = df

    return dfs


def read_history_data(file_path):
    # ��ȡ�ο����ݵ�Excel�ļ�,��������sheet�洢��һ���ֵ���
    history_data = pd.read_excel(file_path, sheet_name=None, index_col=0)

    # ��ÿ��sheet�����ݽ��д���
    for sheet_name, sheet_data in history_data.items():
        sheet_data.index = pd.to_datetime(sheet_data.index).strftime('%Y%m%d')
        history_data[sheet_name] = sheet_data

    return history_data


def static_analysis(dfs, date=None):
    if date is None:
        # ���û��ָ������,ʹ������һ�ڵ�����
        date = max(dfs.keys())

    df = dfs[date]

    # ������ҵӵ���Ⱥ���ҵ����ǿ�ȵ�90%��λ��
    crowdedness_90 = df['��ҵӵ����'].quantile(0.7)
    trend_strength_90 = df['��ҵ����ǿ��'].quantile(0.7)

    # ����ɸѡ����������������
    mask_high_prosperity = df['������ҵ������'] > 0
    mask_low_valuation_PE = df['��ǰPE��ʷ�ٷ�λ'] < 0.5
    mask_low_valuation_PB = df['��ǰPB��ʷ�ٷ�λ'] < 0.5
    mask_low_crowdedness = df['��ҵӵ����'] <= crowdedness_90
    mask_low_trend_strength = df['��ҵ����ǿ��'] <= trend_strength_90

    # �����������������,ѡ������������������
    selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness & mask_low_trend_strength]

    # ����'������ҵ������'�Ӵ�С����
    selected_rows_sorted = selected_rows.sort_values(by='��ǰ��������ʷ�ٷ�λ', ascending=False)

    return selected_rows_sorted


def dynamic_analysis(dfs, history_data, start_date=None, end_date=None):
    if end_date is None:
        # ���û��ָ����������,ʹ������һ�ڵ�����
        end_date = max(dfs.keys())

    if start_date is None:
        # ���û��ָ����ʼ����,ʹ�ý�������7��ǰ������
        start_date = pd.to_datetime(end_date) - pd.Timedelta(days=7)
        start_date = start_date.strftime('%Y%m%d')

    def get_df(dfs, history_data, date):
        if date in dfs.keys():
            return dfs[date]
        else:
            df = pd.DataFrame(index=history_data['��ҵ������'].columns)
            for sheet_name, sheet_data in history_data.items():
                if date in sheet_data.index:
                    df[sheet_name] = sheet_data.loc[date]
            return df

    # ��ȡ��ʼ���ںͽ������ڶ�Ӧ��df
    df_start = get_df(dfs, history_data, start_date)
    df_end = get_df(dfs, history_data, end_date)

    # ��������ӳ���ϵ, ����������
    column_mapping = {
        '������ҵ������': '��ҵ�����ȱ仯',
        '��ҵ������': '��ҵ�����ȱ仯',
        '��ҵƽ��PE(TTM)': '��ҵPE�仯',
        '��ҵPE': '��ҵPE�仯',
        '��ҵƽ��PB(MRQ)': '��ҵPB�仯',
        '��ҵPB': '��ҵPB�仯',
        '��ҵ����ǿ��': '��ҵ����ǿ�ȱ仯',
        '��ҵӵ����': '��ҵӵ���ȱ仯',
        '��ǰ��������ʷ�ٷ�λ': '��������ʷ�ٷ�λ�仯',
        '��ǰPE��ʷ�ٷ�λ': 'PE��ʷ�ٷ�λ�仯',
        '��ǰPB��ʷ�ٷ�λ': 'PB��ʷ�ٷ�λ�仯',
    }
    df_start = df_start.rename(columns=column_mapping)
    df_end = df_end.rename(columns=column_mapping)

    # ����������numeric_columns�б�
    full_numeric_columns = ['��ҵ�����ȱ仯', '��������ʷ�ٷ�λ�仯', '��ҵPE�仯', 'PE��ʷ�ٷ�λ�仯',
                            '��ҵPB�仯', 'PB��ʷ�ٷ�λ�仯', '��ҵ����ǿ�ȱ仯', '��ҵӵ���ȱ仯']
    # �����������ʷ���ݣ���ô����һЩ��
    # ���df_start��df_end�����Ƿ���������numeric_columns�б���ͬ
    if set(df_start.columns) == set(full_numeric_columns) and set(df_end.columns) == set(full_numeric_columns):
        numeric_columns = full_numeric_columns
    else:
        numeric_columns = ['��ҵ�����ȱ仯', '��ҵPE�仯', '��ҵPB�仯', '��ҵ����ǿ�ȱ仯', '��ҵӵ���ȱ仯']

    # ����ָ��仯
    change_df = df_end[numeric_columns] - df_start[numeric_columns]
    change_df = change_df.sort_values(by='��ҵ�����ȱ仯', ascending=False)

    return change_df, start_date, end_date


def save_to_excel(selected_industries, change_df, directory, start_date, end_date):
    # ��ȡ��ǰʱ���,��ȷ��������ʱ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # ��������ļ���
    output_file = os.path.join(directory, 'output', f'output_{end_date}-{start_date}_{timestamp}.xlsx')

    # ����һ���µ� ExcelWriter ����
    with pd.ExcelWriter(output_file) as writer:
        # �� selected_industries д����Ϊ "��̬�������" �Ĺ�����
        selected_industries.to_excel(writer, sheet_name='��̬�������', index=True)

        # �� change_df д����Ϊ "��̬�������" �Ĺ�����
        change_df.to_excel(writer, sheet_name='��̬�������', index=True)

    return output_file


# ����Excel�ļ����ڵ�Ŀ¼
directory = rf"D:\WPS����\WPS����\����-���\���ݿ����\����\�ܲ�"

# ��ȡ����
dfs = read_data(directory)
history_data = read_history_data(rf"{directory}\Industry Prosperity_data.xls")

# ���о�̬����
selected_industries = static_analysis(dfs)
print("�߾������͹�ֵ����ӵ������ҵ:")
print(selected_industries)

# ���ж�̬����
change_df, start_date, end_date = dynamic_analysis(dfs, history_data, start_date='20240517', end_date='20240524')
print("ָ��仯:")
print(change_df)

# ��������浽 Excel �ļ�
output_file = save_to_excel(selected_industries, change_df, directory, start_date, end_date)
print(f"����ѱ��浽 {output_file}")
