# coding=gbk
# Time Created: 2024/4/9 14:00
# Author  : Lucid
# FileName: �ܲ���������.py
# Software: PyCharm
import os
import pandas as pd

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

def static_analysis(dfs, date=None):
    if date is None:
        # ���û��ָ������,ʹ������һ�ڵ�����
        date = max(dfs.keys())

    df = dfs[date]

    # ����ɸѡ����������������
    mask_high_prosperity = df['������ҵ������'] > 0
    mask_low_valuation_PE = df['��ǰPE��ʷ�ٷ�λ'] < 0.5
    mask_low_valuation_PB = df['��ǰPB��ʷ�ٷ�λ'] < 0.5
    mask_low_crowdedness = df['��ҵӵ����'] < 0

    # �����������������,ѡ������������������
    selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness]

    return selected_rows

def dynamic_analysis(dfs, start_date=None, end_date=None):
    if end_date is None:
        # ���û��ָ����������,ʹ������һ�ڵ�����
        end_date = max(dfs.keys())

    if start_date is None:
        # ���û��ָ����ʼ����,ʹ�ý�������7��ǰ������
        start_date = pd.to_datetime(end_date) - pd.Timedelta(days=7)
        start_date = start_date.strftime('%Y%m%d')

    df_start = dfs[start_date]
    df_end = dfs[end_date]

    # ����ָ��仯
    numeric_columns = ['������ҵ������', '��ǰ��������ʷ�ٷ�λ', '��ҵƽ��PE(TTM)', '��ǰPE��ʷ�ٷ�λ',
                       '��ҵƽ��PB(MRQ)', '��ǰPB��ʷ�ٷ�λ', '��ҵ����ǿ��', '��ҵӵ����']

    # ����ָ��仯
    change_df = df_end[numeric_columns] - df_start[numeric_columns]

    return change_df

# ����Excel�ļ����ڵ�Ŀ¼
directory = rf"D:\WPS����\WPS����\����-���\���ݿ����\����\�ܲ�"

# ��ȡ����
dfs = read_data(directory)

# ���о�̬����
selected_industries = static_analysis(dfs)
print("�߾������͹�ֵ����ӵ������ҵ:")
print(selected_industries)

# ���ж�̬����
change_df = dynamic_analysis(dfs, start_date='20240407', end_date='20240409')
print("ָ��仯:")
print(change_df)






# import os
# import pandas as pd
#
# # ����Excel�ļ����ڵ�Ŀ¼
# directory = rf"D:\WPS����\WPS����\����-���\���ݿ����\����\�ܲ�"
#
# # ��ȡĿ¼��������"�����б�����_"��ͷ����".xlsx"��β���ļ���
# files = [f for f in os.listdir(directory) if f.startswith('�����б�����_') and f.endswith('.xlsx')]
#
# # ���ڴ洢��ͬ���ڵ�DataFrame���ֵ�
# dfs = {}
#
# for file in files:
#     # ��ȡ�ļ����е����ڲ���
#     date = file.split('_')[1].split('.')[0]
#
#     # ��ȡExcel�ļ���DataFrame
#     file_path = os.path.join(directory, file)
#     df = pd.read_excel(file_path)
#     df = df.set_index("��ҵ")
#
#     # ��DataFrame�洢���ֵ���,��������Ϊ��
#     dfs[date] = df
#
# # ��ӡÿ�����ڶ�Ӧ��DataFrame����Ϣ
# for date, df in dfs.items():
#     # ����ɸѡ����������������
#     mask_high_prosperity = df['������ҵ������'] > 0
#     mask_low_valuation_PE = df['��ǰPE��ʷ�ٷ�λ'] < 0.5
#     mask_low_valuation_PB = df['��ǰPB��ʷ�ٷ�λ'] < 0.5
#     mask_low_crowdedness = df['��ҵӵ����'] < 0
#
#     # �����������������,ѡ������������������
#     selected_rows = df[mask_high_prosperity & mask_low_valuation_PE & mask_low_valuation_PB & mask_low_crowdedness]