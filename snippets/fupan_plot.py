# coding=gbk
# Time Created: 2023/8/25 11:01
# Author  : Lucid
# FileName: fupan_plot.py
# Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # ָ��Ĭ�����壺���plot������ʾ��������
mpl.rcParams['axes.unicode_minus'] = False           # �������ͼ���Ǹ���'-'��ʾΪ���������


def process_name(name):
    # ��������а��������.���š�����ֻ��ȡ����ǰ�Ĳ���
    if "���.����" in name:
        return name.split("(")[0]
    # ��������а�����������ҵָ��:��������ȡð�ź�Ĳ���
    elif "������ҵָ��:" in name:
        return name.split(":")[1]
    else:
        return name


def plot_trends(start_date, end_date, data, categories='style', specific_categories=None):

    selected_data = data[(data["ָ������"] >= start_date) & (data["ָ������"] <= end_date)]

    if categories == 'style':
        columns_to_plot = data.columns[1:6]
    elif categories == 'industry':
        columns_to_plot = [col for col in data.columns[6:] if
                           specific_categories is None or col.split(":")[1] in specific_categories]


    else:
        raise ValueError("Invalid category. Please choose 'style' or 'industry'.")

    start_values = selected_data[columns_to_plot].iloc[-1]
    aligned_data = ((selected_data[columns_to_plot] - start_values) / start_values) * 100

    plt.figure(figsize=(12, 8))
    for column in aligned_data.columns:
        plt.plot(selected_data["ָ������"], aligned_data[column], label=process_name(column))

    title = f"{'���' if categories == 'style' else '��ҵ'}ָ������ͼ ({start_date} �� {end_date})"
    plt.title(title)
    plt.xlabel("����")
    plt.ylabel("�ǵ��� (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ��ȡ����
    data_path = r"D:\WPS_cloud\WPSDrive\13368898\WPS����\����-���\ר��-����\�������ҵ����.xlsx"
    data = pd.read_excel(data_path)
    # data = data.iloc[1:]
    data.columns = data.iloc[0]
    data = data.drop(data.index[0]).reset_index(drop=True)
    data = data[data['ָ������'] != 'ָ��ID']
    data["ָ������"] = pd.to_datetime(data["ָ������"])

    # �û�����
    # start_date = input("�����뿪ʼ���� (��ʽ: YYYY-MM-DD): ")
    # end_date = input("������������� (��ʽ: YYYY-MM-DD): ")
    # category = input("��ѡ��Ҫ���Ƶ���� ('style' �� 'industry'): ")
    start_date = "2019-12-31"
    end_date = "2020-03-23"
    category = "industry"

    specific_categories = None
    if category == 'industry':
        # specific_categories = input(
        #     "������Ҫ���Ƶ���ҵ���ö��ŷָ� (����: 'ʯ��ʯ��,ú̿'). ���Ҫ����������ҵ����ֱ�Ӱ��س�: ")
        specific_categories = "ú̿,����,�����,����,�����豸������Դ,ҽҩ,����"
        if specific_categories:
            specific_categories = specific_categories.split(',')

    plot_trends(start_date, end_date, data, categories=category, specific_categories=specific_categories)
