# coding=gbk
# Time Created: 2023/8/25 11:01
# Author  : Lucid
# FileName: fupan_plot.py
# Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


def process_name(name):
    # 如果名称中包含“风格.中信”，则只提取括号前的部分
    if "风格.中信" in name:
        return name.split("(")[0]
    # 如果名称中包含“中信行业指数:”，则提取冒号后的部分
    elif "中信行业指数:" in name:
        return name.split(":")[1]
    else:
        return name


def plot_trends(start_date, end_date, data, categories='style', specific_categories=None):

    selected_data = data[(data["指标名称"] >= start_date) & (data["指标名称"] <= end_date)]

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
        plt.plot(selected_data["指标名称"], aligned_data[column], label=process_name(column))

    title = f"{'风格' if categories == 'style' else '行业'}指数走势图 ({start_date} 至 {end_date})"
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("涨跌幅 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 读取数据
    data_path = r"D:\WPS_cloud\WPSDrive\13368898\WPS云盘\工作-麦高\专题-复盘\风格与行业走势.xlsx"
    data = pd.read_excel(data_path)
    # data = data.iloc[1:]
    data.columns = data.iloc[0]
    data = data.drop(data.index[0]).reset_index(drop=True)
    data = data[data['指标名称'] != '指标ID']
    data["指标名称"] = pd.to_datetime(data["指标名称"])

    # 用户输入
    # start_date = input("请输入开始日期 (格式: YYYY-MM-DD): ")
    # end_date = input("请输入结束日期 (格式: YYYY-MM-DD): ")
    # category = input("请选择要绘制的类别 ('style' 或 'industry'): ")
    start_date = "2019-12-31"
    end_date = "2020-03-23"
    category = "industry"

    specific_categories = None
    if category == 'industry':
        # specific_categories = input(
        #     "请输入要绘制的行业，用逗号分隔 (例如: '石油石化,煤炭'). 如果要绘制所有行业，请直接按回车: ")
        specific_categories = "煤炭,钢铁,计算机,电子,电力设备及新能源,医药,银行"
        if specific_categories:
            specific_categories = specific_categories.split(',')

    plot_trends(start_date, end_date, data, categories=category, specific_categories=specific_categories)
