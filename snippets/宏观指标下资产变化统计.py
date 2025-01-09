# coding=gbk
# Time Created: 2024/11/14 14:52
# Author  : Lucid
# FileName: macro.py
# Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from itertools import product
from datetime import datetime
import os
from utils import process_wind_excel
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from current font.")

# TODO:
# 打印出最新的宏观状态和具体的月数（GDP季频的怎么处理的，应该线性插值）
# 把行或者列求和统计

def calculate_states(df, indicators):
    """
    计算每个宏观指标的上行（Up）和下行（Down）状态。
    """
    for indicator in indicators:
        # 计算变化量
        df[f'{indicator}_Change'] = df[indicator].diff()
        # 划分状态
        df[f'{indicator}_State'] = df[f'{indicator}_Change'].apply(lambda x: 'Up' if x > 0 else 'Down')
    # 删除第一行（因 diff 产生 NaN）
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_monthly_returns(asset_df, assets):
    """
    计算每个资产的月度涨跌幅（百分比）。
    """
    asset_df['日期'] = pd.to_datetime(asset_df['日期'])
    asset_df = asset_df.sort_values('日期')
    asset_monthly = asset_df.resample('M', on='日期').last()
    returns_df = asset_monthly[assets].pct_change().dropna() * 100  # 百分比表示
    returns_df.index = returns_df.index.to_period('M').to_timestamp('M')
    return returns_df

def get_combinations(indicators):
    """
    生成状态组合标签的行和列。

    返回：
    - row_labels: 行标签列表 (如 'M2 Up', 'GDP Down', ...)
    - col_labels: 列标签列表 (同上)
    """
    states = ['Up', 'Down']
    # 对每个指标生成状态组合
    combinations = list(product(indicators, states))
    labels = [f"{indicator} {state}" for indicator, state in combinations]
    return labels, labels  # 行和列使用相同的标签


def plot_heatmap(data, title, highlight_rows, highlight_cols, asset, month_type, output_dir='heatmaps'):
    """
    绘制热力图，并高亮显示指定的单元格。

    参数:
    - data: DataFrame，包含平均涨跌幅
    - title: 图表标题
    - highlight_rows: 要高亮的行标签列表
    - highlight_cols: 要高亮的列标签列表
    - asset: 资产名称
    - month_type: 'Current' 或 'Next'
    - output_dir: 保存热力图的目录
    """
    # 设置支持中文的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 根据实际路径调整
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(15, 11))
    sns.set(font_scale=1)

    # 计算色彩映射的范围，以0为中心
    max_abs_value = max(data.max().max(), abs(data.min().min()))
    vmin = -max_abs_value
    vmax = max_abs_value

    # 绘制热力图，fmt='.1f' 仅显示数字，后续添加 '%'
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=.5,
        linecolor='gray',
        vmin=vmin, vmax=vmax,  # 设置色彩映射的范围
        center=0,  # 确保0值在中间
        cbar_kws={'label': 'Average Return (%)'}
    )

    # 高亮显示多个单元格
    for row_label in highlight_rows:
        if row_label in data.index:
            for col_label in highlight_cols:
                if col_label in data.columns:
                    try:
                        # 获取单元格值
                        val = data.at[row_label, col_label]
                        if pd.notna(val):
                            # 添加加粗的黄色边框
                            ax.add_patch(
                                plt.Rectangle((list(data.columns).index(col_label), list(data.index).index(row_label)),
                                              1, 1,
                                              fill=False, edgecolor='yellow', lw=2))
                    except KeyError:
                        print(f"未找到高亮条件: 行='{row_label}', 列='{col_label}'")

    # 在每个注释后添加 %
    for text in ax.texts:
        original_text = text.get_text()
        if original_text != 'nan':
            text.set_text(f"{original_text}%")
        else:
            text.set_text("N/A")

    # 调整图表边距，防止标题和标签被遮挡
    plt.subplots_adjust(top=0.99, bottom=0.45, left=0.2, right=0.95)

    # 使用显式字体属性
    plt.title(f'{asset} - {month_type} Month Asset Performance\n', fontsize=24, fontproperties=prop, y=1.02)
    # plt.xlabel('列宏观状态', fontsize=18, fontproperties=prop)
    # plt.ylabel('行宏观状态', fontsize=18, fontproperties=prop)
    plt.xticks(rotation=45, ha='right', fontsize=14, fontproperties=prop)
    plt.yticks(rotation=0, fontsize=14, fontproperties=prop)

    plt.tight_layout()

    # 显示图表
    plt.show()
    # 如果需要保存图表，可以取消注释以下两行
    # plt.savefig(os.path.join(output_dir, f'{asset}_{month_type}_Performance_Heatmap.png'))
    # plt.close()

def main(excel_path, assets_to_plot=None):
    # 读取宏观指标和资产价格两个 sheet
    macro_meta, macro_df = process_wind_excel(
        excel_file_path=excel_path,
        sheet_name='宏观指标',
        column_name='指标名称'
    )
    asset_meta, asset_df = process_wind_excel(
        excel_file_path=excel_path,
        sheet_name='资产价格',
        column_name='指标名称'
    )

    # 将索引转换为列并重命名
    macro_df.reset_index(inplace=True)
    macro_df.rename(columns={macro_df.columns[0]: '日期'}, inplace=True)

    asset_df.reset_index(inplace=True)
    asset_df.rename(columns={asset_df.columns[0]: '日期'}, inplace=True)

    # 将日期转换为 datetime 类型
    macro_df['日期'] = pd.to_datetime(macro_df['日期'])
    asset_df['日期'] = pd.to_datetime(asset_df['日期'])

    # 按日期排序
    macro_df = macro_df.sort_values('日期').reset_index(drop=True)
    asset_df = asset_df.sort_values('日期').reset_index(drop=True)

    # 处理GDP等季频指标，重新采样为月频并前向填充（填充未来）
    macro_df = macro_df.set_index('日期')
    macro_df = macro_df.resample('M').last()
    macro_df = macro_df.ffill().reset_index()
    # 删除最新月份的数据
    latest_month = macro_df.index.max()
    macro_df = macro_df.drop(latest_month)
    macro_df['年月'] = macro_df['日期'].dt.to_period('M')

    # 确定宏观指标列，假设除了 '日期' 和 '年月' 之外的列都是宏观指标
    macro_indicators = [col for col in macro_df.columns if col not in ['日期', '年月']]

    print(f"宏观指标列表: {macro_indicators}")

    # 处理宏观指标数据，计算每个指标的状态（上行/下行）
    macro_df = calculate_states(macro_df, macro_indicators)

    # 处理资产价格数据，计算月度涨跌幅（百分比）
    assets = [col for col in asset_df.columns if col != '日期']

    # 如果提供了 assets_to_plot，则只处理指定的资产
    if assets_to_plot:
        assets = [asset for asset in assets if asset in assets_to_plot]
        if not assets:
            print(f"未找到匹配的资产: {assets_to_plot}")
            return
    print(f"处理的资产列表: {assets}")

    returns_df = calculate_monthly_returns(asset_df, assets)

    # 将宏观数据的日期设为索引并对齐资产回报数据的日期
    macro_df.set_index('日期', inplace=True)
    returns_df.index = returns_df.index.to_period('M').to_timestamp('M')

    # 删除宏观数据中不在资产回报数据中的日期
    macro_df = macro_df[macro_df.index.isin(returns_df.index)]

    # 生成行和列的标签
    row_labels, col_labels = get_combinations(macro_indicators)

    # 初始化结果字典
    current_results = {asset: pd.DataFrame(np.nan, index=row_labels, columns=col_labels) for asset in assets}
    next_results = {asset: pd.DataFrame(np.nan, index=row_labels, columns=col_labels) for asset in assets}

    # 为每个资产统计当月和次月的平均涨跌幅
    for asset in assets:
        print(f"正在处理资产: {asset} (当前月)")
        for row in row_labels:
            row_indicator, row_state = row.split(' ')
            for col in col_labels:
                col_indicator, col_state = col.split(' ')
                # 定义条件：row_indicator 在 row_state AND col_indicator 在 col_state
                condition = (
                    (macro_df[f'{row_indicator}_State'] == row_state) &
                    (macro_df[f'{col_indicator}_State'] == col_state)
                )
                # 筛选符合条件的月份
                subset = macro_df[condition]
                # 计算平均涨跌幅
                average_return = returns_df.loc[subset.index, asset].mean()
                # 填充结果
                current_results[asset].at[row, col] = round(average_return, 2) if not np.isnan(average_return) else np.nan

        print(f"正在处理资产: {asset} (次月)")
        for row in row_labels:
            row_indicator, row_state = row.split(' ')
            for col in col_labels:
                col_indicator, col_state = col.split(' ')
                # 定义条件
                condition = (
                    (macro_df[f'{row_indicator}_State'] == row_state) &
                    (macro_df[f'{col_indicator}_State'] == col_state)
                )
                # 获取对应的次月
                current_month_indices = macro_df[condition].index
                next_month_indices = current_month_indices + pd.DateOffset(months=1)
                next_month_indices = next_month_indices[next_month_indices.isin(returns_df.index)]

                if not next_month_indices.empty:
                    average_return = returns_df.loc[next_month_indices, asset].mean()
                else:
                    average_return = np.nan

                # 填充结果
                next_results[asset].at[row, col] = round(average_return, 2) if not np.isnan(average_return) else np.nan

    # 确定最新的宏观条件
    latest_macro = macro_df.iloc[-1]
    latest_conditions = {indicator: latest_macro[f'{indicator}_State'] for indicator in macro_indicators}
    latest_date = latest_macro.name
    print(f"最新的宏观条件 (截至 {latest_date.strftime('%Y-%m')}):")
    for indicator in macro_indicators:
        state = latest_conditions[indicator]
        value = latest_macro[indicator]
        print(f"  - {indicator}: {state} (值: {value})")

    # 自动生成需要高亮的行和列标签
    highlight_rows = [f"{indicator} {latest_conditions[indicator]}" for indicator in macro_indicators]
    highlight_cols = highlight_rows.copy()  # 假设行和列的宏观指标相同

    print(f"高亮行标签: {highlight_rows}")
    print(f"高亮列标签: {highlight_cols}")

    # 为每个资产生成热力图并计算高亮单元格的平均值
    for asset in assets:
        print(f"生成 {asset} 的当月热力图...")
        current_data = current_results[asset]
        # 计算被高亮单元格的平均值
        highlighted_values = []
        for row_label in highlight_rows:
            if row_label in current_data.index:
                for col_label in highlight_cols:
                    if col_label in current_data.columns:
                        value = current_data.at[row_label, col_label]
                        if not np.isnan(value):
                            highlighted_values.append(value)

        if highlighted_values:
            average_highlighted_value = sum(highlighted_values) / len(highlighted_values)
            print(f"{asset} (当月) 被高亮单元格的平均值: {average_highlighted_value:.2f}%")
        else:
            print(f"{asset} (当月) 没有被高亮且有值的单元格。")

        plot_heatmap(
            data=current_data,
            title='Current Month Asset Performance',
            highlight_rows=highlight_rows,
            highlight_cols=highlight_cols,
            asset=asset,
            month_type='Current',
            output_dir='heatmaps'
        )

        print(f"生成 {asset} 的次月热力图...")
        next_data = next_results[asset]
        # 计算被高亮单元格的平均值
        highlighted_values = []
        for row_label in highlight_rows:
            if row_label in next_data.index:
                for col_label in highlight_cols:
                    if col_label in next_data.columns:
                        value = next_data.at[row_label, col_label]
                        if not np.isnan(value):
                            highlighted_values.append(value)

        if highlighted_values:
            average_highlighted_value = sum(highlighted_values) / len(highlighted_values)
            print(f"{asset} (次月) 被高亮单元格的平均值: {average_highlighted_value:.2f}%")
        else:
            print(f"{asset} (次月) 没有被高亮且有值的单元格。")

        plot_heatmap(
            data=next_data,
            title='Next Month Asset Performance',
            highlight_rows=highlight_rows,
            highlight_cols=highlight_cols,
            asset=asset,
            month_type='Next',
            output_dir='heatmaps'
        )

    # 准备数据以便上传至 Metabase
    output_data_dir = 'metabase_data'
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    for asset in assets:
        # 当月
        current_avg = current_results[asset].reset_index().melt(
            id_vars='index',
            var_name='Column Indicators',
            value_name='Average_Return'
        )
        current_avg.rename(columns={'index': 'Row Indicators'}, inplace=True)
        current_avg['Month_Type'] = 'Current'
        current_avg['Asset'] = asset

        # 次月
        next_avg = next_results[asset].reset_index().melt(
            id_vars='index',
            var_name='Column Indicators',
            value_name='Average_Return'
        )
        next_avg.rename(columns={'index': 'Row Indicators'}, inplace=True)
        next_avg['Month_Type'] = 'Next'
        next_avg['Asset'] = asset

        # 合并当前月和次月
        combined_data = pd.concat([current_avg, next_avg], ignore_index=True)

        # 拆分行指标和列指标
        for indicator in macro_indicators:
            combined_data[f'Row_{indicator}'] = combined_data['Row Indicators'].apply(
                lambda x: x.split(' ')[1] if f"{indicator} " in x else 'Down')  # 默认填充为 'Down'
        for indicator in macro_indicators:
            combined_data[f'Column_{indicator}'] = combined_data['Column Indicators'].apply(
                lambda x: x.split(' ')[1] if f"{indicator} " in x else 'Down')  # 默认填充为 'Down'

        # 保留需要的列
        metabase_df = combined_data[[f'Row_{indicator}' for indicator in macro_indicators] +
                                    [f'Column_{indicator}' for indicator in macro_indicators] +
                                    ['Average_Return', 'Month_Type', 'Asset']]

        # 重命名列
        rename_dict = {}
        for indicator in macro_indicators:
            rename_dict[f'Row_{indicator}'] = f'Row_{indicator}_State'
            rename_dict[f'Column_{indicator}'] = f'Column_{indicator}_State'
        metabase_df.rename(columns=rename_dict, inplace=True)

        # 填充缺失的列指标状态为 'Down'（或其他逻辑）
        for indicator in macro_indicators:
            metabase_df[f'Row_{indicator}_State'] = metabase_df[f'Row_{indicator}_State'].fillna('Down')
            metabase_df[f'Column_{indicator}_State'] = metabase_df[f'Column_{indicator}_State'].fillna('Down')

        # 保存为 CSV
        metabase_df.to_csv(os.path.join(output_data_dir, f'{asset}_Performance_Data.csv'), index=False)

    print("热力图已生成并保存。Metabase 数据已准备。")

if __name__ == "__main__":
    # 请将以下路径替换为您的实际 Excel 文件路径
    excel_path = r"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\宏观影响风格数据源.xlsx"
    # 如果需要指定要绘制的资产列表，例如 ['资产1', '资产2']，可以传入参数
    # assets_to_plot = None  # 或者 ['资产1', '资产2']
    assets_to_plot = ['中证1000指数/沪深300指数', '中证1000指数', '沪深300指数']
    main(excel_path, assets_to_plot)