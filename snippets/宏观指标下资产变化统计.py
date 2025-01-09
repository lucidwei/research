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

# �����ض��ľ���
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from current font.")

# TODO:
# ��ӡ�����µĺ��״̬�;����������GDP��Ƶ����ô����ģ�Ӧ�����Բ�ֵ��
# ���л��������ͳ��

def calculate_states(df, indicators):
    """
    ����ÿ�����ָ������У�Up�������У�Down��״̬��
    """
    for indicator in indicators:
        # ����仯��
        df[f'{indicator}_Change'] = df[indicator].diff()
        # ����״̬
        df[f'{indicator}_State'] = df[f'{indicator}_Change'].apply(lambda x: 'Up' if x > 0 else 'Down')
    # ɾ����һ�У��� diff ���� NaN��
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_monthly_returns(asset_df, assets):
    """
    ����ÿ���ʲ����¶��ǵ������ٷֱȣ���
    """
    asset_df['����'] = pd.to_datetime(asset_df['����'])
    asset_df = asset_df.sort_values('����')
    asset_monthly = asset_df.resample('M', on='����').last()
    returns_df = asset_monthly[assets].pct_change().dropna() * 100  # �ٷֱȱ�ʾ
    returns_df.index = returns_df.index.to_period('M').to_timestamp('M')
    return returns_df

def get_combinations(indicators):
    """
    ����״̬��ϱ�ǩ���к��С�

    ���أ�
    - row_labels: �б�ǩ�б� (�� 'M2 Up', 'GDP Down', ...)
    - col_labels: �б�ǩ�б� (ͬ��)
    """
    states = ['Up', 'Down']
    # ��ÿ��ָ������״̬���
    combinations = list(product(indicators, states))
    labels = [f"{indicator} {state}" for indicator, state in combinations]
    return labels, labels  # �к���ʹ����ͬ�ı�ǩ


def plot_heatmap(data, title, highlight_rows, highlight_cols, asset, month_type, output_dir='heatmaps'):
    """
    ��������ͼ����������ʾָ���ĵ�Ԫ��

    ����:
    - data: DataFrame������ƽ���ǵ���
    - title: ͼ�����
    - highlight_rows: Ҫ�������б�ǩ�б�
    - highlight_cols: Ҫ�������б�ǩ�б�
    - asset: �ʲ�����
    - month_type: 'Current' �� 'Next'
    - output_dir: ��������ͼ��Ŀ¼
    """
    # ����֧�����ĵ�����
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # ����ʵ��·������
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False  # ������ʾ����

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(15, 11))
    sns.set(font_scale=1)

    # ����ɫ��ӳ��ķ�Χ����0Ϊ����
    max_abs_value = max(data.max().max(), abs(data.min().min()))
    vmin = -max_abs_value
    vmax = max_abs_value

    # ��������ͼ��fmt='.1f' ����ʾ���֣�������� '%'
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=.5,
        linecolor='gray',
        vmin=vmin, vmax=vmax,  # ����ɫ��ӳ��ķ�Χ
        center=0,  # ȷ��0ֵ���м�
        cbar_kws={'label': 'Average Return (%)'}
    )

    # ������ʾ�����Ԫ��
    for row_label in highlight_rows:
        if row_label in data.index:
            for col_label in highlight_cols:
                if col_label in data.columns:
                    try:
                        # ��ȡ��Ԫ��ֵ
                        val = data.at[row_label, col_label]
                        if pd.notna(val):
                            # ��ӼӴֵĻ�ɫ�߿�
                            ax.add_patch(
                                plt.Rectangle((list(data.columns).index(col_label), list(data.index).index(row_label)),
                                              1, 1,
                                              fill=False, edgecolor='yellow', lw=2))
                    except KeyError:
                        print(f"δ�ҵ���������: ��='{row_label}', ��='{col_label}'")

    # ��ÿ��ע�ͺ���� %
    for text in ax.texts:
        original_text = text.get_text()
        if original_text != 'nan':
            text.set_text(f"{original_text}%")
        else:
            text.set_text("N/A")

    # ����ͼ��߾࣬��ֹ����ͱ�ǩ���ڵ�
    plt.subplots_adjust(top=0.99, bottom=0.45, left=0.2, right=0.95)

    # ʹ����ʽ��������
    plt.title(f'{asset} - {month_type} Month Asset Performance\n', fontsize=24, fontproperties=prop, y=1.02)
    # plt.xlabel('�к��״̬', fontsize=18, fontproperties=prop)
    # plt.ylabel('�к��״̬', fontsize=18, fontproperties=prop)
    plt.xticks(rotation=45, ha='right', fontsize=14, fontproperties=prop)
    plt.yticks(rotation=0, fontsize=14, fontproperties=prop)

    plt.tight_layout()

    # ��ʾͼ��
    plt.show()
    # �����Ҫ����ͼ������ȡ��ע����������
    # plt.savefig(os.path.join(output_dir, f'{asset}_{month_type}_Performance_Heatmap.png'))
    # plt.close()

def main(excel_path, assets_to_plot=None):
    # ��ȡ���ָ����ʲ��۸����� sheet
    macro_meta, macro_df = process_wind_excel(
        excel_file_path=excel_path,
        sheet_name='���ָ��',
        column_name='ָ������'
    )
    asset_meta, asset_df = process_wind_excel(
        excel_file_path=excel_path,
        sheet_name='�ʲ��۸�',
        column_name='ָ������'
    )

    # ������ת��Ϊ�в�������
    macro_df.reset_index(inplace=True)
    macro_df.rename(columns={macro_df.columns[0]: '����'}, inplace=True)

    asset_df.reset_index(inplace=True)
    asset_df.rename(columns={asset_df.columns[0]: '����'}, inplace=True)

    # ������ת��Ϊ datetime ����
    macro_df['����'] = pd.to_datetime(macro_df['����'])
    asset_df['����'] = pd.to_datetime(asset_df['����'])

    # ����������
    macro_df = macro_df.sort_values('����').reset_index(drop=True)
    asset_df = asset_df.sort_values('����').reset_index(drop=True)

    # ����GDP�ȼ�Ƶָ�꣬���²���Ϊ��Ƶ��ǰ����䣨���δ����
    macro_df = macro_df.set_index('����')
    macro_df = macro_df.resample('M').last()
    macro_df = macro_df.ffill().reset_index()
    # ɾ�������·ݵ�����
    latest_month = macro_df.index.max()
    macro_df = macro_df.drop(latest_month)
    macro_df['����'] = macro_df['����'].dt.to_period('M')

    # ȷ�����ָ���У�������� '����' �� '����' ֮����ж��Ǻ��ָ��
    macro_indicators = [col for col in macro_df.columns if col not in ['����', '����']]

    print(f"���ָ���б�: {macro_indicators}")

    # ������ָ�����ݣ�����ÿ��ָ���״̬������/���У�
    macro_df = calculate_states(macro_df, macro_indicators)

    # �����ʲ��۸����ݣ������¶��ǵ������ٷֱȣ�
    assets = [col for col in asset_df.columns if col != '����']

    # ����ṩ�� assets_to_plot����ֻ����ָ�����ʲ�
    if assets_to_plot:
        assets = [asset for asset in assets if asset in assets_to_plot]
        if not assets:
            print(f"δ�ҵ�ƥ����ʲ�: {assets_to_plot}")
            return
    print(f"������ʲ��б�: {assets}")

    returns_df = calculate_monthly_returns(asset_df, assets)

    # ��������ݵ�������Ϊ�����������ʲ��ر����ݵ�����
    macro_df.set_index('����', inplace=True)
    returns_df.index = returns_df.index.to_period('M').to_timestamp('M')

    # ɾ����������в����ʲ��ر������е�����
    macro_df = macro_df[macro_df.index.isin(returns_df.index)]

    # �����к��еı�ǩ
    row_labels, col_labels = get_combinations(macro_indicators)

    # ��ʼ������ֵ�
    current_results = {asset: pd.DataFrame(np.nan, index=row_labels, columns=col_labels) for asset in assets}
    next_results = {asset: pd.DataFrame(np.nan, index=row_labels, columns=col_labels) for asset in assets}

    # Ϊÿ���ʲ�ͳ�Ƶ��ºʹ��µ�ƽ���ǵ���
    for asset in assets:
        print(f"���ڴ����ʲ�: {asset} (��ǰ��)")
        for row in row_labels:
            row_indicator, row_state = row.split(' ')
            for col in col_labels:
                col_indicator, col_state = col.split(' ')
                # ����������row_indicator �� row_state AND col_indicator �� col_state
                condition = (
                    (macro_df[f'{row_indicator}_State'] == row_state) &
                    (macro_df[f'{col_indicator}_State'] == col_state)
                )
                # ɸѡ�����������·�
                subset = macro_df[condition]
                # ����ƽ���ǵ���
                average_return = returns_df.loc[subset.index, asset].mean()
                # �����
                current_results[asset].at[row, col] = round(average_return, 2) if not np.isnan(average_return) else np.nan

        print(f"���ڴ����ʲ�: {asset} (����)")
        for row in row_labels:
            row_indicator, row_state = row.split(' ')
            for col in col_labels:
                col_indicator, col_state = col.split(' ')
                # ��������
                condition = (
                    (macro_df[f'{row_indicator}_State'] == row_state) &
                    (macro_df[f'{col_indicator}_State'] == col_state)
                )
                # ��ȡ��Ӧ�Ĵ���
                current_month_indices = macro_df[condition].index
                next_month_indices = current_month_indices + pd.DateOffset(months=1)
                next_month_indices = next_month_indices[next_month_indices.isin(returns_df.index)]

                if not next_month_indices.empty:
                    average_return = returns_df.loc[next_month_indices, asset].mean()
                else:
                    average_return = np.nan

                # �����
                next_results[asset].at[row, col] = round(average_return, 2) if not np.isnan(average_return) else np.nan

    # ȷ�����µĺ������
    latest_macro = macro_df.iloc[-1]
    latest_conditions = {indicator: latest_macro[f'{indicator}_State'] for indicator in macro_indicators}
    latest_date = latest_macro.name
    print(f"���µĺ������ (���� {latest_date.strftime('%Y-%m')}):")
    for indicator in macro_indicators:
        state = latest_conditions[indicator]
        value = latest_macro[indicator]
        print(f"  - {indicator}: {state} (ֵ: {value})")

    # �Զ�������Ҫ�������к��б�ǩ
    highlight_rows = [f"{indicator} {latest_conditions[indicator]}" for indicator in macro_indicators]
    highlight_cols = highlight_rows.copy()  # �����к��еĺ��ָ����ͬ

    print(f"�����б�ǩ: {highlight_rows}")
    print(f"�����б�ǩ: {highlight_cols}")

    # Ϊÿ���ʲ���������ͼ�����������Ԫ���ƽ��ֵ
    for asset in assets:
        print(f"���� {asset} �ĵ�������ͼ...")
        current_data = current_results[asset]
        # ���㱻������Ԫ���ƽ��ֵ
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
            print(f"{asset} (����) ��������Ԫ���ƽ��ֵ: {average_highlighted_value:.2f}%")
        else:
            print(f"{asset} (����) û�б���������ֵ�ĵ�Ԫ��")

        plot_heatmap(
            data=current_data,
            title='Current Month Asset Performance',
            highlight_rows=highlight_rows,
            highlight_cols=highlight_cols,
            asset=asset,
            month_type='Current',
            output_dir='heatmaps'
        )

        print(f"���� {asset} �Ĵ�������ͼ...")
        next_data = next_results[asset]
        # ���㱻������Ԫ���ƽ��ֵ
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
            print(f"{asset} (����) ��������Ԫ���ƽ��ֵ: {average_highlighted_value:.2f}%")
        else:
            print(f"{asset} (����) û�б���������ֵ�ĵ�Ԫ��")

        plot_heatmap(
            data=next_data,
            title='Next Month Asset Performance',
            highlight_rows=highlight_rows,
            highlight_cols=highlight_cols,
            asset=asset,
            month_type='Next',
            output_dir='heatmaps'
        )

    # ׼�������Ա��ϴ��� Metabase
    output_data_dir = 'metabase_data'
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    for asset in assets:
        # ����
        current_avg = current_results[asset].reset_index().melt(
            id_vars='index',
            var_name='Column Indicators',
            value_name='Average_Return'
        )
        current_avg.rename(columns={'index': 'Row Indicators'}, inplace=True)
        current_avg['Month_Type'] = 'Current'
        current_avg['Asset'] = asset

        # ����
        next_avg = next_results[asset].reset_index().melt(
            id_vars='index',
            var_name='Column Indicators',
            value_name='Average_Return'
        )
        next_avg.rename(columns={'index': 'Row Indicators'}, inplace=True)
        next_avg['Month_Type'] = 'Next'
        next_avg['Asset'] = asset

        # �ϲ���ǰ�ºʹ���
        combined_data = pd.concat([current_avg, next_avg], ignore_index=True)

        # �����ָ�����ָ��
        for indicator in macro_indicators:
            combined_data[f'Row_{indicator}'] = combined_data['Row Indicators'].apply(
                lambda x: x.split(' ')[1] if f"{indicator} " in x else 'Down')  # Ĭ�����Ϊ 'Down'
        for indicator in macro_indicators:
            combined_data[f'Column_{indicator}'] = combined_data['Column Indicators'].apply(
                lambda x: x.split(' ')[1] if f"{indicator} " in x else 'Down')  # Ĭ�����Ϊ 'Down'

        # ������Ҫ����
        metabase_df = combined_data[[f'Row_{indicator}' for indicator in macro_indicators] +
                                    [f'Column_{indicator}' for indicator in macro_indicators] +
                                    ['Average_Return', 'Month_Type', 'Asset']]

        # ��������
        rename_dict = {}
        for indicator in macro_indicators:
            rename_dict[f'Row_{indicator}'] = f'Row_{indicator}_State'
            rename_dict[f'Column_{indicator}'] = f'Column_{indicator}_State'
        metabase_df.rename(columns=rename_dict, inplace=True)

        # ���ȱʧ����ָ��״̬Ϊ 'Down'���������߼���
        for indicator in macro_indicators:
            metabase_df[f'Row_{indicator}_State'] = metabase_df[f'Row_{indicator}_State'].fillna('Down')
            metabase_df[f'Column_{indicator}_State'] = metabase_df[f'Column_{indicator}_State'].fillna('Down')

        # ����Ϊ CSV
        metabase_df.to_csv(os.path.join(output_data_dir, f'{asset}_Performance_Data.csv'), index=False)

    print("����ͼ�����ɲ����档Metabase ������׼����")

if __name__ == "__main__":
    # �뽫����·���滻Ϊ����ʵ�� Excel �ļ�·��
    excel_path = r"D:\WPS����\WPS����\����-���\�о�trial\���Ӱ��������Դ.xlsx"
    # �����Ҫָ��Ҫ���Ƶ��ʲ��б����� ['�ʲ�1', '�ʲ�2']�����Դ������
    # assets_to_plot = None  # ���� ['�ʲ�1', '�ʲ�2']
    assets_to_plot = ['��֤1000ָ��/����300ָ��', '��֤1000ָ��', '����300ָ��']
    main(excel_path, assets_to_plot)