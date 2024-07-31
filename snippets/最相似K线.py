# coding=gbk
# Time Created: 2024/7/31 16:42
# Author  : Lucid
# FileName: ������K��.py
# Software: PyCharm
import pandas as pd
import numpy as np

# ��ȡExcel�ļ�
file_path = 'D:\\Downloads\\000852.SH.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')
WINDOW_SIZE = 7

# ��ȡ����к�����������
data = data[['����', '���̼�(Ԫ)', '��߼�(Ԫ)', '��ͼ�(Ԫ)', '���̼�(Ԫ)', '�ɽ���(����)']]
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# ת��������Ϊ���ڸ�ʽ
data['Date'] = pd.to_datetime(data['Date'])

# �����ǵ���
data['Pct_Change'] = data['Close'].pct_change()
data.dropna(inplace=True)

# �����ֵ�ͱ�׼��
means = data.tail(300)[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
std_devs = data.tail(300)[['Open', 'High', 'Low', 'Close', 'Volume']].std()
volatility = std_devs / means
vol_scaler = volatility['Volume'] / volatility['Close']

# ��ȡ������������
recent_data = data.tail(WINDOW_SIZE)

# ��׼������
def standardize(df):
    standardized = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        first_value = df[col].iloc[0]
        standardized[col] = df[col] / first_value
    return standardized
recent_standardized = standardize(recent_data)


# ����������ȡ��ʷ���������
def get_windows(df, window_size=WINDOW_SIZE):
    windows = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size].copy()
        window = standardize(window)
        windows.append(window)
    return windows
windows = get_windows(data)


# �������ƶ�
def calculate_similarity(window, target):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Pct_Change']
    # window_values = window[features].dropna().values.flatten()
    # target_values = target[features].dropna().values.flatten()

    # �����ݽ�������
    window_scaled = window[features].copy()
    target_scaled = target[features].copy()

    window_scaled['Volume'] = window['Volume'] / vol_scaler
    target_scaled['Volume'] = target['Volume'] / vol_scaler

    # չƽ����
    window_values = window_scaled.dropna().values.flatten()
    target_values = target_scaled.dropna().values.flatten()

    # ��������ڵ����ݲ��㣬����һ���ǳ����ֵ
    if len(window_values) != len(target_values):
        return np.inf

    # ����ŷ�Ͼ���
    distance = np.linalg.norm(window_values - target_values)
    return distance


# �������ƶȲ�����
similarities = [calculate_similarity(window, recent_standardized) for window in windows]
sorted_indices = np.argsort(similarities)[:5]
most_similar_windows = [(windows[i], similarities[i]) for i in sorted_indices]

# ��ӡ�����Ƶ��������ڼ������ƶ�
for idx, (window, similarity) in enumerate(most_similar_windows):
    print(f"��{idx}�������ƵĴ�����ʼ���ڣ�{window.iloc[0]['Date']}")
    # print(f"���ƶȣ�{similarity:.3f}")
    # print(window)
    # print()