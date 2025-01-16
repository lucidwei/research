# coding=gbk
# Time Created: 2025/1/16 15:32
# Author  : Lucid
# FileName: jq_results_stats.py
# Software: PyCharm

import re
import csv
from datetime import datetime, timedelta
import chardet
import pandas as pd

def count_trading_days(start_date, end_date, trading_days):
    current_date = start_date
    trading_day_count = 0
    while current_date <= end_date:
        if current_date.strftime("%Y-%m-%d") in trading_days:
            trading_day_count += 1
        current_date += timedelta(days=1)
    return trading_day_count

# ����������ʽƥ��ģʽ
open_trade_pattern = re.compile(r"���տ��֣�(\d{4}-\d{2}-\d{2})")
close_trade_pattern = re.compile(r"ƽ�֣�(\d{4}-\d{2}-\d{2})")
reason_pattern = re.compile(r"(�쳣����|�쳣����)��(\d{4}-\d{2}-\d{2})")
signal_trigger_pattern = re.compile(r"�źŴ����������տ��֣�(\d{4}-\d{2}-\d{2})")

# �������б�
results = []

# ����ļ�����
log_file_path = rf"D:\Downloads\log\log.txt"
with open(log_file_path, "rb") as f:
    raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']

# ��ȡtxt�ļ�
with open(log_file_path, "r", encoding=detected_encoding) as file:
    lines = file.readlines()

# ��ʼ������
open_date = None
open_reason = None
pending_reason = None  # ���ڱ��桰�����տ��֡�������
trading_days = set()

# �����ļ�����
for line in lines:
    if "һ�����" in line:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", line)
        if match:
            trading_days.add(match.group(1))

    # ����Ƿ��п�������
    reason_match = reason_pattern.search(line)
    if reason_match:
        pending_reason = reason_match.group(1)  # ���浱ǰ��������
        continue

    # ����Ƿ����źŴ���
    signal_trigger_match = signal_trigger_pattern.search(line)
    if signal_trigger_match:
        # ������źŴ������������ɣ��ȴ����տ���
        continue

    # ����Ƿ��п���ʱ��
    open_trade_match = open_trade_pattern.search(line)
    if open_trade_match:
        open_date = open_trade_match.group(1)
        open_reason = pending_reason  # ʹ��֮ǰ���������
        pending_reason = None  # ��������
        continue

    # ����Ƿ���ƽ��ʱ��
    close_trade_match = close_trade_pattern.search(line)
    if close_trade_match and open_date:
        close_date = close_trade_match.group(1)

        # ����ֲ�ʱ���ͽ����ճֲ�ʱ��
        open_datetime = datetime.strptime(open_date, "%Y-%m-%d")
        close_datetime = datetime.strptime(close_date, "%Y-%m-%d")
        holding_duration = (close_datetime - open_datetime).days
        trading_duration = count_trading_days(open_datetime, close_datetime, trading_days)

        # ��������浽�б�
        results.append({
            "����ʱ��": open_date,
            "ƽ��ʱ��": close_date,
            "�ֲ�ʱ��": holding_duration,
            "�����ճֲ�ʱ��": trading_duration,
            "��������": open_reason
        })

        # ���ÿ���ʱ�������
        open_date = None
        open_reason = None

# ��ȡƽ��ӯ����CSV�ļ�
profit_loss_file = rf"D:\Downloads\transaction\transaction.csv"
profit_loss_data = {}

# �洢����������۸����ʱ����
last_buy_price = None

with open(profit_loss_file, "r", encoding=detected_encoding) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # ��ȡ�������ͣ���������ͳɽ���
        trade_type = row["��������"]
        trade_price = float(row["�ɽ���"])
        trade_date = datetime.strptime(row["����"], "%Y-%m-%d").strftime("%Y-%m-%d")

        if trade_type == "��":
            # ��¼����۸�
            last_buy_price = trade_price
        elif trade_type == "��" and last_buy_price is not None:
            # ����������
            return_rate = ((trade_price - last_buy_price) / last_buy_price) * 100
            # ��ƽ��ӯ���������ʴ洢���ֵ���
            profit_loss_data[trade_date] = {
                "ƽ��ӯ��": row["ƽ��ӯ��"],
                "������": round(return_rate, 2)  # �洢Ϊ�������������������
            }
            # ��������۸�
            last_buy_price = None

# ��ƽ��ӯ������������Ϣ������
for result in results:
    close_date = result["ƽ��ʱ��"]
    if close_date in profit_loss_data:
        result["ƽ��ӯ��"] = profit_loss_data[close_date]["ƽ��ӯ��"]
        result["������"] = profit_loss_data[close_date]["������"]
    else:
        result["ƽ��ӯ��"] = "�޼�¼"
        result["������"] = None  # ʹ��None�Ա��������

# �����ת��ΪDataFrame
df = pd.DataFrame(results)

# ɾ��������ΪNone�ļ�¼
df = df.dropna(subset=["������"])

# ����һ�����������㿭����λ
def calculate_kelly(win_rate, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    return (win_rate - (1 - win_rate) / (avg_win / avg_loss)) * 100  # ת��Ϊ�ٷֱ�

# ����һ������������ͳ��ָ��
def compute_stats(group):
    count = len(group)
    avg_return = group["������"].mean()
    avg_holding_days = group["�����ճֲ�ʱ��"].mean()  # ʹ�ý����ճֲ�ʱ��

    # ����ʤ��
    wins = group[group["������"] > 0]
    win_rate = len(wins) / count if count > 0 else 0

    # �������ʣ�ƽ��ӯ�� / ƽ������
    avg_win = wins["������"].mean() if not wins.empty else 0
    losses = group[group["������"] <= 0]
    avg_loss = losses["������"].abs().mean() if not losses.empty else 0
    odds = avg_win / avg_loss if avg_loss != 0 else 0

    # ���㿭����λ
    kelly = calculate_kelly(win_rate, avg_win, avg_loss)

    return pd.Series({
        "����": count,
        "ƽ�������� (%)": round(avg_return, 2),
        "ƽ���ֲ�����": round(avg_holding_days, 2),
        "ʤ��": f"{round(win_rate * 100, 2)}%",
        "����": round(odds, 2),
        "������λ (%)": round(kelly, 2)
    })

# ���������ɷ��鲢����ͳ��ָ��
summary_all = df.groupby("��������").apply(compute_stats).reset_index()

# ����Ҫ�޳����ض�����
excluded_trade = {
    "����ʱ��": "2016-01-08",
    "ƽ��ʱ��": "2016-02-29"
}

# �����쳣���������࣬��Ϊȫ�����޳��ض�����
# ��ȡ���С��쳣����������
shrink_all = df[df["��������"] == "�쳣����"]

# ��ȡ�޳��ض����׺�ġ��쳣����������
shrink_excluded = shrink_all[
    ~((shrink_all["����ʱ��"] == excluded_trade["����ʱ��"]) &
      (shrink_all["ƽ��ʱ��"] == excluded_trade["ƽ��ʱ��"]))
]

# ���㡰�쳣������ȫ��ͳ��
stats_shrink_all = compute_stats(shrink_all)
stats_shrink_all.name = "�쳣������ȫ����"

# ���㡰�쳣�������޳��ض����׺��ͳ��
stats_shrink_excluded = compute_stats(shrink_excluded)
stats_shrink_excluded.name = "�쳣�������޳��ض����ף�"

# ��������ͳ�ƽ��ת��ΪDataFrame
summary_shrink_all = stats_shrink_all.to_frame().T
summary_shrink_excluded = stats_shrink_excluded.to_frame().T

# �ϲ�������ͳ�ƽ��
summary_shrink = pd.concat([summary_shrink_all, summary_shrink_excluded], ignore_index=True)

# ����summary_all���Ƴ����쳣��������ԭʼͳ�ƣ����滻Ϊ�µ�ͳ��
summary_all = summary_all[summary_all["��������"] != "�쳣����"]
summary_all = pd.concat([summary_all, summary_shrink], ignore_index=True)

# ׼����ϸ�ķ������ݣ�������Ҫ����Ϊ�Ѻϲ���summary_all��

# ʹ��ExcelWriter���浽Excel�ļ��Ĳ�ͬSheet
output_excel = rf"D:\WPS����\WPS����\����-���\ר���о�\��Ƶ��ʱ\trading_summary1.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # д����ϸ���׼�¼��Sheet1
    df.to_excel(writer, sheet_name='���׼�¼', index=False)

    # д���ܽᵽSheet2
    summary_all.to_excel(writer, sheet_name='ͳ���ܽ�', index=False)

    # ���⣬Ҳ���Ե���д�롰�쳣����������ϸͳ��
    # д�롰�쳣�������ܽᵽSheet3
    summary_shrink.to_excel(writer, sheet_name='�쳣����', index=False)

    # д�롰�쳣�������ܽᵽSheet4
    summary_expand = summary_all[summary_all["��������"] == "�쳣����"]
    summary_expand.to_excel(writer, sheet_name='�쳣����', index=False)

print(f"ͳ�ƽ���ѱ��浽 {output_excel} �ļ��С�")