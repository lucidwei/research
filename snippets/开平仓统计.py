# coding=gbk
# Time Created: 2024/12/27 10:53
# Author  : Lucid
# FileName: ��ƽ��ͳ��.py
# Software: PyCharm

import re
import csv
from datetime import datetime
import chardet
from datetime import timedelta


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

        # ����ֲ�ʱ��
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
                "������": f"{return_rate:.2f}%"
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
        result["������"] = "�޼�¼"

# д�뵽csv�ļ�
with open(rf"D:\Downloads\log\trading_summary3.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["����ʱ��", "ƽ��ʱ��", "�ֲ�ʱ��", '�����ճֲ�ʱ��', "��������", "ƽ��ӯ��", "������"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # д���ͷ
    writer.writeheader()

    # д������
    writer.writerows(results)

print("ͳ�ƽ���ѱ��浽 trading_summary3.csv �ļ��С�")