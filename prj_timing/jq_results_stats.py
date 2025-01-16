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


# 定义正则表达式匹配模式
open_trade_pattern = re.compile(r"次日开仓：(\d{4}-\d{2}-\d{2})")
close_trade_pattern = re.compile(r"平仓：(\d{4}-\d{2}-\d{2})")
reason_pattern = re.compile(r"(异常缩量|异常放量)：(\d{4}-\d{2}-\d{2})")
signal_trigger_pattern = re.compile(r"信号触发，待次日开仓：(\d{4}-\d{2}-\d{2})")

# 定义结果列表
results = []

# 检测文件编码
log_file_path = rf"D:\Downloads\log\log.txt"
with open(log_file_path, "rb") as f:
    raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']

# 读取txt文件
with open(log_file_path, "r", encoding=detected_encoding) as file:
    lines = file.readlines()

# 初始化变量
open_date = None
open_reason = None
pending_reason = None  # 用于保存“待次日开仓”的理由
trading_days = set()

# 遍历文件内容
for line in lines:
    if "一天结束" in line:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", line)
        if match:
            trading_days.add(match.group(1))

    # 检查是否有开仓理由
    reason_match = reason_pattern.search(line)
    if reason_match:
        pending_reason = reason_match.group(1)  # 保存当前开仓理由
        continue

    # 检查是否有信号触发
    signal_trigger_match = signal_trigger_pattern.search(line)
    if signal_trigger_match:
        # 如果有信号触发，保存理由，等待次日开仓
        continue

    # 检查是否有开仓时间
    open_trade_match = open_trade_pattern.search(line)
    if open_trade_match:
        open_date = open_trade_match.group(1)
        open_reason = pending_reason  # 使用之前保存的理由
        pending_reason = None  # 重置理由
        continue

    # 检查是否有平仓时间
    close_trade_match = close_trade_pattern.search(line)
    if close_trade_match and open_date:
        close_date = close_trade_match.group(1)

        # 计算持仓时长
        open_datetime = datetime.strptime(open_date, "%Y-%m-%d")
        close_datetime = datetime.strptime(close_date, "%Y-%m-%d")
        holding_duration = (close_datetime - open_datetime).days
        trading_duration = count_trading_days(open_datetime, close_datetime, trading_days)

        # 将结果保存到列表
        results.append({
            "开仓时间": open_date,
            "平仓时间": close_date,
            "持仓时长": holding_duration,
            "交易日持仓时长": trading_duration,
            "开仓理由": open_reason
        })

        # 重置开仓时间和理由
        open_date = None
        open_reason = None

# 读取平仓盈亏的CSV文件
profit_loss_file = rf"D:\Downloads\transaction\transaction.csv"
profit_loss_data = {}

# 存储买入和卖出价格的临时变量
last_buy_price = None

with open(profit_loss_file, "r", encoding=detected_encoding) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 提取交易类型（买或卖）和成交价
        trade_type = row["交易类型"]
        trade_price = float(row["成交价"])
        trade_date = datetime.strptime(row["日期"], "%Y-%m-%d").strftime("%Y-%m-%d")

        if trade_type == "买":
            # 记录买入价格
            last_buy_price = trade_price
        elif trade_type == "卖" and last_buy_price is not None:
            # 计算收益率
            return_rate = ((trade_price - last_buy_price) / last_buy_price) * 100
            # 将平仓盈亏和收益率存储在字典中
            profit_loss_data[trade_date] = {
                "平仓盈亏": row["平仓盈亏"],
                "收益率": round(return_rate, 2)  # 存储为浮点数，方便后续计算
            }
            # 重置买入价格
            last_buy_price = None

# 将平仓盈亏和收益率信息加入结果
for result in results:
    close_date = result["平仓时间"]
    if close_date in profit_loss_data:
        result["平仓盈亏"] = profit_loss_data[close_date]["平仓盈亏"]
        result["收益率"] = profit_loss_data[close_date]["收益率"]
    else:
        result["平仓盈亏"] = "无记录"
        result["收益率"] = None  # 使用None以便后续处理

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 删除收益率为None的记录
df = df.dropna(subset=["收益率"])


# 定义一个函数来计算凯利仓位
def calculate_kelly(win_rate, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    return (win_rate - (1 - win_rate) / (avg_win / avg_loss)) * 100  # 转换为百分比


# 定义一个函数来计算统计指标
def compute_stats(group):
    count = len(group)
    avg_return = group["收益率"].mean()
    avg_holding_days = group["交易日持仓时长"].mean()

    # 计算胜率
    wins = group[group["收益率"] > 0]
    win_rate = len(wins) / count if count > 0 else 0

    # 计算赔率（平均盈利 / 平均亏损）
    avg_win = wins["收益率"].mean() if not wins.empty else 0
    losses = group[group["收益率"] <= 0]
    avg_loss = losses["收益率"].abs().mean() if not losses.empty else 0
    odds = avg_win / avg_loss if avg_loss != 0 else 0

    # 计算凯利仓位
    kelly = calculate_kelly(win_rate, avg_win, avg_loss)

    return pd.Series({
        "次数": count,
        "平均收益率 (%)": round(avg_return, 2),
        "平均持仓天数": round(avg_holding_days, 2),
        "胜率": f"{round(win_rate * 100, 2)}%",
        "赔率": round(odds, 2),
        "凯利仓位 (%)": round(kelly, 2)
    })


# 按开仓理由分组并计算统计指标
summary = df.groupby("开仓理由").apply(compute_stats).reset_index()

# 准备详细的分类数据
shrink = df[df["开仓理由"] == "异常缩量"]
expand = df[df["开仓理由"] == "异常放量"]

# 使用ExcelWriter保存到Excel文件的不同Sheet
output_excel = rf"D:\WPS云盘\WPS云盘\工作-麦高\专题研究\低频择时\trading_summary.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # 写入详细交易记录到Sheet1
    df.to_excel(writer, sheet_name='交易记录', index=False)

    # 写入“异常缩量”总结到Sheet2
    summary_shrink = compute_stats(shrink)
    summary_shrink.name = "异常缩量"
    summary_shrink_df = summary_shrink.to_frame().T
    summary_shrink_df.to_excel(writer, sheet_name='异常缩量', index=False)

    # 写入“异常放量”总结到Sheet3
    summary_expand = compute_stats(expand)
    summary_expand.name = "异常放量"
    summary_expand_df = summary_expand.to_frame().T
    summary_expand_df.to_excel(writer, sheet_name='异常放量', index=False)

print(f"统计结果已保存到 {output_excel} 文件中。")