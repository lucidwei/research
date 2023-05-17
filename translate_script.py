# coding=gbk
# Time Created: 2023/5/11 16:00
# Author  : Lucid
# FileName: translate_script.py
# Software: PyCharm
import json, re
from utils import generate_column_name_dict

manual_translations = {
    '金额': 'Value',
    '数量': 'Quantity',
    '当月值': 'Current Month Value',
    '当月同比': 'Current Month YoY',
    '折年数': 'Annualized',
    '季调': 'SeasonAdj',
    '美国': 'US',
    '中国香港': 'Hongkong',
    '中国台湾': 'Taiwan',
    '出口': 'Export',
    '进口': 'Import',
}

if __name__ == "__main__":
    chinese_column_names = """
    中国:出口金额:拉丁美洲:当月值	中国:进口金额:拉丁美洲:当月值	中国:出口金额:非洲:当月值	中国:进口金额:非洲:当月值
    """

    chinese_column_names = re.split(r'\s+', chinese_column_names.strip())
    column_name_dict = generate_column_name_dict(chinese_column_names, manual_translations)

    print(column_name_dict)
    column_name_dict_str = json.dumps(column_name_dict, ensure_ascii=False, indent=2)

    a = 1
