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
    '总指数': 'AggIndex',
}

if __name__ == "__main__":
    chinese_column_names = """
出口金额:食品及主要供食用的活动物	出口金额:饮料及烟类	出口金额:非食用原料	出口金额:矿物燃料、润滑油及有关原料	出口金额:动、植物油脂及蜡	出口金额:化学品及有关产品	出口金额:轻纺产品、橡胶制品矿冶产品及其制品	出口金额:机械及运输设备	出口金额:杂项制品	出口金额:未分类的其他商品	进口金额:食品及主要供食用的活动物	进口金额:饮料及烟类	进口金额:非食用原料	进口金额:矿物燃料、润滑油及有关原料	进口金额:动、植物油脂及蜡	进口金额:化学品及有关产品	进口金额:轻纺产品、橡胶制品矿冶产品及其制品	进口金额:机械及运输设备	进口金额:杂项制品	进口金额:未分类的其他商品

    """

    chinese_column_names = re.split(r'\s+', chinese_column_names.strip())
    column_name_dict = generate_column_name_dict(chinese_column_names, manual_translations)

    print(column_name_dict)
    column_name_dict_str = json.dumps(column_name_dict, ensure_ascii=False, indent=2)

    a = 1
