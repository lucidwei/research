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
    出口价值指数(HS2):总指数
出口数量指数(HS2):总指数
出口价格指数(HS2):总指数
出口价值指数(HS2):同比
出口数量指数(HS2):同比
出口价格指数(HS2):同比
工业企业:出口交货值:当月同比    
工业企业:出口交货值:当月值
PPI:全部工业品:当月同比:+3月
全球:摩根大通全球制造业PMI
OECD综合领先指标
印度:出口金额:商品:美元    
越南:出口金额:总金额:当月值    
韩国:出口总额:百万    
日本:出口金额:当月值:美元    
德国:出口金额:美元:百万
投入产出基本流量:最终使用:建筑/合计    
投入产出基本流量:最终使用:其他服务/合计    
投入产出基本流量:最终使用:机械设备制造/合计    
投入产出基本流量:最终使用:出口/合计
最终消费率(消费率)    
资本形成率(投资率)    
净出口率
服务贸易差额:占GDP比重:当季值    
货物贸易差额:占GDP比重:当季值    
经常账户差额:占GDP比重:当季值    
投资收益差额:占GDP比重:当季值
GDP当季同比贡献率:货物和服务净出口    
对GDP当季同比的拉动:最终消费支出    
对GDP当季同比的拉动:资本形成总额    
对GDP当季同比的拉动:货物和服务净出口
    """

    chinese_column_names = re.split(r'\s+', chinese_column_names.strip())
    column_name_dict = generate_column_name_dict(chinese_column_names, manual_translations)

    print(column_name_dict)
    column_name_dict_str = json.dumps(column_name_dict, ensure_ascii=False, indent=2)

    a = 1
