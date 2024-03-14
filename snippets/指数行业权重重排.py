import pandas as pd

# 读取Excel文件的“成分行业分布明细”sheet
file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\数据库相关\基金仓位测算\000985.CSI-成分行业分布-20240312.xlsx"
sheet_name = '成分行业分布明细'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 定义行业名称的新顺序
new_order = [
    '有色金属', '石油石化', '煤炭', '基础化工', '钢铁', '建材', '建筑', '电力设备及新能源', '机械', '国防军工',
    '电力及公用事业', '轻工制造', '医药', '食品饮料', '家电', '消费者服务', '汽车', '交通运输', '商贸零售', '农林牧渔',
    '纺织服装', '电子', '计算机', '传媒', '通信', '房地产', '银行', '非银行金融', '综合金融', '综合', '港股', '债券',
    '现金'
]

# 重建DataFrame以匹配新的顺序
# 根据新顺序对DataFrame进行排序
df = df.set_index('行业名称').reindex(new_order).reset_index()

# 保存新的DataFrame到一个新的Excel文件
new_file_path = rf"D:\WPS云盘\WPS云盘\工作-麦高\数据库相关\基金仓位测算\000985.CSI-new成分行业分布-20240312.xlsx"
with pd.ExcelWriter(new_file_path) as writer:
    df.to_excel(writer, index=False)

