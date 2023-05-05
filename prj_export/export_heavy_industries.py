# coding=gbk
# Time Created: 2023/2/22 14:56
# Author  : Lucid
# FileName: export_heavy_industries.py
# Software: PyCharm

import pandas as pd
from base_config import BaseConfig
from mapping_industry import IndustryMap

code_guomin = BaseConfig('code_processing').structural_data['code_guomin']
code_guomin.columns = ['代码', '大类', '中类', '小类', '类别名称', '说明']
code_guomin = code_guomin[['大类', '类别名称']].dropna()

export_demand = BaseConfig('export_heavy').structural_data
export_demand = export_demand[['部门名称', '代码', 'EX', 'TFU']]
export_demand['大类代码'] = [code[:2] for code in export_demand['代码']]
export_demand['大类int'] = [int(code) for code in export_demand['大类代码']]
export_demand.sort_values(by='大类代码', inplace=True)

export_demand_agg = pd.DataFrame(data=code_guomin, columns=['大类', '类别名称', 'EX', 'TFU', 'ex_ratio']).set_index(['大类'])
for code in export_demand_agg.index:
    sub_df = export_demand[export_demand['大类int'] == int(code)]
    # 外需比内需的概念。TFU-GCF资本形成则是生产/供给的概念。
    export_demand_agg.loc[code, ['EX', 'TFU', 'ex_ratio']] = [sub_df['EX'].astype(int).sum(),
                                                              sub_df['TFU'].astype(int).sum(),
                                                              sub_df['EX'].astype(int).sum()/sub_df['TFU'].astype(int).sum()]
export_demand_agg = export_demand_agg.dropna().sort_values(by='ex_ratio', ascending=False)
export_demand_agg.loc[:, '类别名称'] = export_demand_agg['类别名称'].str.strip()


firms_cls_info = BaseConfig('firms_cls').structural_data
map_info = IndustryMap(firms_cls_info, '国民二级', '中信二级').map_info
# 股票分类和国民分类文字不能完全对应得上，需要对其中一个修正。筛选出来的国民分类数量更少只剩7个，因此对它进行修正。应该是股票那边被筛选掉了。不用找了，反正不是应该关心的
# intersect = set(map_info['国民二级']) & set(selected_indu['类别名称'])
# mismatch_in_gm = set(selected_indu['类别名称']) - intersect
# mismatch_in_stk = set(map_info['国民二级']) - intersect

ratio_array = []
for indu in map_info['国民二级']:
    try:
        ratio = export_demand_agg[export_demand_agg['类别名称']==indu]['ex_ratio'].values[0]
    except:
        ratio = 0
    ratio_array.append(round(ratio, 2))

map_info.set_index('国民二级', inplace=True)
map_info.insert(0, '外需比内需', ratio_array)
map_info.sort_values(by='外需比内需', ascending=False, inplace=True)
# 因为全部EX占全部TFU均值的概念为15.9%，因此筛选均值以上的行业。
selected_indu = map_info[map_info['外需比内需'] > 0.16]

pass