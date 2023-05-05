# coding=gbk
# Time Created: 2023/2/22 14:56
# Author  : Lucid
# FileName: export_heavy_industries.py
# Software: PyCharm

import pandas as pd
from base_config import BaseConfig
from mapping_industry import IndustryMap

code_guomin = BaseConfig('code_processing').structural_data['code_guomin']
code_guomin.columns = ['����', '����', '����', 'С��', '�������', '˵��']
code_guomin = code_guomin[['����', '�������']].dropna()

export_demand = BaseConfig('export_heavy').structural_data
export_demand = export_demand[['��������', '����', 'EX', 'TFU']]
export_demand['�������'] = [code[:2] for code in export_demand['����']]
export_demand['����int'] = [int(code) for code in export_demand['�������']]
export_demand.sort_values(by='�������', inplace=True)

export_demand_agg = pd.DataFrame(data=code_guomin, columns=['����', '�������', 'EX', 'TFU', 'ex_ratio']).set_index(['����'])
for code in export_demand_agg.index:
    sub_df = export_demand[export_demand['����int'] == int(code)]
    # ���������ĸ��TFU-GCF�ʱ��γ���������/�����ĸ��
    export_demand_agg.loc[code, ['EX', 'TFU', 'ex_ratio']] = [sub_df['EX'].astype(int).sum(),
                                                              sub_df['TFU'].astype(int).sum(),
                                                              sub_df['EX'].astype(int).sum()/sub_df['TFU'].astype(int).sum()]
export_demand_agg = export_demand_agg.dropna().sort_values(by='ex_ratio', ascending=False)
export_demand_agg.loc[:, '�������'] = export_demand_agg['�������'].str.strip()


firms_cls_info = BaseConfig('firms_cls').structural_data
map_info = IndustryMap(firms_cls_info, '�������', '���Ŷ���').map_info
# ��Ʊ����͹���������ֲ�����ȫ��Ӧ���ϣ���Ҫ������һ��������ɸѡ�����Ĺ��������������ֻʣ7������˶�������������Ӧ���ǹ�Ʊ�Ǳ߱�ɸѡ���ˡ��������ˣ���������Ӧ�ù��ĵ�
# intersect = set(map_info['�������']) & set(selected_indu['�������'])
# mismatch_in_gm = set(selected_indu['�������']) - intersect
# mismatch_in_stk = set(map_info['�������']) - intersect

ratio_array = []
for indu in map_info['�������']:
    try:
        ratio = export_demand_agg[export_demand_agg['�������']==indu]['ex_ratio'].values[0]
    except:
        ratio = 0
    ratio_array.append(round(ratio, 2))

map_info.set_index('�������', inplace=True)
map_info.insert(0, '���������', ratio_array)
map_info.sort_values(by='���������', ascending=False, inplace=True)
# ��Ϊȫ��EXռȫ��TFU��ֵ�ĸ���Ϊ15.9%�����ɸѡ��ֵ���ϵ���ҵ��
selected_indu = map_info[map_info['���������'] > 0.16]

pass