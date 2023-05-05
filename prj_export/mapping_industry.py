# coding=gbk
# Time Created: 2023/2/21 9:36
# Author  : Lucid
# FileName: mapping_industry.py
# Software: PyCharm
import pandas as pd


class IndustryMap:
    def __init__(self, firms_cls_info, from_, to):
        self.firms_cls_info = firms_cls_info
        self.map_info = self.get_industry_map(firms_cls_info, from_, to)

    def get_industry_map(self, firms_cls_info: pd.DataFrame, from_: str, to: str):
        """
        :param firms_cls_info: ��wind���������Ĺ�˾��������
        :param from_: ����ѡ��'����һ��', '���Ŷ���', '��������', '����һ��', '�������', '��������'
        :param to: ͬfrom_������ѡ��
        :return: ӳ����
        """
        map_dic = {}
        for indu_from in firms_cls_info[from_].unique():
            map_dic[indu_from] = firms_cls_info[firms_cls_info[from_] == indu_from][to]
        map_count = pd.DataFrame(columns=firms_cls_info[to].unique())
        map_percent = map_count.copy(deep=True)
        mapping = pd.DataFrame(columns=[f'��1���{to}', '��2', '��3', '��1ռ��', '��1����'])
        for indu_from in map_dic:
            map_count.loc[indu_from] = map_dic[indu_from].value_counts().astype(int)
            map_percent.loc[indu_from] = map_dic[indu_from].value_counts(normalize=True)
            # ÿ��������ҵȡռ��ǰ������˾����̫��������
            count = map_count.loc[indu_from].copy(deep=True).sort_values(ascending=False).head(3).fillna(value=0)
            if count[0] < 5:
                continue
            ratio = map_percent.loc[indu_from].copy(deep=True).sort_values(ascending=False).head(3).fillna(value=0)
            names = list(ratio.index)
            info_list = []
            for i in range(3):
                if ratio[i] > 0.03:
                    info_list.append(
                        'ռ��:' + str(int(100 * ratio[i])) + '%' + ' ��˾����:' + str(int(count[i])) + names[i])
                else:
                    info_list.append('')
            info_list.append(int(100 * ratio[0]))
            info_list.append(count[0])
            mapping.loc[indu_from] = info_list

        mapping = mapping.sort_values(by=['��1ռ��', '��1����'], ascending=False).reset_index(names=from_).iloc[:, :4]
        return mapping


# data = DataBase('firms_cls')
# firms_cls_info = data.structural_data
# z1_g2_map = get_industry_map(firms_cls_info, '����һ��', '����һ��')
# g1_z1_map = get_industry_map(firms_cls_info, '����һ��', '����һ��')
# g1_z2_map = get_industry_map(firms_cls_info, '����һ��', '���Ŷ���')
# g2_z2_map = get_industry_map(firms_cls_info, '�������', '���Ŷ���')

