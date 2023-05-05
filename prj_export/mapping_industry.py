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
        :param firms_cls_info: 从wind拷贝下来的公司介绍数据
        :param from_: 从中选择：'中信一级', '中信二级', '中信三级', '国民一级', '国民二级', '国民三级'
        :param to: 同from_，从中选择
        :return: 映射结果
        """
        map_dic = {}
        for indu_from in firms_cls_info[from_].unique():
            map_dic[indu_from] = firms_cls_info[firms_cls_info[from_] == indu_from][to]
        map_count = pd.DataFrame(columns=firms_cls_info[to].unique())
        map_percent = map_count.copy(deep=True)
        mapping = pd.DataFrame(columns=[f'第1相近{to}', '第2', '第3', '第1占比', '第1个数'])
        for indu_from in map_dic:
            map_count.loc[indu_from] = map_dic[indu_from].value_counts().astype(int)
            map_percent.loc[indu_from] = map_dic[indu_from].value_counts(normalize=True)
            # 每个中信行业取占比前三，公司个数太少则跳过
            count = map_count.loc[indu_from].copy(deep=True).sort_values(ascending=False).head(3).fillna(value=0)
            if count[0] < 5:
                continue
            ratio = map_percent.loc[indu_from].copy(deep=True).sort_values(ascending=False).head(3).fillna(value=0)
            names = list(ratio.index)
            info_list = []
            for i in range(3):
                if ratio[i] > 0.03:
                    info_list.append(
                        '占比:' + str(int(100 * ratio[i])) + '%' + ' 公司个数:' + str(int(count[i])) + names[i])
                else:
                    info_list.append('')
            info_list.append(int(100 * ratio[0]))
            info_list.append(count[0])
            mapping.loc[indu_from] = info_list

        mapping = mapping.sort_values(by=['第1占比', '第1个数'], ascending=False).reset_index(names=from_).iloc[:, :4]
        return mapping


# data = DataBase('firms_cls')
# firms_cls_info = data.structural_data
# z1_g2_map = get_industry_map(firms_cls_info, '中信一级', '国民一级')
# g1_z1_map = get_industry_map(firms_cls_info, '国民一级', '中信一级')
# g1_z2_map = get_industry_map(firms_cls_info, '国民一级', '中信二级')
# g2_z2_map = get_industry_map(firms_cls_info, '国民二级', '中信二级')

