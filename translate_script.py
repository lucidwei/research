# coding=gbk
# Time Created: 2023/5/11 16:00
# Author  : Lucid
# FileName: translate_script.py
# Software: PyCharm
import json, re
from utils import generate_column_name_dict

manual_translations = {
    '���': 'Value',
    '����': 'Quantity',
    '����ֵ': 'Current Month Value',
    '����ͬ��': 'Current Month YoY',
    '������': 'Annualized',
    '����': 'SeasonAdj',
    '����': 'US',
    '�й����': 'Hongkong',
    '�й�̨��': 'Taiwan',
    '����': 'Export',
    '����': 'Import',
    '��ָ��': 'AggIndex',
}

if __name__ == "__main__":
    chinese_column_names = """
���ڽ��:ʳƷ����Ҫ��ʳ�õĻ��	���ڽ��:���ϼ�����	���ڽ��:��ʳ��ԭ��	���ڽ��:����ȼ�ϡ����ͼ��й�ԭ��	���ڽ��:����ֲ����֬����	���ڽ��:��ѧƷ���йز�Ʒ	���ڽ��:��Ĳ�Ʒ������Ʒ��ұ��Ʒ������Ʒ	���ڽ��:��е�������豸	���ڽ��:������Ʒ	���ڽ��:δ�����������Ʒ	���ڽ��:ʳƷ����Ҫ��ʳ�õĻ��	���ڽ��:���ϼ�����	���ڽ��:��ʳ��ԭ��	���ڽ��:����ȼ�ϡ����ͼ��й�ԭ��	���ڽ��:����ֲ����֬����	���ڽ��:��ѧƷ���йز�Ʒ	���ڽ��:��Ĳ�Ʒ������Ʒ��ұ��Ʒ������Ʒ	���ڽ��:��е�������豸	���ڽ��:������Ʒ	���ڽ��:δ�����������Ʒ

    """

    chinese_column_names = re.split(r'\s+', chinese_column_names.strip())
    column_name_dict = generate_column_name_dict(chinese_column_names, manual_translations)

    print(column_name_dict)
    column_name_dict_str = json.dumps(column_name_dict, ensure_ascii=False, indent=2)

    a = 1
