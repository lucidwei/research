# coding=gbk
# Time Created: 2024/4/3 14:27
# Author  : Lucid
# FileName: run_boom.py
# Software: PyCharm
from base_config import BaseConfig
from prj_boom.preprocess import DataPreprocessor
from prj_boom.modeler import DynamicFactorModeler
from docx import Document
from docx.shared import Inches
from datetime import datetime
import warnings

# �����ض��� FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning,
                        message='verbose is deprecated since functions should not print results')

base_config = BaseConfig('boom')

SINGLE_BATCH_MODE = 'single'
# SINGLE_BATCH_MODE = 'batch'

if SINGLE_BATCH_MODE == 'single':
    # �ֵ�洢��ͬ�����ã�ÿ�����ð���industry��compare_to����Ϣ
    # key������industry, compare_to, stationary, date_start, single_line
    configs = [
        # {'industry': 'ʯ��ʯ��', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': 'ʯ��ʯ��', 'compare_to': '����ĸ��˾�ɶ��ľ�����ͬ��������', 'stationary': False},
        # {'industry': 'ú̿', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��ɫ����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��������', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��ý', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '�����', 'compare_to': '���ʲ�������ROE'},
        # {'industry': 'ͨ��', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��е', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '��������', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��ͨ����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '�ҵ�', 'compare_to': '���ʲ�������ROE'},
        # {'industry': 'ҽҩ', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': 'ʳƷ����', 'compare_to': '���ʲ�������ROE'},
        # {'industry': 'ũ������', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '�׾�', 'compare_to': '���ʲ�������ROE'},
        # {'industry': 'ʳƷ', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '����', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '���ѷ���', 'compare_to': '���ʲ�������ROE'},
        # {'industry': '�ķ�', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��ó����', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '���ݻ���', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '���ز�', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��������', 'compare_to': '���ʲ�������ROE', 'stationary': False},
        # {'industry': '��ҵ״��', 'compare_to': '�й�:�������ʧҵ��', 'stationary': False, 'date_start': '2018-01-01'},
        # {'industry': '������ָ', 'compare_to': '�й�:�������Ʒ�����ܶ�:����ͬ��', 'stationary': False, 'date_start': '2020-01-01'},
        # {'industry': '����', 'compare_to': '�й�:���ڽ��:����ͬ��', 'stationary': False, 'date_start': '2010-01-01'},
        # {'industry': 'PPI', 'compare_to': '�й�:PPI:ȫ����ҵƷ:����ͬ��', 'stationary': False, 'date_start': '2010-01-01'},
        # {'industry': '����', 'compare_to': '�й�:�������ۼ۸�ָ��:����סլ:70�����г���:����ͬ��', 'stationary': False, 'date_start': '2010-01-01'},
        # {'industry': '��ҵ����ֵ', 'compare_to': '�й�:��ģ���Ϲ�ҵ����ֵ:����ͬ��', 'stationary': False, 'date_start': '2010-01-01'},
        {'industry': '����ҵͶ��', 'compare_to': '(�¶Ȼ�)�й�:�̶��ʲ�Ͷ����ɶ�:����ҵ:�ۼ�ͬ��', 'stationary': False,
         'date_start': '2014-01-01', 'leading_prediction': False},
        # {'industry': '����ҵͶ��', 'compare_to': '�й�:�̶��ʲ�Ͷ����ɶ�:����ҵ:�ۼ�ͬ��', 'stationary': False,
        #  'date_start': '2014-01-01', 'leading_prediction': False},
    ]

    for config in configs:
        preprocessor = DataPreprocessor(
            base_config,
            industry=config['industry'],
            stationary=config.get('stationary', False),
            date_start=config.get('date_start', '2013-01-01'),
            compare_to=config['compare_to']
        )
        preprocessor.preprocess()

        modeler = DynamicFactorModeler(
            preprocessor,
            k_factors=1,
            factor_orders=2,
            compare_to=config['compare_to'],
            leading_prediction=config['leading_prediction'],
        )
        modeler.run()

if SINGLE_BATCH_MODE == 'batch':
    industries = ['ʯ��ʯ��', 'ú̿', '��ɫ����', '����', '��������', '����', 'ũ������',
                  '����', '��ý', '�����',
                  '��е', '����', '����', '��������',
                  '����', '�ҵ�', 'ҽҩ', 'ʳƷ����', '�׾�', 'ʳƷ', '����', '���ѷ���', '�ķ�', '��ó����',
                  '��ͨ����',
                  '���ز�', '��������']
    financial_list = ['���ʲ�������ROE']  # , '����ĸ��˾�ɶ��ľ�����ͬ��������', 'Ӫҵ����ͬ��������']


    def process_industry(industry, stationary, base_config, financial_list):
        preprocessor = DataPreprocessor(base_config, industry=industry, stationary=stationary)
        preprocessor.preprocess()

        results = []
        for financial in financial_list:
            modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to=financial)
            modeler.apply_dynamic_factor_model()
            modeler.evaluate_model()

            # start_date = '2024-02-29'
            # end_date = '2024-04-30'
            start_date = None
            end_date = None

            contribution_text = f"{industry}��ҵ �й����ݶ��ۺϾ���ָ����Ӱ����({start_date} to {end_date}):\n"
            contribution_text += modeler.analyze_factor_contribution(start_date, end_date)

            results.append({
                'industry': industry,
                'stationary': stationary,
                'financial': financial,
                'figure': modeler.plot_factors(save_or_show='save'),
                'contribution_text': contribution_text
            })

        return results


    all_results = []
    for industry in industries:
        print(f'Processing industry {industry}')
        for stationary in [True, False]:
            results = process_industry(industry, stationary, base_config, financial_list)
            all_results.extend(results)

    document = Document()
    for result in all_results:
        stationary_str = '������������(�й�����ƽ����)' if result['stationary'] else '�����仯����(�й�����δƽ��)'
        document.add_heading(f"{result['industry']} ({stationary_str}, financial={result['financial']})", level=1)
        document.add_picture(result['figure'], width=Inches(6))
        document.add_paragraph(result['contribution_text'])
        document.add_page_break()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    document.save(rf'{base_config.excels_path}/����/����ҵ�����ۺ�ָ��_{current_time}.docx')
