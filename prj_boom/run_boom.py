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

base_config = BaseConfig('boom')

SINGLE_BATCH_MODE = 'single'
# SINGLE_BATCH_MODE = 'batch'

if SINGLE_BATCH_MODE == 'single':
    # preprocessor = DataPreprocessor(base_config, industry='ʯ��ʯ��')
    # preprocessor = DataPreprocessor(base_config, industry='ʯ��ʯ��', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='ú̿')
    # preprocessor = DataPreprocessor(base_config, industry='ú̿', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��ɫ����')
    # preprocessor = DataPreprocessor(base_config, industry='��ɫ����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��������')
    # preprocessor = DataPreprocessor(base_config, industry='��������', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��ý')
    # preprocessor = DataPreprocessor(base_config, industry='�����')
    # preprocessor = DataPreprocessor(base_config, industry='ͨ��')
    # preprocessor = DataPreprocessor(base_config, industry='ͨ��', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��е')
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='��������')
    # preprocessor = DataPreprocessor(base_config, industry='��������', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��ͨ����')
    # preprocessor = DataPreprocessor(base_config, industry='��ͨ����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='�ҵ�')
    # preprocessor = DataPreprocessor(base_config, industry='ҽҩ')
    # preprocessor = DataPreprocessor(base_config, industry='ҽҩ', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='ʳƷ����')
    # preprocessor = DataPreprocessor(base_config, industry='ũ������')
    # preprocessor = DataPreprocessor(base_config, industry='ũ������', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='�׾�')
    # preprocessor = DataPreprocessor(base_config, industry='ʳƷ')
    # preprocessor = DataPreprocessor(base_config, industry='����')
    # preprocessor = DataPreprocessor(base_config, industry='���ѷ���')
    # preprocessor = DataPreprocessor(base_config, industry='�ķ�')
    # preprocessor = DataPreprocessor(base_config, industry='�ķ�', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��ó����')
    # preprocessor = DataPreprocessor(base_config, industry='��ó����', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='���ز�')
    # preprocessor = DataPreprocessor(base_config, industry='��������')
    # preprocessor = DataPreprocessor(base_config, industry='��������', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='��ҵ״��')
    # preprocessor = DataPreprocessor(base_config, industry='��ҵ״��', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='������ָ')
    # preprocessor = DataPreprocessor(base_config, industry='������ָ', stationary=False, date_start='2020-01-01')
    preprocessor = DataPreprocessor(base_config, industry='����', stationary=False, date_start='2010-01-01')

    preprocessor.preprocess()
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='���ʲ�������ROE')
    modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='�й�:���ڽ��:����ͬ��')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='����:�����ܶ�:����:ͬ��-����:����ܶ�:����:ͬ��:+6��')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='����ĸ��˾�ɶ��ľ�����ͬ��������')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='Ӫҵ����ͬ��������')
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

            start_date = '2024-02-29'
            end_date = '2024-04-30'

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
