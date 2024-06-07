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
    # preprocessor = DataPreprocessor(base_config, industry='石油石化')
    # preprocessor = DataPreprocessor(base_config, industry='石油石化', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='煤炭')
    # preprocessor = DataPreprocessor(base_config, industry='煤炭', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='有色金属')
    # preprocessor = DataPreprocessor(base_config, industry='有色金属', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='钢铁')
    # preprocessor = DataPreprocessor(base_config, industry='钢铁', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='基础化工')
    # preprocessor = DataPreprocessor(base_config, industry='基础化工', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='建材')
    # preprocessor = DataPreprocessor(base_config, industry='建材', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='电子')
    # preprocessor = DataPreprocessor(base_config, industry='电子', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='传媒')
    # preprocessor = DataPreprocessor(base_config, industry='计算机')
    # preprocessor = DataPreprocessor(base_config, industry='通信')
    # preprocessor = DataPreprocessor(base_config, industry='通信', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='机械')
    # preprocessor = DataPreprocessor(base_config, industry='电新')
    # preprocessor = DataPreprocessor(base_config, industry='电新', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='电力')
    # preprocessor = DataPreprocessor(base_config, industry='国防军工')
    # preprocessor = DataPreprocessor(base_config, industry='国防军工', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='交通运输')
    # preprocessor = DataPreprocessor(base_config, industry='交通运输', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='汽车')
    # preprocessor = DataPreprocessor(base_config, industry='家电')
    # preprocessor = DataPreprocessor(base_config, industry='医药')
    # preprocessor = DataPreprocessor(base_config, industry='医药', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='食品饮料')
    # preprocessor = DataPreprocessor(base_config, industry='农林牧渔')
    # preprocessor = DataPreprocessor(base_config, industry='农林牧渔', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='白酒')
    # preprocessor = DataPreprocessor(base_config, industry='食品')
    # preprocessor = DataPreprocessor(base_config, industry='饮料')
    # preprocessor = DataPreprocessor(base_config, industry='消费服务')
    # preprocessor = DataPreprocessor(base_config, industry='纺服')
    # preprocessor = DataPreprocessor(base_config, industry='纺服', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='商贸零售')
    # preprocessor = DataPreprocessor(base_config, industry='商贸零售', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='房地产')
    # preprocessor = DataPreprocessor(base_config, industry='非银金融')
    # preprocessor = DataPreprocessor(base_config, industry='非银金融', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='就业状况')
    # preprocessor = DataPreprocessor(base_config, industry='就业状况', stationary=False)
    # preprocessor = DataPreprocessor(base_config, industry='社零综指')
    # preprocessor = DataPreprocessor(base_config, industry='社零综指', stationary=False, date_start='2020-01-01')
    preprocessor = DataPreprocessor(base_config, industry='出口', stationary=False, date_start='2010-01-01')

    preprocessor.preprocess()
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='净资产收益率ROE')
    modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='中国:出口金额:当月同比')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, compare_to='美国:销售总额:季调:同比-美国:库存总额:季调:同比:+6月')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='归属母公司股东的净利润同比增长率')
    # modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='营业收入同比增长率')
    modeler.run()

if SINGLE_BATCH_MODE == 'batch':
    industries = ['石油石化', '煤炭', '有色金属', '钢铁', '基础化工', '建材', '农林牧渔',
                  '电子', '传媒', '计算机',
                  '机械', '电新', '电力', '国防军工',
                  '汽车', '家电', '医药', '食品饮料', '白酒', '食品', '饮料', '消费服务', '纺服', '商贸零售',
                  '交通运输',
                  '房地产', '非银金融']
    financial_list = ['净资产收益率ROE']  # , '归属母公司股东的净利润同比增长率', '营业收入同比增长率']


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

            contribution_text = f"{industry}行业 中观数据对综合景气指数的影响拆解({start_date} to {end_date}):\n"
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
        stationary_str = '景气趋势慢线(中观数据平滑后)' if result['stationary'] else '景气变化快线(中观数据未平滑)'
        document.add_heading(f"{result['industry']} ({stationary_str}, financial={result['financial']})", level=1)
        document.add_picture(result['figure'], width=Inches(6))
        document.add_paragraph(result['contribution_text'])
        document.add_page_break()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    document.save(rf'{base_config.excels_path}/景气/各行业景气综合指数_{current_time}.docx')
