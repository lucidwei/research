# coding=gbk
# Time Created: 2024/4/3 14:27
# Author  : Lucid
# FileName: run_boom.py
# Software: PyCharm
from base_config import BaseConfig
from prj_boom.preprocess import DataPreprocessor
from prj_boom.modeler import DynamicFactorModeler

base_config = BaseConfig('boom')
# preprocessor = DataPreprocessor(base_config, industry='石油石化')
# preprocessor = DataPreprocessor(base_config, industry='煤炭')
# preprocessor = DataPreprocessor(base_config, industry='有色金属')
# preprocessor = DataPreprocessor(base_config, industry='钢铁')
# preprocessor = DataPreprocessor(base_config, industry='基础化工')
# preprocessor = DataPreprocessor(base_config, industry='建材')
# preprocessor = DataPreprocessor(base_config, industry='电子')
# preprocessor = DataPreprocessor(base_config, industry='机械')
# preprocessor = DataPreprocessor(base_config, industry='电新')
# preprocessor = DataPreprocessor(base_config, industry='国防军工')
# preprocessor = DataPreprocessor(base_config, industry='汽车')
# preprocessor = DataPreprocessor(base_config, industry='白酒')
# preprocessor = DataPreprocessor(base_config, industry='食品')
# preprocessor = DataPreprocessor(base_config, industry='饮料')
preprocessor = DataPreprocessor(base_config, industry='房地产')

preprocessor.preprocess()
modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='净资产收益率ROE')
# modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='归属母公司股东的净利润同比增长率')
# modeler = DynamicFactorModeler(preprocessor, k_factors=1, factor_orders=2, financial='营业收入同比增长率')
modeler.run()
