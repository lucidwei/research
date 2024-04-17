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
preprocessor = DataPreprocessor(base_config, industry='煤炭')
preprocessor.preprocess()
modeler = DynamicFactorModeler(preprocessor, k_factors=1, financial='净资产收益率ROE')
modeler.run()
