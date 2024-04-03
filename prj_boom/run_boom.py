# coding=gbk
# Time Created: 2024/4/3 14:27
# Author  : Lucid
# FileName: run_boom.py
# Software: PyCharm
from base_config import BaseConfig
from prj_boom.preprocess import DataPreprocessor

base_config = BaseConfig('boom')
preprocessor = DataPreprocessor(base_config)
preprocessor.preprocess()