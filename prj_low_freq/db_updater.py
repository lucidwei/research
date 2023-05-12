# coding=gbk
# Time Created: 2023/4/25 9:45
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re, math
from WindPy import w
import pandas as pd
from utils import timeit, get_nearest_dates_from_contract, check_wind
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text, MetaData, Table
from pypinyin import lazy_pinyin


class DatabaseUpdater(PgDbUpdaterBase):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        self.update_pmi()
        self.update_gdp()
        self.set_all_nan_to_null()
        self.close()

    def update_gdp(self):
        pass

    def update_pmi(self):
        # 创建一个字典，键为指标 ID，值为手动映射的英文列名 (交给chatgpt完成)
        map_name_to_english = {
            "欧元区:综合PMI": "pmi_comprehensive_eurozone",
            "日本:综合PMI": "pmi_comprehensive_japan",
            "美国:综合PMI": "pmi_comprehensive_usa",
            "美国:供应管理协会(ISM):制造业PMI": "pmi_manufacturing_ism_usa",
            "欧元区:制造业PMI": "pmi_manufacturing_eurozone",
            "日本:制造业PMI": "pmi_manufacturing_japan"
        }

        self.update_low_freq_from_excel_meta('博士PMI.xlsx', map_name_to_english)

    def update_export(self):
        map_name_to_english = {
            "中国:出口金额:当月值": "China_ExportValue_CurrentMonthValue",
            "中国:进口金额:当月值": "China_ImportValue_CurrentMonthValue",
            "中国:贸易差额:当月值": "China_TradeBalance_CurrentMonthValue",
            "中国:出口金额:当月同比": "China_ExportValue_CurrentMonthYoy",
            "中国:进口金额:当月同比": "China_ImportValue_CurrentMonthYoy",
            "中国:贸易差额:当月同比": "China_TradeBalance_CurrentMonthYoy",
            "中国:出口金额:机电产品:当月同比": "China_ExportValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "中国:出口金额:高新技术产品:当月同比": "China_ExportValue_High-techProducts_Current",
            "中国:出口金额:服装及衣着附件:当月同比": "China_ExportValue_ClothingAndClothingAccessories_CurrentMonthYoy",
            "中国:出口金额:纺织纱线织物及其制品:当月同比": "China_ExportValue_TextileYarnFabricAndItsProducts_CurrentMonthYoy",
            "中国:出口金额:集成电路:当月值": "China_ExportValue_IntegratedCircuit_CurrentMonthValue",
            "中国:出口金额:集成电路:当月同比": "China_ExportValue_IntegratedCircuit_CurrentMonthYoy",
            "中国:出口数量:集成电路:当月值": "China_ExportQuantity_IntegratedCircuit_CurrentMonthValue",
            "中国:出口金额:塑料制品:当月值": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "中国:出口金额:塑料制品:当月同比": "China_ExportValue_PlasticProducts_CurrentMonthYoy",
            "中国:出口金额:医疗仪器及器械:当月值": "China_ExportValue_MedicalInstrumentsAndDevices_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月同比": "China_ExportValue_MedicalInstrumentsAndDevices_CurrentMonthYoy",
            "中国:出口数量:汽车包括底盘:当月值": "China_ExportQuantity_AutomobileIncludingChassis_CurrentMonthValue",
            "中国:出口数量:汽车包括底盘:当月同比": "China_ExportQuantity_AutomobileIncludingChassis_CurrentMonthYoy",
            "中国:出口金额:汽车包括底盘:当月值": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthValue",
            "中国:出口金额:汽车包括底盘:当月同比": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthYoy",
            "中国:出口金额:汽车零配件:当月值": "China_ExportValue_AutoParts_CurrentMonthValue",
            "中国:出口金额:汽车零配件:当月同比": "China_ExportValue_AutoParts_CurrentMonthYoy",
            "中国:进口金额:农产品:当月值": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "中国:进口金额:农产品:当月同比": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "中国:进口数量:大豆:当月值": "China_ImportQuantity_Soybean_CurrentMonthValue",
            "中国:进口数量:大豆:当月同比": "China_ImportQuantity_Soybean_CurrentMonthYoy",
            "中国:进口金额:铁矿砂及其精矿:当月值": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthValue",
            "中国:进口金额:铁矿砂及其精矿:当月同比": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:铜矿砂及其精矿:当月值": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "中国:进口金额:铜矿砂及其精矿:当月同比": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:原油:当月值": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "中国:进口金额:原油:当月同比": "China_Current_ImportValue_CoalAndLignite_CurrentMonthValue",
            "中国:进口金额:煤及褐煤:当月值": "China_ImportValue_CoalAndLignite_CurrentMonthYoy",
            "中国:进口金额:煤及褐煤:当月同比": "China_ImportValue_NaturalGas_CurrentMonthValue",
            "中国:进口金额:天然气:当月值": "China_ImportValue_NaturalGas_CurrentMonthYoy",
            "中国:进口金额:天然气:当月同比": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthValue",
            "中国:进口金额:机电产品:当月值": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthYoyIntegration",
            "中国:进口金额:机电产品:当月同比": "China_ImportValue_CurrentMonthValue",
            "中国:进口金额:集成电路:当月值": "China_ImportValue_IntegratedCircuit_CurrentMonthYoy",
            "中国:进口金额:集成电路:当月同比": "China_ImportValue_High-techProduct_CurrentMonthValue",
            "中国:进口金额:高新技术产品:当月值": "China_ImportValue_High-techProduct_CurrentMonthYoy",
            "中国:进口金额:高新技术产品:当月同比": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "中国:出口金额:东南亚国家联盟:当月值": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "中国:进口金额:东南亚国家联盟:当月值": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "中国:出口金额:东南亚国家联盟:当月同比": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "中国:进口金额:东南亚国家联盟:当月同比": "China_ExportValue_Eu_CurrentMonthValue",
            "中国:出口金额:欧盟:当月值": "China_ImportValue_Eu_CurrentMonthValue",
            "中国:进口金额:欧盟:当月值": "China_ExportValue_Eu_CurrentMonthYoy",
            "中国:出口金额:欧盟:当月同比": "China_ImportValue_Eu_CurrentMonthYoy",
            "中国:进口金额:欧盟:当月同比": "China_ExportValue_UnitedStates_CurrentMonthValue",
            "中国:出口金额:美国:当月值": "China_ImportValue_UnitedStates_CurrentMonthValue_ExportValue_Us_CurrentMonthYoy",
            "中国:进口金额:美国:当月值": "China_ImportValue_Us_CurrentMonthYoy",
            "中国:出口金额:美国:当月同比": "China_ExportValue_HongKong",
            "中国:进口金额:美国:当月同比": "China_CurrentMonthValue",
            "中国:出口金额:中国香港:当月值": "China_ImportValue_HongKong",
            "中国:进口金额:中国香港:当月值": "China_CurrentMonthValue",
            "中国:出口金额:中国香港:当月同比": "China_ExportValue_HongKong",
            "中国:进口金额:中国香港:当月同比": "China_CurrentMonthYoy",
            "中国:出口金额:日本:当月值": "China_ImportValue_HongKong",
            "中国:进口金额:日本:当月值": "China_CurrentMonthYoy",
            "中国:出口金额:日本:当月同比": "China_ExportMonthValue_Japan_Current",
            "中国:进口金额:日本:当月同比": "China_ImportValue_Japan_CurrentMonthValue",
            "中国:出口金额:韩国:当月值": "China_ExportValue_Japan_CurrentMonthYoy",
            "中国:进口金额:韩国:当月值": "China_ImportValue_Japan_CurrentMonthYoy",
            "中国:出口金额:韩国:当月同比": "China_ExportValue_Korea_CurrentMonthValue",
            "中国:进口金额:韩国:当月同比": "China_ImportValue_Korea_CurrentMonthValue",
            "中国:出口金额:中国台湾:当月值": "China_ExportValue_Korea_CurrentMonthYoy",
            "中国:进口金额:中国台湾:当月值": "China_ImportValue_Korea_Current_ExportValue_TaiwanChina_CurrentMonthValue",
            "中国:出口金额:中国台湾:当月同比": "China_ImportValue_TaiwanChina_CurrentMonthValue",
            "中国:进口金额:中国台湾:当月同比": "China_ExportValue_TaiwanChina_CurrentMonthYoy"
        }
        self.update_low_freq_from_excel_meta('中信出口模板.xlsx', map_name_to_english)
