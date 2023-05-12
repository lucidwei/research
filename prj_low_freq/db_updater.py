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
        self.update_export()
        self.set_all_nan_to_null()
        self.close()

    def update_pmi(self):
        # 创建一个字典，键为指标 ID，值为手动映射的英文列名 (利用translate_script.py得到)
        map_name_to_english = {
            "欧元区:综合PMI": "Eurozone_CompositePmi",
            "日本:综合PMI": "Japan_CompositePmi",
            "美国:综合PMI": "US_CompositePmi",
            "美国:供应管理协会(ISM):制造业PMI": "US_InstituteForSupplyManagement(ism)_ManufacturingPmi",
            "日本:制造业PMI": "Japan_ManufacturingPmi",
            "欧元区:制造业PMI": "Eurozone_ManufacturingPmi",
            "日本:GDP:现价": "Japan_Gdp_CurrentPrice",
            "日本:GDP:现价:美元": "Japan_Gdp_CurrentPrice_Usd",
            "欧元区:GDP:现价": "Eurozone_Gdp_CurrentPrice",
            "欧元区:GDP:现价:美元": "Eurozone_Gdp_CurrentPrice_Usd",
            "美国:GDP:现价:折年数:季调": "US_Gdp_CurrentPrice_Annualized_SeasonAdj"
        }
        self.update_low_freq_from_excel_meta('博士PMI.xlsx', map_name_to_english)

    def update_export(self):
        # 利用translate_script.py得到
        map_name_to_english = {
            "中国:出口金额:当月值": "China_ExportsValue_CurrentMonthValue",
            "中国:进口金额:当月值": "China_ImportsValue_CurrentMonthValue",
            "中国:贸易差额:当月值": "China_BalanceOfTrade_CurrentMonthValue",
            "中国:出口金额:当月同比": "China_ExportsValue_CurrentMonthYoy",
            "中国:进口金额:当月同比": "China_ImportsValue_CurrentMonthYoy",
            "中国:贸易差额:当月同比": "China_BalanceOfTrade_CurrentMonthYoy",
            "中国:出口金额:机电产品:当月同比": "China_ExportsValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "中国:出口金额:高新技术产品:当月同比": "China_ExportsValue_High-techProducts_CurrentMonthYoy",
            "中国:出口金额:服装及衣着附件:当月同比": "China_ExportValue_ClothingAndClothingAccessories_CurrentMonthYoy",
            "中国:出口金额:纺织纱线织物及其制品:当月同比": "China_ExportValue_TextileYarnFabricsAndProducts_CurrentMonthYoy",
            "中国:出口金额:塑料制品:当月同比": "China_ExportsValue_PlasticProducts_CurrentMonthYoy",
            "中国:出口金额:集成电路:当月值": "China_ExportValue_Ic_CurrentMonthValue",
            "中国:出口金额:集成电路:当月同比": "China_ExportValue_Ic_CurrentMonthYoy",
            "中国:出口数量:集成电路:当月值": "China_ExportQuantity_Ic_CurrentMonthValue",
            "中国:出口金额:塑料制品:当月值": "China_ExportsValue_PlasticProducts_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月值": "China_ExportsValue_MedicalInstrumentsAndInstruments_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月同比": "China_ExportValue_MedicalInstrumentsAndInstruments_CurrentMonthYoy",
            "中国:出口数量:汽车包括底盘:当月值": "China_ExportQuantity_AutomobilesIncludingChassis_CurrentMonthValue",
            "中国:出口数量:汽车包括底盘:当月同比": "China_ExportsQuantity_AutomobilesIncludingChassis_CurrentMonthYoy",
            "中国:出口金额:汽车包括底盘:当月值": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthValue",
            "中国:出口金额:汽车包括底盘:当月同比": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthYoy",
            "中国:出口金额:汽车零配件:当月值": "China_ExportValue_AutomobileParts_CurrentMonthValue",
            "中国:出口金额:汽车零配件:当月同比": "China_ExportsValue_AutoParts_CurrentMonthYoy",
            "中国:进口金额:农产品:当月值": "China_ImportsValue_AgriculturalProducts_CurrentMonthValue",
            "中国:进口金额:农产品:当月同比": "China_ImportsValue_AgriculturalProducts_CurrentMonthYoy",
            "中国:进口数量:大豆:当月值": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "中国:进口数量:大豆:当月同比": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "中国:进口金额:铁矿砂及其精矿:当月值": "China_ImportedValue_IronOreAndConcentrate_CurrentMonthValue",
            "中国:进口金额:铁矿砂及其精矿:当月同比": "China_ImportsValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:铜矿砂及其精矿:当月值": "China_ImportsValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "中国:进口金额:铜矿砂及其精矿:当月同比": "China_ImportsValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:原油:当月值": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "中国:进口金额:原油:当月同比": "China_ImportValue_CrudeOil_CurrentMonthYoy",
            "中国:进口金额:煤及褐煤:当月值": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            "中国:进口金额:煤及褐煤:当月同比": "China_ImportsValue_CoalAndLignite_CurrentMonthYoy",
            "中国:进口金额:天然气:当月值": "China_ImportsValue_NaturalGas_CurrentMonthValue",
            "中国:进口金额:天然气:当月同比": "China_ImportsValue_NaturalGas_CurrentMonthYoy",
            "中国:进口金额:机电产品:当月值": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthValue",
            "中国:进口金额:机电产品:当月同比": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "中国:进口金额:集成电路:当月值": "China_ImportValue_Ic_CurrentMonthValue",
            "中国:进口金额:集成电路:当月同比": "China_ImportValue_Ic_CurrentMonthYoy",
            "中国:进口金额:高新技术产品:当月值": "China_ImportValue_High-techProducts_CurrentMonthValue",
            "中国:进口金额:高新技术产品:当月同比": "China_ImportValue_High-techProducts_CurrentMonthYoy",
            "中国:出口金额:东南亚国家联盟:当月值": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "中国:进口金额:东南亚国家联盟:当月值": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "中国:出口金额:东南亚国家联盟:当月同比": "China_ExportsValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "中国:进口金额:东南亚国家联盟:当月同比": "China_ImportsValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "中国:出口金额:欧盟:当月值": "China_ExportsValue_EuropeanUnion_CurrentMonthValue",
            "中国:进口金额:欧盟:当月值": "China_ImportValue_Eu_CurrentMonthValue",
            "中国:出口金额:欧盟:当月同比": "China_ExportValue_Eu_CurrentMonthYoy",
            "中国:进口金额:欧盟:当月同比": "China_ImportValue_Eu_CurrentMonthYoy",
            "中国:出口金额:美国:当月值": "China_ExportsValue_Us_CurrentMonthValue",
            "中国:进口金额:美国:当月值": "China_ImportsValue_Us_CurrentMonthValue",
            "中国:出口金额:美国:当月同比": "China_ExportsValue_Us_CurrentMonthYoy",
            "中国:进口金额:美国:当月同比": "China_ImportsValue_Us_CurrentMonthYoy",
            "中国:出口金额:中国香港:当月值": "China_ExportsValue_Hongkong_CurrentMonthValue",
            "中国:进口金额:中国香港:当月值": "China_ImportsValue_Hongkong_CurrentMonthValue",
            "中国:出口金额:中国香港:当月同比": "China_ExportsValue_Hongkong_CurrentMonthYoy",
            "中国:进口金额:中国香港:当月同比": "China_ImportsValue_Hongkong_CurrentMonthYoy",
            "中国:出口金额:日本:当月值": "China_ExportsValue_Japan_CurrentMonthValue",
            "中国:进口金额:日本:当月值": "China_ImportsValue_Japan_CurrentMonthValue",
            "中国:出口金额:日本:当月同比": "China_ExportsValue_Japan_CurrentMonthYoy",
            "中国:进口金额:日本:当月同比": "China_ImportValue_Japan_CurrentMonthYoy",
            "中国:出口金额:韩国:当月值": "China_ExportValue_Korea_CurrentMonthValue",
            "中国:进口金额:韩国:当月值": "China_ImportValue_SouthKorea_CurrentMonthValue",
            "中国:出口金额:韩国:当月同比": "China_ExportsValue_SouthKorea_CurrentMonthYoy",
            "中国:进口金额:韩国:当月同比": "China_ImportsValue_SouthKorea_CurrentMonthYoy",
            "中国:出口金额:中国台湾:当月值": "China_ExportsValue_Taiwan_CurrentMonthValue",
            "中国:进口金额:中国台湾:当月值": "China_ImportValue_Taiwan_CurrentMonthValue",
            "中国:出口金额:中国台湾:当月同比": "China_ExportValue_Taiwan_CurrentMonthYoy",
            "中国:进口金额:中国台湾:当月同比": "China_ImportValue_Taiwan_CurrentMonthYoy"
        }
        self.update_low_freq_from_excel_meta('中信出口模板.xlsx', map_name_to_english)
