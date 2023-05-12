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
        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ�ֶ�ӳ���Ӣ������ (����chatgpt���)
        map_name_to_english = {
            "ŷԪ��:�ۺ�PMI": "pmi_comprehensive_eurozone",
            "�ձ�:�ۺ�PMI": "pmi_comprehensive_japan",
            "����:�ۺ�PMI": "pmi_comprehensive_usa",
            "����:��Ӧ����Э��(ISM):����ҵPMI": "pmi_manufacturing_ism_usa",
            "ŷԪ��:����ҵPMI": "pmi_manufacturing_eurozone",
            "�ձ�:����ҵPMI": "pmi_manufacturing_japan"
        }

        self.update_low_freq_from_excel_meta('��ʿPMI.xlsx', map_name_to_english)

    def update_export(self):
        map_name_to_english = {
            "�й�:���ڽ��:����ֵ": "China_ExportValue_CurrentMonthValue",
            "�й�:���ڽ��:����ֵ": "China_ImportValue_CurrentMonthValue",
            "�й�:ó�ײ��:����ֵ": "China_TradeBalance_CurrentMonthValue",
            "�й�:���ڽ��:����ͬ��": "China_ExportValue_CurrentMonthYoy",
            "�й�:���ڽ��:����ͬ��": "China_ImportValue_CurrentMonthYoy",
            "�й�:ó�ײ��:����ͬ��": "China_TradeBalance_CurrentMonthYoy",
            "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ExportValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ExportValue_High-techProducts_Current",
            "�й�:���ڽ��:��װ�����Ÿ���:����ͬ��": "China_ExportValue_ClothingAndClothingAccessories_CurrentMonthYoy",
            "�й�:���ڽ��:��֯ɴ��֯�Ｐ����Ʒ:����ͬ��": "China_ExportValue_TextileYarnFabricAndItsProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ExportValue_IntegratedCircuit_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ExportValue_IntegratedCircuit_CurrentMonthYoy",
            "�й�:��������:���ɵ�·:����ֵ": "China_ExportQuantity_IntegratedCircuit_CurrentMonthValue",
            "�й�:���ڽ��:������Ʒ:����ֵ": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "�й�:���ڽ��:������Ʒ:����ͬ��": "China_ExportValue_PlasticProducts_CurrentMonthYoy",
            "�й�:���ڽ��:ҽ����������е:����ֵ": "China_ExportValue_MedicalInstrumentsAndDevices_CurrentMonthValue",
            "�й�:���ڽ��:ҽ����������е:����ͬ��": "China_ExportValue_MedicalInstrumentsAndDevices_CurrentMonthYoy",
            "�й�:��������:������������:����ֵ": "China_ExportQuantity_AutomobileIncludingChassis_CurrentMonthValue",
            "�й�:��������:������������:����ͬ��": "China_ExportQuantity_AutomobileIncludingChassis_CurrentMonthYoy",
            "�й�:���ڽ��:������������:����ֵ": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthValue",
            "�й�:���ڽ��:������������:����ͬ��": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthYoy",
            "�й�:���ڽ��:���������:����ֵ": "China_ExportValue_AutoParts_CurrentMonthValue",
            "�й�:���ڽ��:���������:����ͬ��": "China_ExportValue_AutoParts_CurrentMonthYoy",
            "�й�:���ڽ��:ũ��Ʒ:����ֵ": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "�й�:���ڽ��:ũ��Ʒ:����ͬ��": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "�й�:��������:��:����ֵ": "China_ImportQuantity_Soybean_CurrentMonthValue",
            "�й�:��������:��:����ͬ��": "China_ImportQuantity_Soybean_CurrentMonthYoy",
            "�й�:���ڽ��:����ɰ���侫��:����ֵ": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:����ɰ���侫��:����ͬ��": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ֵ": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ͬ��": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ԭ��:����ֵ": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "�й�:���ڽ��:ԭ��:����ͬ��": "China_Current_ImportValue_CoalAndLignite_CurrentMonthValue",
            "�й�:���ڽ��:ú����ú:����ֵ": "China_ImportValue_CoalAndLignite_CurrentMonthYoy",
            "�й�:���ڽ��:ú����ú:����ͬ��": "China_ImportValue_NaturalGas_CurrentMonthValue",
            "�й�:���ڽ��:��Ȼ��:����ֵ": "China_ImportValue_NaturalGas_CurrentMonthYoy",
            "�й�:���ڽ��:��Ȼ��:����ͬ��": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthValue",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthYoyIntegration",
            "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ImportValue_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ImportValue_IntegratedCircuit_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ImportValue_High-techProduct_CurrentMonthValue",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ImportValue_High-techProduct_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ExportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ImportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ExportValue_Eu_CurrentMonthYoy",
            "�й�:���ڽ��:ŷ��:����ͬ��": "China_ImportValue_Eu_CurrentMonthYoy",
            "�й�:���ڽ��:ŷ��:����ͬ��": "China_ExportValue_UnitedStates_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_UnitedStates_CurrentMonthValue_ExportValue_Us_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_Us_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ͬ��": "China_ExportValue_HongKong",
            "�й�:���ڽ��:����:����ͬ��": "China_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ֵ": "China_ImportValue_HongKong",
            "�й�:���ڽ��:�й����:����ֵ": "China_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ͬ��": "China_ExportValue_HongKong",
            "�й�:���ڽ��:�й����:����ͬ��": "China_CurrentMonthYoy",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ImportValue_HongKong",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_CurrentMonthYoy",
            "�й�:���ڽ��:�ձ�:����ͬ��": "China_ExportMonthValue_Japan_Current",
            "�й�:���ڽ��:�ձ�:����ͬ��": "China_ImportValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_Japan_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_Japan_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ͬ��": "China_ExportValue_Korea_CurrentMonthValue",
            "�й�:���ڽ��:����:����ͬ��": "China_ImportValue_Korea_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ExportValue_Korea_CurrentMonthYoy",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ImportValue_Korea_Current_ExportValue_TaiwanChina_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ImportValue_TaiwanChina_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ExportValue_TaiwanChina_CurrentMonthYoy"
        }
        self.update_low_freq_from_excel_meta('���ų���ģ��.xlsx', map_name_to_english)
