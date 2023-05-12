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
        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ�ֶ�ӳ���Ӣ������ (����translate_script.py�õ�)
        map_name_to_english = {
            "ŷԪ��:�ۺ�PMI": "Eurozone_CompositePmi",
            "�ձ�:�ۺ�PMI": "Japan_CompositePmi",
            "����:�ۺ�PMI": "US_CompositePmi",
            "����:��Ӧ����Э��(ISM):����ҵPMI": "US_InstituteForSupplyManagement(ism)_ManufacturingPmi",
            "�ձ�:����ҵPMI": "Japan_ManufacturingPmi",
            "ŷԪ��:����ҵPMI": "Eurozone_ManufacturingPmi",
            "�ձ�:GDP:�ּ�": "Japan_Gdp_CurrentPrice",
            "�ձ�:GDP:�ּ�:��Ԫ": "Japan_Gdp_CurrentPrice_Usd",
            "ŷԪ��:GDP:�ּ�": "Eurozone_Gdp_CurrentPrice",
            "ŷԪ��:GDP:�ּ�:��Ԫ": "Eurozone_Gdp_CurrentPrice_Usd",
            "����:GDP:�ּ�:������:����": "US_Gdp_CurrentPrice_Annualized_SeasonAdj"
        }
        self.update_low_freq_from_excel_meta('��ʿPMI.xlsx', map_name_to_english)

    def update_export(self):
        # ����translate_script.py�õ�
        map_name_to_english = {
            "�й�:���ڽ��:����ֵ": "China_ExportsValue_CurrentMonthValue",
            "�й�:���ڽ��:����ֵ": "China_ImportsValue_CurrentMonthValue",
            "�й�:ó�ײ��:����ֵ": "China_BalanceOfTrade_CurrentMonthValue",
            "�й�:���ڽ��:����ͬ��": "China_ExportsValue_CurrentMonthYoy",
            "�й�:���ڽ��:����ͬ��": "China_ImportsValue_CurrentMonthYoy",
            "�й�:ó�ײ��:����ͬ��": "China_BalanceOfTrade_CurrentMonthYoy",
            "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ExportsValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ExportsValue_High-techProducts_CurrentMonthYoy",
            "�й�:���ڽ��:��װ�����Ÿ���:����ͬ��": "China_ExportValue_ClothingAndClothingAccessories_CurrentMonthYoy",
            "�й�:���ڽ��:��֯ɴ��֯�Ｐ����Ʒ:����ͬ��": "China_ExportValue_TextileYarnFabricsAndProducts_CurrentMonthYoy",
            "�й�:���ڽ��:������Ʒ:����ͬ��": "China_ExportsValue_PlasticProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ExportValue_Ic_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ExportValue_Ic_CurrentMonthYoy",
            "�й�:��������:���ɵ�·:����ֵ": "China_ExportQuantity_Ic_CurrentMonthValue",
            "�й�:���ڽ��:������Ʒ:����ֵ": "China_ExportsValue_PlasticProducts_CurrentMonthValue",
            "�й�:���ڽ��:ҽ����������е:����ֵ": "China_ExportsValue_MedicalInstrumentsAndInstruments_CurrentMonthValue",
            "�й�:���ڽ��:ҽ����������е:����ͬ��": "China_ExportValue_MedicalInstrumentsAndInstruments_CurrentMonthYoy",
            "�й�:��������:������������:����ֵ": "China_ExportQuantity_AutomobilesIncludingChassis_CurrentMonthValue",
            "�й�:��������:������������:����ͬ��": "China_ExportsQuantity_AutomobilesIncludingChassis_CurrentMonthYoy",
            "�й�:���ڽ��:������������:����ֵ": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthValue",
            "�й�:���ڽ��:������������:����ͬ��": "China_ExportValue_AutomobileIncludingChassis_CurrentMonthYoy",
            "�й�:���ڽ��:���������:����ֵ": "China_ExportValue_AutomobileParts_CurrentMonthValue",
            "�й�:���ڽ��:���������:����ͬ��": "China_ExportsValue_AutoParts_CurrentMonthYoy",
            "�й�:���ڽ��:ũ��Ʒ:����ֵ": "China_ImportsValue_AgriculturalProducts_CurrentMonthValue",
            "�й�:���ڽ��:ũ��Ʒ:����ͬ��": "China_ImportsValue_AgriculturalProducts_CurrentMonthYoy",
            "�й�:��������:��:����ֵ": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "�й�:��������:��:����ͬ��": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "�й�:���ڽ��:����ɰ���侫��:����ֵ": "China_ImportedValue_IronOreAndConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:����ɰ���侫��:����ͬ��": "China_ImportsValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ֵ": "China_ImportsValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ͬ��": "China_ImportsValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ԭ��:����ֵ": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "�й�:���ڽ��:ԭ��:����ͬ��": "China_ImportValue_CrudeOil_CurrentMonthYoy",
            "�й�:���ڽ��:ú����ú:����ֵ": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            "�й�:���ڽ��:ú����ú:����ͬ��": "China_ImportsValue_CoalAndLignite_CurrentMonthYoy",
            "�й�:���ڽ��:��Ȼ��:����ֵ": "China_ImportsValue_NaturalGas_CurrentMonthValue",
            "�й�:���ڽ��:��Ȼ��:����ͬ��": "China_ImportsValue_NaturalGas_CurrentMonthYoy",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthValue",
            "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ImportValue_MechanicalAndElectricalProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ImportValue_Ic_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ImportValue_Ic_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ImportValue_High-techProducts_CurrentMonthValue",
            "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ImportValue_High-techProducts_CurrentMonthYoy",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ExportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ImportValue_AssociationOfSoutheastAsianNations_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ExportsValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ImportsValue_AssociationOfSoutheastAsianNations_CurrentMonthYoy",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ExportsValue_EuropeanUnion_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ImportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ͬ��": "China_ExportValue_Eu_CurrentMonthYoy",
            "�й�:���ڽ��:ŷ��:����ͬ��": "China_ImportValue_Eu_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ExportsValue_Us_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportsValue_Us_CurrentMonthValue",
            "�й�:���ڽ��:����:����ͬ��": "China_ExportsValue_Us_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ͬ��": "China_ImportsValue_Us_CurrentMonthYoy",
            "�й�:���ڽ��:�й����:����ֵ": "China_ExportsValue_Hongkong_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ֵ": "China_ImportsValue_Hongkong_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ͬ��": "China_ExportsValue_Hongkong_CurrentMonthYoy",
            "�й�:���ڽ��:�й����:����ͬ��": "China_ImportsValue_Hongkong_CurrentMonthYoy",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ExportsValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ImportsValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:�ձ�:����ͬ��": "China_ExportsValue_Japan_CurrentMonthYoy",
            "�й�:���ڽ��:�ձ�:����ͬ��": "China_ImportValue_Japan_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_Korea_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_SouthKorea_CurrentMonthValue",
            "�й�:���ڽ��:����:����ͬ��": "China_ExportsValue_SouthKorea_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ͬ��": "China_ImportsValue_SouthKorea_CurrentMonthYoy",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ExportsValue_Taiwan_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ImportValue_Taiwan_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ExportValue_Taiwan_CurrentMonthYoy",
            "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ImportValue_Taiwan_CurrentMonthYoy"
        }
        self.update_low_freq_from_excel_meta('���ų���ģ��.xlsx', map_name_to_english)
