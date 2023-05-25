# coding=gbk
# Time Created: 2023/4/25 9:45
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase


class DatabaseUpdater(PgDbUpdaterBase):
    """
    ע�⣬pgsql��֧����������~45���ַ��������ضϣ�����Ӱ��calculate_yet_updated�����߼������Ӣ������������Ҫ�ֵ���
    """

    def __init__(self, base_config: BaseConfig, if_rename):
        super().__init__(base_config)
        self.if_rename = if_rename
        self.get_stored_metrics()
        # self.update_pmi()
        # self.update_export()
        self.update_export2()
        self.update_export3()

        self.set_all_nan_to_null()
        self.close()

    def update_pmi(self):
        self.update_low_freq_from_excel_meta('��ʿPMI.xlsx', self.pmi_map_windname_to_english)

    def update_export(self):
        self.update_low_freq_from_excel_meta('���ų���ģ��.xlsx', self.export_required_windname_to_english,
                                             if_rename=self.if_rename)
        self.calculate_yoy(value_str='CurrentMonthValue', yoy_str='CurrentMonthYoy', cn_value_str='����ֵ',
                           cn_yoy_str='����ͬ��', cn_names_to_exhibit=self.export_chinese_names_for_view)

        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export_wide',
                                    self.export_chinese_names_for_view)

    def update_export2(self):
        self.update_low_freq_from_excel_meta('need_update���������ݿ�.xlsx', self.export2_required_windname_to_english,
                                             sheet_name='����', if_rename=self.if_rename)
        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export2_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export2_wide',
                                    self.export2_chinese_names_for_view)

    def update_export3(self):
        self.update_low_freq_from_excel_meta('need_update���������ݿ�.xlsx', self.export3_required_windname_to_english,
                                             sheet_name='�ṹ', if_rename=self.if_rename)
        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export3_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export3_wide',
                                    self.export3_chinese_names_for_view)

    def get_stored_metrics(self):
        """
        Ŀ���ǰ����ݴ����߼��ͱ�������¼�ֿ���������������
        �ֵ�map��������;����excel��ȡ��Ӧ��Ԫ���ݣ�Ȼ��ݴ˴�wind���ء�
        �б���һ����;�����ɰ�����Щ���ݵĿ��ʽ��������metabaseչʾ��
        """
        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ�ֶ�ӳ���Ӣ������ (����translate_script.py�õ�)
        self.pmi_map_windname_to_english = {
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
        # ����translate_script.py�õ�
        self.export_required_windname_to_english = {
            "�й�:���ڽ��:����ֵ": "China_ExportValue_CurrentMonthValue",
            "�й�:���ڽ��:����ֵ": "China_ImportValue_CurrentMonthValue",
            "�й�:ó�ײ��:����ֵ": "China_BalanceOfTrade_CurrentMonthValue",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ExportValue_MechanicalAndElectrical_CurrentMonthValue",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ExportValue_High_techProducts_CurrentMonthValue",
            "�й�:���ڽ��:��װ�����Ÿ���:����ֵ": "China_ExportValue_ClothingAndAccessories_CurrentMonthValue",
            "�й�:���ڽ��:��֯ɴ�ߡ�֯�Ｐ��Ʒ:����ֵ": "China_ExportValue_TextileYarnFabrics_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ExportValue_Ic_CurrentMonthValue",
            "�й�:���ڽ��:������Ʒ:����ֵ": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "�й�:���ڽ��:ҽ����������е:����ֵ": "China_ExportValue_MedicalInstruments_CurrentMonthValue",
            "�й�:���ڽ��:������������:����ֵ": "China_ExportValue_AutomobileInclChassis_CurrentMonthValue",
            "�й�:���ڽ��:���������:����ֵ": "China_ExportValue_AutoParts_CurrentMonthValue",
            "�й�:���ڽ��:ũ��Ʒ:����ֵ": "China_ImportValue_AgriculturalProducts_CurrentMonthValue",
            "�й�:���ڽ��:��:����ֵ": "China_ImportValue_Soybean_CurrentMonthValue",
            "�й�:���ڽ��:����ɰ���侫��:����ֵ": "China_ImportValue_IronOreAndConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ֵ": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "�й�:���ڽ��:ԭ��:����ֵ": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "�й�:���ڽ��:ú����ú:����ֵ": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            "�й�:���ڽ��:��Ȼ��:����ֵ": "China_ImportValue_NaturalGas_CurrentMonthValue",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ImportValue_MechanicalAndElectrical_CurrentMonthValue",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ImportValue_Ic_CurrentMonthValue",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ImportValue_High_techProducts_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ExportValue_ASEAN_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ImportValue_ASEAN_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ExportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ImportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_Us_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_Us_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ֵ": "China_ExportValue_Hongkong_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ֵ": "China_ImportValue_Hongkong_CurrentMonthValue",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ExportValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ImportValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_SouthKorea_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_SouthKorea_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ExportValue_Taiwan_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ImportValue_Taiwan_CurrentMonthValue",
            "�й�:���ڽ��:��������:����ֵ": "China_ExportValue_LatinAmerica_CurrentMonthValue",
            "�й�:���ڽ��:��������:����ֵ": "China_ImportValue_LatinAmerica_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_Africa_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_Africa_CurrentMonthValue",
            "�й�:���ڽ��:����˹:����ֵ": "China_ExportValue_Russia_CurrentMonthValue",
            "�й�:���ڽ��:����˹:����ֵ": "China_ImportValue_Russia_CurrentMonthValue",
        }

        self.export2_required_windname_to_english = {
            "���ڼ�ֵָ��(HS2):��ָ��": "ExportValueIndex(hs2)_Aggindex",
            "��������ָ��(HS2):��ָ��": "ExportQuantityIndex(hs2)_Aggindex",
            "���ڼ۸�ָ��(HS2):��ָ��": "ExportPriceIndex(hs2)_Aggindex",
            "���ڼ�ֵָ��(HS2):ͬ��": "ExportValueIndex(hs2)_Yoy",
            "��������ָ��(HS2):ͬ��": "ExportquantityIndex(hs2))_Yoy",
            "���ڼ۸�ָ��(HS2):ͬ��": "ExportPriceIndex(hs2)_Yoy",
            "��ҵ��ҵ:���ڽ���ֵ:����ͬ��": "IndustrialEnterprises_ExportDeliveryValue_CurrentMonthYoy",
            "��ҵ��ҵ:���ڽ���ֵ:����ֵ": "IndustrialEnterprises_ExportDeliveryValue_CurrentMonthValue",
            "PPI:ȫ����ҵƷ:����ͬ��:+3��": "Ppi_TotalIndustrialGoods_CurrentMonthYoy+3M",
            "ȫ��:Ħ����ͨȫ������ҵPMI": "Global_JPMorganGlobalManufacturingPmi",
            "OECD�ۺ�����ָ��": "OecdCompositeLeadingIndicators",
            "���ڽ��:����ֵ:����": "China_ExportValue_CurrentMonthValue_MillionUSD",
            "ӡ��:���ڽ��:��Ʒ:��Ԫ": "India_ExportValue_CurrentMonthValue",
            "Խ��:���ڽ��:�ܽ��:����ֵ": "Vietnam_ExportValue_CurrentMonthValue",
            "����:�����ܶ�:����": "Korea_ExportValue_CurrentMonthValue",
            "�ձ�:���ڽ��:����ֵ:��Ԫ": "Japan_ExportValue_CurrentMonthValue",
            "�¹�:���ڽ��:��Ԫ:����": "Germany_ExportValue_CurrentMonthValue",
            "Ͷ�������������:����ʹ��:����/�ϼ�": "Input-outputBasicTraffic_FinalUse_Construction/Total",
            "Ͷ�������������:����ʹ��:��������/�ϼ�": "Input-outputBasicTraffic_FinalUse_OtherServices/Total",
            "Ͷ�������������:����ʹ��:��е�豸����/�ϼ�": "Input-outputBasicTraffic_FinalUse_Manufacturing/Total",
            "Ͷ�������������:����ʹ��:����/�ϼ�": "Input-outputBasicTraffic_FinalUse_Export/Total",
            "����������(������)": "FinalConsumptionRate",
            "�ʱ��γ���(Ͷ����)": "CapitalFormationRate",
            "��������": "NetExportRate",
            "����ó�ײ��:ռGDP����:����ֵ": "ServiceTradeBalance_ShareOfGdp_CurrentQuarterValue",
            "����ó�ײ��:ռGDP����:����ֵ": "GoodsTradeBalance_ShareOfGdp_CurrentQuarterValue",
            "�����˻����:ռGDP����:����ֵ": "CurrentAccountBalance_ShareOfGdp_CurrentQuarterValue",
            "Ͷ��������:ռGDP����:����ֵ": "InvestmentIncomeBalance_ShareOfGdp_CurrentQuarterValue",
            "GDP����ͬ�ȹ�����:����ͷ��񾻳���": "YoyQuarterlyContributionToGdp_GoodsAndServicesNetExport",
            "��GDP����ͬ�ȵ�����:��������֧��": "YoyQuarterlyGdpBoost_FinalConsumptionExpenditure",
            "��GDP����ͬ�ȵ�����:�ʱ��γ��ܶ�": "YoyQuarterlyGdpBoost_GrossCapitalFormation",
            "��GDP����ͬ�ȵ�����:����ͷ��񾻳���": "YoyQuarterlyGdpBoost_GoodsAndServicesNetExport"
        }

        self.export3_required_windname_to_english = {
            "���ʷ���ó�ײ��:����ֵ": "BalanceOfTradeInServices_CurrentMonthValue",
            "���ʷ���ó�ײ��:�ӹ�����:����ֵ": "BalanceOfTradeInServices_ProcessingServices_CurrentMonthValue",
            "���ʷ���ó�ײ��:����:����ֵ": "BalanceOfTradeInServices_Transportation_CurrentMonthValue",
            "���ʷ���ó�ײ��:����:����ֵ": "BalanceOfTradeInServices_Travel_CurrentMonthValue",
            "���ʷ���ó�ײ��:֪ʶ��Ȩʹ�÷�:����ֵ": "BalanceOfTradeInServices_IntellectualProperty_CurrentMonthValue",
            "���ʷ���ó�ײ��:���š����������Ϣ����:����ֵ": "BalanceOfTradeInServices_Telecommunications_CurrentMonthValue",
            "���ʷ���ó�ײ��:������ҵ����:����ֵ": "BalanceOfTradeInServices_OtherBusinessServices_CurrentMonthValue_TheValue",
            "SITC:���ڽ��:����ֵ:һ��������Ʒ": "Sitc_Exportvalue_CurrentMonthValue_PrimaryProducts",
            "SITC:ó�ײ��:����ֵ:һ��������Ʒ": "Sitc_BalanceOfTrade_CurrentMonthValue_PrimaryProducts",
            "SITC:���ڽ��:����ֵ:������ҵ��Ʒ": "Sitc_Exportvalue_CurrentMonthValue_IndustrialProducts",
            "SITC:ó�ײ��:����ֵ:������ҵ��Ʒ": "Sitc_BalanceOfTrade_CurrentMonthValue_IndustrialProducts",
            ###
            "���ڽ��:ʳƷ����Ҫ��ʳ�õĻ��": "Exportvalue_FoodAndLiveAnimalsMainlyForConsumption",
            "���ڽ��:���ϼ�����": "Exportvalue_BeveragesAndTobacco",
            "���ڽ��:��ʳ��ԭ��": "Exportvalue_Non-edibleRawMaterials",
            "���ڽ��:����ȼ�ϡ����ͼ��й�ԭ��": "Exportvalue_FossilFuelLubricantsAndRelatedRawMaterials",
            "���ڽ��:����ֲ����֬����": "Exportvalue_AnimalAndPlantOilsAndWaxes",
            "���ڽ��:��ѧƷ���йز�Ʒ": "Exportvalue_ChemistryProductsAndRelatedProducts",
            "���ڽ��:��Ĳ�Ʒ������Ʒ��ұ��Ʒ������Ʒ": "Exportvalue_TextileRubberMiningAndMetal",
            "���ڽ��:��е�������豸": "Exportvalue_MachineryAndTransportEquipment",
            "���ڽ��:������Ʒ": "Exportvalue_MiscellaneousProducts",
            "���ڽ��:δ�����������Ʒ": "Exportvalue_OtherGoodsNotClassified",
            "���ڽ��:ʳƷ����Ҫ��ʳ�õĻ��": "Importvalue_FoodstuffsAndLiveAnimalsMainlyForConsumption",
            "���ڽ��:���ϼ�����": "IMportvalue_BeveragesAndTobacco",
            "���ڽ��:��ʳ��ԭ��": "Importvalue_Non-edibleRawMaterials",
            "���ڽ��:����ȼ�ϡ����ͼ��й�ԭ��": "Importvalue_FossilFuelLubricantsAndRelatedRawMaterials",
            "���ڽ��:����ֲ����֬����": "Importvalue_AnimalAndPlantOilsAndWaxes",
            "���ڽ��:��ѧƷ���йز�Ʒ": "Importvalue_ChemicalsAndRelatedProducts",
            "���ڽ��:��Ĳ�Ʒ������Ʒ��ұ��Ʒ������Ʒ": "Importvalue_TextileRubberMiningAndMetal",
            "���ڽ��:��е�������豸": "Importvalue_MachineryAndTransportationEquipment",
            "���ڽ��:������Ʒ": "Importvalue_MiscellaneousProducts",
            "���ڽ��:δ�����������Ʒ": "Importvalue_OtherGoodsNotClassified"
        }

        # ���¿�����view������չʾ������
        self.export_chinese_names_for_view = [
            '�й�:���ڽ��:����ͬ��', '�й�:���ڽ��:����ͬ��', '�й�:ó�ײ��:����ͬ��',
            '�й�:���ڽ��:����ֵ', '�й�:���ڽ��:����ֵ', '�й�:ó�ײ��:����ֵ',
            ###
            '�й�:���ڽ��:�����Ʒ:����ͬ��', '�й�:���ڽ��:���¼�����Ʒ:����ͬ��',
            '�й�:���ڽ��:��װ�����Ÿ���:����ͬ��', '�й�:���ڽ��:��֯ɴ�ߡ�֯�Ｐ��Ʒ:����ͬ��',
            '�й�:���ڽ��:���ɵ�·:����ͬ��', '�й�:���ڽ��:������Ʒ:����ͬ��',
            '�й�:���ڽ��:ҽ����������е:����ͬ��', '�й�:���ڽ��:������������:����ͬ��',
            '�й�:���ڽ��:���������:����ͬ��', '�й�:���ڽ��:�����Ʒ:����ֵ', '�й�:���ڽ��:���¼�����Ʒ:����ֵ',
            '�й�:���ڽ��:��װ�����Ÿ���:����ֵ', '�й�:���ڽ��:��֯ɴ�ߡ�֯�Ｐ��Ʒ:����ֵ',
            '�й�:���ڽ��:���ɵ�·:����ֵ', '�й�:���ڽ��:������Ʒ:����ֵ', '�й�:���ڽ��:ҽ����������е:����ֵ',
            '�й�:���ڽ��:������������:����ֵ', '�й�:���ڽ��:���������:����ֵ',
            ###
            '�й�:���ڽ��:ũ��Ʒ:����ֵ',
            '�й�:���ڽ��:��:����ֵ', '�й�:���ڽ��:����ɰ���侫��:����ֵ', '�й�:���ڽ��:ͭ��ɰ���侫��:����ֵ',
            '�й�:���ڽ��:ԭ��:����ֵ', '�й�:���ڽ��:ú����ú:����ֵ', '�й�:���ڽ��:��Ȼ��:����ֵ',
            '�й�:���ڽ��:�����Ʒ:����ֵ', '�й�:���ڽ��:���ɵ�·:����ֵ', '�й�:���ڽ��:���¼�����Ʒ:����ֵ',
            ###
            '�й�:���ڽ��:�����ǹ�������:����ֵ', '�й�:���ڽ��:�����ǹ�������:����ֵ', '�й�:���ڽ��:ŷ��:����ֵ',
            '�й�:���ڽ��:ŷ��:����ֵ', '�й�:���ڽ��:����:����ֵ', '�й�:���ڽ��:����:����ֵ',
            '�й�:���ڽ��:�й����:����ֵ', '�й�:���ڽ��:�й����:����ֵ', '�й�:���ڽ��:�ձ�:����ֵ',
            '�й�:���ڽ��:�ձ�:����ֵ', '�й�:���ڽ��:����:����ֵ', '�й�:���ڽ��:����:����ֵ',
            '�й�:���ڽ��:�й�̨��:����ֵ', '�й�:���ڽ��:�й�̨��:����ֵ', '�й�:���ڽ��:��������:����ֵ',
            '�й�:���ڽ��:��������:����ֵ', '�й�:���ڽ��:����:����ֵ', '�й�:���ڽ��:����:����ֵ',
            '�й�:���ڽ��:����˹:����ֵ', '�й�:���ڽ��:����˹:����ֵ',
            ####
            '�й�:���ڽ��:ũ��Ʒ:����ͬ��', '�й�:���ڽ��:��:����ͬ��', '�й�:���ڽ��:����ɰ���侫��:����ͬ��',
            '�й�:���ڽ��:ͭ��ɰ���侫��:����ͬ��', '�й�:���ڽ��:ԭ��:����ͬ��', '�й�:���ڽ��:ú����ú:����ͬ��',
            '�й�:���ڽ��:��Ȼ��:����ͬ��', '�й�:���ڽ��:�����Ʒ:����ͬ��', '�й�:���ڽ��:���ɵ�·:����ͬ��',
            '�й�:���ڽ��:���¼�����Ʒ:����ͬ��',
            ###
            '�й�:���ڽ��:�����ǹ�������:����ͬ��',
            '�й�:���ڽ��:�����ǹ�������:����ͬ��', '�й�:���ڽ��:ŷ��:����ͬ��', '�й�:���ڽ��:ŷ��:����ͬ��',
            '�й�:���ڽ��:����:����ͬ��', '�й�:���ڽ��:����:����ͬ��', '�й�:���ڽ��:�й����:����ͬ��',
            '�й�:���ڽ��:�й����:����ͬ��', '�й�:���ڽ��:�ձ�:����ͬ��', '�й�:���ڽ��:�ձ�:����ͬ��',
            '�й�:���ڽ��:����:����ͬ��', '�й�:���ڽ��:����:����ͬ��', '�й�:���ڽ��:�й�̨��:����ͬ��',
            '�й�:���ڽ��:�й�̨��:����ͬ��', '�й�:���ڽ��:��������:����ͬ��', '�й�:���ڽ��:��������:����ͬ��',
            '�й�:���ڽ��:����:����ͬ��', '�й�:���ڽ��:����:����ͬ��',
            '�й�:���ڽ��:����˹:����ͬ��', '�й�:���ڽ��:����˹:����ͬ��'
        ]

        self.export2_chinese_names_for_view = [
            '���ڼ�ֵָ��(HS2):��ָ��', '��������ָ��(HS2):��ָ��', '���ڼ۸�ָ��(HS2):��ָ��',
            '���ڼ�ֵָ��(HS2):ͬ��', '��������ָ��(HS2):ͬ��', '���ڼ۸�ָ��(HS2):ͬ��', '�й�:���ڽ��:����ͬ��',
            '�й�:���ڽ��:����ֵ', '��ҵ��ҵ:���ڽ���ֵ:����ͬ��', '��ҵ��ҵ:���ڽ���ֵ:����ֵ',
            'PPI:ȫ����ҵƷ:����ͬ��:+3��', 'ȫ��:Ħ����ͨȫ������ҵPMI', 'OECD�ۺ�����ָ��', 'ӡ��:���ڽ��:��Ʒ:��Ԫ',
            'Խ��:���ڽ��:�ܽ��:����ֵ', '����:�����ܶ�:����', '�ձ�:���ڽ��:����ֵ:��Ԫ', '�¹�:���ڽ��:��Ԫ:����',
            '���ڽ��:����ֵ:����',
            'Ͷ�������������:����ʹ��:����/�ϼ�', 'Ͷ�������������:����ʹ��:��������/�ϼ�',
            'Ͷ�������������:����ʹ��:��е�豸����/�ϼ�', 'Ͷ�������������:����ʹ��:����/�ϼ�', '����������(������)',
            '�ʱ��γ���(Ͷ����)', '��������', '����ó�ײ��:ռGDP����:����ֵ', '����ó�ײ��:ռGDP����:����ֵ',
            '�����˻����:ռGDP����:����ֵ', 'Ͷ��������:ռGDP����:����ֵ', 'GDP����ͬ�ȹ�����:����ͷ��񾻳���',
            '��GDP����ͬ�ȵ�����:��������֧��', '��GDP����ͬ�ȵ�����:�ʱ��γ��ܶ�',
            '��GDP����ͬ�ȵ�����:����ͷ��񾻳���']

        self.export3_chinese_names_for_view = list(self.export3_required_windname_to_english.keys())