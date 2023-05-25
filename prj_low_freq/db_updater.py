# coding=gbk
# Time Created: 2023/4/25 9:45
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase


class DatabaseUpdater(PgDbUpdaterBase):
    """
    注意，pgsql不支持列名超过~45个字符，否则会截断，进而影响calculate_yet_updated函数逻辑。因此英文名过长的需要手调。
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
        self.update_low_freq_from_excel_meta('博士PMI.xlsx', self.pmi_map_windname_to_english)

    def update_export(self):
        self.update_low_freq_from_excel_meta('中信出口模板.xlsx', self.export_required_windname_to_english,
                                             if_rename=self.if_rename)
        self.calculate_yoy(value_str='CurrentMonthValue', yoy_str='CurrentMonthYoy', cn_value_str='当月值',
                           cn_yoy_str='当月同比', cn_names_to_exhibit=self.export_chinese_names_for_view)

        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export_wide',
                                    self.export_chinese_names_for_view)

    def update_export2(self):
        self.update_low_freq_from_excel_meta('need_update进出口数据库.xlsx', self.export2_required_windname_to_english,
                                             sheet_name='总量', if_rename=self.if_rename)
        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export2_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export2_wide',
                                    self.export2_chinese_names_for_view)

    def update_export3(self):
        self.update_low_freq_from_excel_meta('need_update进出口数据库.xlsx', self.export3_required_windname_to_english,
                                             sheet_name='结构', if_rename=self.if_rename)
        # useful to check if next line reports error.
        missing_metrics = self.get_missing_metrics('metric_static_info', 'chinese_name',
                                                   self.export3_chinese_names_for_view)
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export3_wide',
                                    self.export3_chinese_names_for_view)

    def get_stored_metrics(self):
        """
        目的是把数据处理逻辑和变量名记录分开，代码更简洁清晰
        字典map有两个用途：从excel获取对应的元数据，然后据此从wind下载。
        列表有一个用途：生成包含这些数据的宽格式表，方便在metabase展示。
        """
        # 创建一个字典，键为指标 ID，值为手动映射的英文列名 (利用translate_script.py得到)
        self.pmi_map_windname_to_english = {
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
        # 利用translate_script.py得到
        self.export_required_windname_to_english = {
            "中国:出口金额:当月值": "China_ExportValue_CurrentMonthValue",
            "中国:进口金额:当月值": "China_ImportValue_CurrentMonthValue",
            "中国:贸易差额:当月值": "China_BalanceOfTrade_CurrentMonthValue",
            "中国:出口金额:机电产品:当月值": "China_ExportValue_MechanicalAndElectrical_CurrentMonthValue",
            "中国:出口金额:高新技术产品:当月值": "China_ExportValue_High_techProducts_CurrentMonthValue",
            "中国:出口金额:服装及衣着附件:当月值": "China_ExportValue_ClothingAndAccessories_CurrentMonthValue",
            "中国:出口金额:纺织纱线、织物及制品:当月值": "China_ExportValue_TextileYarnFabrics_CurrentMonthValue",
            "中国:出口金额:集成电路:当月值": "China_ExportValue_Ic_CurrentMonthValue",
            "中国:出口金额:塑料制品:当月值": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月值": "China_ExportValue_MedicalInstruments_CurrentMonthValue",
            "中国:出口金额:汽车包括底盘:当月值": "China_ExportValue_AutomobileInclChassis_CurrentMonthValue",
            "中国:出口金额:汽车零配件:当月值": "China_ExportValue_AutoParts_CurrentMonthValue",
            "中国:进口金额:农产品:当月值": "China_ImportValue_AgriculturalProducts_CurrentMonthValue",
            "中国:进口金额:大豆:当月值": "China_ImportValue_Soybean_CurrentMonthValue",
            "中国:进口金额:铁矿砂及其精矿:当月值": "China_ImportValue_IronOreAndConcentrate_CurrentMonthValue",
            "中国:进口金额:铜矿砂及其精矿:当月值": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "中国:进口金额:原油:当月值": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "中国:进口金额:煤及褐煤:当月值": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            "中国:进口金额:天然气:当月值": "China_ImportValue_NaturalGas_CurrentMonthValue",
            "中国:进口金额:机电产品:当月值": "China_ImportValue_MechanicalAndElectrical_CurrentMonthValue",
            "中国:进口金额:集成电路:当月值": "China_ImportValue_Ic_CurrentMonthValue",
            "中国:进口金额:高新技术产品:当月值": "China_ImportValue_High_techProducts_CurrentMonthValue",
            "中国:出口金额:东南亚国家联盟:当月值": "China_ExportValue_ASEAN_CurrentMonthValue",
            "中国:进口金额:东南亚国家联盟:当月值": "China_ImportValue_ASEAN_CurrentMonthValue",
            "中国:出口金额:欧盟:当月值": "China_ExportValue_Eu_CurrentMonthValue",
            "中国:进口金额:欧盟:当月值": "China_ImportValue_Eu_CurrentMonthValue",
            "中国:出口金额:美国:当月值": "China_ExportValue_Us_CurrentMonthValue",
            "中国:进口金额:美国:当月值": "China_ImportValue_Us_CurrentMonthValue",
            "中国:出口金额:中国香港:当月值": "China_ExportValue_Hongkong_CurrentMonthValue",
            "中国:进口金额:中国香港:当月值": "China_ImportValue_Hongkong_CurrentMonthValue",
            "中国:出口金额:日本:当月值": "China_ExportValue_Japan_CurrentMonthValue",
            "中国:进口金额:日本:当月值": "China_ImportValue_Japan_CurrentMonthValue",
            "中国:出口金额:韩国:当月值": "China_ExportValue_SouthKorea_CurrentMonthValue",
            "中国:进口金额:韩国:当月值": "China_ImportValue_SouthKorea_CurrentMonthValue",
            "中国:出口金额:中国台湾:当月值": "China_ExportValue_Taiwan_CurrentMonthValue",
            "中国:进口金额:中国台湾:当月值": "China_ImportValue_Taiwan_CurrentMonthValue",
            "中国:出口金额:拉丁美洲:当月值": "China_ExportValue_LatinAmerica_CurrentMonthValue",
            "中国:进口金额:拉丁美洲:当月值": "China_ImportValue_LatinAmerica_CurrentMonthValue",
            "中国:出口金额:非洲:当月值": "China_ExportValue_Africa_CurrentMonthValue",
            "中国:进口金额:非洲:当月值": "China_ImportValue_Africa_CurrentMonthValue",
            "中国:出口金额:俄罗斯:当月值": "China_ExportValue_Russia_CurrentMonthValue",
            "中国:进口金额:俄罗斯:当月值": "China_ImportValue_Russia_CurrentMonthValue",
        }

        self.export2_required_windname_to_english = {
            "出口价值指数(HS2):总指数": "ExportValueIndex(hs2)_Aggindex",
            "出口数量指数(HS2):总指数": "ExportQuantityIndex(hs2)_Aggindex",
            "出口价格指数(HS2):总指数": "ExportPriceIndex(hs2)_Aggindex",
            "出口价值指数(HS2):同比": "ExportValueIndex(hs2)_Yoy",
            "出口数量指数(HS2):同比": "ExportquantityIndex(hs2))_Yoy",
            "出口价格指数(HS2):同比": "ExportPriceIndex(hs2)_Yoy",
            "工业企业:出口交货值:当月同比": "IndustrialEnterprises_ExportDeliveryValue_CurrentMonthYoy",
            "工业企业:出口交货值:当月值": "IndustrialEnterprises_ExportDeliveryValue_CurrentMonthValue",
            "PPI:全部工业品:当月同比:+3月": "Ppi_TotalIndustrialGoods_CurrentMonthYoy+3M",
            "全球:摩根大通全球制造业PMI": "Global_JPMorganGlobalManufacturingPmi",
            "OECD综合领先指标": "OecdCompositeLeadingIndicators",
            "出口金额:当月值:百万": "China_ExportValue_CurrentMonthValue_MillionUSD",
            "印度:出口金额:商品:美元": "India_ExportValue_CurrentMonthValue",
            "越南:出口金额:总金额:当月值": "Vietnam_ExportValue_CurrentMonthValue",
            "韩国:出口总额:百万": "Korea_ExportValue_CurrentMonthValue",
            "日本:出口金额:当月值:美元": "Japan_ExportValue_CurrentMonthValue",
            "德国:出口金额:美元:百万": "Germany_ExportValue_CurrentMonthValue",
            "投入产出基本流量:最终使用:建筑/合计": "Input-outputBasicTraffic_FinalUse_Construction/Total",
            "投入产出基本流量:最终使用:其他服务/合计": "Input-outputBasicTraffic_FinalUse_OtherServices/Total",
            "投入产出基本流量:最终使用:机械设备制造/合计": "Input-outputBasicTraffic_FinalUse_Manufacturing/Total",
            "投入产出基本流量:最终使用:出口/合计": "Input-outputBasicTraffic_FinalUse_Export/Total",
            "最终消费率(消费率)": "FinalConsumptionRate",
            "资本形成率(投资率)": "CapitalFormationRate",
            "净出口率": "NetExportRate",
            "服务贸易差额:占GDP比重:当季值": "ServiceTradeBalance_ShareOfGdp_CurrentQuarterValue",
            "货物贸易差额:占GDP比重:当季值": "GoodsTradeBalance_ShareOfGdp_CurrentQuarterValue",
            "经常账户差额:占GDP比重:当季值": "CurrentAccountBalance_ShareOfGdp_CurrentQuarterValue",
            "投资收益差额:占GDP比重:当季值": "InvestmentIncomeBalance_ShareOfGdp_CurrentQuarterValue",
            "GDP当季同比贡献率:货物和服务净出口": "YoyQuarterlyContributionToGdp_GoodsAndServicesNetExport",
            "对GDP当季同比的拉动:最终消费支出": "YoyQuarterlyGdpBoost_FinalConsumptionExpenditure",
            "对GDP当季同比的拉动:资本形成总额": "YoyQuarterlyGdpBoost_GrossCapitalFormation",
            "对GDP当季同比的拉动:货物和服务净出口": "YoyQuarterlyGdpBoost_GoodsAndServicesNetExport"
        }

        self.export3_required_windname_to_english = {
            "国际服务贸易差额:当月值": "BalanceOfTradeInServices_CurrentMonthValue",
            "国际服务贸易差额:加工服务:当月值": "BalanceOfTradeInServices_ProcessingServices_CurrentMonthValue",
            "国际服务贸易差额:运输:当月值": "BalanceOfTradeInServices_Transportation_CurrentMonthValue",
            "国际服务贸易差额:旅行:当月值": "BalanceOfTradeInServices_Travel_CurrentMonthValue",
            "国际服务贸易差额:知识产权使用费:当月值": "BalanceOfTradeInServices_IntellectualProperty_CurrentMonthValue",
            "国际服务贸易差额:电信、计算机和信息服务:当月值": "BalanceOfTradeInServices_Telecommunications_CurrentMonthValue",
            "国际服务贸易差额:其他商业服务:当月值": "BalanceOfTradeInServices_OtherBusinessServices_CurrentMonthValue_TheValue",
            "SITC:出口金额:当月值:一、初级产品": "Sitc_Exportvalue_CurrentMonthValue_PrimaryProducts",
            "SITC:贸易差额:当月值:一、初级产品": "Sitc_BalanceOfTrade_CurrentMonthValue_PrimaryProducts",
            "SITC:出口金额:当月值:二、工业制品": "Sitc_Exportvalue_CurrentMonthValue_IndustrialProducts",
            "SITC:贸易差额:当月值:二、工业制品": "Sitc_BalanceOfTrade_CurrentMonthValue_IndustrialProducts",
            ###
            "出口金额:食品及主要供食用的活动物": "Exportvalue_FoodAndLiveAnimalsMainlyForConsumption",
            "出口金额:饮料及烟类": "Exportvalue_BeveragesAndTobacco",
            "出口金额:非食用原料": "Exportvalue_Non-edibleRawMaterials",
            "出口金额:矿物燃料、润滑油及有关原料": "Exportvalue_FossilFuelLubricantsAndRelatedRawMaterials",
            "出口金额:动、植物油脂及蜡": "Exportvalue_AnimalAndPlantOilsAndWaxes",
            "出口金额:化学品及有关产品": "Exportvalue_ChemistryProductsAndRelatedProducts",
            "出口金额:轻纺产品、橡胶制品矿冶产品及其制品": "Exportvalue_TextileRubberMiningAndMetal",
            "出口金额:机械及运输设备": "Exportvalue_MachineryAndTransportEquipment",
            "出口金额:杂项制品": "Exportvalue_MiscellaneousProducts",
            "出口金额:未分类的其他商品": "Exportvalue_OtherGoodsNotClassified",
            "进口金额:食品及主要供食用的活动物": "Importvalue_FoodstuffsAndLiveAnimalsMainlyForConsumption",
            "进口金额:饮料及烟类": "IMportvalue_BeveragesAndTobacco",
            "进口金额:非食用原料": "Importvalue_Non-edibleRawMaterials",
            "进口金额:矿物燃料、润滑油及有关原料": "Importvalue_FossilFuelLubricantsAndRelatedRawMaterials",
            "进口金额:动、植物油脂及蜡": "Importvalue_AnimalAndPlantOilsAndWaxes",
            "进口金额:化学品及有关产品": "Importvalue_ChemicalsAndRelatedProducts",
            "进口金额:轻纺产品、橡胶制品矿冶产品及其制品": "Importvalue_TextileRubberMiningAndMetal",
            "进口金额:机械及运输设备": "Importvalue_MachineryAndTransportationEquipment",
            "进口金额:杂项制品": "Importvalue_MiscellaneousProducts",
            "进口金额:未分类的其他商品": "Importvalue_OtherGoodsNotClassified"
        }

        # 更新宽数据view，用来展示的数据
        self.export_chinese_names_for_view = [
            '中国:出口金额:当月同比', '中国:进口金额:当月同比', '中国:贸易差额:当月同比',
            '中国:出口金额:当月值', '中国:进口金额:当月值', '中国:贸易差额:当月值',
            ###
            '中国:出口金额:机电产品:当月同比', '中国:出口金额:高新技术产品:当月同比',
            '中国:出口金额:服装及衣着附件:当月同比', '中国:出口金额:纺织纱线、织物及制品:当月同比',
            '中国:出口金额:集成电路:当月同比', '中国:出口金额:塑料制品:当月同比',
            '中国:出口金额:医疗仪器及器械:当月同比', '中国:出口金额:汽车包括底盘:当月同比',
            '中国:出口金额:汽车零配件:当月同比', '中国:出口金额:机电产品:当月值', '中国:出口金额:高新技术产品:当月值',
            '中国:出口金额:服装及衣着附件:当月值', '中国:出口金额:纺织纱线、织物及制品:当月值',
            '中国:出口金额:集成电路:当月值', '中国:出口金额:塑料制品:当月值', '中国:出口金额:医疗仪器及器械:当月值',
            '中国:出口金额:汽车包括底盘:当月值', '中国:出口金额:汽车零配件:当月值',
            ###
            '中国:进口金额:农产品:当月值',
            '中国:进口金额:大豆:当月值', '中国:进口金额:铁矿砂及其精矿:当月值', '中国:进口金额:铜矿砂及其精矿:当月值',
            '中国:进口金额:原油:当月值', '中国:进口金额:煤及褐煤:当月值', '中国:进口金额:天然气:当月值',
            '中国:进口金额:机电产品:当月值', '中国:进口金额:集成电路:当月值', '中国:进口金额:高新技术产品:当月值',
            ###
            '中国:出口金额:东南亚国家联盟:当月值', '中国:进口金额:东南亚国家联盟:当月值', '中国:出口金额:欧盟:当月值',
            '中国:进口金额:欧盟:当月值', '中国:出口金额:美国:当月值', '中国:进口金额:美国:当月值',
            '中国:出口金额:中国香港:当月值', '中国:进口金额:中国香港:当月值', '中国:出口金额:日本:当月值',
            '中国:进口金额:日本:当月值', '中国:出口金额:韩国:当月值', '中国:进口金额:韩国:当月值',
            '中国:出口金额:中国台湾:当月值', '中国:进口金额:中国台湾:当月值', '中国:出口金额:拉丁美洲:当月值',
            '中国:进口金额:拉丁美洲:当月值', '中国:出口金额:非洲:当月值', '中国:进口金额:非洲:当月值',
            '中国:出口金额:俄罗斯:当月值', '中国:进口金额:俄罗斯:当月值',
            ####
            '中国:进口金额:农产品:当月同比', '中国:进口金额:大豆:当月同比', '中国:进口金额:铁矿砂及其精矿:当月同比',
            '中国:进口金额:铜矿砂及其精矿:当月同比', '中国:进口金额:原油:当月同比', '中国:进口金额:煤及褐煤:当月同比',
            '中国:进口金额:天然气:当月同比', '中国:进口金额:机电产品:当月同比', '中国:进口金额:集成电路:当月同比',
            '中国:进口金额:高新技术产品:当月同比',
            ###
            '中国:出口金额:东南亚国家联盟:当月同比',
            '中国:进口金额:东南亚国家联盟:当月同比', '中国:出口金额:欧盟:当月同比', '中国:进口金额:欧盟:当月同比',
            '中国:出口金额:美国:当月同比', '中国:进口金额:美国:当月同比', '中国:出口金额:中国香港:当月同比',
            '中国:进口金额:中国香港:当月同比', '中国:出口金额:日本:当月同比', '中国:进口金额:日本:当月同比',
            '中国:出口金额:韩国:当月同比', '中国:进口金额:韩国:当月同比', '中国:出口金额:中国台湾:当月同比',
            '中国:进口金额:中国台湾:当月同比', '中国:出口金额:拉丁美洲:当月同比', '中国:进口金额:拉丁美洲:当月同比',
            '中国:出口金额:非洲:当月同比', '中国:进口金额:非洲:当月同比',
            '中国:出口金额:俄罗斯:当月同比', '中国:进口金额:俄罗斯:当月同比'
        ]

        self.export2_chinese_names_for_view = [
            '出口价值指数(HS2):总指数', '出口数量指数(HS2):总指数', '出口价格指数(HS2):总指数',
            '出口价值指数(HS2):同比', '出口数量指数(HS2):同比', '出口价格指数(HS2):同比', '中国:出口金额:当月同比',
            '中国:出口金额:当月值', '工业企业:出口交货值:当月同比', '工业企业:出口交货值:当月值',
            'PPI:全部工业品:当月同比:+3月', '全球:摩根大通全球制造业PMI', 'OECD综合领先指标', '印度:出口金额:商品:美元',
            '越南:出口金额:总金额:当月值', '韩国:出口总额:百万', '日本:出口金额:当月值:美元', '德国:出口金额:美元:百万',
            '出口金额:当月值:百万',
            '投入产出基本流量:最终使用:建筑/合计', '投入产出基本流量:最终使用:其他服务/合计',
            '投入产出基本流量:最终使用:机械设备制造/合计', '投入产出基本流量:最终使用:出口/合计', '最终消费率(消费率)',
            '资本形成率(投资率)', '净出口率', '服务贸易差额:占GDP比重:当季值', '货物贸易差额:占GDP比重:当季值',
            '经常账户差额:占GDP比重:当季值', '投资收益差额:占GDP比重:当季值', 'GDP当季同比贡献率:货物和服务净出口',
            '对GDP当季同比的拉动:最终消费支出', '对GDP当季同比的拉动:资本形成总额',
            '对GDP当季同比的拉动:货物和服务净出口']

        self.export3_chinese_names_for_view = list(self.export3_required_windname_to_english.keys())