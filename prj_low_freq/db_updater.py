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
    """
    注意，pgsql不支持列名超过~45个字符，否则会截断，进而影响calculate_yet_updated函数逻辑。因此英文名过长的需要手调。
    """
    def __init__(self, base_config: BaseConfig, if_rename):
        super().__init__(base_config)
        self.if_rename = if_rename
        self.update_pmi()
        self.update_export()
        self.calculate_yet_updated()
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
            "中国:出口金额:当月值": "China_ExportValue_CurrentMonthValue",
            "中国:进口金额:当月值": "China_ImportValue_CurrentMonthValue",
            "中国:贸易差额:当月值": "China_BalanceOfTrade_CurrentMonthValue",
            "中国:出口金额:当月同比": "China_ExportValue_CurrentMonthYoy",
            "中国:进口金额:当月同比": "China_ImportValue_CurrentMonthYoy",
            "中国:贸易差额:当月同比": "China_BalanceOfTrade_CurrentMonthYoy",
            "中国:出口金额:机电产品:当月同比": "China_ExportValue_MechanicalAndElectrical_CurrentMonthYoy",
            "中国:出口金额:高新技术产品:当月同比": "China_ExportValue_High_techProducts_CurrentMonthYoy",
            "中国:出口金额:服装及衣着附件:当月同比": "China_ExportValue_ClothingAndAccessories_CurrentMonthYoy",
            "中国:出口金额:纺织纱线织物及其制品:当月同比": "China_ExportValue_TextileYarnFabrics_CurrentMonthYoy",
            "中国:出口金额:塑料制品:当月同比": "China_ExportValue_PlasticProducts_CurrentMonthYoy",
            "中国:出口金额:集成电路:当月值": "China_ExportValue_Ic_CurrentMonthValue",
            "中国:出口金额:集成电路:当月同比": "China_ExportValue_Ic_CurrentMonthYoy",
            "中国:出口数量:集成电路:当月值": "China_ExportQuantity_Ic_CurrentMonthValue",
            "中国:出口金额:塑料制品:当月值": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月值": "China_ExportValue_MedicalInstruments_CurrentMonthValue",
            "中国:出口金额:医疗仪器及器械:当月同比": "China_ExportValue_MedicalInstruments_CurrentMonthYoy",
            "中国:出口数量:汽车包括底盘:当月值": "China_ExportQuantity_AutomobilesInclChassis_CurrentMonthValue",
            "中国:出口数量:汽车包括底盘:当月同比": "China_ExportQuantity_AutomobilesInclChassis_CurrentMonthYoy",
            "中国:出口金额:汽车包括底盘:当月值": "China_ExportValue_AutomobileInclChassis_CurrentMonthValue",
            "中国:出口金额:汽车包括底盘:当月同比": "China_ExportValue_AutomobileInclChassis_CurrentMonthYoy",
            "中国:出口金额:汽车零配件:当月值": "China_ExportValue_AutoParts_CurrentMonthValue",
            "中国:出口金额:汽车零配件:当月同比": "China_ExportValue_AutoParts_CurrentMonthYoy",
            "中国:进口金额:农产品:当月值": "China_ImportValue_AgriculturalProducts_CurrentMonthValue",
            "中国:进口金额:农产品:当月同比": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "中国:进口数量:大豆:当月值": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "中国:进口数量:大豆:当月同比": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "中国:进口金额:铁矿砂及其精矿:当月值": "China_ImportValue_IronOreAndConcentrate_CurrentMonthValue",
            "中国:进口金额:铁矿砂及其精矿:当月同比": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:铜矿砂及其精矿:当月值": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            "中国:进口金额:铜矿砂及其精矿:当月同比": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "中国:进口金额:原油:当月值": "China_ImportValue_CrudeOil_CurrentMonthValue",
            "中国:进口金额:原油:当月同比": "China_ImportValue_CrudeOil_CurrentMonthYoy",
            "中国:进口金额:煤及褐煤:当月值": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            "中国:进口金额:煤及褐煤:当月同比": "China_ImportValue_CoalAndLignite_CurrentMonthYoy",
            "中国:进口金额:天然气:当月值": "China_ImportValue_NaturalGas_CurrentMonthValue",
            "中国:进口金额:天然气:当月同比": "China_ImportValue_NaturalGas_CurrentMonthYoy",
            "中国:进口金额:机电产品:当月值": "China_ImportValue_MechanicalAndElectrical_CurrentMonthValue",
            "中国:进口金额:机电产品:当月同比": "China_ImportValue_MechanicalAndElectrical_CurrentMonthYoy",
            "中国:进口金额:集成电路:当月值": "China_ImportValue_Ic_CurrentMonthValue",
            "中国:进口金额:集成电路:当月同比": "China_ImportValue_Ic_CurrentMonthYoy",
            "中国:进口金额:高新技术产品:当月值": "China_ImportValue_High_techProducts_CurrentMonthValue",
            "中国:进口金额:高新技术产品:当月同比": "China_ImportValue_High_techProducts_CurrentMonthYoy",
            "中国:出口金额:东南亚国家联盟:当月值": "China_ExportValue_ASEAN_CurrentMonthValue",
            "中国:进口金额:东南亚国家联盟:当月值": "China_ImportValue_ASEAN_CurrentMonthValue",
            "中国:出口金额:东南亚国家联盟:当月同比": "China_ExportValue_ASEAN_CurrentMonthYoy",
            "中国:进口金额:东南亚国家联盟:当月同比": "China_ImportValue_ASEAN_CurrentMonthYoy",
            "中国:出口金额:欧盟:当月值": "China_ExportValue_Eu_CurrentMonthValue",
            "中国:进口金额:欧盟:当月值": "China_ImportValue_Eu_CurrentMonthValue",
            "中国:出口金额:欧盟:当月同比": "China_ExportValue_Eu_CurrentMonthYoy",
            "中国:进口金额:欧盟:当月同比": "China_ImportValue_Eu_CurrentMonthYoy",
            "中国:出口金额:美国:当月值": "China_ExportValue_Us_CurrentMonthValue",
            "中国:进口金额:美国:当月值": "China_ImportValue_Us_CurrentMonthValue",
            "中国:出口金额:美国:当月同比": "China_ExportValue_Us_CurrentMonthYoy",
            "中国:进口金额:美国:当月同比": "China_ImportValue_Us_CurrentMonthYoy",
            "中国:出口金额:中国香港:当月值": "China_ExportValue_Hongkong_CurrentMonthValue",
            "中国:进口金额:中国香港:当月值": "China_ImportValue_Hongkong_CurrentMonthValue",
            "中国:出口金额:中国香港:当月同比": "China_ExportValue_Hongkong_CurrentMonthYoy",
            "中国:进口金额:中国香港:当月同比": "China_ImportValue_Hongkong_CurrentMonthYoy",
            "中国:出口金额:日本:当月值": "China_ExportValue_Japan_CurrentMonthValue",
            "中国:进口金额:日本:当月值": "China_ImportValue_Japan_CurrentMonthValue",
            "中国:出口金额:日本:当月同比": "China_ExportValue_Japan_CurrentMonthYoy",
            "中国:进口金额:日本:当月同比": "China_ImportValue_Japan_CurrentMonthYoy",
            "中国:出口金额:韩国:当月值": "China_ExportValue_SouthKorea_CurrentMonthValue",
            "中国:进口金额:韩国:当月值": "China_ImportValue_SouthKorea_CurrentMonthValue",
            "中国:出口金额:韩国:当月同比": "China_ExportValue_SouthKorea_CurrentMonthYoy",
            "中国:进口金额:韩国:当月同比": "China_ImportValue_SouthKorea_CurrentMonthYoy",
            "中国:出口金额:中国台湾:当月值": "China_ExportValue_Taiwan_CurrentMonthValue",
            "中国:进口金额:中国台湾:当月值": "China_ImportValue_Taiwan_CurrentMonthValue",
            "中国:出口金额:中国台湾:当月同比": "China_ExportValue_Taiwan_CurrentMonthYoy",
            "中国:进口金额:中国台湾:当月同比": "China_ImportValue_Taiwan_CurrentMonthYoy"
        }
        self.update_low_freq_from_excel_meta('中信出口模板.xlsx', map_name_to_english, if_rename=self.if_rename)

    def calculate_yet_updated(self):
        # Step 1: Find the columns with null values in the latest row of export_wide
        query = "SELECT * FROM export_wide ORDER BY date DESC LIMIT 1"
        df = pd.read_sql_query(text(query), self.alch_conn)
        null_columns = df.columns[df.isnull().any()].tolist()

        for col in null_columns:
            # Step 2: Go to low_freq_long and find the corresponding column (replace "Yoy" with "Value")
            metric_name_value = col.replace("Yoy", "Value")

            # Step 3: Find the latest value for this metric_name_value and the value from the same period last year
            query = f"""
            SELECT l1.value AS current_value, l2.value AS last_year_value, l1.date AS current_date
            FROM low_freq_long l1
            LEFT JOIN low_freq_long l2
            ON EXTRACT(YEAR FROM l1.date) = EXTRACT(YEAR FROM l2.date) + 1
            AND EXTRACT(MONTH FROM l1.date) = EXTRACT(MONTH FROM l2.date)
            WHERE l1.metric_name = '{metric_name_value}'
            ORDER BY l1.date DESC
            LIMIT 1
            """
            df = pd.read_sql_query(text(query), self.alch_conn)

            if df.empty or pd.isnull(df.loc[0, 'current_value']):
                missing_date = df.loc[0, 'current_date'] if not df.empty else "unknown"
                print(f"The latest value for {metric_name_value} is missing for date: {missing_date}.")
                continue

            # Calculate YoY change
            current_value = df.loc[0, 'current_value']
            last_year_value = df.loc[0, 'last_year_value']
            yoy_change = (current_value - last_year_value) / last_year_value * 100

            # Step 4: Write the calculated value into low_freq_long
            query = f"""
            INSERT INTO low_freq_long (date, metric_name, value, data_version)
            VALUES ('{df.loc[0, 'date']}', '{col}', {yoy_change}, 'calculated from CurrentMonthValue')
            """
            self.alch_conn.execute(text(query))
            self.alch_conn.commit()
