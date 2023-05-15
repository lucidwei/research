# coding=gbk
# Time Created: 2023/4/25 9:45
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re, math
from WindPy import w
import pandas as pd
from base_config import BaseConfig
from pgdb_updater_base import PgDbUpdaterBase
from sqlalchemy import text
from utils import timeit


class DatabaseUpdater(PgDbUpdaterBase):
    """
    ע�⣬pgsql��֧����������~45���ַ��������ضϣ�����Ӱ��calculate_yet_updated�����߼������Ӣ������������Ҫ�ֵ���
    """

    def __init__(self, base_config: BaseConfig, if_rename):
        super().__init__(base_config)
        self.if_rename = if_rename
        self.get_stored_metrics()
        # self.update_pmi()
        self.update_export()

        self.set_all_nan_to_null()
        self.close()

    def update_pmi(self):
        self.update_low_freq_from_excel_meta('��ʿPMI.xlsx', self.pmi_map_name_to_english)

    def update_export(self):
        self.update_low_freq_from_excel_meta('���ų���ģ��.xlsx', self.export_map_name_to_english,
                                             if_rename=self.if_rename)
        self.calculate_yoy()
        self.execute_pgsql_function('processed_data.create_wide_view_from_chinese', 'low_freq_long', 'export_wide',
                                    self.export_chinese_names_to_display)

    @timeit
    def calculate_yoy(self):
        # Step 1: Select all "*CurrentMonthValue" data from low_freq_long
        query = "SELECT * FROM low_freq_long WHERE metric_name LIKE '%CurrentMonthValue'"
        df = pd.read_sql_query(text(query), self.alch_conn)

        for _, row in df.iterrows():
            metric_name_value = row['metric_name']
            metric_name_yoy = metric_name_value.replace("CurrentMonthValue", "CurrentMonthYoy")

            # Step 2: Find the value from the same period last year
            query = f"""
            SELECT value
            FROM low_freq_long
            WHERE metric_name = '{metric_name_value}'
            AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CAST('{row['date']}' AS DATE)) - 1
            AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CAST('{row['date']}' AS DATE))
            """
            df_last_year = pd.read_sql_query(text(query), self.alch_conn)

            if df_last_year.empty or pd.isnull(df_last_year.loc[0, 'value']):
                # print(f"No data for {metric_name_value} '{row['date']}' from the same period last year.")
                continue

            # Calculate YoY change
            current_value = row['value']
            last_year_value = df_last_year.loc[0, 'value']
            yoy_change = (current_value - last_year_value) / last_year_value * 100

            # Step 3: Insert the calculated YoY data into low_freq_long
            query = f"""
            INSERT INTO low_freq_long (date, metric_name, value)
            VALUES ('{row['date']}', '{metric_name_yoy}', {yoy_change})
            ON CONFLICT (date, metric_name) DO UPDATE SET value = EXCLUDED.value
            """
            self.alch_conn.execute(text(query))

            # Step 4.1: Get the source_code of the corresponding Value variable
            query = f"""
            SELECT source_code
            FROM metric_static_info
            WHERE english_name = '{metric_name_value}'
            """
            df = pd.read_sql_query(text(query), self.alch_conn)
            source_code_value = df.loc[0, 'source_code']

            # Step 4.2: Update source_code in metric_static_info
            new_source_code = f'calculated from {source_code_value}'
            query = f"""
            UPDATE metric_static_info
            SET source_code = '{new_source_code}'
            WHERE english_name = '{metric_name_yoy}'
            """
            self.alch_conn.execute(text(query))
            self.alch_conn.commit()

    def get_stored_metrics(self):
        """
        Ŀ���ǰ����ݴ����߼��ͱ�������¼�ֿ���������������
        """
        # ����һ���ֵ䣬��Ϊָ�� ID��ֵΪ�ֶ�ӳ���Ӣ������ (����translate_script.py�õ�)
        self.pmi_map_name_to_english = {
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
        self.export_map_name_to_english = {
            "�й�:���ڽ��:����ֵ": "China_ExportValue_CurrentMonthValue",
            "�й�:���ڽ��:����ֵ": "China_ImportValue_CurrentMonthValue",
            "�й�:ó�ײ��:����ֵ": "China_BalanceOfTrade_CurrentMonthValue",
            "�й�:���ڽ��:����ͬ��": "China_ExportValue_CurrentMonthYoy",
            "�й�:���ڽ��:����ͬ��": "China_ImportValue_CurrentMonthYoy",
            "�й�:ó�ײ��:����ͬ��": "China_BalanceOfTrade_CurrentMonthYoy",
            # "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ExportValue_MechanicalAndElectrical_CurrentMonthYoy",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ExportValue_MechanicalAndElectrical_CurrentMonthValue",
            # "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ExportValue_High_techProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ExportValue_High_techProducts_CurrentMonthValue",
            # "�й�:���ڽ��:��װ�����Ÿ���:����ͬ��": "China_ExportValue_ClothingAndAccessories_CurrentMonthYoy",
            "�й�:���ڽ��:��װ�����Ÿ���:����ֵ": "China_ExportValue_ClothingAndAccessories_CurrentMonthValue",
            # "�й�:���ڽ��:��֯ɴ��֯�Ｐ����Ʒ:����ͬ��": "China_ExportValue_TextileYarnFabrics_CurrentMonthYoy",
            "�й�:���ڽ��:��֯ɴ�ߡ�֯�Ｐ��Ʒ:����ֵ": "China_ExportValue_TextileYarnFabrics_CurrentMonthValue",
            # "�й�:���ڽ��:������Ʒ:����ͬ��": "China_ExportValue_PlasticProducts_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ExportValue_Ic_CurrentMonthValue",
            # "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ExportValue_Ic_CurrentMonthYoy",
            "�й�:��������:���ɵ�·:����ֵ": "China_ExportQuantity_Ic_CurrentMonthValue",
            "�й�:���ڽ��:������Ʒ:����ֵ": "China_ExportValue_PlasticProducts_CurrentMonthValue",
            "�й�:���ڽ��:ҽ����������е:����ֵ": "China_ExportValue_MedicalInstruments_CurrentMonthValue",
            # "�й�:���ڽ��:ҽ����������е:����ͬ��": "China_ExportValue_MedicalInstruments_CurrentMonthYoy",
            "�й�:��������:������������:����ֵ": "China_ExportQuantity_AutomobilesInclChassis_CurrentMonthValue",
            # "�й�:��������:������������:����ͬ��": "China_ExportQuantity_AutomobilesInclChassis_CurrentMonthYoy",
            "�й�:���ڽ��:������������:����ֵ": "China_ExportValue_AutomobileInclChassis_CurrentMonthValue",
            # "�й�:���ڽ��:������������:����ͬ��": "China_ExportValue_AutomobileInclChassis_CurrentMonthYoy",
            "�й�:���ڽ��:���������:����ֵ": "China_ExportValue_AutoParts_CurrentMonthValue",
            # "�й�:���ڽ��:���������:����ͬ��": "China_ExportValue_AutoParts_CurrentMonthYoy",
            "�й�:���ڽ��:ũ��Ʒ:����ֵ": "China_ImportValue_AgriculturalProducts_CurrentMonthValue",
            # "�й�:���ڽ��:ũ��Ʒ:����ͬ��": "China_ImportValue_AgriculturalProducts_CurrentMonthYoy",
            "�й�:��������:��:����ֵ": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            # "�й�:��������:��:����ͬ��": "China_ImportedQuantity_Soybean_CurrentMonthYoy",
            "�й�:���ڽ��:����ɰ���侫��:����ֵ": "China_ImportValue_IronOreAndConcentrate_CurrentMonthValue",
            # "�й�:���ڽ��:����ɰ���侫��:����ͬ��": "China_ImportValue_IronOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ͭ��ɰ���侫��:����ֵ": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthValue",
            # "�й�:���ڽ��:ͭ��ɰ���侫��:����ͬ��": "China_ImportValue_CopperOreAndItsConcentrate_CurrentMonthYoy",
            "�й�:���ڽ��:ԭ��:����ֵ": "China_ImportValue_CrudeOil_CurrentMonthValue",
            # "�й�:���ڽ��:ԭ��:����ͬ��": "China_ImportValue_CrudeOil_CurrentMonthYoy",
            "�й�:���ڽ��:ú����ú:����ֵ": "China_ImportValue_CoalAndLignite_CurrentMonthValue",
            # "�й�:���ڽ��:ú����ú:����ͬ��": "China_ImportValue_CoalAndLignite_CurrentMonthYoy",
            "�й�:���ڽ��:��Ȼ��:����ֵ": "China_ImportValue_NaturalGas_CurrentMonthValue",
            # "�й�:���ڽ��:��Ȼ��:����ͬ��": "China_ImportValue_NaturalGas_CurrentMonthYoy",
            "�й�:���ڽ��:�����Ʒ:����ֵ": "China_ImportValue_MechanicalAndElectrical_CurrentMonthValue",
            # "�й�:���ڽ��:�����Ʒ:����ͬ��": "China_ImportValue_MechanicalAndElectrical_CurrentMonthYoy",
            "�й�:���ڽ��:���ɵ�·:����ֵ": "China_ImportValue_Ic_CurrentMonthValue",
            # "�й�:���ڽ��:���ɵ�·:����ͬ��": "China_ImportValue_Ic_CurrentMonthYoy",
            "�й�:���ڽ��:���¼�����Ʒ:����ֵ": "China_ImportValue_High_techProducts_CurrentMonthValue",
            # "�й�:���ڽ��:���¼�����Ʒ:����ͬ��": "China_ImportValue_High_techProducts_CurrentMonthYoy",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ExportValue_ASEAN_CurrentMonthValue",
            "�й�:���ڽ��:�����ǹ�������:����ֵ": "China_ImportValue_ASEAN_CurrentMonthValue",
            # "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ExportValue_ASEAN_CurrentMonthYoy",
            # "�й�:���ڽ��:�����ǹ�������:����ͬ��": "China_ImportValue_ASEAN_CurrentMonthYoy",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ExportValue_Eu_CurrentMonthValue",
            "�й�:���ڽ��:ŷ��:����ֵ": "China_ImportValue_Eu_CurrentMonthValue",
            # "�й�:���ڽ��:ŷ��:����ͬ��": "China_ExportValue_Eu_CurrentMonthYoy",
            # "�й�:���ڽ��:ŷ��:����ͬ��": "China_ImportValue_Eu_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_Us_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_Us_CurrentMonthValue",
            # "�й�:���ڽ��:����:����ͬ��": "China_ExportValue_Us_CurrentMonthYoy",
            # "�й�:���ڽ��:����:����ͬ��": "China_ImportValue_Us_CurrentMonthYoy",
            "�й�:���ڽ��:�й����:����ֵ": "China_ExportValue_Hongkong_CurrentMonthValue",
            "�й�:���ڽ��:�й����:����ֵ": "China_ImportValue_Hongkong_CurrentMonthValue",
            # "�й�:���ڽ��:�й����:����ͬ��": "China_ExportValue_Hongkong_CurrentMonthYoy",
            # "�й�:���ڽ��:�й����:����ͬ��": "China_ImportValue_Hongkong_CurrentMonthYoy",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ExportValue_Japan_CurrentMonthValue",
            "�й�:���ڽ��:�ձ�:����ֵ": "China_ImportValue_Japan_CurrentMonthValue",
            # "�й�:���ڽ��:�ձ�:����ͬ��": "China_ExportValue_Japan_CurrentMonthYoy",
            # "�й�:���ڽ��:�ձ�:����ͬ��": "China_ImportValue_Japan_CurrentMonthYoy",
            "�й�:���ڽ��:����:����ֵ": "China_ExportValue_SouthKorea_CurrentMonthValue",
            "�й�:���ڽ��:����:����ֵ": "China_ImportValue_SouthKorea_CurrentMonthValue",
            # "�й�:���ڽ��:����:����ͬ��": "China_ExportValue_SouthKorea_CurrentMonthYoy",
            # "�й�:���ڽ��:����:����ͬ��": "China_ImportValue_SouthKorea_CurrentMonthYoy",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ExportValue_Taiwan_CurrentMonthValue",
            "�й�:���ڽ��:�й�̨��:����ֵ": "China_ImportValue_Taiwan_CurrentMonthValue",
            # "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ExportValue_Taiwan_CurrentMonthYoy",
            # "�й�:���ڽ��:�й�̨��:����ͬ��": "China_ImportValue_Taiwan_CurrentMonthYoy"
        }

        # ���¿�����view������չʾ������
        self.export_chinese_names_to_display = ['�й�:���ڽ��:����ͬ��', '�й�:���ڽ��:����ͬ��',
                                                '�й�:ó�ײ��:����ͬ��',
                                                '�й�:���ڽ��:�����Ʒ:����ͬ��',
                                                '�й�:���ڽ��:���¼�����Ʒ:����ͬ��',
                                                '�й�:���ڽ��:��װ�����Ÿ���:����ͬ��',
                                                '�й�:���ڽ��:��֯ɴ��֯�Ｐ����Ʒ:����ͬ��',
                                                '�й�:���ڽ��:���ɵ�·:����ͬ��', '�й�:���ڽ��:������Ʒ:����ͬ��',
                                                '�й�:���ڽ��:ҽ����������е:����ͬ��',
                                                '�й�:���ڽ��:������������:����ͬ��',
                                                '�й�:���ڽ��:���������:����ͬ��', '�й�:���ڽ��:�����Ʒ:����ֵ',
                                                '�й�:���ڽ��:���¼�����Ʒ:����ֵ',
                                                '�й�:���ڽ��:��װ�����Ÿ���:����ֵ',
                                                '�й�:���ڽ��:��֯ɴ�ߡ�֯�Ｐ��Ʒ:����ֵ',
                                                '�й�:���ڽ��:���ɵ�·:����ֵ',
                                                '�й�:���ڽ��:������Ʒ:����ֵ', '�й�:���ڽ��:ҽ����������е:����ֵ',
                                                '�й�:���ڽ��:������������:����ֵ', '�й�:���ڽ��:���������:����ֵ']

    # def calculate_yet_updated(self):
    #     # Step 1: Find the columns with null values in the latest row of export_wide
    #     query = "SELECT * FROM export_wide ORDER BY date DESC LIMIT 1"
    #     df = pd.read_sql_query(text(query), self.alch_conn)
    #     null_columns = df.columns[df.isnull().any()].tolist()
    #
    #     for col in null_columns:
    #         # Step 2: Go to low_freq_long and find the corresponding column (replace "Yoy" with "Value")
    #         metric_name_value = col.replace("Yoy", "Value")
    #
    #         # Step 3: Find the latest value for this metric_name_value and the value from the same period last year
    #         query = f"""
    #         SELECT l1.value AS current_value, l2.value AS last_year_value, l1.date AS current_date
    #         FROM low_freq_long l1
    #         LEFT JOIN low_freq_long l2
    #         ON EXTRACT(YEAR FROM l1.date) = EXTRACT(YEAR FROM l2.date) + 1
    #         AND EXTRACT(MONTH FROM l1.date) = EXTRACT(MONTH FROM l2.date)
    #         WHERE l1.metric_name = '{metric_name_value}'
    #         ORDER BY l1.date DESC
    #         LIMIT 1
    #         """
    #         df = pd.read_sql_query(text(query), self.alch_conn)
    #
    #         if df.empty or pd.isnull(df.loc[0, 'current_value']):
    #             missing_date = df.loc[0, 'current_date'] if not df.empty else "unknown"
    #             print(f"The latest value for {metric_name_value} is missing for date: {missing_date}.")
    #             continue
    #
    #         # Calculate YoY change
    #         current_value = df.loc[0, 'current_value']
    #         last_year_value = df.loc[0, 'last_year_value']
    #         yoy_change = (current_value - last_year_value) / last_year_value * 100
    #
    #         # Step 4: Write the calculated value into low_freq_long
    #         query = f"""
    #         INSERT INTO low_freq_long (date, metric_name, value, data_version)
    #         VALUES ('{df.loc[0, 'date']}', '{col}', {yoy_change}, 'calculated from CurrentMonthValue')
    #         """
    #         self.alch_conn.execute(text(query))
    #         self.alch_conn.commit()
