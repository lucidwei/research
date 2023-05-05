# coding=gbk
# Time Created: 2023/3/24 20:25
# Author  : Lucid
# FileName: db_updater.py
# Software: PyCharm
import datetime, re
from WindPy import w
import pandas as pd
from utils import timeit, get_nearest_dates_from_contract, check_wind
from base_config import BaseConfig
from pgdb_manager import PgDbManager
from sqlalchemy import text


class DatabaseUpdater(PgDbManager):
    def __init__(self, base_config: BaseConfig):
        super().__init__(base_config)
        check_wind()
        self.set_tradedays()
        self.update_contract_stats_ts()
        self.update_bond_info_ts()
        self.update_rates_ts()
        self.set_all_nan_to_null()
        self.close()

    def set_tradedays(self):
        self.tradedays = self.base_config.tradedays
        self.tradedays_str = self.base_config.tradedays_str

    @timeit
    def update_contract_stats_ts(self):
        """
        �Ȼ�ȡȱʧ�������б�
        wind fetchȱʧ���ڵĻ�Ծ�ʹλ�Ծ��Լ���룬д��DB��
        �����У�contract_prefix, active_num,
        ��������deliver_date��
        ��ȡȱʧ��deliver_date��wind fetch
        """
        def write_in(downloaded_df: pd.DataFrame, act_num, date):
            regex = r'^(?:T|TF|TS)\d+(?!.*����)'  # ����������ʽ,����'T', 'TF','TS'��������Щ��ĸ����������֡����Ҳ�����"����"����
            try:
                mask = downloaded_df['sec_name'].str.contains(regex)  # Ӧ��������ʽ
                result = downloaded_df[mask]  # ɸѡ��������������
                for contract in result['sec_name'].values.tolist():
                    contract_prefix = re.findall('[a-zA-Z]+', contract)[0]
                    query = f"INSERT INTO contract_stats_ts (date, contract_code, contract_prefix, active_num) VALUES ('{date}', '{contract}', '{contract_prefix}', {act_num});"
                    self.write_data(query)
            except Exception as e:
                print(e)
                if downloaded_df[0][0].__contains__('quota exceeded'):
                    print('WindPy quota����')
            return

        # ����date, contract_code, contract_prefix, active_num
        sql = f"SELECT DISTINCT date FROM contract_stats_ts;"
        df = pd.read_sql_query(text(sql), con=self.alch_conn)
        dates_from_db = df['date'].tolist()
        dates_missing = sorted(list(set(self.tradedays) - set(dates_from_db)), reverse=True)
        if len(dates_missing) != 0:
            for date in dates_missing:
                print(f'Wind downloading �����ڻ���Ծ�º�Լfor {date}')
                # �����ڻ���Ծ�º�Լ
                active1_contracts = \
                    w.wset("sectorconstituent", f"date={date};sectorid=1000015510000000;field=sec_name",
                           usedf=True)[1]
                # �����ڻ��λ�Ծ�º�Լ
                active2_contracts = \
                    w.wset("sectorconstituent", f"date={date};sectorid=1000039191000000;field=sec_name",
                           usedf=True)[1]
                write_in(active1_contracts, '1', date)
                write_in(active2_contracts, '2', date)

        # �����֪�Ŀɽ�������
        sql = '''
            UPDATE contract_stats_ts 
            SET deliver_date = (
                SELECT t2.deliver_date 
                FROM contract_stats_ts AS t2 
                WHERE t2.contract_code = contract_stats_ts.contract_code 
                AND t2.deliver_date IS NOT NULL 
                LIMIT 1
            )
            WHERE deliver_date IS NULL;
        '''
        self.write_data(sql)
        # ���δ֪�ɽ������ڣ���wind�ϲ���
        sql = "SELECT contract_code FROM contract_stats_ts WHERE deliver_date IS NULL;"
        df = pd.read_sql_query(text(sql), con=self.alch_conn)
        if not df.empty:
            for contract in df['contract_code'].values.tolist():
                print(f"Wind downloading �ڻ���Լ�б� for {contract}")
                # �ڻ���Լ�б�
                df = w.wset("futurecc",
                            f"startdate={get_nearest_dates_from_contract(contract)[0]};enddate={get_nearest_dates_from_contract(contract)[1]};wind_code={contract}.CFE;field=code,contract_issue_date,last_trade_date,last_delivery_month",
                            usedf=True)[1]
                deli_date = datetime.date.fromtimestamp(df[df['code'] == contract].loc[:, 'last_delivery_month'][0].timestamp())
                sql = f"UPDATE contract_stats_ts SET deliver_date = '{deli_date}' WHERE contract_code = '{contract}' AND deliver_date IS NULL;"
                self.write_data(sql)

    @timeit
    def update_bond_info_ts(self):
        """
        ����contract_stats_ts���µ�date��contract_code�õ���Ҫ���µ�df
        ͨ��deliverablebondlist��ȡbasis, irr, ytm, amount������.
        ����bond_act_num����ȡ�����contract_act_num, contract_prefix
        # ��amountΪnull��NAN��row
        TODO: ͨ��������������ֵ
        ����bnoc
        """
        # ��ѯ��Ҫ���µ����ݣ���������contract_code��date
        sql = '''
        SELECT DISTINCT contract_stats_ts.date, contract_stats_ts.contract_code
        FROM contract_stats_ts
        LEFT JOIN bond_info_ts ON (bond_info_ts.date = contract_stats_ts.date AND bond_info_ts.contract_code = contract_stats_ts.contract_code)
        WHERE bond_info_ts.date IS NULL
        ORDER BY contract_stats_ts.date DESC;
        '''
        df1 = pd.read_sql_query(text(sql), con=self.alch_conn)
        # Wind�����ڔ��������^�������ȱʧ��transaction_amount���࣬��Ҫ���½���ȱʧ����
        sql = '''
            SELECT
                date,
                contract_code
            FROM
                bond_info_ts
            WHERE
                date >= CURRENT_DATE - INTERVAL '2 month'
                AND (
                    SELECT COUNT(*) FROM bond_info_ts t2
                    WHERE t2.date = bond_info_ts.date
                    AND t2.contract_code = bond_info_ts.contract_code
                    AND t2.transaction_amount IS NULL
                ) / (
                    SELECT COUNT(*) FROM bond_info_ts t3
                    WHERE t3.date = bond_info_ts.date
                    AND t3.contract_code = bond_info_ts.contract_code
                ) >= 0.8
            GROUP BY
                date,
                contract_code
            ORDER BY
                date DESC,
                contract_code
            '''
        df2 = pd.read_sql_query(text(sql), con=self.alch_conn)
        # combine ����df
        df = pd.concat([df1, df2])
        for _, row in df.iterrows():
            date = row['date']
            contract = row['contract_code']
            print(f"Wind downloading deliverables for {contract} on {date}")
            deliverables_info = w.wset("deliverablebondlist",
                                       f"windcode={contract}.CFE;date={date};flag=interbank;pricetype=close;field=code,name,factor,irr,basegap,closedirtyprice,closenetprice,yield,term,duration,convexity,amount,volume,rate",
                                       usedf=True)[1]
            # ����DataFrame�������ͱ��������Ķ�Ӧ��ϵ
            column_map = {
                'code': 'bond_code',
                'name': 'bond_name',
                'factor': 'conversion_factor',
                'basegap': 'basis',
                'amount': 'transaction_amount',
                'rate': 'coupon_rate',
                'yield': 'ytm',
            }
            # �������ݵ����ݿ����
            deliverables_info['date'] = date
            deliverables_info['contract_code'] = contract
            deliverables_info = deliverables_info.rename(columns=column_map)
            # ɾ����ͻ������
            conflict_rows = deliverables_info[['date', 'bond_code', 'contract_code']].apply(tuple, axis=1).tolist()
            conflict_tuples = ', '.join(str((row[0].strftime('%Y-%m-%d'), row[1], row[2])) for row in conflict_rows)

            # ��Ԫ���ʽ���� SQL ��ѯ��
            delete_sql = f"""
                DELETE FROM bond_info_ts
                WHERE (date, bond_code, contract_code) IN
                (SELECT t.date_text::date, t.bond_code, t.contract_code
                 FROM (VALUES {conflict_tuples}) AS t(date_text, bond_code, contract_code)
                 JOIN bond_info_ts b ON (t.date_text::date = b.date AND t.bond_code = b.bond_code AND t.contract_code = b.contract_code));
            """
            with self.alch_engine.connect() as conn:
                conn.execute(text(delete_sql))
                conn.commit()

            # ����������
            deliverables_info.to_sql('bond_info_ts', self.alch_engine, if_exists='append', index=False)

        # ����bond_act_num
        sql = '''
            WITH ranked_bond_info AS (
              SELECT
                *,
                CASE
                  WHEN transaction_amount IS NULL OR transaction_amount::text !~ '^[^-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' THEN NULL
                  ELSE ROW_NUMBER() OVER (
                    PARTITION BY date, contract_code
                    ORDER BY (CASE WHEN transaction_amount IS NOT NULL AND transaction_amount::text ~ '^[^-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' THEN transaction_amount END) DESC NULLS LAST
                  )
                END AS bond_act_num_rank
              FROM bond_info_ts
            )
            UPDATE bond_info_ts
            SET bond_act_num = ranked_bond_info.bond_act_num_rank
            FROM ranked_bond_info
            WHERE
              bond_info_ts.date = ranked_bond_info.date AND
              bond_info_ts.bond_code = ranked_bond_info.bond_code AND
              bond_info_ts.contract_code = ranked_bond_info.contract_code;
        '''
        self.write_data(sql)
        # ��ȡ����ȫcontract_act_num
        sql = '''
            UPDATE bond_info_ts
            SET contract_act_num = contract_stats_ts.active_num
            FROM contract_stats_ts
            WHERE
              bond_info_ts.date = contract_stats_ts.date AND
              bond_info_ts.contract_code = contract_stats_ts.contract_code AND
              bond_info_ts.contract_act_num IS NULL AND
              contract_stats_ts.active_num IS NOT NULL;
        '''
        self.write_data(sql)

        # ��ȡ����ȫcontract_prefix
        sql = '''
            UPDATE bond_info_ts
            SET contract_prefix = SUBSTRING(contract_code FROM '^[A-Za-z]+')
            WHERE contract_prefix IS NULL;
            '''
        self.write_data(sql)

        # ����bnocʱ����R007�Ļ������ַ�����һ���ǰѽ���R007����δ������ģ�һ������ʵδ��R007��������ʵҲ���岻����Ϊ�����ܳ�����������ﰴǰ�ߴ����㷨����
        sql = '''
        WITH rates_info AS (
            SELECT
                r.date,
                r.r007 AS today_r007
            FROM
                rates_ts r
        ),
        deliver_date_info AS (
            SELECT
                c.date,
                c.contract_code,
                c.deliver_date
            FROM
                contract_stats_ts c
        ),
        bond_rate_info AS (
            SELECT
                b.date,
                b.bond_code,
                b.contract_code,
                b.coupon_rate,
                r.today_r007,
                d.deliver_date
            FROM
                bond_info_ts b
            JOIN
                rates_info r ON b.date = r.date
            JOIN
                deliver_date_info d ON b.date = d.date AND b.contract_code = d.contract_code
        ),
        future_carry_info AS (
            SELECT
                b.date,
                b.bond_code,
                b.contract_code,
                (EXTRACT(EPOCH FROM (b.deliver_date::timestamp - b.date::timestamp))/86400) / 360 * (b.coupon_rate - b.today_r007) AS future_carry
            FROM
                bond_rate_info b
        )
        UPDATE
            bond_info_ts b
        SET
            bnoc = fc.future_carry - b.basis
        FROM
            future_carry_info fc
        WHERE
            b.date = fc.date AND b.bond_code = fc.bond_code AND b.contract_code = fc.contract_code;
        '''
        self.write_data(sql)


    @timeit
    def update_rates_ts(self):
        # ɾ��rates_ts���н�ʮ���ڰ�����ֵ����
        ten_days_ago = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
        with self.alch_engine.begin() as conn:
            conn.execute(text(f"DELETE FROM rates_ts WHERE r007 is NULL AND date >= '{ten_days_ago}'"))
            conn.commit()
        # ��������
        sql = '''
        SELECT *
        FROM rates_ts
        '''
        read_df = pd.read_sql_query(text(sql), con=self.alch_conn)
        dates_from_db = read_df['date'].tolist()
        dates_missing = sorted(list(set(self.tradedays) - set(dates_from_db)))
        if len(dates_missing) != 0:
            print('Wind downloading rates_ts')
            downloaded_df = w.wsd("R007.IB,TB1Y.WI,TB3Y.WI,TB5Y.WI,TB7Y.WI,TB10Y.WI", "vwap",
                                  dates_missing[0], dates_missing[-1], "", usedf=True)[1]
            # wind���ص�df������Ϊһ��Ͷ���ĸ�ʽ��һ��
            if dates_missing[0] == dates_missing[-1]:
                downloaded_df = downloaded_df.T
                downloaded_df.index = dates_missing

            # ����DataFrame�������ͱ��������Ķ�Ӧ��ϵ
            column_map = {
                'R007.IB': 'r007',
                'TB1Y.WI': 'tb1y',
                'TB3Y.WI': 'tb3y',
                'TB5Y.WI': 'tb5y',
                'TB7Y.WI': 'tb7y',
                'TB10Y.WI': 'tb10y',
            }

            # �������ݵ����ݿ����
            downloaded_df.reset_index(inplace=True)
            downloaded_df.rename(columns={'index': 'date'}, inplace=True)
            downloaded_df.rename(columns=column_map, inplace=True)
            # ѡ�������������в����ڵ�����
            new_rows = downloaded_df[~downloaded_df['date'].isin(read_df['date'])]

            # �����в������ݿ���
            new_rows.to_sql('rates_ts', self.alch_engine, if_exists='append', index=False)

