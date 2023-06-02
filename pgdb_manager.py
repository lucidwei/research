# coding=gbk
# Time Created: 2023/4/7 16:12
# Author  : Lucid
# FileName: pg_database.py
# Software: PyCharm
from base_config import BaseConfig
import psycopg2, sqlalchemy, subprocess
from sqlalchemy import create_engine, text, select, case, String, cast, and_, column
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.orm import Session
from utils import timeit
import pandas as pd

class PgDbManager:
    """
    �������ݿ�����ķ���
    connect, disconnect,
    insert, insert_many,
    """

    def __init__(self, base_config: BaseConfig, engine='sqlalchemy'):
        self.base_config = base_config
        self.engine = engine
        self.db_config = self.base_config.db_config
        self.check_server_running()
        self.connect()

    def connect(self):
        if not self.db_running:
            self.start_sql_server()

        if self.engine == 'sqlalchemy':
            connection_string = f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.alch_engine = create_engine(connection_string)
            self.alch_conn = self.alch_engine.connect()
            print("sqlalchemy engine successfully connected to the database.")

        elif self.engine == 'psycopg2':
            try:
                # ����PostgreSQL������
                self.conn = psycopg2.connect(**self.base_config.db_config)
                # ������ݿ�������Ƿ�������
                self.cur = self.conn.cursor()
                self.cur.execute("SELECT 1")
                print("PostgreSQL database server is running and connected.")
            except psycopg2.Error as e:
                print(f"Error: {e}")
                raise Exception("Unable to connect to the PostgreSQL database server.")

    # def execute_query(self, query):
    #     self.alch_conn.execute(query)

    def write_data(self, sql, data=None):
        if self.engine == 'sqlalchemy':
            self.alch_conn.execute(text(sql))
            self.alch_conn.commit()
        elif self.engine == 'psycopg2':
            if not self.conn:
                print("Error: No connection to the database. Call the 'connect()' method first.")
                return
            try:
                self.cur.execute(sql, data)
                self.conn.commit()
            except psycopg2.Error as e:
                print(f"Error: {e}")
                self.conn.rollback()

    def close(self):
        if self.engine == 'sqlalchemy':
            self.alch_engine.dispose()
            print("sqlalchemy engine is successfully disposed.")
        elif self.engine == 'psycopg2':
            if self.conn:
                self.cur.close()
                self.conn.close()
                self.conn = None
                self.cur = None
                print("Connection and cursor successfully closed.")
            else:
                print("Connection is not open so close activity is void.")
    def check_server_running(self):
        """
        ��� PostgreSQL server �Ƿ�������
        """
        try:
            # �������ӵ� PostgreSQL server
            subprocess.check_call(['pg_isready', '-h', 'localhost', '-U', 'postgres', '-p', '5432'])
            print("PostgreSQL database server is running.")
            self.db_running = True
            return True
        except subprocess.CalledProcessError:
            # �������ʧ�ܣ�˵�� PostgreSQL server δ����
            print("PostgreSQL database server is NOT running.")
            self.db_running = False
            return False

    def start_sql_server(self):
        """
        ���� PostgreSQL server
        """
        if not self.db_running:
            # ��������
            cmd = [self.base_config.pg_ctl, "-D", self.base_config.data_dir, "start"]

            # �����ӽ���
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            # ��ȡ�����Ϣ
            out, err = p.communicate()
            print(out.decode('utf-8'))
            print(err.decode('utf-8'))



    @timeit
    def set_all_nan_to_null(self):
        '''
        metabase�o������nan�����M���L�D��������nan��null
        '''
        if self.engine != 'sqlalchemy':
            raise Exception('��sqlalchemy�ѽ�����֧�֣���׃��sql engine')

        # ��ȡ���Ԫ����
        metadata = MetaData()

        # �ų��ı����
        excluded = {}
        # ʾ����    {
        #     "excluded_table1": ["excluded_column1", "excluded_column2"],
        #     "excluded_table2": ["excluded_column3", "excluded_column4"],
        # }

        # �������б�
        metadata.reflect(self.alch_engine)

        # �������б�
        for table_name, table in metadata.tables.items():
            if table_name not in excluded:
                update_stmt = table.update()
                for column in table.columns:
                    update_stmt = update_stmt.values({column: case((cast(column, String) == "NaN", None), else_=column)})
                # �����ݿ���ִ�и������
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)
            else:
                excluded_columns = excluded[table_name]
                update_stmt = table.update()
                for column in table.columns:
                    if column.name not in excluded_columns:
                        update_stmt = update_stmt.values({column: case((cast(column, String) == "NaN", None), else_=column)})
                # �����ݿ���ִ�и������
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)

    def select_existing_values_in_target_column(self, target_table: str, target_column: str, *where_conditions):
        """
        ��ȡĿ�����ָ���е�����ֵ���ϡ�

        Parameters:
            - target_table (str): Ŀ�������ơ�
            - target_column (str): Ŀ���е����ơ�
            - *where_conditions (tuple or str): �ɱ䳤�Ȳ�����ÿ��������һ��Ԫ�� (where_column, where_value) ��ʾɸѡ������

        Returns:
            set: Ŀ���е�����ֵ���ϡ�
        """
        # Create session
        session = Session(self.alch_engine)

        try:
            # Get metadata of the target table
            metadata = MetaData()
            target_table = Table(target_table, metadata, autoload_with=self.alch_engine)

            # Build the select statement
            where_clauses = []
            for where_condition in where_conditions:
                if isinstance(where_condition, tuple):
                    where_column, where_value = where_condition
                    if where_value is not None:
                        where_clauses.append(target_table.c[where_column] == where_value)
                    else:
                        where_clauses.append(target_table.c[where_column].is_(None))
                else:
                    where_clauses.append(text(where_condition))

            if where_clauses:
                stmt = select(target_table.c[target_column]).where(and_(*where_clauses))
            else:
                stmt = select(target_table.c[target_column])

            # Execute the select statement
            result = session.execute(stmt)

            existing_values = {row[0] for row in result}

            return existing_values
        finally:
            session.close()

    def select_existing_dates_from_long_table(self, table_name, metric_name=None, product_name=None, field=None, return_df=False):
        """
        �����ݿ��л�ȡ���е������б�

        ������
        - table_name��������Ҫ���л�ȡ�����б�ı�����ơ�
        - metric_name���������ƣ�Ҫɸѡ�Ķ������ƣ���ѡ�������ڼ�¼���ڲ�Ʒ��table�����Զ�ת��Ϊ����product_name
        - field���ֶ����ƣ�Ҫɸѡ���ֶ����ƣ���ѡ����

        ���أ�
        - existing_dates�������б������ݿ��м��������������ڡ�

        �쳣��
        - ValueError������ڱ���м�û��'metric_name'��Ҳû��'product_name'�С�

        """
        columns = self.alch_conn.execute(text(f"SELECT * FROM {table_name} LIMIT 0")).keys()

        conditions = []

        if metric_name:
            conditions.append(f"metric_name = '{metric_name}'")
        if product_name:
            conditions.append(f"product_name = '{product_name}'")
        if field and 'field' in columns:
            conditions.append(f"field = '{field}'")

        condition = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        existing_dates_df = self.alch_conn.execute(text(f"SELECT date FROM {table_name} {condition}")).fetchall()
        existing_dates = [row[0] for row in existing_dates_df]

        if return_df:
            return existing_dates_df
        else:
            return existing_dates

    def select_column_from_joined_table(self, target_table_name: str, target_join_column: str, join_table_name: str,
                                        join_column: str, selected_column: str, filter_condition: str = ""):
        """
        ��ȡ���ӱ��е������б�

        Args:
            target_table_name (str): Ŀ�������
            target_join_column (str): Ŀ������������
            join_table_name (str): ���ӱ�����
            join_column (str): ���ӵ�����
            filter_condition (str, optional): ��������. Defaults to "". e.g."product_static_info.type_identifier = 'fund'"

        Returns:
            list: ���ӱ��е������б�
        """
        # Get the joined table as a DataFrame
        df = self.get_joined_table_as_dataframe(target_table_name, target_join_column, join_table_name,
                                                join_column, filter_condition)
        if selected_column not in df.columns:
            raise ValueError("The selected column does not exist in either join_table or target_table.")

        # Return the selected column values
        return sorted(df[selected_column].drop_duplicates().tolist())

    def get_joined_table_as_dataframe(self, target_table_name: str, target_join_column: str, join_table_name: str,
                                      join_column: str, filter_condition: str = ""):
        """
        Retrieves the joined table data as a DataFrame.

        Parameters:
            target_table_name (str): Name of the target table.
            target_join_column (str): Column in the target table for the join condition.
            join_table_name (str): Name of the join table.
            join_column (str): Column in the join table for the join condition.
            filter_condition (str, optional): Additional filtering condition for the query. Defaults to "".

        Returns:
            pandas.DataFrame: Joined table data.
        """
        # Build query
        query = f"SELECT * FROM {target_table_name} INNER JOIN {join_table_name} ON {target_table_name}.{target_join_column} = {join_table_name}.{join_column}"

        # Add filter condition
        if filter_condition:
            query = f"{query} WHERE {filter_condition}"

        # Execute query and fetch result as pandas DataFrame
        result_df = pd.read_sql_query(text(query), self.alch_conn)

        return result_df

    def get_missing_dates(self, all_dates, existing_dates):
        """
        Get the missing dates between before and after existing_dates.

        Args:
            all_dates (list): List of all dates to consider.
            existing_dates (list): List of existing dates in the database.

        Returns:
            list: Sorted list of missing dates.
        """
        # ������ǿյģ�û�����ݣ���ȫ��missing
        if not existing_dates:
            return all_dates

        min_date = min(existing_dates)
        max_date = max(existing_dates)

        missing_dates = []
        # �� all_dates ���������ڵ����ݿ��е���������
        missing_dates.extend([d for d in all_dates if d < min_date])

        # �����ݿ��е��������ڵ� all_dates ����������
        missing_dates.extend([d for d in all_dates if d > max_date])

        return sorted(missing_dates)

    def get_missing_months_ends(self, all_month_ends, earliest_available, table_name, column_name):
        existing_dates = self.select_existing_dates_from_long_table(table_name, metric_name=column_name)
        missing_dates = self.get_missing_dates(all_month_ends, existing_dates=existing_dates)
        # �� missing_dates ת��Ϊ pandas �� DatetimeIndex
        missing_dates = pd.DatetimeIndex(missing_dates)
        # ���˳� earliest_available ֮�������
        missing_dates = missing_dates[missing_dates >= earliest_available]
        return sorted(missing_dates)

    def adjust_seq_val(self, seq_name='metric_static_info_internal_id_seq'):
        """
        ����metric_static_info��internal_id����ִ��INSERT ... ON CONFLICTʱ���µĶ���
        :param seq_name:����һ���������������ı�ʱ��PostgreSQL ͨ�����Զ�Ϊ�ñ���һ����Ϊ '����_����_seq' �����С�
        ����ʹ������Ĳ�ѯ���Զ���ȡ 'metric_static_info' ��� 'internal_id' �е���������
        SELECT pg_get_serial_sequence('metric_static_info', 'internal_id');
        """
        # ʹ������ internal_id ��Ϊ�µ�����ֵ
        query_max = text("SELECT MAX(internal_id) FROM metric_static_info")
        result = self.alch_conn.execute(query_max)
        max_id = result.scalar()+1 or 1  # �����Ϊ�գ���ʹ��1��ΪĬ��ֵ

        # ��������ֵ
        query_setval = text(f"SELECT setval('{seq_name}', {max_id}, false)")
        self.alch_conn.execute(query_setval)
