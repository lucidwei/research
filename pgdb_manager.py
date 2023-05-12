# coding=gbk
# Time Created: 2023/4/7 16:12
# Author  : Lucid
# FileName: pg_database.py
# Software: PyCharm
from base_config import BaseConfig
import psycopg2, sqlalchemy, subprocess
from sqlalchemy import create_engine, text, select, case, String, cast
from sqlalchemy.schema import MetaData, Table
from utils import timeit
import pandas as pd

class PgDbManager:
    """
    定义数据库操作的方法
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
                # 连接PostgreSQL服务器
                self.conn = psycopg2.connect(**self.base_config.db_config)
                # 检查数据库服务器是否在运行
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
        检查 PostgreSQL server 是否在运行
        """
        try:
            # 尝试连接到 PostgreSQL server
            subprocess.check_call(['pg_isready', '-h', 'localhost', '-U', 'postgres', '-p', '5432'])
            print("PostgreSQL database server is running.")
            self.db_running = True
            return True
        except subprocess.CalledProcessError:
            # 如果连接失败，说明 PostgreSQL server 未运行
            print("PostgreSQL database server is NOT running.")
            self.db_running = False
            return False

    def start_sql_server(self):
        """
        启动 PostgreSQL server
        """
        if not self.db_running:
            # 构建命令
            cmd = [self.base_config.pg_ctl, "-D", self.base_config.data_dir, "start"]

            # 启动子进程
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            # 获取输出信息
            out, err = p.communicate()
            print(out.decode('utf-8'))
            print(err.decode('utf-8'))



    @timeit
    def set_all_nan_to_null(self):
        '''
        metabase無法對有nan的列進行繪圖，需重置nan為null
        '''
        if self.engine != 'sqlalchemy':
            raise Exception('非sqlalchemy已經不再支持，需變更sql engine')

        # 获取表的元数据
        metadata = MetaData()

        # 排除的表和列
        excluded = {}
        # 示例：    {
        #     "excluded_table1": ["excluded_column1", "excluded_column2"],
        #     "excluded_table2": ["excluded_column3", "excluded_column4"],
        # }

        # 反射所有表
        metadata.reflect(self.alch_engine)

        # 遍历所有表
        for table_name, table in metadata.tables.items():
            if table_name not in excluded:
                update_stmt = table.update()
                for column in table.columns:
                    update_stmt = update_stmt.values({column: case((cast(column, String) == "NaN", None), else_=column)})
                # 在数据库中执行更新语句
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)
            else:
                excluded_columns = excluded[table_name]
                update_stmt = table.update()
                for column in table.columns:
                    if column.name not in excluded_columns:
                        update_stmt = update_stmt.values({column: case((cast(column, String) == "NaN", None), else_=column)})
                # 在数据库中执行更新语句
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)

    def get_existing_dates_from_db(self, table_name, metric_name=None, field=None):
        columns = self.alch_conn.execute(text(f"SELECT * FROM {table_name} LIMIT 0")).keys()

        conditions = []

        if metric_name:
            if 'metric_name' in columns:
                conditions.append(f"metric_name = '{metric_name}'")
            elif 'product_name' in columns:
                conditions.append(f"product_name = '{metric_name}'")
            else:
                raise ValueError("Neither 'metric_name' nor 'product_name' columns were found in the table.")

        if field and 'field' in columns:
            conditions.append(f"field = '{field}'")

        condition = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        existing_dates = self.alch_conn.execute(text(f"SELECT date FROM {table_name} {condition}")).fetchall()
        existing_dates = [row[0] for row in existing_dates]

        return existing_dates

    def get_missing_dates(self, all_dates, table_name, english_id, field=None):
        existing_dates = self.get_existing_dates_from_db(table_name, english_id, field)

        # 如果表是空的，没有数据，则全部missing
        if not existing_dates:
            return all_dates

        min_date = min(existing_dates)
        max_date = max(existing_dates)

        missing_dates = []
        # 从 all_dates 的最早日期到数据库中的最早日期
        missing_dates.extend([d for d in all_dates if d < min_date])

        # 从数据库中的最晚日期到 all_dates 的最晚日期
        missing_dates.extend([d for d in all_dates if d > max_date])

        return sorted(missing_dates)

    def get_missing_months_ends(self, all_month_ends, earliest_available, table_name, column_name):
        missing_dates = self.get_missing_dates(all_month_ends, table_name, column_name)
        # 将 missing_dates 转换为 pandas 的 DatetimeIndex
        missing_dates = pd.DatetimeIndex(missing_dates)
        # 过滤出 earliest_available 之后的日期
        missing_dates = missing_dates[missing_dates >= earliest_available]
        return sorted(missing_dates)

