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

    def select_existing_values_in_target_column(self, target_table: str, target_column: str, *where_conditions):
        """
        获取目标表中指定列的现有值集合。

        Parameters:
            - target_table (str): 目标表的名称。
            - target_column (str): 目标列的名称。
            - *where_conditions (tuple or str): 可变长度参数，每个参数是一个元组 (where_column, where_value) 表示筛选条件。

        Returns:
            set: 目标列的现有值集合。
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
        从数据库中获取现有的日期列表。

        参数：
        - table_name：表名，要从中获取日期列表的表格名称。
        - metric_name：度量名称，要筛选的度量名称（可选）。对于记录金融产品的table，会自动转换为搜索product_name
        - field：字段名称，要筛选的字段名称（可选）。

        返回：
        - existing_dates：日期列表，从数据库中检索到的现有日期。

        异常：
        - ValueError：如果在表格中既没有'metric_name'列也没有'product_name'列。

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
        获取连接表中的日期列表

        Args:
            target_table_name (str): 目标表名称
            target_join_column (str): 目标表的连接列名
            join_table_name (str): 连接表名称
            join_column (str): 连接的列名
            filter_condition (str, optional): 过滤条件. Defaults to "". e.g."product_static_info.type_identifier = 'fund'"

        Returns:
            list: 连接表中的日期列表
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
        existing_dates = self.select_existing_dates_from_long_table(table_name, metric_name=column_name)
        missing_dates = self.get_missing_dates(all_month_ends, existing_dates=existing_dates)
        # 将 missing_dates 转换为 pandas 的 DatetimeIndex
        missing_dates = pd.DatetimeIndex(missing_dates)
        # 过滤出 earliest_available 之后的日期
        missing_dates = missing_dates[missing_dates >= earliest_available]
        return sorted(missing_dates)

    def adjust_seq_val(self, seq_name='metric_static_info_internal_id_seq'):
        """
        避免metric_static_info中internal_id列在执行INSERT ... ON CONFLICT时导致的断续
        :param seq_name:创建一个带有自增主键的表时，PostgreSQL 通常会自动为该表创建一个名为 '表名_列名_seq' 的序列。
        可以使用下面的查询来自动获取 'metric_static_info' 表的 'internal_id' 列的序列名：
        SELECT pg_get_serial_sequence('metric_static_info', 'internal_id');
        """
        # 使用最大的 internal_id 作为新的序列值
        query_max = text("SELECT MAX(internal_id) FROM metric_static_info")
        result = self.alch_conn.execute(query_max)
        max_id = result.scalar()+1 or 1  # 如果表为空，则使用1作为默认值

        # 调整序列值
        query_setval = text(f"SELECT setval('{seq_name}', {max_id}, false)")
        self.alch_conn.execute(query_setval)
