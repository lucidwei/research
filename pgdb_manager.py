# coding=gbk
# Time Created: 2023/4/7 16:12
# Author  : Lucid
# FileName: pg_database.py
# Software: PyCharm
from typing import Union, List
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
        except:
            # 如果pg_isready检查失败，尝试检查Docker容器
            try:
                result = subprocess.check_output(['docker', 'ps', '--filter', 'publish=5432', '--format', '{{.Names}}'])

                # 如果输出不为空，说明有容器在运行并发布到5432端口
                if result:
                    print("PostgreSQL database container is running.")
                    self.db_running = True
                    return True
                else:
                    print("PostgreSQL database container is NOT running.")
                    self.db_running = False
                    return False

            except subprocess.CalledProcessError:
                print("Error checking Docker container.")
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
                    update_stmt = update_stmt.values(
                        {column: case((cast(column, String) == "NaN", None), else_=column)})
                # 在数据库中执行更新语句
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)
            else:
                excluded_columns = excluded[table_name]
                update_stmt = table.update()
                for column in table.columns:
                    if column.name not in excluded_columns:
                        update_stmt = update_stmt.values(
                            {column: case((cast(column, String) == "NaN", None), else_=column)})
                # 在数据库中执行更新语句
                with self.alch_engine.begin() as connection:
                    connection.execute(update_stmt)

    def select_existing_values_in_target_column(self, target_table: str, target_columns,
                                                *where_conditions) -> Union[list, pd.DataFrame]:
        """
        获取目标表中指定列的现有值集合。

        Parameters:
            - target_table (str): 目标表的名称。
            - target_columns (Union[str, List[str]]): 目标列的名称，可以是单个列名或多个列名组成的列表。
            - *where_conditions (tuple or str): 可变长度参数，每个参数是一个元组 (where_column, where_value) 表示筛选条件，
              或者是一个字符串表示额外的筛选条件。

        Returns:
            list: 目标列的现有值集合（已排序）。
        """
        # Build the SELECT statement
        select_columns = ', '.join(target_columns) if isinstance(target_columns, list) else target_columns
        where_clauses = []
        for where_condition in where_conditions:
            if isinstance(where_condition, tuple):
                where_column, where_value = where_condition
                if where_value is not None:
                    where_clauses.append(f"{where_column} = '{where_value}'")
                else:
                    where_clauses.append(f"{where_column} IS NULL")
            else:
                where_clauses.append(where_condition)

        where_clause = ' AND '.join(where_clauses) if where_clauses else ''
        if where_clause:
            query = f"SELECT {select_columns} FROM {target_table} WHERE {where_clause};"
        else:
            query = f"SELECT {select_columns} FROM {target_table};"

        # Execute the SQL statement
        result = self.alch_conn.execute(text(query))

        if isinstance(target_columns, list):
            # Return a DataFrame with the specified columns
            columns = [col[0] for col in result.cursor.description]
            data = result.fetchall()
            df = pd.DataFrame(data, columns=columns)
            return df
        else:
            # Return a list of column values
            existing_values = sorted(set(row[0] for row in result if row[0] is not None))
            return existing_values

    def select_existing_dates_from_long_table(self, table_name, metric_name=None, product_name=None, field=None,
                                              return_df=False):
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
        df = self.read_joined_table_as_dataframe(target_table_name, target_join_column, join_table_name,
                                                 join_column, filter_condition)
        if selected_column not in df.columns:
            raise ValueError("The selected column does not exist in either join_table or target_table.")

        # Return the selected column values
        return sorted(df[selected_column].dropna().drop_duplicates().tolist())

    def read_from_high_freq_view(self, code_list: List[str]) -> pd.DataFrame:
        """
        从 high_freq_view (wide format) 读取数据，根据 code_list。
        :param code_list: 要查询的代码列表，例如 ['S0059749', 'G0005428']
        :return: 返回一个包含所请求数据的 DataFrame
        """
        # 获取英文列名
        column_names = [self.conversion_dicts['id_to_english'][code] for code in code_list]

        # 构建 SQL 查询
        sql_query = f"SELECT date, {', '.join(column_names)} FROM high_freq_wide"

        # 从数据库中读取数据
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        return result_df.set_index('date').sort_index(ascending=False)

    def read_from_markets_daily_long(self, code: str, fields: str) -> pd.DataFrame:
        """
        从 markets_daily_long (long format) 读取数据，根据 code 和 fields。
        :param code: 要查询的代码，例如 'S0059749'
        :param fields: 要查询的字段，例如 "close,volume,pe_ttm"
        :return: 返回一个包含所请求数据的 DataFrame
        """
        # 将逗号分隔的 fields 转换为列表
        field_list = fields.split(',')

        # 获取 product_name
        product_name = self.conversion_dicts['id_to_english'][code]

        # 构建 SQL 查询
        sql_query = f"""SELECT date, field, value
                       FROM markets_daily_long
                       WHERE product_name = '{product_name}'
                       AND field IN ({', '.join([f"'{field}'" for field in field_list])})"""

        # 从数据库中读取数据
        result_df = pd.read_sql_query(text(sql_query), self.alch_conn)

        # 转换为宽格式
        result_df = result_df.pivot_table(index='date', columns='field', values='value')

        return result_df

    def select_rows_by_column_strvalue(self, table_name: str, column_name: str, search_value: str,
                                       selected_columns: list = None, filter_condition: str = None):
        """
        选取某个表的某个列中包含指定字段的所有行，并根据指定的列名筛选出 DataFrame。

        Parameters:
            - table_name (str): 表的名称。
            - column_name (str): 列的名称。
            - search_value (str): 要搜索的字段。
            - selected_columns (list): 可选参数，要筛选的列名列表，默认为 None，表示选取所有列。
            - filter_condition (str): 可选参数，额外的筛选条件字符串，默认为 None。

        Returns:
            pd.DataFrame: 筛选后的 DataFrame。
        """
        # 创建元数据
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=self.alch_engine)

        # 构建查询语句
        if selected_columns is None:
            stmt = select(table)
        else:
            selected_columns = [table.c[col] for col in selected_columns]
            stmt = select(*selected_columns)

        stmt = stmt.where(table.c[column_name].ilike(f'%{search_value}%'))

        # 添加额外的筛选条件
        if filter_condition is not None:
            stmt = stmt.where(text(filter_condition))

        # 执行查询并返回结果
        with self.alch_engine.connect() as conn:
            result = conn.execute(stmt)
            data = result.fetchall()

        # 将查询结果转为 DataFrame
        if data:
            df = pd.DataFrame(data, columns=result.keys())
        else:
            df = pd.DataFrame(columns=result.keys())

        return df

    def read_joined_table_as_dataframe(self, target_table_name: str, target_join_column: str, join_table_name: str,
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

    def read_table_from_schema(self, schema_name, table_name):
        # 构造SQL查询语句
        query = f"SELECT * FROM {schema_name}.{table_name}"

        # 执行查询并读取数据到DataFrame
        df = pd.read_sql_query(text(query), self.alch_conn)

        # 返回DataFrame
        return df

    # def get_MA_df_upload(self, joined_df, MA_period: int):
    #     df_long = joined_df[["date", 'chinese_name', 'field', "value"]]
    #     unique_field = df_long['field'].unique()
    #     if len(unique_field) > 1:
    #         raise ValueError("列'field'的值不唯一，无法正确进行长转宽")
    #
    #     df_wide = df_long.pivot(index='date', columns='chinese_name', values='value')
    #     # 对每一列求10日移动平均
    #     df_ma = df_wide.rolling(window=MA_period).mean()
    #     df_ma = df_ma.dropna(how='all').reset_index()
    #     df_upload = df_ma.melt(id_vars=['date'], var_name='chinese_name',
    #                            value_name='value').sort_values(by="date", ascending=False)
    #     df_upload['field'] = f'{unique_field[0]}_MA{MA_period}'
    #     return df_upload

    def get_MA_df_long(self, original_long_df, field_col_name, value_col_name, MA_period: int):
        df_long = original_long_df.copy()
        df_long = df_long.groupby(field_col_name).rolling(MA_period, on='date').mean()
        df_long = df_long.reset_index().drop('level_1', axis=1).sort_values('date', ascending=False).dropna()
        df_long = df_long.rename(columns={value_col_name: f'{value_col_name}_MA{MA_period}'})
        # 把original_long_df中的value_col_name列对应地填回到df_long
        df_long = df_long.merge(original_long_df[['date', field_col_name, value_col_name]], on=['date', field_col_name],
                                how='left')
        return df_long

    def get_missing_dates_old(self, all_dates, existing_dates):
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

        missing_dates = sorted(set(all_dates) - set(existing_dates))
        return missing_dates

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
        match seq_name:
            case 'metric_static_info_internal_id_seq':
                query_max = text("SELECT MAX(internal_id) FROM metric_static_info")
            case 'product_static_info_internal_id_seq':
                query_max = text("SELECT MAX(internal_id) FROM product_static_info")
            case _:
                raise Exception(f'seq_name={seq_name} not supported.')
        result = self.alch_conn.execute(query_max)
        max_id = result.scalar() + 1 or 1  # 如果表为空，则使用1作为默认值

        # 调整序列值
        query_setval = text(f"SELECT setval('{seq_name}', {max_id}, false)")
        self.alch_conn.execute(query_setval)
