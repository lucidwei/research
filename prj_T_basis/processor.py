# from prj_T_basis.db_updater import DatabaseUpdater
import numpy as np

from prj_T_basis.db_reader import DatabaseReader
from utils import timeit
import pandas as pd
import datetime, pickle, traceback
import statsmodels.formula.api as smf
from sqlalchemy import text
from sklearn.linear_model import LinearRegression


class Processor:
    """
    calculate代表计算中间变量
    get代表计算暴露给下一流程的接口数据
    """
    def __init__(self, db_reader: DatabaseReader):
        self.loaded_data = db_reader
        self.date_index_str = self.loaded_data.base_config.tradedays_str
        self.date_index = self.loaded_data.base_config.tradedays
        # self.calc_contract_dict_bnoc_ts()
        self.get_contract_dict_bonds_ts()
        self.get_contract_dict_bonds_stats()

    # @property
    # def contracts_dict_bnoc(self):
    #     return self._contracts_dict_bnoc
    #
    # @property
    # def contract_dict_bonds_ts(self):
    #     return self._contract_dict_bonds_ts

    @property
    def measure_dict_visual_data(self):
        return self._measure_dict_visual_data

    # @property
    # def max_bond_num_in_a_contract(self):
    #     return self._max_bond_num_in_a_contract

    @property
    def contract_dict_bonds_stats(self):
        return self._contract_dict_bonds_stats

    @property
    def rates_ts(self):
        return self.loaded_data.rates_timeseries

    @timeit
    def get_contract_dict_bonds_ts(self):
        """
        某一天的净基差走廊应该至多有两种颜色，一组是活跃合约对应的现券bnoc，一组是次活跃。
        每一条线应该代表一个现券
        按合约循环时，是否会出现某一组合约其实已经不活跃了，但是仍然画了出来？
        我觉得是可能的。因此需要在用来画图的dataframe中增加一列"if_active12"在画图时加一个mask。
        index为日期，
        column包括['if_active12', 'ctd_bnoc', 'dtd_bnoc',
            'bond1_code', 'bond1_basis', 'bond1_irr', 'bond1_ytm', 'bond1_bnoc',...,'bond{n}']
        """
        sql = f'''
                SELECT * FROM bond_info_ts
                ORDER BY date DESC, transaction_amount DESC;
                '''
        bond_info_ts = pd.read_sql_query(text(sql), con=self.loaded_data.alch_conn)
        bond_info_ts = bond_info_ts[bond_info_ts['date'].isin(self.date_index)]

        # 首先，找到每个合约中出现过的所有 bond_code
        all_bond_codes = bond_info_ts.groupby('contract_code')['bond_code'].unique()

        # 然后，对每个合约创建一个固定的 bond_code 列表
        fixed_contract_code_dfs = []
        for contract_code, bond_codes in all_bond_codes.items():
            df = bond_info_ts[bond_info_ts['contract_code'] == contract_code]

            # 使用 reindex 将数据扩展到所有 bond_code
            fixed_df = df.set_index(['date', 'bond_code']).reindex(
                pd.MultiIndex.from_product(
                    [df['date'].unique(), bond_codes],
                    names=['date', 'bond_code']
                )
            ).reset_index()

            # 将缺失值设为 NaN
            fixed_df.fillna(value=np.nan, inplace=True)

            # 保留合约代码
            fixed_df['contract_code'] = contract_code

            fixed_contract_code_dfs.append(fixed_df)

        # 最后，将所有合约的数据合并到一个 DataFrame 中
        all_fixed_data = pd.concat(fixed_contract_code_dfs)

        # 为了方便可视化，我们可以使用 pivot_table 函数将数据转换为宽格式
        visual_data_basis = all_fixed_data.pivot_table(index='date', columns=['contract_code', 'bond_code'],
                                                       values='basis')
        visual_data_bnoc = all_fixed_data.pivot_table(index='date', columns=['contract_code', 'bond_code'],
                                                      values='bnoc')
        visual_data_duration = all_fixed_data.pivot_table(index='date', columns=['contract_code', 'bond_code'],
                                                          values='duration')
        visual_data_ytm = all_fixed_data.pivot_table(index='date', columns=['contract_code', 'bond_code'],
                                                     values='ytm')

        # 重命名列
        visual_data_basis.columns = [f'{col[0]}_basis_{col[1]}' for col in visual_data_basis.columns]
        visual_data_bnoc.columns = [f'{col[0]}_bnoc_{col[1]}' for col in visual_data_bnoc.columns]
        visual_data_duration.columns = [f'{col[0]}_duration_{col[1]}' for col in visual_data_duration.columns]
        visual_data_ytm.columns = [f'{col[0]}_ytm_{col[1]}' for col in visual_data_ytm.columns]

        # 获取所有的 contract_code
        contract_codes = all_fixed_data['contract_code'].unique()

        # 初始化一个字典来保存每个 contract_code 的宽格式 DataFrame
        visual_data_basis_by_contract = {}
        visual_data_bnoc_by_contract = {}
        visual_data_duration_by_contract = {}
        visual_data_ytm_by_contract = {}

        # 遍历每个 contract_code 并将对应的列分配给单独的 DataFrame
        for contract_code in contract_codes:
            contract_columns_basis = [col for col in visual_data_basis.columns if
                                      col.startswith(contract_code + '_basis')]
            contract_columns_bnoc = [col for col in visual_data_bnoc.columns if col.startswith(contract_code + '_bnoc')]
            contract_columns_duration = [col for col in visual_data_duration.columns if
                                         col.startswith(contract_code + '_duration')]
            contract_columns_ytm = [col for col in visual_data_ytm.columns if col.startswith(contract_code + '_ytm')]

            visual_data_basis_by_contract[contract_code] = visual_data_basis[contract_columns_basis]
            visual_data_bnoc_by_contract[contract_code] = visual_data_bnoc[contract_columns_bnoc]
            visual_data_duration_by_contract[contract_code] = visual_data_duration[contract_columns_duration]
            visual_data_ytm_by_contract[contract_code] = visual_data_ytm[contract_columns_ytm]
            # 遍历列表中的每一个 DataFrame，并重命名列名只保留bond_code
            dataframes = [visual_data_basis_by_contract[contract_code], visual_data_bnoc_by_contract[contract_code],
                          visual_data_duration_by_contract[contract_code], visual_data_ytm_by_contract[contract_code]]
            for df in dataframes:
                new_columns = {col: col.split('_')[-1] for col in df.columns}
                with pd.option_context('mode.chained_assignment', None):
                    df.rename(columns=new_columns, inplace=True)

        self._measure_dict_visual_data = {
            'basis': visual_data_basis_by_contract,
            'bnoc': visual_data_bnoc_by_contract,
            'duration': visual_data_duration_by_contract,
            'ytm': visual_data_ytm_by_contract
        }

        for metric in ['bnoc', 'basis']:
            for contract, data in self._measure_dict_visual_data[metric].items():
                with pd.option_context('mode.chained_assignment', None):
                    data['ctd'] = data.min(axis=1)
                    data['dtd'] = data.max(axis=1)

    @timeit
    def get_contract_dict_bonds_stats(self):
        contract_dict_bonds_stats = {}
        for contract, data in self.measure_dict_visual_data['duration'].items():
            bond_list = data.columns
            bond_stats_df = pd.DataFrame(index=bond_list, columns=['avg_dura', 'coef', 'n_obs', 'n_norm'])
            duration_df = self.measure_dict_visual_data['duration'][contract]
            bnoc_df = self.measure_dict_visual_data['bnoc'][contract]
            ytm_df = self.measure_dict_visual_data['ytm'][contract]

            max_n_obs = 0
            for bond_code in bond_list:
                if duration_df.empty or ytm_df.empty or bnoc_df.empty:
                    continue

                avg_dura = duration_df[bond_code].mean()
                # 计算回归系数
                x = ytm_df[bond_code].dropna()
                try:
                    y = bnoc_df[bond_code].dropna()
                except:
                    a=1

                # 删除具有空值的数据点
                try:
                    non_null_indices = x.index.intersection(y.index)
                except:
                    a=1
                x = x.loc[non_null_indices]
                y = y.loc[non_null_indices]

                # 设置阈值
                threshold = 10  # 您可以根据需要修改此值

                if len(x) > threshold:
                    x = x.values.reshape(-1, 1)
                    y = y.values.reshape(-1, 1)
                    reg = LinearRegression().fit(x, y)
                    coef = reg.coef_[0][0]
                else:
                    coef = None

                # 计算 n_obs
                n_obs = duration_df[bond_code].count()
                # 更新 max_n_obs
                max_n_obs = max(max_n_obs, n_obs)

                bond_stats_df.loc[bond_code, 'avg_dura'] = avg_dura
                bond_stats_df.loc[bond_code, 'coef'] = coef
                bond_stats_df.loc[bond_code, 'n_obs'] = n_obs
            # 计算 n_norm, 如果 bond_stats_df 全部为 NaN，则将其设置为空 DataFrame
            if bond_stats_df.isna().all().all():
                bond_stats_df = pd.DataFrame()
            else:
                bond_stats_df['n_norm'] = bond_stats_df['n_obs'] / max_n_obs

            contract_dict_bonds_stats[contract] = bond_stats_df
        self._contract_dict_bonds_stats = contract_dict_bonds_stats


