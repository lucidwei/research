import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors, re
from pylab import mpl
from prj_T_basis.processor import Processor
from utils import backup_file
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


class Plotter():
    def __init__(self, data: Processor):
        self.base_config = data.loaded_data.base_config
        self.image_folder = self.base_config.image_folder
        self.processed_data = data
        self.plot_bnoc_corridor()
        self.plot_basis_corridor()
        self.plot_season_bnoc_corridors()
        self.plot_season_basis_corridors()
        self.plot_dura_coef_scatter()
        self.plot_yield_basis_line()

    def plot_bnoc_corridor(self):
        self.corridor('bnoc')

    def plot_basis_corridor(self):
        self.corridor('basis', add_r007=True)

    def plot_irr_corridor(self):
        self.corridor('irr', set_ylimit=True)

    def plot_season_bnoc_corridors(self):
        self.season_corridor('bnoc')

    def plot_season_basis_corridors(self):
        self.season_corridor('basis')

    def season_corridor(self, target: str):
        '''
        :param target: basis, bnoc, duration, ytm
        '''
        target_dic = self.processed_data.measure_dict_visual_data[target]
        cmap = plt.cm.get_cmap('rainbow')

        # 创建画布对象for T TF TS
        fig1 = plt.figure(figsize=(10, 6))
        fig2 = plt.figure(figsize=(10, 6))
        fig3 = plt.figure(figsize=(10, 6))

        # 在画布对象上分别绘制图形
        def ax_plot(fig, code):
            for month_num in [0, 1, 2, 3]:
                month = ['03', '06', '09', '12'][month_num]
                ax = fig.add_subplot(2, 2, month_num+1)
                color_num = 0
                # 三种合约
                color_num_max = len(target_dic.keys()) / 12
                x_axis = list(range(140))
                keys = sorted(target_dic.keys())
                for contract in keys:
                    if re.match(rf"^{code}\d", contract) and re.match(rf".*{month}$", contract):
                        for bond_code, data in target_dic[contract].items():
                            arr = data[~np.isnan(data)]
                            ax.plot(x_axis[:len(arr)], arr, color=cmap(color_num / color_num_max), alpha=0.1)
                        ctd = target_dic[contract][f'ctd'].values.astype(float)
                        ctd = ctd[~np.isnan(ctd)]
                        ax.plot(x_axis[:len(ctd)], ctd, color=cmap(color_num / color_num_max), alpha=1, label=f'{contract}')
                        dtd = target_dic[contract][f'dtd'].values.astype(float)
                        dtd = dtd[~np.isnan(dtd)]
                        ax.plot(x_axis[:len(dtd)], dtd, color=cmap(color_num / color_num_max), alpha=0.4)
                        # ax.fill_between(x_axis, pd.to_numeric(ctd.values), pd.to_numeric(dtd.values),
                        #                 color=cmap(color_num / color_num_max), alpha=0.2)
                        color_num += 1
                    ax.plot(x_axis, [0 for _ in x_axis], color='black', alpha=0.2, linewidth=0.3)
                ax.set_title(code + f'{month}月合约各可交割券{target}')
                ax.legend()

        fig_list = [fig1, fig2, fig3]
        code_list = ['T', 'TF', 'TS']
        for i in range(3):
            ax_plot(fig_list[i], code_list[i])
            fig_list[i].tight_layout()
            # 一定要先备份再保存
            backup_file(f'{self.image_folder}season_corridor_{target}_{code_list[i]}')
            fig_list[i].savefig(f'{self.image_folder}season_corridor_{target}_{code_list[i]}')

        if not self.base_config.auto_save_fig:
            plt.show()

    def corridor(self, target: str, set_ylimit=False, add_r007=False):
        target_dic = self.processed_data.measure_dict_visual_data[target]
        rates_timeseries = self.processed_data.rates_ts
        cmap = plt.cm.get_cmap('rainbow')

        # 创建画布对象for T TF TS
        fig1 = plt.figure(figsize=(10, 6))
        fig2 = plt.figure(figsize=(10, 6))
        fig3 = plt.figure(figsize=(10, 6))
        x_axis = pd.to_datetime(self.processed_data.date_index_str)
        # 从 rates_timeseries 中获取 R007.IB 列的数据，并去除重复行
        r007 = rates_timeseries.loc[:, 'r007']

        # 在画布对象上分别绘制图形
        def ax_plot(fig, code):
            ax = fig.add_subplot(111)
            color_num = 0
            # 三种合约除以3
            color_num_max = len(target_dic.keys()) / 3
            keys = sorted(target_dic.keys())
            for contract in keys:
                if re.match(rf"^{code}\d", contract):
                    for bond_code, data in target_dic[contract].items():
                        data = data.reindex(x_axis)
                        ax.plot(x_axis, data.values, color=cmap(color_num/color_num_max), alpha=0.1, label='')

                    ctd = target_dic[contract][f'ctd']
                    ctd = ctd.reindex(x_axis)
                    ax.plot(x_axis, ctd.values, color=cmap(color_num/color_num_max), alpha=1, label=f'{contract}')
                    dtd = target_dic[contract][f'dtd']
                    dtd = dtd.reindex(x_axis)
                    ax.plot(x_axis, dtd.values, color=cmap(color_num/color_num_max), alpha=0.4, label='')
                    ax.fill_between(x_axis, pd.to_numeric(ctd.values), pd.to_numeric(dtd.values),
                                    color=cmap(color_num/color_num_max), alpha=0.2, label='')
                    if set_ylimit:
                        ax.set_ylim(-3, 8)
                    if add_r007:
                        ax2 = ax.twinx()
                        ax2.plot(x_axis, r007, color='red', label='R007(右轴)')
                        ax2.set_ylim(-5, 5)
                    color_num += 1
                ax.plot(x_axis, [0 for _ in x_axis], color='black', alpha=0.5, linewidth=0.3, label='')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0, title='Contracts')
            if add_r007:
                ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.2), borderaxespad=0)
            fig.suptitle(code+f' 各可交割券{target}')

        fig_list = [fig1, fig2, fig3]
        code_list = ['T', 'TF', 'TS']
        for i in range(3):
            ax_plot(fig_list[i], code_list[i])
            fig_list[i].tight_layout()

            # 一定要先备份再保存
            backup_file(f'{self.image_folder}corridor_{target}_{code_list[i]}')
            fig_list[i].savefig(f'{self.image_folder}corridor_{target}_{code_list[i]}')

        if not self.base_config.auto_save_fig:
            plt.show()

    def plot_dura_coef_scatter(self):
        df_dic = self.processed_data.contract_dict_bonds_stats
        cmap = plt.cm.get_cmap('rainbow')

        # 创建画布对象for T TF TS
        fig1 = plt.figure(figsize=(10, 6))
        fig2 = plt.figure(figsize=(10, 6))
        fig3 = plt.figure(figsize=(10, 6))

        # 在画布对象上分别绘制图形
        def ax_plot(fig, code):
            ax = fig.add_subplot(111)
            color_num = 0
            # 三种合约
            color_num_max = len(df_dic.keys()) / 3
            for contract in df_dic.keys():
                df = df_dic[contract].dropna()
                if re.match(rf"^{code}\d", contract) and not df.empty:# and not df.isna().any().any():
                    ax.scatter(df['avg_dura'].values, df['coef'].values, color=cmap(color_num / color_num_max),
                               alpha=df['n_norm'].values)
                    color_num += 1
            fig.suptitle(code + ' 久期与(delta基差/delta收益率)关系')

        fig_list = [fig1, fig2, fig3]
        code_list = ['T', 'TF', 'TS']
        for i in range(3):
            fig_list[i].tight_layout()
            ax_plot(fig_list[i], code_list[i])
            # 一定要先备份再保存
            backup_file(f'{self.image_folder}dura_coef_scatter_{code_list[i]}')
            fig_list[i].savefig(f'{self.image_folder}dura_coef_scatter_{code_list[i]}')

        if not self.base_config.auto_save_fig:
            plt.show()

    def plot_yield_basis_line(self):
        dic_duration = self.processed_data.measure_dict_visual_data['duration']
        dic_ytm = self.processed_data.measure_dict_visual_data['ytm']
        dic_basis = self.processed_data.measure_dict_visual_data['basis']
        cmap = plt.cm.get_cmap('rainbow')

        # 创建一个3x3的子图
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        labels = ['低', '中', '高']
        color_num_max = len(dic_duration.keys())

        for code_num, code in enumerate(['T', 'TF', 'TS']):
            for group, label in enumerate(labels):
                ax = axes[code_num, group]
                color_num = 0

                for contract in dic_duration.keys():
                    if re.match(rf"^{code}\d", contract):
                        # 计算每个 bond_code 的 duration 的平均值
                        avg_durations = dic_duration[contract].mean()
                        # 使用 pd.cut() 将平均duration划分为三档
                        categories = pd.cut(avg_durations, bins=3, labels=labels)

                        for bond_code, data in dic_duration[contract].items():
                            if data.notna().sum() < 5:
                                continue

                            if categories[bond_code] == label:
                                ytm = dic_ytm[contract][bond_code].dropna()
                                basis = dic_basis[contract][bond_code].dropna()

                                # 排序
                                sort_index = np.argsort(ytm)

                                # 绘制
                                ax.plot(ytm[sort_index], basis[sort_index], color=cmap(color_num / color_num_max),
                                        alpha=min(len(ytm[sort_index]) / 200, 1))
                                ax.scatter(ytm[sort_index], basis[sort_index], color=cmap(color_num / color_num_max),
                                           alpha=min(len(ytm[sort_index]) / 200, 1), s=3)

                                color_num += 1
                ax.set_title(f'{code}合约，{label}久期组')
                ax.set_xlabel('收益率')  # 添加横轴标签
                ax.set_ylabel('基差')  # 添加纵轴标签

        fig.tight_layout()  # 自动调整子图之间的间距
        backup_file(f'{self.image_folder}yield_basis')
        fig.savefig(f'{self.image_folder}yield_basis')

        if not self.base_config.auto_save_fig:
            plt.show()
