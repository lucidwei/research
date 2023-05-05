# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.dates as mdates
import funcs, utils
from base_config import BaseConfig

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = BaseConfig().structural_data
    data_copy = data.squeeze().iloc[::-1].copy(deep=True)
    bull_dates = funcs.get_bull_or_bear(data_copy, 60)
    bull_dates['2009-02-05':'2009-05-18'] = True  # 手动修改
    stats_df = funcs.get_stats(bull_dates, data_copy, True)
    stats_df = stats_df.sort_values(by='magnitude').reset_index()

    # 震荡市回调幅度X持续时长scatter图带标记
    max_min_scaler = lambda x: 100 * (x - np.min(x)) / (np.max(x) - np.min(x))
    chopping_df = stats_df.iloc[0:12, :].copy(deep=True)
    chopping_df['norm_mag'] = chopping_df[['magnitude']].apply(max_min_scaler)
    chopping_df['norm_len'] = chopping_df[['length']].apply(max_min_scaler)
    print(chopping_df['magnitude'].mean())
    print(chopping_df['magnitude'].median())
    print(chopping_df['length'].mean())
    print(chopping_df['length'].median())
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(chopping_df['norm_mag'], chopping_df['norm_len'], c='r', label='No.历史序列编号:调整幅度bp, 调整月数')
    for i in range(len(chopping_df)):
        if chopping_df['index'][i] == 21:
            plt.annotate('No.%d:%dbp,%.1fm' % (
                chopping_df['index'][i], chopping_df['magnitude'][i], chopping_df['length'][i] / 30),
                         xy=(chopping_df['norm_mag'][i], chopping_df['norm_len'][i]),
                         xytext=(chopping_df['norm_mag'][i] - 6, chopping_df['norm_len'][i] - 5))
        else:
            plt.annotate('No.%d:%dbp,%.1fm' % (
                chopping_df['index'][i], chopping_df['magnitude'][i], chopping_df['length'][i] / 30),
                         xy=(chopping_df['norm_mag'][i], chopping_df['norm_len'][i]),
                         xytext=(chopping_df['norm_mag'][i] - 13, chopping_df['norm_len'][i] + 2))
    plt.vlines(chopping_df['norm_mag'].mean(), 0, 100, colors="r", linestyles="dashed", label='历史均值')
    plt.hlines(chopping_df['norm_len'].mean(), 0, 100, colors="r", linestyles="dashed")
    plt.vlines(chopping_df['norm_mag'].median(), 0, 100, colors="y", linestyles="dashed", label='历史中位数')
    plt.hlines(chopping_df['norm_len'].median(), 0, 100, colors="y", linestyles="dashed")

    ax.grid()
    ax.set_xlabel("上调幅度历史比例/%")
    ax.set_ylabel("调整时长历史比例/%")
    fig.legend(loc='upper right')#, bbox_to_anchor=(0.85,100))
    # plt.tight_layout()
    plt.show(block=True)

    # 历史行情与牛熊划分图
    fig = plt.figure(figsize=(17, 5))
    ax = fig.add_subplot()
    data.plot(c='b', ax=ax, legend=False, ylabel='%', xlabel='')
    for ind, row in stats_df.iterrows():
        if ind in range(14, 23):  # [1, 3, 5, 7, 8, 9, 13, 16, 20]:
            ax.axvspan(stats_df['start'][ind], stats_df['end'][ind], color='g', alpha=0.7,
                       label="_" * ind + "利率上行区间")
        else:
            ax.axvspan(stats_df['start'][ind], stats_df['end'][ind], color='g', alpha=0.3,
                       label="_" * ind + "利率上行区间")
        plt.text(stats_df['start'][ind] + (stats_df['end'][ind] - stats_df['start'][ind]) / 2, 5.6,
                 str(stats_df['index'][ind]),
                 ha='center')
    # 有bug
    # plt.fill_between(data.index, 2, 6, where=(bull_dates.astype('int')==1), facecolor='g', alpha=0.3)
    fig.legend(loc=8)
    plt.show(block=True)

    '''
    # 两个不同颜色线图，有断层直线，没法用。
    # data[bull_dates].plot(c='r', ax=ax)
    # data[~bull_dates].plot(c='g', ax=ax)
    # data[~bull_dates].plot(secondary_y=True, c='r')

    # 多色线图：有bug，跑不通，改试背景阴影。
    # from matplotlib.collections import LineCollection
    # from matplotlib.colors import ListedColormap, BoundaryNorm
    #
    # cmap = ListedColormap(['r', 'g'])
    # norm = BoundaryNorm([-2, 0.5, 2], cmap.N)
    # x = mdates.date2num(data.index.to_pydatetime())
    # points = np.array([x, data.values.ravel()]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #
    # # Create the line collection object, setting the colormapping parameters.
    # # Have to set the actual values used for colormapping separately.
    # lc = LineCollection(segments, cmap=cmap, norm=norm)
    # z = bull_dates[1:].astype('int').values.ravel()
    # lc.set_array(z)
    # lc.set_linewidth(3)
    #
    # fig1 = plt.figure()
    # plt.gca().add_collection(lc)
    # # plt.xlim(data.index.min(), data.index.max())
    # # plt.ylim(1.1, 5.1)
    # plt.show(block=True)
    '''

    # 回调幅度bar图和回调时长scatter图
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(stats_df.index, stats_df['magnitude'], label='上调幅度')
    ax.grid(axis='y')

    ax2 = ax.twinx()
    ax2.scatter(stats_df.index, stats_df['length'], c='r', label='持续时间(右轴)')

    fig.legend(loc=2)
    ax.set_xlabel("历史序列编号")
    ax.set_ylabel("上调幅度/bp")
    ax2.set_ylabel("持续时间/天")

    plt.xticks(stats_df.index, stats_df['index'])

    plt.show(block=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
