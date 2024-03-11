# coding=gbk
# Time Created: 2024/3/8 21:10
# Author  : Lucid

import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date, DateFormatter
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
print(matplotlib.get_backend())
# plt.ion()  # 开启交互模式



# 假设我们有一个DataFrame 'df'，其中包含了比特币历史价格数据
df = pd.read_csv(rf"D:\WPS云盘\WPS云盘\工作-麦高\研究trial\btc history.csv", encoding='GBK', thousands=',') # 数据路径需自己提供
df['Date'] = pd.to_datetime(df['日期'])
df = df.set_index('Date')
df = df.sort_index(ascending=True)
df["Close"] = pd.to_numeric(df["收市"], errors='coerce')
df["Open"] = pd.to_numeric(df["開市"], errors='coerce')
df["High"] = pd.to_numeric(df["高"], errors='coerce')
df["Low"] = pd.to_numeric(df["低"], errors='coerce')
last_date = df.index.max()  # 获取最后一个日期
future_dates = pd.date_range(start=last_date, periods=800, closed='right')  # 生成未来1年的日期
future_df = pd.DataFrame(index=future_dates)
future_df = future_df.join(df[['Close', 'Open', 'High', 'Low']], how='left')  # 以空值加入，保持列结构一致
df = pd.concat([df, future_df])  # 将未来数据合并到原始DataFrame中

class InteractivePlot:
    def __init__(self, df, n, m):
        self.original_df = copy.deepcopy(df[df.index >= pd.to_datetime('2016-01-01')])
        # 使用过滤后的数据
        self.df = self.original_df.copy()

        self.fig, self.ax = plt.subplots()
        self.n = n
        self.m = m
        self.lines = []
        self.user_data = pd.DataFrame(columns=['Date', 'Close'])

        # 绘制历史价格和均线
        self.plot_price_and_moving_averages()

        # 连接事件
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def plot_price_and_moving_averages(self):
        self.df['n_MA'] = self.df['Close'].rolling(window=self.n).mean()
        self.df['2*m_MA'] = self.df['Close'].rolling(window=self.m).mean()*2
        self.ax.grid(True)
        # 手动设置y轴的初始范围
        # self.ax.set_ylim(top=1e6)

        self.mark_crossing_points()

        # 绘制价格和均线
        # mpf.plot(self.df, type='line', ax=self.ax)
        self.ax.plot(self.df.index, self.df['Close'], label='Close Price')

        # 手动绘制移动平均线
        self.ax.plot(self.df.index, self.df['n_MA'], label=f'{self.n}-day MA')
        self.ax.plot(self.df.index, self.df['2*m_MA'], label=f'{self.m}-day MA * 2')

        # 设置 Y 轴为对数刻度
        self.ax.set_yscale('log')
        # 设置图例
        self.ax.legend()

        # 设置Y轴为对数刻度
        self.ax.set_yscale('log')

        # 优化日期轴显示
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.figure.autofmt_xdate()  # 自动调整日期显示角度以避免重叠
        # 重新绘制画布以更新对数刻度
        self.fig.canvas.draw()

        # 绘制按钮
        axclear = plt.axes([0.8, 0.0, 0.1, 0.05])
        self.button = Button(axclear, 'Clear')
        self.button.on_clicked(self.clear)

        # 为日期标签添加更多的空间
        plt.subplots_adjust(bottom=0.2)

    def __call__(self, event):
        if event.inaxes is not self.button.ax:  # 如果点击不在按钮上
            clicked_date_num = event.xdata  # 鼠标点击的x轴位置

            # 使用自定义方法获取最接近的日期
            clicked_date = self.get_closest_date(clicked_date_num)

            print(f'clicked_date {clicked_date}')
            print(f'clicked_date_num {clicked_date_num}')

            # 找到最近的历史数据点
            last_data_date = self.df.loc[self.df.index < clicked_date].last_valid_index()

            # 获取点击日期最接近的未来索引（考虑可能存在未来数据的情况）
            future_date_index = self.df.index.get_loc(clicked_date, method='pad')
            closest_future_date = self.df.index[future_date_index]

            # 更新点击位置的数据
            clicked_y_value = event.ydata
            self.df.at[closest_future_date, 'Close'] = clicked_y_value

            # 如果点击位置和最近历史数据点之间有间隔，则进行线性插值
            if last_data_date is not None and (closest_future_date - last_data_date).days > 1:
                # 线性插值填充
                self.linear_interpolate_data(last_data_date, closest_future_date, clicked_y_value)

            # 重新绘制图表
            self.redraw_chart()

    def get_closest_date(self, xdata):
        # 首先，将df的索引（即日期）转换为浮点数
        dates_num = date2num(self.df.index.to_pydatetime())

        # 接下来，找到xdata最接近的日期
        # 计算xdata与每个日期数值之间的差的绝对值
        abs_diff = np.abs(dates_num - xdata)

        # 找到最小差值对应的索引
        closest_index = np.argmin(abs_diff)

        # 使用这个索引来获取最接近的日期
        closest_date = self.df.index[closest_index]

        return closest_date

    def linear_interpolate_data(self, start_date, end_date, end_value):
        start_value = self.df.at[start_date, 'Close']
        # 计算需要插值的天数
        days = (end_date - start_date).days
        # 生成线性插值的值
        interpolated_values = np.linspace(start_value, end_value, days + 1)

        # 遍历每一天，更新 DataFrame
        for i in range(1, days):
            interpolate_date = start_date + pd.Timedelta(days=i)
            self.df.at[interpolate_date, 'Close'] = interpolated_values[i]

    def redraw_chart(self):
        self.ax.clear()  # 清除当前轴的内容

        # 使用mpf.plot绘制价格线，但不再传入mav参数
        # mpf.plot(self.df, type='line', ax=self.ax)

        self.ax.grid(True)
        # self.ax.set_ylim(top=1e6)

        # 绘制原始价格线
        self.ax.plot(self.df.index, self.df['Close'], label='Close Price')

        # 手动绘制移动平均线
        self.df['n_MA'] = self.df['Close'].rolling(window=self.n).mean()
        self.df['2*m_MA'] = self.df['Close'].rolling(window=self.m).mean()*2
        self.ax.plot(self.df.index, self.df['n_MA'], label=f'{self.n}-day MA')
        self.ax.plot(self.df.index, self.df['2*m_MA'], label=f'{self.m}-day MA * 2')
        self.mark_crossing_points()

        # 添加图例
        self.ax.legend()

        self.ax.set_yscale('log')  # 保持Y轴为对数刻度
        # 优化日期轴显示
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.figure.autofmt_xdate()  # 自动调整日期显示角度以避免重叠
        plt.subplots_adjust(bottom=0.2)  # 重新调整布局
        self.fig.canvas.draw()  # 更新画布显示

    def mark_crossing_points(self):
        # 找到2*m_MA > n_MA的所有位置
        crossing_indices = np.where(self.df['2*m_MA'] < self.df['n_MA'])[0]
        # print(crossing_indices)
        for idx in crossing_indices:
            # 使用df的索引找到对应的x值（日期）
            date = self.df.index[idx]
            # 获取2*m_MA的值作为y坐标
            y_value = self.df['2*m_MA'].iloc[idx]
            # 在图中标记
            self.ax.annotate('', xy=(date, y_value), xytext=(date, y_value + (y_value * 0.05)),
                             arrowprops=dict(facecolor='green', shrink=0.05))

    def clear(self, event):
        # 循环遍历所有线条，并移除
        while self.ax.lines:
            self.ax.lines[0].remove()

        # 清除所有的注释（包括箭头标记）
        for annotation in reversed(self.ax.texts):
            annotation.remove()

        # 清除所有箭头（注解）
        for artist in reversed(self.ax.artists):
            artist.remove()

        # 清除其他图表元素，例如图例
        self.ax.legend_ = None

        # 恢复df为原始数据的副本
        self.df = self.original_df.copy()

        # 重新绘制图表以显示清除后的状态
        self.plot_price_and_moving_averages()
        self.fig.canvas.draw()

    def on_mouse_move(self, event):
        def on_mouse_move(self, event):
            if event.inaxes:  # 确保鼠标在轴内部
                # 将鼠标位置转换为对应的日期
                closest_date = self.get_closest_date(event.xdata)
                if closest_date:
                    # 获取对应的价格
                    price = self.df.at[closest_date, 'Close']
                    # 更新状态栏信息
                    self.ax.format_coord = lambda x, y: f'Date: {closest_date.strftime("%Y-%m-%d")} Price: {price:.2f}'
                else:
                    # 处理找不到对应日期的情况
                    self.ax.format_coord = lambda x, y: 'Date: N/A Price: N/A'
            self.fig.canvas.draw_idle()  # 仅当需要时重绘图表


# 创建并展示交云图
interactive_plot = InteractivePlot(df, n=111, m=350)
plt.show()
