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
matplotlib.use('TkAgg')  # �� 'Qt5Agg'
print(matplotlib.get_backend())
# plt.ion()  # ��������ģʽ



# ����������һ��DataFrame 'df'�����а����˱��ر���ʷ�۸�����
df = pd.read_csv(rf"D:\WPS����\WPS����\����-���\�о�trial\btc history.csv", encoding='GBK', thousands=',') # ����·�����Լ��ṩ
df['Date'] = pd.to_datetime(df['����'])
df = df.set_index('Date')
df = df.sort_index(ascending=True)
df["Close"] = pd.to_numeric(df["����"], errors='coerce')
df["Open"] = pd.to_numeric(df["�_��"], errors='coerce')
df["High"] = pd.to_numeric(df["��"], errors='coerce')
df["Low"] = pd.to_numeric(df["��"], errors='coerce')
last_date = df.index.max()  # ��ȡ���һ������
future_dates = pd.date_range(start=last_date, periods=800, closed='right')  # ����δ��1�������
future_df = pd.DataFrame(index=future_dates)
future_df = future_df.join(df[['Close', 'Open', 'High', 'Low']], how='left')  # �Կ�ֵ���룬�����нṹһ��
df = pd.concat([df, future_df])  # ��δ�����ݺϲ���ԭʼDataFrame��

class InteractivePlot:
    def __init__(self, df, n, m):
        self.original_df = copy.deepcopy(df[df.index >= pd.to_datetime('2016-01-01')])
        # ʹ�ù��˺������
        self.df = self.original_df.copy()

        self.fig, self.ax = plt.subplots()
        self.n = n
        self.m = m
        self.lines = []
        self.user_data = pd.DataFrame(columns=['Date', 'Close'])

        # ������ʷ�۸�;���
        self.plot_price_and_moving_averages()

        # �����¼�
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def plot_price_and_moving_averages(self):
        self.df['n_MA'] = self.df['Close'].rolling(window=self.n).mean()
        self.df['2*m_MA'] = self.df['Close'].rolling(window=self.m).mean()*2
        self.ax.grid(True)
        # �ֶ�����y��ĳ�ʼ��Χ
        # self.ax.set_ylim(top=1e6)

        self.mark_crossing_points()

        # ���Ƽ۸�;���
        # mpf.plot(self.df, type='line', ax=self.ax)
        self.ax.plot(self.df.index, self.df['Close'], label='Close Price')

        # �ֶ������ƶ�ƽ����
        self.ax.plot(self.df.index, self.df['n_MA'], label=f'{self.n}-day MA')
        self.ax.plot(self.df.index, self.df['2*m_MA'], label=f'{self.m}-day MA * 2')

        # ���� Y ��Ϊ�����̶�
        self.ax.set_yscale('log')
        # ����ͼ��
        self.ax.legend()

        # ����Y��Ϊ�����̶�
        self.ax.set_yscale('log')

        # �Ż���������ʾ
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.figure.autofmt_xdate()  # �Զ�����������ʾ�Ƕ��Ա����ص�
        # ���»��ƻ����Ը��¶����̶�
        self.fig.canvas.draw()

        # ���ư�ť
        axclear = plt.axes([0.8, 0.0, 0.1, 0.05])
        self.button = Button(axclear, 'Clear')
        self.button.on_clicked(self.clear)

        # Ϊ���ڱ�ǩ��Ӹ���Ŀռ�
        plt.subplots_adjust(bottom=0.2)

    def __call__(self, event):
        if event.inaxes is not self.button.ax:  # ���������ڰ�ť��
            clicked_date_num = event.xdata  # �������x��λ��

            # ʹ���Զ��巽����ȡ��ӽ�������
            clicked_date = self.get_closest_date(clicked_date_num)

            print(f'clicked_date {clicked_date}')
            print(f'clicked_date_num {clicked_date_num}')

            # �ҵ��������ʷ���ݵ�
            last_data_date = self.df.loc[self.df.index < clicked_date].last_valid_index()

            # ��ȡ���������ӽ���δ�����������ǿ��ܴ���δ�����ݵ������
            future_date_index = self.df.index.get_loc(clicked_date, method='pad')
            closest_future_date = self.df.index[future_date_index]

            # ���µ��λ�õ�����
            clicked_y_value = event.ydata
            self.df.at[closest_future_date, 'Close'] = clicked_y_value

            # ������λ�ú������ʷ���ݵ�֮���м������������Բ�ֵ
            if last_data_date is not None and (closest_future_date - last_data_date).days > 1:
                # ���Բ�ֵ���
                self.linear_interpolate_data(last_data_date, closest_future_date, clicked_y_value)

            # ���»���ͼ��
            self.redraw_chart()

    def get_closest_date(self, xdata):
        # ���ȣ���df�������������ڣ�ת��Ϊ������
        dates_num = date2num(self.df.index.to_pydatetime())

        # ���������ҵ�xdata��ӽ�������
        # ����xdata��ÿ��������ֵ֮��Ĳ�ľ���ֵ
        abs_diff = np.abs(dates_num - xdata)

        # �ҵ���С��ֵ��Ӧ������
        closest_index = np.argmin(abs_diff)

        # ʹ�������������ȡ��ӽ�������
        closest_date = self.df.index[closest_index]

        return closest_date

    def linear_interpolate_data(self, start_date, end_date, end_value):
        start_value = self.df.at[start_date, 'Close']
        # ������Ҫ��ֵ������
        days = (end_date - start_date).days
        # �������Բ�ֵ��ֵ
        interpolated_values = np.linspace(start_value, end_value, days + 1)

        # ����ÿһ�죬���� DataFrame
        for i in range(1, days):
            interpolate_date = start_date + pd.Timedelta(days=i)
            self.df.at[interpolate_date, 'Close'] = interpolated_values[i]

    def redraw_chart(self):
        self.ax.clear()  # �����ǰ�������

        # ʹ��mpf.plot���Ƽ۸��ߣ������ٴ���mav����
        # mpf.plot(self.df, type='line', ax=self.ax)

        self.ax.grid(True)
        # self.ax.set_ylim(top=1e6)

        # ����ԭʼ�۸���
        self.ax.plot(self.df.index, self.df['Close'], label='Close Price')

        # �ֶ������ƶ�ƽ����
        self.df['n_MA'] = self.df['Close'].rolling(window=self.n).mean()
        self.df['2*m_MA'] = self.df['Close'].rolling(window=self.m).mean()*2
        self.ax.plot(self.df.index, self.df['n_MA'], label=f'{self.n}-day MA')
        self.ax.plot(self.df.index, self.df['2*m_MA'], label=f'{self.m}-day MA * 2')
        self.mark_crossing_points()

        # ���ͼ��
        self.ax.legend()

        self.ax.set_yscale('log')  # ����Y��Ϊ�����̶�
        # �Ż���������ʾ
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.figure.autofmt_xdate()  # �Զ�����������ʾ�Ƕ��Ա����ص�
        plt.subplots_adjust(bottom=0.2)  # ���µ�������
        self.fig.canvas.draw()  # ���»�����ʾ

    def mark_crossing_points(self):
        # �ҵ�2*m_MA > n_MA������λ��
        crossing_indices = np.where(self.df['2*m_MA'] < self.df['n_MA'])[0]
        # print(crossing_indices)
        for idx in crossing_indices:
            # ʹ��df�������ҵ���Ӧ��xֵ�����ڣ�
            date = self.df.index[idx]
            # ��ȡ2*m_MA��ֵ��Ϊy����
            y_value = self.df['2*m_MA'].iloc[idx]
            # ��ͼ�б��
            self.ax.annotate('', xy=(date, y_value), xytext=(date, y_value + (y_value * 0.05)),
                             arrowprops=dict(facecolor='green', shrink=0.05))

    def clear(self, event):
        # ѭ�������������������Ƴ�
        while self.ax.lines:
            self.ax.lines[0].remove()

        # ������е�ע�ͣ�������ͷ��ǣ�
        for annotation in reversed(self.ax.texts):
            annotation.remove()

        # ������м�ͷ��ע�⣩
        for artist in reversed(self.ax.artists):
            artist.remove()

        # �������ͼ��Ԫ�أ�����ͼ��
        self.ax.legend_ = None

        # �ָ�dfΪԭʼ���ݵĸ���
        self.df = self.original_df.copy()

        # ���»���ͼ������ʾ������״̬
        self.plot_price_and_moving_averages()
        self.fig.canvas.draw()

    def on_mouse_move(self, event):
        def on_mouse_move(self, event):
            if event.inaxes:  # ȷ����������ڲ�
                # �����λ��ת��Ϊ��Ӧ������
                closest_date = self.get_closest_date(event.xdata)
                if closest_date:
                    # ��ȡ��Ӧ�ļ۸�
                    price = self.df.at[closest_date, 'Close']
                    # ����״̬����Ϣ
                    self.ax.format_coord = lambda x, y: f'Date: {closest_date.strftime("%Y-%m-%d")} Price: {price:.2f}'
                else:
                    # �����Ҳ�����Ӧ���ڵ����
                    self.ax.format_coord = lambda x, y: 'Date: N/A Price: N/A'
            self.fig.canvas.draw_idle()  # ������Ҫʱ�ػ�ͼ��


# ������չʾ����ͼ
interactive_plot = InteractivePlot(df, n=111, m=350)
plt.show()
