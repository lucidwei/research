# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# from it_data_tool.data_tool import *
# from lowrisk_it_tools.WindPy import *
# from macrotoolchain.mainAPI import *
# from lowrisk.CustomPlot import *
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
# from lowrisk.CustomECharts import *
warnings.filterwarnings("ignore")

di.HTML(getCatalogHtml())

h1Title('公募基金仓位与股指关系模型')
h2Title('更新时间： {}年{}月{}日'.format(Date.today().year, Date.today().month, Date.today().day))

# 模型描述：
# + 将市场上所有参与金融市场活动的对象看作一只大的公募基金整体，由于没有交易对手方所以股票数量恒定，因此仓位上升的原因有且仅有股票价格上涨。已知本期仓位和前期仓位，且股票数量不变，可以列方程求出本期和前期股票价格变动百分比（注意持有的资产金额总数也发生了变化），公式为$\frac{n_2}{1-n_2}\frac{1-n_1}{n_1}-1$，其中$n$为仓位。

start_date = '20120930'
end_date = datetime.today().strftime('%Y%m%d')

# 增量资金
# 主动权益型基金发行规模
def get_fund_issue(sd):
    sql = '''
        select fund_code as fcode, setup_date as ann_date, issue_total_unit as 主动权益型基金发行规模
        from ads_fund_baseinfo_invest_a
        where custom_first_type = '主动' and custom_second_type = '权益' and is_initial = '1' and setup_date >= '{}'
        order by setup_date
        '''.format(sd)
    df = get_quant_db_hz(sql).dropna()
    df.ann_date = pd.to_datetime(df.ann_date)
    df = df.set_index('ann_date').resample('Q').sum()
    return df

fund_issue_q = get_fund_issue(start_date)

# 主动权益型基金净申赎规模
def get_fund_netshares(sd,ed):
    sql = '''
        select fund_code as fcode
        from ads_fund_baseinfo_invest_a
        where custom_first_type = '主动' and  custom_second_type = '权益' and is_initial = '1'
        order by fund_code
        '''.format(sd)
    df = get_quant_db_hz(sql).dropna()
    fund_code_list = df.fcode.tolist()
    # 份额（亿份）
    sql = '''
        select a.F_INFO_WINDCODE as fcode, a.CHANGE_DATE as ann_date, a.FUNDSHARE_TOTAL/10000 as share
        from ads_fund_ChinaMutualFundShare a
        where a.F_INFO_WINDCODE in {} and a.CHANGE_DATE >= '{}'
        order by a.F_INFO_WINDCODE, a.CHANGE_DATE
        '''.format(tuple(fund_code_list),sd)
    df = get_quant_db_hz(sql).dropna()
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    sql1 = '''
        select F_INFO_WINDCODE as fcode,PRICE_DATE as ann_date, F_NAV_UNIT as nav
        from ads_fund_ChinaMutualFundNAV
        where F_INFO_WINDCODE in {} and PRICE_DATE >= '{}'
        order by f_info_windcode,price_date
        '''.format(tuple(fund_code_list),sd)
    nav = get_quant_db_hz(sql1)
    nav['ann_date'] = pd.to_datetime(nav['ann_date'])
    nav = nav.groupby(['fcode']).apply(lambda x:x.set_index(['ann_date']).resample('D').fillna(method='ffill')).drop(['fcode'],axis=1).reset_index()
    df = pd.merge(df,nav,on=['fcode','ann_date'])
    # 季度金额变动
    date_list = pd.date_range(sd,ed,freq='Q')
    df = df[df.ann_date.isin(date_list)]
    df1 = df.groupby(['fcode'])['share'].diff().dropna()
    df2 = df.groupby(['fcode'])['nav'].rolling(2,1).mean().dropna().reset_index().drop(['fcode','level_1'],axis=1)
    fund = pd.concat([df[['fcode','ann_date']],df1],axis=1).reset_index(drop=True)
    fund = pd.concat([fund,df2],axis=1).dropna()
    fund['netbuy'] = fund['share']*fund['nav']
    return fund
fund_netshares = get_fund_netshares(start_date,end_date)

def cal_nettotal(df):
    df = df.copy()
    df = df.groupby(['ann_date'])[['netbuy']].sum().rename(columns={'netbuy':'主动权益型基金净申赎规模'})
    return df
fund_nettotal = cal_nettotal(fund_netshares)

# 私募基金管理规模变化
hedge_fund_aum_df = w.edb("M5543215", "2012-12-31",end_date,"Fill=Previous",usedf=True)[1]
hedge_fund_aum_df.columns = ['私募基金管理规模变化']
hedge_fund_aum_df.index = pd.to_datetime(hedge_fund_aum_df.index)
hedge_fund_aum_df['私募基金管理规模变化'] = hedge_fund_aum_df['私募基金管理规模变化'].diff().resample('Q').sum()
hedge_fund_aum_df = hedge_fund_aum_df.dropna()

# 陆股通净买入规模
data1 = edbData(seriesCode='M0329497', startingDate=Date.fromString(date='20020101',format='%Y%m%d'), title='沪股通:当日资金净流入(人民币)')
data2 = edbData(seriesCode='M0329499', startingDate=Date.fromString(date='20100101',format='%Y%m%d'), title='深股通:当日资金净流入(人民币)')
data = data1.addOtherData(data2, ignoreMissing=True, title='陆股通净买入规模')
df1 = pd.DataFrame(data).rename(columns={0:'trade_dt',1:data.title}).set_index('trade_dt').resample('Q').sum()

# 两融净增量
def get_rzye(sd,ed):
    sql = '''
        select TRADE_DT,S_MARSUM_EXCHMARKET,S_MARSUM_TRADINGBALANCE as yue, S_MARSUM_SECLENDINGBALANCE as rongquanyue
        from ads_wind_asharemargintradesum_a
        where TRADE_DT >= '{}' and TRADE_DT <='{}'
        order by TRADE_DT
        '''.format(sd,ed)
    df = get_quant_db_hz(sql)
    df['trade_dt'] = pd.to_datetime(df['trade_dt'])
    df['yue'] /= 1e8
    df['rongquanyue'] /= 1e8
    # 用前一天数据补充深交所没有出来的当天两融数据
    date_list = sorted(list(set(df.trade_dt.tolist())))
    df = df.append({'trade_dt':date_list[-1],'s_marsum_exchmarket':'SZSE',
                    'yue':df.loc[(df.trade_dt==date_list[-2]) & (df.s_marsum_exchmarket=='SZSE'),'yue'].tolist()[0],
                    'rongquanyue':df.loc[(df.trade_dt==date_list[-2]) & (df.s_marsum_exchmarket=='SZSE'),'rongquanyue'].tolist()[0]},
                   ignore_index=True)
    df = df.drop_duplicates(subset=['trade_dt','s_marsum_exchmarket'],keep='first')
    df = df.groupby(['trade_dt'])[['yue','rongquanyue']].sum()
    return df

rzye = get_rzye(start_date,end_date)
lrjzl = pd.DataFrame(rzye['yue']+rzye['rongquanyue']).rename(columns={0:'两融净增量'})
lrjzl = lrjzl.diff().resample('Q').sum()

# 增量资金净流入
zlzj = pd.concat([fund_issue_q,fund_nettotal,hedge_fund_aum_df,df1,lrjzl],axis=1)
zlzj['增量资金净流入'] = zlzj[['主动权益型基金发行规模','主动权益型基金净申赎规模','私募基金管理规模变化','陆股通净买入规模','两融净增量']].sum(axis=1)
zlzj['增量资金净流入（不含私募）'] = zlzj[['主动权益型基金发行规模','主动权益型基金净申赎规模','陆股通净买入规模','两融净增量']].sum(axis=1)

# 股票融资
data = w.edb("M5206737", "2012-10-31",end_date,"Fill=Previous",usedf=True)[1].rename(columns={'CLOSE':'股票融资'})
data.index = pd.to_datetime(data.index)
data = -data.resample('Q').sum()
zlzj = pd.concat([zlzj,data],axis=1)

# 公募基金仓位
start_date = '20121231'
end_date = datetime.today().strftime('%Y%m%d')

def get_fund_cangwei(sd,ed):
    sql = '''
        select fund_code as fcode
        from ads_fund_baseinfo_invest_a
        where custom_first_type = '主动' and  custom_second_type = '权益' and is_initial = '1'
        order by fcode
        '''
    df = get_quant_db_hz(sql).dropna()
    fund_code_list = df.fcode.tolist()
    date_list = pd.date_range(sd,ed,freq='Q')
    date_list = [x.strftime('%Y%m%d') for x in date_list]
    sql = '''
        select S_INFO_WINDCODE as fcode, F_PRT_ENDDATE as ann_date, F_PRT_STOCKTONAV as stock_ratio, F_PRT_NETASSET as nav
        from ads_fund_ChinaMutualFundAssetPortfolio
        where S_INFO_WINDCODE in {} and F_PRT_ENDDATE in {}
        order by S_INFO_WINDCODE,F_PRT_ENDDATE
        '''.format(tuple(fund_code_list),tuple(date_list))
    df = get_quant_db_hz(sql).dropna()
    df.ann_date = pd.to_datetime(df.ann_date)
    df = df.assign(tot_nav = lambda x:x.groupby(['ann_date'])['nav'].transform('sum'),
                   w = lambda x:x['nav']/x['tot_nav'])
    df['公募基金仓位%'] = df['w']*df['stock_ratio']
    df = df.groupby(['ann_date'])[['公募基金仓位%']].sum()
    return df

fund_cangwei = get_fund_cangwei(start_date,end_date)

fund_cangwei['now'] = fund_cangwei['公募基金仓位%']/100/(1-fund_cangwei['公募基金仓位%']/100)
fund_cangwei['pre'] = fund_cangwei['now'].shift(1)
fund_cangwei['公募基金持股季度涨跌幅%'] = (fund_cangwei['now']/fund_cangwei['pre']-1)*100
fund_cangwei = fund_cangwei.drop(['now','pre'],axis=1)

# 指数月度ret
kjzs = ['881001.WI','8841415.WI','000016.SH','000300.SH','000905.SH','399303.SZ','399006.SZ','399296.SZ']
kjzs_name = ['Wind全A','茅指数','上证50','沪深300','中证500','国证2000','创业板指','创成长']
sql1 = '''
    select a.s_info_windcode as scode, a.trade_dt, a.s_dq_close, b.s_info_name as sname
    from ads_wind_aindexeodprices_a a
        join ads_wind_aindexdescription_a b
        on a.s_info_windcode = b.s_info_windcode
    where a.s_info_windcode in {} and a.trade_dt >= '{}' and a.trade_dt <='{}'
    '''.format(tuple(kjzs),start_date,end_date)
df1 = get_quant_db_hz(sql1)
sql2 = '''
    select a.s_info_windcode as scode, a.trade_dt, a.s_dq_close
    from ads_wind_AIndexWindIndustriesEOD_a a
    where a.s_info_windcode in {} and a.trade_dt >= '{}' and a.trade_dt <='{}'
    '''.format(tuple(kjzs),start_date,end_date)
df2 = get_quant_db_hz(sql2)
df2.loc[df2.scode=='881001.WI','sname'] = 'Wind全A'
df2.loc[df2.scode=='8841415.WI','sname'] = '茅指数'
index_data = pd.concat([df1,df2],axis=0)

def cal_index_pctchg(sd,ed,name,df):
    df = df.copy()
    df = df.pivot_table(index=['trade_dt'],columns=['sname'],values=['s_dq_close'])['s_dq_close']
    df = df.reset_index('trade_dt').sort_values('trade_dt').set_index('trade_dt')
    df.index = pd.to_datetime(df.index)
    # 月涨跌幅
    df = ((df.pct_change()+1).resample('Q').prod()-1)
    df = df.iloc[1:,:].reset_index()
    df = df.set_index(['trade_dt']).T
    df = df.T
    df = df[name]
    df = np.round(df*100,2)
    return df

index_pctchg = cal_index_pctchg(start_date,end_date,kjzs_name,index_data)

res = pd.concat([zlzj,fund_cangwei,index_pctchg],axis=1)
res = np.round(res,2)

# 美化倒数第二个（如果为空）
last_day = res.index[-2]
if np.isnan(res.loc[last_day,'公募基金仓位%']):
    last_day_2 = res.index[-3]
    res.loc[last_day,'公募基金持股季度涨跌幅%'] = res.loc[last_day,'Wind全A']
    # 仓位每变动1%对应的股价涨跌幅
    cangwei = res.loc[last_day_2,'公募基金仓位%']/100
    x = 1-(1-cangwei)/cangwei*(cangwei-0.01)/(1-cangwei+0.01)
    res.loc[last_day,'公募基金仓位%'] = res.loc[last_day_2,'公募基金仓位%']+res.loc[last_day,'公募基金持股季度涨跌幅%']/x/100

# 美化最后一个
last_day = res.index[-1]
last_day_2 = res.index[-2]
res.loc[last_day,'公募基金持股季度涨跌幅%'] = res.loc[last_day,'Wind全A']
# 仓位每变动1%对应的股价涨跌幅
cangwei = res.loc[last_day_2,'公募基金仓位%']/100
x = 1-(1-cangwei)/cangwei*(cangwei-0.01)/(1-cangwei+0.01)
res.loc[last_day,'公募基金仓位%'] = res.loc[last_day_2,'公募基金仓位%']+res.loc[last_day,'公募基金持股季度涨跌幅%']/x/100

res = res.sort_index(ascending=False)
np.round(res,2)

