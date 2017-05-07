import pandas as pd
import numpy as np
import json
from datetime import *

'''
Helper functions for Analytics.py and app.py created by Adam Lieberman
'''

def filter_data(file,ls_fixed_syms,ls_blacklist):
    '''
    Description: Filters data from a csv file based on the fixed symbols and blacklist
    :return: ls_all_syms: list of all tickers
    :return: nd_all_data: list of filtered ticker symbols form fixed_syms and blacklist
    :return: df_data: dataframe of all data from the csv file
    '''
    df_all_data = pd.read_csv(file,header=None)
    df_all_data = df_all_data[1:]
    df_data = df_all_data
    for i in ls_fixed_syms:
        df_all_data = df_all_data[df_all_data[1]!=i]
    for i in ls_blacklist:
        df_all_data = df_all_data[df_all_data[1]!=i]
    df_all_data = df_all_data.reset_index(drop=True)
    ls_all_syms = df_all_data[1].values.tolist()
    nd_all_data  = df_all_data.values
    return ls_all_syms, nd_all_data,df_data

def year_list(start,stop):
    years = []
    start = int(start)
    stop = int(stop)
    years.append(str(start))
    i = start+1
    while i > start and i <= stop:
        years.append(str(i))
        i = i+1
    return years

def compute_df_sharpe(df,weights,money):
    rf = 0
    samples_per_year = 252
    normed = df/df.ix[0,:]
    alloced = normed*weights
    pos_vals = alloced*money
    series_portfolio_value = pos_vals.sum(axis=1)
    daily_returns = (series_portfolio_value/series_portfolio_value.shift(1))-1
    daily_returns = daily_returns[1:]
    mean_avg_daily_rets = (daily_returns - rf).mean()
    volatility = daily_returns.std()
    sharpe = np.sqrt(samples_per_year)*(mean_avg_daily_rets/volatility)
    return sharpe

def rolling_sharpe_to_json(series_sharpe):
    vals = list(series_sharpe.values)
    indx = list(series_sharpe.index.values)
    indx_lst = []
    for i in indx:
        d = pd.to_datetime(str(i))
        indx_lst.append(d.strftime('%Y/%m/%d'))
    jsn_rolling_sharpe = json.dumps([{'date': date, 'sharpe': sharpe} for date, sharpe in zip(indx_lst, vals)])
    return jsn_rolling_sharpe


def rolling_ret_vol_to_json(series_sharpe):
    vals = list(series_sharpe.values)
    ret = []
    vol = []
    tickers = []   #Now, we can do the same for the stock columns in the dataframe and append to each
    for v in vals:
        tickers.append("portfolio")
        ret.append(v[0])
        vol.append(v[1])
    indx = list(series_sharpe.index.values)
    indx_lst = []
    for i in indx:
        d = pd.to_datetime(str(i))
        indx_lst.append(d.strftime('%Y/%m/%d'))

    jsn_rolling_sharpe = json.dumps([{'ticker':ticker, 'date': date, 'returns': returnz,'volatility':volatility} for ticker, date, returnz,volatility in zip(tickers,indx_lst, ret,vol)])
    return jsn_rolling_sharpe

def dt_to_date(indx):
    indx_lst = []
    for i in indx:
        d = pd.to_datetime(str(i))
        indx_lst.append(d.strftime('%Y/%m/%d'))
    return indx_lst

def motion_chart_data(n,df):
    dte = list(df.index.values)
    dte = dt_to_date(dte)
    ret = list(df['return'].values)
    vol = list(df['volatility'].values)
    ret_feat = [list(a) for a in zip(dte,ret)]
    vol_feat = [list(a) for a in zip(dte,vol)]
    dct = {'name':n,'feat1':ret_feat,'feat2':vol_feat}
    #dct = json.dumps(dct)
    return dct



#Calculates the rolling sharpe ratio of a dataframe
def rolling_sharpe(pv):
    def inner_sharpe(pv):
        rf = 0
        samples_per_year = 252
        daily_returns = (pv/pv.shift(1))-1
        daily_returns = daily_returns[1:]
        volatility = daily_returns.std()
        avg_daily_returns = daily_returns.mean()
        sharpe  = np.sqrt(samples_per_year)*(avg_daily_returns/volatility)
        return sharpe
    df_rolling_sharpe = pd.Series([inner_sharpe(pv[pv.index <= d]) for d in pv.index.values],index=pd.DatetimeIndex(pv.index.values,name='Date'))
    return df_rolling_sharpe

#Calcualtes volatility from a pandas series on rolling basis
def rolling_volatility(pv):
    def inner_vol(pv):
        daily_rets = (pv/pv.shift(1))-1
        daily_rets = daily_rets[1:]
        volatility = daily_rets.std()
        return volatility
    df_volatility = pd.Series([inner_vol(pv[pv.index <= d]) for d in pv.index.values],index=pd.DatetimeIndex(pv.index.values,name='Date'))
    return df_volatility


#Calcualtes the cumulative returns from a pandas series on a rolling basis
def rolling_cum_rets(pv):
    def inner_cum_rets(pv):
        cum_rets = (pv[-1]/pv[0])-1
        return cum_rets
    df_cum_rets = pd.Series([inner_cum_rets(pv[pv.index <= d]) for d in pv.index.values],index=pd.DatetimeIndex(pv.index.values,name='Date'))
    return df_cum_rets

def line_chart_json(dr):
    headers = list(dr.columns.values)
    rows = list(dr.values)
    indexes = list(dr.index.values)
    symbols = []
    dates = []
    prices = []
    for i in range(len(headers)):
        for j in range(len(rows)):
            symbols.append(headers[i])
            t = pd.to_datetime(str(indexes[j]))  #We need to reformat the datetime
            ts = t.strftime('%Y-%m-%d')
            dates.append(ts)
            prices.append(dr[headers[i]][j])
    jsn_line_chart = json.dumps([{'symbol':symbol, 'date': date, 'price': price} for symbol, date, price,in zip(symbols,dates,prices)], sort_keys=False)
    return jsn_line_chart

def sector_weights(stocks):
        energy = 0.0
        basic_materials = 0.0
        industrials = 0.0
        cyclical = 0.0
        non_cyclical = 0.0
        financials = 0.0
        healthcare = 0.0
        tech = 0.0
        telecom = 0.0
        utilities = 0.0
        for i in stocks:
            if i == 'Energy':
                energy = energy + 1
            elif i == 'Basic Materials':
                basic_materials = basic_materials + 1
            elif i == 'Industrials':
                industrials = industrials + 1
            elif i == 'Cyclical Consumer Goods & Services':
                cyclical = cyclical + 1
            elif i =='Non-Cyclical Consumer Goods & Services':
                non_cyclical = non_cyclical + 1
            elif i == 'Healthcare':
                healthcare = healthcare + 1
            elif i == 'Financials':
                financials = financials + 1
            elif i == 'Technology':
                tech = tech + 1
            elif i == 'Telecommunications Services':
                telecom = telecom + 1
            elif i == 'Utilities':
                utilities = utilities + 1
        l = float(len(stocks))
        weights = [energy/l,basic_materials/l,industrials/l,cyclical/l,non_cyclical/l,financials/l,healthcare/l,
                   tech/l,telecom/l,utilities/l]
        sectors = ['Energy','Basic Materials','Industrials','Cyclical Cons. G&S','Non-Cyclical Cons. G&S','Financials','Healthcare','Technology',
                   'Telecommunications Services','Utilities']
        json_radar_data = json.dumps([{'axis':axis, 'value': float(value)} for axis, value in zip(sectors,weights)], sort_keys=False)

        return json_radar_data


#concat([ (Series(rolling_sharpe(pv.iloc[i:i+window]),
#                      index=[pv.index[i+window]])) for i in xrange(len(pv)-window) ])
if __name__ == "__main__":
    a,b,c = filter_data('etfs.csv',['EWD','TLT'],['NKY'])
    import numpy as np
    import pandas as pd
    #data = pd.Series([100000.000000, 100500.648302, 100481.450478, 99550.193742, 101913.648567],
                 #index=pd.DatetimeIndex(['2016-11-01', '2016-11-02', '2016-11-03',
                                      #'2016-11-04', '2016-11-07'], name='Date'))
    #x = rolling_sharpe(data)
    #print x

    #dr = pd.DataFrame([1,2,3,4,5,6],columns=['AAPL'],index=['a','b','c','d','e','f'])
    #line = line_chart_json(dr)
    #print line
    x = ['Technology','Energy','Energy','Technology','Healthcare','Healthcare']
    y = ['Energy','Basic Materials','Basic Materials','Utilities']
    z1 = sector_weights(x)
    print z1
    z2 = sector_weights(y)
    print z2
    print [z1+z2]
    #print len(c)
    #print type(a)
    #print type(b)

