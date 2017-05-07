import pandas as pd
import pandas_datareader.data as web
import numpy as np
import math
from urllib2 import Request, urlopen
import matplotlib.pyplot as plt
import json
import datetime
import itertools
from datetime import datetime
import scipy.optimize as spo
from helper_functions import filter_data, year_list, compute_df_sharpe, rolling_sharpe_to_json,rolling_sharpe,rolling_volatility,rolling_cum_rets,rolling_ret_vol_to_json
from helper_functions import *
from scrape_description import *
import time
import csv
import time
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import pool
'''
 A Class for Quant portfolio metrics and backtests
 Created by Adam Justin Lieberman
 adam.justin.lieberman@gmail.com
'''

#TODO: FIX ALL fuction within class being called by other functions to compute metrics
class Portfolio():


    def __init__(self,equities,weights,start,end,money): #Portfolio Construction
        self.equities = equities
        self.num_shares = weights
        w = []
        s = sum(weights)
        for i in weights:
            w.append(float(i)/s)

        self.weights = w
        self.start = start #'2014-12-01' format
        self.end = end
        self.money = money #Initial money
        self.weights_copy = self.weights

    def Portfolio_Adj_Close(self): #Adj Close
        d = {}
        for ticker in self.equities:
            d[ticker] = web.DataReader(ticker, "yahoo", self.start, self.end)
        pan = pd.Panel(d)
        df_adj_close = pan.minor_xs('Adj Close')
        self.df_adj_close = df_adj_close
        return df_adj_close


    def Portfolio_Volume(self): #Volume
        d = {}
        for ticker in self.equities:
            d[ticker] = web.DataReader(ticker, "yahoo", self.start, self.end)
        pan = pd.Panel(d)
        df_volume = pan.minor_xs('Volume')
        self.df_volume = df_volume
        return df_volume


    def Portfolio_Value(self):
        self.df_adj_close = self.Portfolio_Adj_Close()
        normed = self.df_adj_close/self.df_adj_close.ix[0,:] #Norming the prices
        alloced = normed*self.weights
        pos_vals = alloced*self.money
        series_portfolio_value = pos_vals.sum(axis=1)
        self.series_portfolio_value = series_portfolio_value
        return series_portfolio_value


    def Plot_Portfolio_Value(self):
        plt.style.use('ggplot')
        port_val = self.Portfolio_Value()
        port_val = port_val.to_frame()
        dates =  port_val.index
        port_val = port_val.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('Portfolio Value')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        plt.plot(dates,port_val)
        #TROUBLE HERE
        fig.savefig('static/plot_port_val.png')
        plt.close()
        return


    def daily_returns(self):
        self.series_portfolio_value = self.Portfolio_Value()
        daily_returns = (self.series_portfolio_value/self.series_portfolio_value.shift(1))-1
        daily_returns = daily_returns[1:]
        self.df_daily_returns = daily_returns
        return daily_returns


    def cumulative_returns(self):
        self.series_portfolio_value = self.Portfolio_Value()
        #print self.series_portfolio_value
        cumulative_returns = (self.series_portfolio_value[-1]/self.series_portfolio_value[0])-1
        self.df_cumulative_returns = cumulative_returns
        return cumulative_returns


    def average_daily_returns(self):
        average_daily_returns = self.daily_returns().mean()
        self.df_average_daily_returns = average_daily_returns
        return average_daily_returns


    def Volatility(self):
        volatility = self.daily_returns()
        volatility = volatility.std()
        self.volatility = volatility
        return volatility


    def Sharpe_Ratio(self,rf=0):  #Assuming a risk free rate of 0
        #rf = 0 #The Risk Free Rate
        samples_per_year = 252
        mean_avg_daily_rets = (self.average_daily_returns() - rf).mean()
        vol = self.Volatility()
        sharpe = np.sqrt(samples_per_year)*(mean_avg_daily_rets/vol)
        self.sharpe_ratio = sharpe
        return sharpe


    #TODO: Check to make sure that this is the correct value
    def Conditional_Sharpe_Ratio(self,expected_return, risk_free_rate=0):
        cSR = (expected_return - risk_free_rate) / self.Condtition_Value_At_Risk()
        self.conditional_sharpe_ratio = cSR
        return cSR


    def Treynor_Ratio(self,excess_returns,risk_free_rate=0):
        treynor_ratio = (excess_returns - risk_free_rate) / self.Beta()
        self.treynor_ratio = treynor_ratio
        return treynor_ratio


    def Information_Ratio(self,benchmark='SPY'):
        Market_Portfolio = Portfolio([benchmark],[sum(self.num_shares)], self.start, self.end,self.money)
        market_returns = np.array(Market_Portfolio.daily_returns())
        returns = np.array(self.daily_returns())
        difference = returns - market_returns
        vol_differnece = difference.std()
        information_ratio = np.mean(difference) / vol_differnece
        self.information_ratio = information_ratio
        return information_ratio


    def Modigliani_Ratio(self,expected_returns, benchmark='SPY', risk_free_rate=0):
        returns = np.array(self.daily_returns())
        Market_Portfolio = Portfolio([benchmark],1, self.start, self.end,self.money)
        market_returns = np.array(Market_Portfolio.daily_returns())
        arry = np.empty(len(returns))
        arry.fill(risk_free_rate)
        difference = returns - arry
        benchmark_difference = market_returns - arry
        modigliani_ratio = (expected_returns - risk_free_rate) * (difference.std() / benchmark_difference.std()) + risk_free_rate
        self.modigliani_ratio = modigliani_ratio
        return modigliani_ratio


    def Beta(self):
        Market_Portfolio = Portfolio(['SPY'],[sum(self.num_shares)], self.start, self.end,self.money)
        market_returns = Market_Portfolio.daily_returns() #Market Returns
        self.market_returns = market_returns
        portfolio_returns = self.daily_returns()  #Our Portfolio Returns
        covariance = np.cov(portfolio_returns,market_returns)[0][1]
        variance = np.var(market_returns)
        beta = covariance / variance
        self.beta = beta
        return beta


    def Alpha(self,rf=0): #TODO: Optionally make risk free rate a parameter
        #Also called Jensen's Alpha
        rf = 0
        beta = self.Beta()
        Market_Portfolio = Portfolio(['SPY'],[sum(self.num_shares)], self.start, self.end,self.money)
        market_returns = Market_Portfolio.cumulative_returns()
        portfolio_returns = self.cumulative_returns()
        alpha = portfolio_returns - (rf + (market_returns-rf)*beta)
        self.alpha = alpha
        return alpha



    def Omega_Ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = self.Lower_Partial_Moment(target,order=1)
        omega = (expected_returns-risk_free_rate)/lpm
        self.omega = omega
        return omega


    def Sortino_Ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = math.sqrt(self.Lower_Partial_Moment(target,order=2))
        sortino_ratio = (expected_returns - risk_free_rate) / lpm
        self.sortino_ratio = sortino_ratio
        return sortino_ratio


    def Calmar_Ratio(self,expected_returns,risk_free_rate=0):
        max_draw_down = self.Max_Draw_Down()
        calmar_ratio = (expected_returns - risk_free_rate) / max_draw_down
        self.calmar_ratio = calmar_ratio
        return calmar_ratio

    #TODO: Slow due to drawdown calculation
    def Sterling_Ratio(self,expected_returns,periods,risk_free_rate=0):
        average_draw_down = self.Average_Drawdown(periods)
        sterling_ratio = (expected_returns - risk_free_rate) / average_draw_down
        return sterling_ratio


    #TODO: Slow due to drawdown caluclation
    def Burke_Ratio(self,expected_returns,periods, risk_free_rate=0):
        average_drawdown_squared = math.sqrt(self.Average_Drawdown_Squared(periods))
        burke_ratio = (expected_returns - risk_free_rate) / average_drawdown_squared
        self.burke_ratio = burke_ratio
        return burke_ratio


    #TODO: Test this to make sure it is working, giving difference between expected_returns and risk_free_rate
    #TODO: lpm keeps equaling 1
    def Kappa_Three_ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = math.pow(self.Lower_Partial_Moment(target,order=3), float(1/3))
        kappa_three_ratio = (expected_returns - risk_free_rate) / lpm
        self.kappa_three_ratio = kappa_three_ratio
        return kappa_three_ratio


    def Lower_Partial_Moment(self,target=0,order=1):
        returns = np.array(self.daily_returns()) #an np array so we can use .clip()
        threshold_array = np.empty(len(returns))
        threshold_array.fill(target)
        diff = threshold_array - returns
        diff = diff.clip(min=0)
        lpm = np.sum(diff ** order) / len(returns)
        self.lower_partial_moment = lpm
        return lpm


    def Higher_Partial_Moment(self,target=0,order=1):
        returns = np.array(self.daily_returns())
        threshold_array = np.empty(len(returns))
        threshold_array.fill(target)
        diff = returns - threshold_array
        diff = diff.clip(min=0) # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        hpm = np.sum(diff ** order) / len(returns)
        self.higher_partial_moment = hpm
        return hpm


    def Gain_Loss_Ratio(self,target=0):
        hpm = self.Higher_Partial_Moment(target,1)
        lpm = self.Lower_Partial_Moment(target,1)
        gain_loss_ratio = hpm / lpm
        self.gain_loss_ratio = gain_loss_ratio
        return gain_loss_ratio


    def Upside_Potential_Ratio(self,target=0):
        hpm = self.Higher_Partial_Moment(target,order=1)
        lpm = math.sqrt(self.Lower_Partial_Moment(target,order=2))
        upside_potential_ratio = hpm / lpm
        self.upside_potential_ratio = upside_potential_ratio
        return upside_potential_ratio


    def Value_At_Risk(self):
        returns = np.array(self.daily_returns())
        alpha = self.Alpha()
        sort_returns = np.sort(returns)
        indx = alpha*(len(sort_returns))
        indx = int(indx)
        var = abs(sort_returns[indx])
        self.value_at_risk = var
        return var


    #TODO: Make daily_returns() Daily_Returns() and update in all corresponding functions
    def Excess_Return_On_Value_At_Risk(self,excess_return,risk_free_rate=0):
        alpha = self.Alpha()
        returns = np.array(self.daily_returns())

        sorted_returns = np.sort(returns)
        indx = int(self.Alpha()*len(sorted_returns))
        vari = abs(sorted_returns[indx])
        ervar = (excess_return - risk_free_rate) / vari
        self.excess_return_on_value_at_risk = ervar
        return ervar


    def Condtition_Value_At_Risk(self):
        returns = np.array(self.daily_returns())
        alpha = self.Alpha()
        sort_returns = np.sort(returns)
        indx = alpha*(len(sort_returns))
        indx = int(indx)
        sigma_var = sort_returns[0]
        for i in range(1,indx):
            sigma_var += sort_returns[i]
        mcvar = abs(sigma_var/indx)
        self.condition_value_at_risk = mcvar
        return mcvar


    def Drawdown(self,tau):
        returns = np.array(self.daily_returns())
        s = [100]
        for i in range(len(returns)):
            s.append(100 * (1 + returns[i]))
        values = np.array(s)
        pos  = len(values) - 1
        pre = pos - tau
        drawdown = float('+inf')
        while pre >= 0:
            dd_i = (values[pos]/values[pre])-1
            if dd_i < drawdown:
                drawdown = dd_i
            pos, pre = pos - 1, pre - 1
        drawdown = abs(drawdown)
        self.drawdown = drawdown
        return drawdown


    def Max_Draw_Down(self): #TODO: Very Slow If there are a lot of returns
        returns = np.array(self.daily_returns())
        max_drawdown = float('-inf')
        for i in range(0,len(returns)):
            drawdown_i = self.Drawdown(i)
            if drawdown_i > max_drawdown:
                max_drawdown = drawdown_i
        max_drawdown = abs(max_drawdown)
        self.max_drawdown = max_drawdown
        return max_drawdown


    def Average_Drawdown(self,periods):
        drawdowns = []
        returns  = self.daily_returns()
        for i in range(0,len(returns)):
            drawdown_i = self.Drawdown(i)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_drawdown = abs(drawdowns[0])
        for i in range(1,periods):
            total_drawdown += abs(drawdowns[i])
        average_drawdown = total_drawdown / periods
        self.average_drawdown = average_drawdown
        return average_drawdown


    def Average_Drawdown_Squared(self,periods):
        drawdowns = []
        returns  = self.daily_returns()
        for i in range(0,len(returns)):
            drawdown_i = math.pow(self.Drawdown(i),2.0)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_drawdown = abs(drawdowns[0])
        for i in range(1,periods):
            total_drawdown += abs(drawdowns[i])
        average_drawdown_squared = total_drawdown / periods
        self.average_drawdown_squared = average_drawdown_squared
        return average_drawdown_squared


    def Portfolio_Price_To_Book(self):
        ptb_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'p6')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            ptb_list.append(ptb)
        average_price_to_book = np.mean(ptb_list)
        self.average_price_to_book = average_price_to_book
        return average_price_to_book


    def Portfolio_Price_to_Earnings(self):
        pe_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'r')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            pe_list.append(ptb)
        average_price_to_earnings = np.mean(pe_list)
        self.average_price_to_book = average_price_to_earnings
        return average_price_to_earnings

    def Portfolio_PEG(self):
        peg_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'r5')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            peg_list.append(ptb)
        average_PEG = np.mean(peg_list)
        self.average_PEG = average_PEG
        return average_PEG

    def ticker_Name(self):
        names = []
        for stock in self.forward_selected_stocks:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'n')
            req = Request(yhf_link)
            resp = urlopen(req)
            name = str(resp.read().decode().strip())
            names.append(name)
        self.forward_selected_stocks_name = names
        return names

    # def sector_weights(self,jsn=False):
    #     energy = 0
    #     basic_materials = 0
    #     industrials = 0
    #     cyclical = 0
    #     non_cyclical = 0
    #     financials = 0
    #     healthcare = 0
    #     tech = 0
    #     telecom = 0
    #     utilities = 0
    #     for i in self.forward_selected_stocks_name:
    #         if i == 'Energy':
    #             energy = energy + 1
    #         elif i == 'Basic Materials':
    #             basic_materials = basic_materials + 1
    #         elif i == 'Industrials':
    #             industrials = industrials + 1
    #         elif i == 'Cyclical Consumer Goods & Services':
    #             cyclical = cyclical + 1
    #         elif i =='Non-Cyclical Consumer Goods & Services':
    #             non_cyclical = non_cyclical + 1
    #         elif i == 'Healthcare':
    #             healthcare = healthcare + 1
    #         elif i == 'Financials':
    #             financials = financials + 1
    #         elif i == 'Technology':
    #             tech = tech + 1
    #         elif i == 'Telecommunications Services':
    #             telecom = telecom + 1
    #         elif i == 'Utilities':
    #             utilities = utilities + 1
    #     l = float(len(self.forward_selected_stocks_name))
    #     weights = [energy/l,basic_materials/l,industrials/l,cyclical/l,non_cyclical/l,financials/l,healthcare/l,
    #                tech/l,telecom/l,utilities/l]
    #     self.forward_selected_stocks_sector_weights = weights
    #     if jsn==True:
    #         jsn_line_chart = json.dumps([{'symbol':symbol, 'date': date, 'price': price} for symbol, date, price,in zip(symbols,dates,prices)], sort_keys=False)
    #
    #     return weights
    ####################################################################################################################
    #decimals is the number of decimals we want the final allocations to be rounded to
    def find_optimal_allocations(self):
        prices = self.Portfolio_Adj_Close()
        columns = len(prices.columns)
        allocs = self.weights           #Our initial assigment is the original weights (can uniformly allocate if wish)
        def SR(allocs): #Inner function which me minimize using MVO
            self.weights = allocs
            sharpe_ratio = self.Sharpe_Ratio()   #This changes the weigths that will recompute the Sharpe Ratio
            sharpe_ratio = (-1)*sharpe_ratio
            return sharpe_ratio
        bnds = tuple((0,1) for x in range(columns))
        cons = ({ 'type': 'eq', 'fun': lambda inputs: np.sum(inputs) - 1.0})
        min = spo.minimize(SR, allocs, method = 'SLSQP', bounds= bnds, constraints= cons)
        allocs = min.x
        self.optimized_weights = allocs
        self.weights = self.weights_copy #reset class weights to oringial weights
        return allocs
    ####################################################################################################################


    #We will call an outside sharpe function that handles dataframes for the forward selection from helper_functions.py
    def forward_selection(self,num):
        universe_df,lst_equities = self.df_filtered_universe()
        port_equities = self.equities                                    #our portfolio stocks
        universe = [x for x in lst_equities if (x not in port_equities)] #our test stocks
        max_sharpe = self.sharpe_ratio
        max_stocks = port_equities[:]
        port_equities1 = port_equities[:]
        for j in range(num):
            for i in universe:
                x = port_equities1[:]
                x.append(i)
                weights = [float(1)/len(x)]*len(x)
                df = universe_df[x]  #Get all rows for the columns that we want
                sharpe_ratio = compute_df_sharpe(df,weights,self.money)  #******** IMPLEMENT THIS IN HELPER FUNCTIONS ********
                if sharpe_ratio > max_sharpe:
                    max_sharpe = sharpe_ratio
                    max_stocks = x
                x = []
            if max_stocks == port_equities:
                break
            else:
                port_equities1.append(max_stocks[-1])
                universe = [x for x in universe if (x not in max_stocks)]
        self.forward_selected_stocks = max_stocks
        self.forward_selected_sharpe = max_sharpe
        return max_stocks, max_sharpe

    #Depricated
    def get_universe(self):
        start = str(datetime.strptime(self.start, '%Y-%m-%d').year)
        end = str(datetime.strptime(self.end, '%Y-%m-%d').year)
        df = pd.read_csv('constituents.csv')
        start_df_indx =  set((df[start])[df[start] == 1].index.values)
        end_df_indx =  set((df[end])[df[end] == 1].index.values)
        indx = list(start_df_indx.intersection(end_df_indx))
        df = df.loc[indx,['Ticker']]
        universe = df[df.Ticker.str.contains(" ") == False].values.T.tolist()[0] #Remove entries such as (old) which will mess up Yahoo Finance Pull
        return universe

    #DO not use get_universe it has been depricated. This finds tickers trading during all years froms start to end
    def get_filtered_universe(self):
        start = str(datetime.strptime(self.start, '%Y-%m-%d').year)
        end = str(datetime.strptime(self.end, '%Y-%m-%d').year)
        years = year_list(start,end)
        df1 = pd.read_csv('constituents.csv')
        df = df1[years]
        indx = list(df[df == 1].dropna().index)
        stocks =  df1.loc[indx,['Ticker']].values
        stocks = list(itertools.chain.from_iterable(stocks))
        self.fs_universe = stocks
        return stocks

    #This will find all the stocks which yahoo can pull (from 2008 to 2016)
    def see_yahoo_stocks(self):
        start = str(datetime.strptime(self.start, '%Y-%m-%d').year)
        end = str(datetime.strptime(self.end, '%Y-%m-%d').year)
        years = year_list(start,end)
        df1 = pd.read_csv('constituents.csv')
        df = df1[years]
        stocks =  df1.loc[:,['Ticker']].values
        stocks = list(itertools.chain.from_iterable(stocks))
        d = {}
        i = 0
        for ticker in stocks:
            try:
                d[ticker] = web.DataReader(ticker, "yahoo", start, end)
            except:
                pass
            i = i+1;
            if i % 49 == 0:
                time.sleep(3)
        pan = pd.Panel(d)
        df_adj_close = pan.minor_xs('Adj Close')
        filtered_equities = list(df_adj_close.columns.values) #the tickers
        with open('yahoostocks.csv',"wb") as f:
            writer = csv.writer(f,quoting=csv.QUOTE_ALL)
            writer.writerow(filtered_equities)
        return filtered_equities


    def df_filtered_universe(self):
        equities = self.get_filtered_universe()
        equities = list(set().union(equities,self.equities)) #Add our stocks in to get a complete dataframe of all stocks (just incase our stocks get filtered out)
        with open('testyahoostocks.csv','rb') as f:
            reader = csv.reader(f)
            yahoo_stocks = list(reader)[0]
        equities = [x for x in equities if (x in yahoo_stocks)]  #Filters out all stocks not in yahoo
        start = self.start
        end = self.end
        d = {}
        i = 0
        for ticker in equities:  #Building the dataframes one by one
            try:
                d[ticker] = web.DataReader(ticker, "yahoo", start, end)
            except:
                pass
            i = i+1;
            if i % 49 == 0:
                time.sleep(3) #So we do not get blocked by yahoo
        pan = pd.Panel(d)
        df_adj_close = pan.minor_xs('Adj Close')
        self.df_filtered_equities = df_adj_close
        self.filtered_equities = list(df_adj_close.columns.values) #the tickers
        return df_adj_close, list(df_adj_close.columns.values)

    def covariance(self,binary,bool):
        if binary == 0:
            equities = self.equities
        else:  #Forward selection needs to be run before this
            equities = self.equities + self.forward_selected_stocks
        stock_indices = ['^IXIC','^GSPC','^DJI','^RUT','^FTSE','^TNX']
        equities = equities+stock_indices
        d = {}
        for ticker in equities:  #Building the dataframes one by one
            d[ticker] = web.DataReader(ticker, "yahoo", self.start, self.end)
        pan = pd.Panel(d)
        df_stock_indices = pan.minor_xs('Adj Close')
        frames = [self.df_adj_close,df_stock_indices]
        df_stocks = pd.concat(frames)
        cov_list = []
        for i in equities:
            for j in equities:
                df1 = df_stocks[i].pct_change()
                df2 = df_stocks[j].pct_change()
                cov = df1.corr(df2) #correlation coefficient
                cov_list.append([i,j,cov])
        #Remove the duplicates now
        seen = set()
        covariances = [x for x in cov_list if frozenset(x) not in seen and not seen.add(frozenset(x))]
        self.covariances = covariances
        sources = []
        targets = []
        cov = []
        for i in covariances:
            sources.append(i[0])
            targets.append(i[1])
            cov.append(i[2])

        if bool == 'JSON': #We will return covariances as JSON format
            #d = {}
            #for i in range(len(covariances)):
                #d[i] = {"source":covariances[i][0],"target":covariances[i][1],"covariance":covariances[i][2]}
            #return d
            json_sankey_data = json.dumps([{'source':source, 'target': target, 'value':value} for source, target, value in zip(sources,targets,cov)], sort_keys=False)
            return json_sankey_data
        return covariances

    def individual_stock_returns(self):
        df = self.df_adj_close
        returns = []
        headers = list(df.columns.values)
        for i in headers:
            d = df[i]
            ret = ((d[-1]/d[0])-1)*100    #Get this in % for the html pass through
            if ret > 0:
                returns.append((str(ret)+' %','green'))
            elif ret < 0:
                returns.append((str(ret)+' %','red'))
            else:
                returns.append((str(ret)+' %','yellow'))
            #ret = str(ret)+'%'
            #returns.append(ret)
        return returns

    def individual_stock_dollar_returns(self):
        df = self.df_adj_close
        ticker_weights = self.num_shares
        returns = []
        headers = list(df.columns.values)
        for i in range(len(headers)):
            d = df[headers[i]]
            ret = (d[-1]-d[0])*ticker_weights[i]
            if ret < 0.0001 and ret > -0.0001:
                ret = 0.00000 #To filter out very small returns from the 0 weight being e-17 or e-16 in magnitude
            #ret = ((d[-1]/d[0])-1)*100    #Get this in % for the html pass through
            x = '$ '+str(ret)
            if ret > 0:
                returns.append((x,'green'))
            elif ret < 0:
                returns.append((x,'red'))
            else:
                returns.append((x,'orange'))
            #ret = str(ret)+'%'
            #returns.append(ret)
        return returns

    def PnL(self,type='NO'):
        pv = self.series_portfolio_value
        monthly_sample = pv.resample('BM').apply(lambda x: x[-1])
        monthly_pct_chage = monthly_sample.pct_change()
        self.monthly_pct_change = monthly_pct_chage
        if type == "JSON":
            dates1 = []
            dates = list(monthly_pct_chage.index.values)
            for i in dates:
                t = pd.to_datetime(str(i))  #We need to reformat the datetime
                print t
                ts = t.strftime('%b %Y')
                dates1.append(ts)
            pct_change = list(monthly_pct_chage.values)
            pct_change = [x*100 for x in pct_change]
            pct_change.pop(0)
            dates1.pop(0)
            json_pnl_data = json.dumps([{'name':name,'value':value} for name, value in zip(dates1,pct_change)], sort_keys=False)
            return json_pnl_data
        return monthly_pct_chage
if __name__ == '__main__':
    #start = time.time()
    port = Portfolio(['AAPL','BAC','GILD','SLB'],[100,100,100,100],'2015-01-01','2016-11-01',1000000)
    pv = port.Portfolio_Value()
    print(port.Sharpe_Ratio())
    #print port.df_adj_close
    #x = port.individual_stock_dollar_returns()
    #print x
    # pv = port.Portfolio_Value()
    # force_data_original = port.covariance(0,'JSON')
    # print force_data_original
    # Russell2000 = Portfolio(['^RUT'],[400],'2012-11-01','2016-11-07',100000)
    # Russell2000 = Russell2000.Portfolio_Value()
    # Russell2000_cum_rets = rolling_cum_rets(Russell2000)
    # Russell2000_vol = rolling_volatility(Russell2000)
    # Russell2000_frames = [Russell2000_cum_rets,Russell2000_vol]
    # Russell2000_result = pd.concat(Russell2000_frames,axis=1)
    # Russell2000_result.columns = ['return','volatility']
    # Russell2000_result = Russell2000_result.dropna()
    # r2k = motion_chart_data('russell 2000',Russell2000_result)
    # print r2k
    #
    # DJI = Portfolio(['^DJI'],[400],'2012-11-01','2016-11-07',100000)
    # DJI = DJI.Portfolio_Value()
    # DJI_cum_rets = rolling_cum_rets(DJI)
    # DJI_vol = rolling_volatility(DJI)
    # DJI_frames = [DJI_cum_rets,DJI_vol]
    # DJI_result = pd.concat(DJI_frames,axis=1)
    # DJI_result.columns = ['return','volatility']
    # DJI_result = DJI_result.dropna()
    # dj = motion_chart_data('Dow Jones',DJI_result)
    # print dj
    # print type(dj)
    # data = json.dumps([r2k,dj])
    # print data
    #dte = list(Russell2000_result.index.values)
    #dte = dt_to_date(dte)
    #ret = list(Russell2000_result['return'].values)
    #vol = list(Russell2000_result['volatility'].values)
    #ret_feat = [list(a) for a in zip(dte,ret)]
    #vol_feat = [list(a) for a in zip(dte,vol)]
    #print ret_feat
    #print vol_feat
    #tech = Portfolio(['AAPL','GILD','MSFT','KO'],[.25,.25,.25,.25],'2010-01-01','2015-12-01',100000)
    # port = Portfolio(['AAPL','CMG','KO','MSFT'],[100,100,100,100],'2012-11-01','2016-11-07',100000)
    # a = port.cumulative_returns()
    # b = port.average_daily_returns()
    # c = port.Volatility()
    # d = port.Beta()
    # e = port.Gain_Loss_Ratio()
    # f = port.Upside_Potential_Ratio()
    # g = port.Information_Ratio()
    # #h = port.Modigliani_Ratio()
    # #i = port.Sortino_Ratio()
    # j = port.Alpha()
    # print a,b,c,d,e,f,g,j
    #Our data for The Motion Chart: Index Circles
    # NASDAQ = Portfolio(['^IXIC'],[400],'2012-11-01','2016-11-07',100000)
    # NASDAQ = NASDAQ.Portfolio_Value()
    # NASDAQ_cum_rets = rolling_cum_rets(NASDAQ)
    # NASDAQ_vol = rolling_volatility(NASDAQ)
    # SP500 = Portfolio(['^GSPC'],[400],'2012-11-01','2016-11-07',100000)
    # SP500 = SP500.Portfolio_Value()
    # SP500_cum_rets = rolling_cum_rets(SP500)
    # SP500_vol = rolling_volatility(SP500)
    # DJI = Portfolio(['^DJI'],[400],'2012-11-01','2016-11-07',100000)
    # DJI = DJI.Portfolio_Value()
    # DJI_cum_rets = rolling_cum_rets(DJI)
    # DJI_vol = rolling_volatility(DJI)
    # Russell2000 = Portfolio(['^RUT'],[400],'2012-11-01','2016-11-07',100000)
    # Russell2000 = Russell2000.Portfolio_Value()
    # Russell2000_cum_rets = rolling_cum_rets(Russell2000)
    # Russell2000_vol = rolling_volatility(Russell2000)
    # FTSE100 = Portfolio(['^FTSE'],[400],'2012-11-01','2016-11-07',100000)
    # FTSE100 = FTSE100.Portfolio_Value()
    # FTSE100_cum_rets = rolling_cum_rets(FTSE100)
    # FTSE100_vol = rolling_volatility(FTSE100)
    # Treasury = Portfolio(['^TNX'],[400],'2012-11-01','2016-11-07',100000)
    # Treasury = Treasury.Portfolio_Value()
    # Treasury_cum_rets = rolling_cum_rets(Treasury)
    # Treasury_vol = rolling_volatility(Treasury)
    #pv = port.Portfolio_Value()
    #pnl = port.PnL("JSON")
    #print pnl
    #sharpe = port.Sharpe_Ratio()
    #cov = port.covariance(0,'JSON')

    #print cov
    #org_ret_colors = port.individual_stock_returns()
    #print org_ret_colors
    #num_discover_stocks = 2
    #stocks = port.equities
    #discover_stocks, discover_sharpe = port.forward_selection(num_discover_stocks)
    #discover_descriptions = scrape_google_description(discover_stocks)
    #discover_names = port.ticker_Name()
    #discover_sectors = scrape_google_sector(discover_stocks)
    #org_sectors = scrape_google_sector(stocks)
    #ls_discover = zip(discover_names,discover_stocks,discover_sectors,discover_descriptions)
    #opt_sec_weights = sector_weights(discover_sectors)
    #org_sec_weights = sector_weights(org_sectors)

    #RUNNING THE ROLLING CALCULATIONS
    #s = rolling_sharpe(pv)
    #cum_rets = rolling_cum_rets(pv)
    #vol = rolling_volatility(pv)
    #frames = [cum_rets,vol]
    #result = pd.concat(frames,axis=1)
    #result.columns = ['return','volatility']
    #result = result.dropna() #Drop the NaN values
    #print rolling_ret_vol_to_json(result)

    #rs = concat([ (Series(rolling_sharpe(pv.iloc[i:i+window]),index=[pv.index[i+window]])) for i in xrange(len(pv)-window) ])

    #pv = port.Portfolio_Value()
    #x = rolling_sharpe_to_json(pv)
    #print x
    #new_weights = port.find_optimal_allocations()
    #new_weights = [ '%.4f' % elem for elem in new_weights ]
    #print new_weights
    #weights = port.find_optimal_allocations()
    #universe,stocks = port.df_filtered_universe()
    #print stocks
    #stocks = port.see_yahoo_stocks()
    #print stocks
    #s = port.Sharpe_Ratio()  #call before the forward selection
    #a, b = port.forward_selection(2)
    #print a
    #print b
    #end = time.time()
    #print end - start
    #universe = port.df_filtered_universe()
    #print universe

    #print weights
    #weights = tech.find_optimal_allocations()
    #print weights
    #print tech.weights
    #y = tech.Portfolio_Adj_Close()
    #print y
    #x = tech.cumulative_returns()
    #print x
    #print x.values
    #print x.index
    #print type(x.index)
    #print type(list(x.values))
    #l = []
    #for i in list(x.index):
        #j = i.strftime("%Y-%m-%d")
        #l.append(j)
    #print l




    #sharpe = tech.Sharpe_Ratio()
    #print sharpe
    #print tech.equities
    #print tech.weights
    #a = tech.Beta()
    #print a
    #b = tech.Alpha()
    #c = tech.Higher_Partial_Moment()
    #d = tech.Sterling_Ratio(.005,5)
    #print c
    #print b
    #print d
    #f = tech.Portfolio_Price_To_Book()
    #print f

    #i = tech.Portfolio_Adj_Close()
    #j = tech.Portfolio_Volume()
    #k = tech.Portfolio_Value()
    #l = tech.volatility
    #print l

    #avg = tech.average_daily_returns()
    #print avg