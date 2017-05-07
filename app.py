from flask import Flask, render_template, request, flash, send_file, redirect, url_for, session
from scrape_description import scrape_google_description,scrape_google_sector
from Analytics import *
from helper_functions import *
import io
import json
import datetime
import os
import time
from pandas_datareader._utils import RemoteDataError
app = Flask(__name__)
app.secret_key = 'redpotato230atlnycsfdc' #required for flashing


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolioinformation')
def portfolioinformation():
    return render_template('portfolioinformation.html')

@app.route('/login.html')
def loginpage():
    return render_template('login.html')

@app.route('/login.html',methods=['GET','POST'])
def logincheck():
    log_in_un = request.form.get('loginusername',type=str)
    log_in_pw = request.form.get('loginpassword', type=str)

    if str(log_in_un)+','+str(log_in_pw) in open('accounts.csv').read() and str(log_in_un) != "" and str(log_in_pw)!="":
    #if log_in_un == 'Adam' and log_in_pw == 'password':
        return redirect('portinfo.html')
    else:
        flash('Incorrect username or password, please try again.')
        return redirect('login.html')


@app.route('/ca.html')
def registerpage():
    return render_template('ca.html')

@app.route('/ca.html',methods=['GET','POST'])
def registercheck():
    username = request.form.get('un',type=str)
    email = request.form.get('email',type=str)
    pw = request.form.get('pw',type=str)

    if not str(username):
        flash('Please enter a valid username')
        return redirect('ca.html')
    elif not str(email) or '@' not in str(email):
        flash('Please enter a valud email address')
        return redirect('ca.html')
    elif not str(pw):
        flash('Please enter a valid password')
        return redirect('ca.html')
    else:
        with open('accounts.csv','a') as f:
            f.write(str(username)+','+str(pw)+','+str(email)+'\n')
        f.close()
        return redirect('login.html')

#This is where we enter information for the portfolio such as the portfolio name, the tickers, the shares, cash, and the  start and end date for the analysis
@app.route('/portinfo.html')
def infoport():
    return render_template('portinfo.html')


@app.route('/portinfo.html',methods=['GET','POST'])
def process_port_info():
    #Making sure that there are no fields are left empty
    try:
        port_name = request.form.get('portfolioname',type=str)
        stocks0 = request.form.get('stocks',type=str)  #Get first stock
        weights0 = request.form.get('weights',type=float)
        stocks1 = request.form.getlist('stocks1')    #Get all additional stocks the user enters in with dynamic input fields
        weights1 = request.form.getlist('weights1')
        money = float(request.form.get('startingcash',type=str))
        start = str(request.form.get('startdate', type=str))
        end = str(request.form.get('enddate',type=str))
        num_discover_stocks = request.form.get('numdiscoverstocks',type=str)
    except ValueError:
        flash('Please make sure no boxes are empty')
        return render_template('portinfo.html')

    stocks = [stocks0]+stocks1
    stocks = filter(None, stocks) #Remove empty elements
    upper_case_stocks = []
    for i in stocks:
        upper_case_stocks.append(i.upper())
    stocks = upper_case_stocks
    weights = [weights0]+weights1
    try:
        weights = [int(weight) for weight in weights]
    except ValueError:
        flash('Please make sure there is not an empty weight. Each weight must be greater than zero.')
        return render_template('portinfo.html')
    weights = list(filter(None,weights))

    #Making sure that the number of stocks to discover is an integer
    try:
        num_discover_stocks = int(num_discover_stocks)
    except ValueError:
        flash('Please make sure # of stocks to discover is an integer (i.e. 1 or 2 or 8, etc..)')
        return redirect('portinfo.html')
    #Make sure that stocks are in the yahoo finance database
    with open('yahoostocks.csv', 'r') as f:
        reader = csv.reader(f)
        file = list(reader)
    file = set(file[0])
    s = set(list(stocks))
    if s.issubset(file) == False:
        flash('Ticker not found in database please enter a valid yahoo finance ticker.')
        return render_template('portinfo.html')

    #Making sure that the start and end date are in the form YYYY-mm-dd
    try:
        datetime.datetime.strptime(start,'%Y-%m-%d')
        datetime.datetime.strptime(end,'%Y-%m-%d')
    except ValueError:
        flash('Please make sure start and end date are in the form YYYY-mm-dd (Year-Month-Day)')
        return redirect('portinfo.html')

    if start > end:
        flash('Please make sure start date is before end date')
        return redirect('portinfo.html')


    if not port_name:
        flash('Please enter a name for your portfolio.')
        return render_template('portinfo.html')
    elif len(weights) != len(stocks):
        flash('Please make sure every there is not an empty ticker in the ticker column.')
        return redirect('portinfo.html')
    elif int(money) <= 0:
        flash('Please make sure Starting Cash is greater than zero.')
        return redirect('portinfo.html')

    #create the portfolio
    # port = Portfolio(stocks,weights,start,end,money)
    # sharpe = port.Sharpe_Ratio()
    # optimal_allocs = port.find_optimal_allocations()
    # discover_stocks, discover_sharpe = port.forward_selection(num_discover_stocks)
    session['name'] = port_name
    session['stocks'] = stocks
    session['weights'] = weights
    session['start'] = start
    session['end'] = end
    session['money'] = money
    session['num_discover_stocks'] = num_discover_stocks
    return redirect(url_for('loadingtest'))
    #Get user input
    # port_name = request.form.get('portfolioname',type=str)
    # stocks0 = request.form.get('stocks',type=str)  #Get first stock
    # weights0 = request.form.get('weights',type=float)
    # stocks1 = request.form.getlist('stocks1')    #Get all additional stocks the user enters in with dynamic input fields
    # weights1 = request.form.getlist('weights1')
    # money = float(request.form.get('startingcash',type=str))
    # start = str(request.form.get('startdate', type=str))
    # end = str(request.form.get('enddate',type=str))
    #
    # stocks = [stocks0]+stocks1
    # stocks = filter(None, stocks) #Remove empty elements
    # weights = [weights0]+weights1
    # weights = [int(weight) for weight in weights]
    # weights = list(filter(None,weights))
    # #por = Portfolio(stocks,weights,start,end,money)
    # #returns = por.cumulative_returns()
    # if len(stocks) != len(weights):
    #     #flash('Please make sure stock tickers have a weight component')
    #     return render_template('portinfo.html')
    #
    # # Check for errors, if error exists then flash error
    # if money != 100:
    #     flash('wrong!')
    #     redirect('portinfo.html')
    #
    # try:
    #     money = float(money)
    # except TypeError:
    #     flash('Please Enter a non negative number for Starting Cash.')
    #     return redirect('portinfo.html')
    #
    # if start >= end:
    #     flash('Please make sure end date is after the start date or that stocks are trading during this time period.')
    #     return render_template('portinfo.html')
    #
    #
    #
    # #Flash Errors and return page if the user enters wrong data
    # # Create tuple of all stocks and weights
    # # Delete spaces from strings and turn weights into floats
    # # Create portfolio object
    # #Write into HTML
    #
    # #Handle stock tickers and weights:
    #
    #
    #
    #
    # #Handle and format stock tickers
    # for stock in stocks1:
    #     if " " in stock: #Make sure stock is not blank
    #         stocks1.remove(stock)
    # tickers = [stocks0]+stocks1
    # tickers = list(tickers)
    #
    # #Handle and format stock weights
    # #for weight in weights1:
    #     #weight = float(weight)
    #     #if " "
    #
    # #port = Portfolio(tickers,weights,start,end,money)
    # #returns  = port.cumulative_returns()
    # len_check = len(stocks1)
    # if len_check > 0:
    #     return render_template('page.html',stocklist=tickers,portname=port_name)
    # #lent = str(len(stocks1))
    # #stock = str(stocks[0])
    # #stock1 = str(stocks1[2])
    # #x = ''
    # #for i in stock:
    #     #x = x+i
    # else:
    #     return '<h1>'+port_name+'</h1>'+'<br><br><p>'+start+'</p>'


@app.route('/lt1')
def loadingtest():
    #Load in data from previous session
    port_name = session.get('name')
    stocks = session.get('stocks')
    weights = session.get('weights')
    start = session.get('start')
    end = session.get('end')
    money = session.get('money')
    num_discover_stocks = session.get('num_discover_stocks')

    #Create Original Portfolio
    port = Portfolio(stocks,weights,start,end,money)
    sharpe = port.Sharpe_Ratio()
    stocks = port.equities
    wghts = port.weights

    #Find and Create Optimized Portfolio
    optimal_allocs = port.find_optimal_allocations()
    optimal_allocs1 = optimal_allocs
    x = sum(weights)
    optimal_allocs = optimal_allocs*x
    #optimal_allocs = [ '%.6f' % elem for elem in optimal_allocs ]
    opt_port = Portfolio(stocks,optimal_allocs,start,end,money)
    opt_sharpe = opt_port.Sharpe_Ratio()

    #Forward Selection on Original Portfolio for Discovery Section
    discover_stocks, discover_sharpe = port.forward_selection(num_discover_stocks)
    discover_descriptions = scrape_google_description(discover_stocks)
    discover_names = port.ticker_Name()
    discover_sectors = scrape_google_sector(discover_stocks)
    org_sectors = scrape_google_sector(stocks)
    ls_discover = zip(discover_names,discover_stocks,discover_sectors,discover_descriptions)
    opt_sec_weights = sector_weights(discover_sectors)
    org_sec_weights = sector_weights(org_sectors)


    #Profit and Loss Chart
    pnl_original_pv = port.Portfolio_Value()
    pnl_original = port.PnL("JSON")
    pnl_optimized_pv = opt_port.Portfolio_Value()
    pnl_optimized = opt_port.PnL("JSON")

    #Line Chart Data (original and optimized chart)
    dr = port.Portfolio_Adj_Close()
    org_line_chart_data = line_chart_json(dr)
    dr1 = opt_port.Portfolio_Adj_Close()
    opt_line_chart_data = line_chart_json(dr1)


    #Metric box under line chart (original and optimized)
    org_ret_colors = port.individual_stock_dollar_returns()
    opt_ret_colors = opt_port.individual_stock_dollar_returns()
    optimal_allocs = [ '%.6f' % elem for elem in optimal_allocs1 ] #use the decimals not shares

    # #Motion Chart Rolling Returns and Volatility Data
    # #Original Portfolio Motion Chart Data
    org_pv = port.Portfolio_Value()
    org_cum_rets = rolling_cum_rets(org_pv)
    org_vol = rolling_volatility(org_pv)
    org_frames = [org_cum_rets,org_vol]
    org_result = pd.concat(org_frames,axis=1)
    org_result.columns = ['return','volatility']
    org_result = org_result.dropna() #Drop the NaN values
    org_mc = motion_chart_data('Original',org_result)

    # #Optimized Portfolio Motion Chart Data
    opt_pv = opt_port.Portfolio_Value()
    opt_cum_rets = rolling_cum_rets(opt_pv)
    opt_vol = rolling_volatility(opt_pv)
    opt_frames = [opt_cum_rets,opt_vol]
    opt_result = pd.concat(opt_frames,axis=1)
    opt_result.columns = ['return','volatility']
    opt_result = opt_result.dropna()
    opt_mc = motion_chart_data('Optimized',opt_result)
    #
    # #NASDAQ Motion Chart Data
    NASDAQ = Portfolio(['^IXIC'],[x],start,end,money) #x is the sum of our weights instantiated above
    NASDAQ = NASDAQ.Portfolio_Value()
    NASDAQ_cum_rets = rolling_cum_rets(NASDAQ)
    NASDAQ_vol = rolling_volatility(NASDAQ)
    NASDAQ_frames = [NASDAQ_cum_rets,NASDAQ_vol]
    NASDAQ_result = pd.concat(NASDAQ_frames,axis=1)
    NASDAQ_result.columns = ['return','volatility']
    NASDAQ_result = NASDAQ_result.dropna()
    nasdaq_mc = motion_chart_data('NASDAQ',NASDAQ_result)
    #
    # #S&P500 Motion Chart Data
    SP500 = Portfolio(['^GSPC'],[x],start,end,money)
    SP500 = SP500.Portfolio_Value()
    SP500_cum_rets = rolling_cum_rets(SP500)
    SP500_vol = rolling_volatility(SP500)
    SP500_frames = [SP500_cum_rets,SP500_vol]
    SP500_result = pd.concat(SP500_frames,axis=1)
    SP500_result.columns = ['return','volatility']
    SP500_result = SP500_result.dropna()
    sp500_mc = motion_chart_data('S&P500',SP500_result)
    #
    # #Dow Jones Industrial Average Motion Chart Data
    DJI = Portfolio(['^DJI'],[x],start,end,money)
    DJI = DJI.Portfolio_Value()
    DJI_cum_rets = rolling_cum_rets(DJI)
    DJI_vol = rolling_volatility(DJI)
    DJI_frames = [DJI_cum_rets,DJI_vol]
    DJI_result = pd.concat(DJI_frames,axis=1)
    DJI_result.columns = ['return','volatility']
    DJI_result = DJI_result.dropna()
    dji_mc = motion_chart_data('Dow Jones',DJI_result)
    #
    # #Russell2000 Motion Chart Data
    Russell2000 = Portfolio(['^RUT'],[x],start,end,money)
    Russell2000 = Russell2000.Portfolio_Value()
    Russell2000_cum_rets = rolling_cum_rets(Russell2000)
    Russell2000_vol = rolling_volatility(Russell2000)
    Russell2000_frames = [Russell2000_cum_rets,Russell2000_vol]
    Russell2000_result = pd.concat(Russell2000_frames,axis=1)
    Russell2000_result.columns = ['return','volatility']
    Russell2000_result = Russell2000_result.dropna()
    russell2000_mc = motion_chart_data('Russell 2000',Russell2000_result)
    #
    # #FTSE100 Motion Chart Data
    FTSE100 = Portfolio(['^FTSE'],[x],start,end,money)
    FTSE100 = FTSE100.Portfolio_Value()
    FTSE100_cum_rets = rolling_cum_rets(FTSE100)
    FTSE100_vol = rolling_volatility(FTSE100)
    FTSE100_frames = [FTSE100_cum_rets,FTSE100_vol]
    FTSE100_result = pd.concat(FTSE100_frames,axis=1)
    FTSE100_result.columns = ['return','volatility']
    FTSE100_result = FTSE100_result.dropna()
    ftse_mc = motion_chart_data('FTSE100',FTSE100_result)
    #
    Treasury = Portfolio(['^TNX'],[x],start,end,money)
    Treasury = Treasury.Portfolio_Value()
    Treasury_cum_rets = rolling_cum_rets(Treasury)
    Treasury_vol = rolling_volatility(Treasury)
    Treasury_frames = [Treasury_cum_rets,Treasury_vol]
    Treasury_result = pd.concat(Treasury_frames,axis=1)
    Treasury_result.columns = ['return','volatility']
    Treasury_result = Treasury_result.dropna()
    treasury_mc = motion_chart_data('Treasury',Treasury_result)

    l = [org_mc,opt_mc,nasdaq_mc,sp500_mc,dji_mc,russell2000_mc,ftse_mc,treasury_mc]
    mc_data = json.dumps(l)

    #Force Diagram Chart Covariance Data
    force_data_original = port.covariance(0,'JSON')
    force_data_optimized = opt_port.covariance(0,'JSON')

    #Comparison Chart
    org_cum_rets = port.cumulative_returns()
    opt_cum_rets = opt_port.cumulative_returns()
    org_port_val = '$ '+str(pnl_original_pv[-1]) #Final dollar value
    opt_port_val = '$ '+str(pnl_optimized_pv[-1]) #Final dollar value
    org_adr = port.average_daily_returns()
    opt_adr = opt_port.average_daily_returns()
    org_vol = port.Volatility()
    opt_vol = opt_port.Volatility()
    org_beta = port.Beta()
    opt_beta = opt_port.Beta()
    org_gl = port.Gain_Loss_Ratio()
    opt_gl = opt_port.Gain_Loss_Ratio()
    org_upl = port.Upside_Potential_Ratio()
    opt_upl = opt_port.Upside_Potential_Ratio()
    org_ir = port.Information_Ratio()
    opt_ir = opt_port.Information_Ratio()
    org_shape = sharpe
    opt_sharpe = opt_sharpe
    org_jen_alpha = port.Alpha()
    opt_jen_alpha = opt_port.Alpha()


    return render_template('semantic_index.html',port_name=port_name,stocks=stocks,wghts=wghts, optimal_allocs=optimal_allocs,sharpe=sharpe,
                           org_line_chart_data=org_line_chart_data,opt_line_chart_data=opt_line_chart_data,
                           org_ret_colors=org_ret_colors,opt_ret_colors=opt_ret_colors,ls_discover=ls_discover,
                           org_sec_weights=org_sec_weights,opt_sec_weights=opt_sec_weights,force_data_original=force_data_original,
                           force_data_optimized=force_data_optimized,pnl_original=pnl_original,pnl_optimized=pnl_optimized,
                           org_cum_rets=org_cum_rets,opt_cum_rets=opt_cum_rets,org_port_val=org_port_val,opt_port_val=opt_port_val,
                           org_adr=org_adr,opt_adr=opt_adr,org_vol=org_vol,opt_vol=opt_vol,org_beta=org_beta,opt_beta=opt_beta,
                           org_gl=org_gl,opt_gl=opt_gl,org_upl=org_upl,opt_upl=opt_upl,org_ir=org_ir,opt_ir=opt_ir,org_sharpe=org_shape,
                           opt_sharpe=opt_sharpe,org_jen_alpha=org_jen_alpha,opt_jen_alpha=opt_jen_alpha,motion_chart_data=mc_data)
    #return render_template('semantic_index.html',port_name=port_name,stocks=stocks,wghts=wghts,optimal_allocs=optimal_allocs,sharpe=sharpe,
                           #opt_sharpe=opt_sharpe,discover_stocks=discover_stocks,discover_sharpe=discover_sharpe,discover_descriptions=discover_descriptions,
                           #org_line_chart_data=org_line_chart_data)

@app.route('/lt1')
def loadthign():
    return render_template('semantic_index.html')

@app.route('/portfolioinformation.html', methods=['GET','POST'])
def portfolio():

    stocks=request.form.get('stocks',type=str)
    shares=request.form.get('shares',type=str)

    #Make sure that the user enters an integer or a float value for their Starting Cash
    try:
        money=request.form.get('money',type=float)
        money = float(money)
    except TypeError:
        flash('Please Enter a non negative number for Starting Cash.')
        return render_template('portfolioinformation.html')



    start = request.form.get('start',type=str)
    end = request.form.get('end',type=str)

    if start >= end:
        flash('Please make sure end date is after the start date.')
        return render_template('portfolioinformation.html')

    #Making sure that user enters numbers for shares
    try:
        stocks = stocks.split(',')
        stocks = [stock.strip(' ') for stock in stocks] #making sure that there are no spaces in between stocks (there can be periods for certain stocks)
        shares = shares.split(',')
        s = []
        pos_shares = []
        for i in shares:
            s.append(float(i))
            pos_shares.append(abs(float(i)))
        shares = s
        total = sum(pos_shares)
        shares = [i / total for i in shares]
    except ValueError:
        flash('Please make sure that the number of shares corresponds to the stocks')
        return render_template('portfolioinformation.html')

    #Checks to make stocks and shares match up
    if len(stocks) != len(shares):
        flash('Number of Stocks and Shares do not match. Please try again.')
        return render_template('portfolioinformation.html')

    #Creating Portfolio Object
    port = Portfolio(stocks,shares,start,end,money)


    #Portfolio Returns
    try:
        returns = port.daily_returns()
    except:
        flash('Please check entry fields as portfolio could not be constructed. Make sure that stock trades during the timeframe.')
        return render_template('portfolioinformation.html')
    port_val = port.Portfolio_Value()
    prices = list(port_val.values)
    dates = list(port_val.index)
    str_dates = []
    for i in dates:
        j = i.strftime("%Y-%m-%d")
        str_dates.append(j)
    prices = json.dumps(prices)
    dates = json.dumps(str_dates)

    #x = returns.index
    #x1 = json.dumps(x)
    #y = returns.values    #PROBLEM HERE WITH DATETIME

    #y1 = []
    #for i in y:
        #j = i.strftime('%Y/%m/%d')
        #y1.append(j)
    #y1 = json.dumps(y1)
    #graph_returns = port.Plot_Portfolio_Value()

    #Calculating Metrics
    Sharpe_Ratio = port.Sharpe_Ratio(rf=0)
    #s1 = []
    #for i in shares:
        #s1.append(str(i))
    #return ','.join(s1)+'TEST'
    #return Sharpe_Ratio
    coords = [150,140,141,145,138,100,100]
    coords = json.dumps(coords)
    return render_template('chartjs.html',stocks=stocks,weights=shares,sharpe_ratio=Sharpe_Ratio,coords=coords,prices=prices,dates=dates )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


    #To get rid of the Errno[48] go to the terminal and type in lsof -i tcp:5000 and then kill the running numbers