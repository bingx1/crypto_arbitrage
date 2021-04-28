import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
import os
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import time as time
from multiprocessing import Pool
from functools import reduce
import plotting
from collections import Counter
from strategy import make_sample, do_pca, get_eigenportfolio_returns

PATH_TO_COINS_DATA = "../data/coinmarketcap_data/"


def clean_coin_data(coin_data: pd.DataFrame) -> None:
    # coin_data['time'] = pd.to_datetime(coin_data['time'])
    coin_data.index = pd.to_datetime(coin_data['time'])

def load_coin_data(path:str):
    '''
    Returns a list of dataframes containing the price data of each coin,
    and a dictionary where the keys are TICKERS and values are the ICO date of the coin.
    '''
    coins_data = []
    ico_dates = {}
    # read in the price data
    for filename in glob.glob(os.path.join(path, '*.csv')):
        fname = filename.split('/')[-1]
        ticker = fname.split('.')[0]
        # print(ticker)
        coin_data = pd.read_csv(filename)
    #     clean up the data
        clean_coin_data(coin_data)
        coin_data.name = ticker
        coins_data.append(coin_data)
    #     Do not include coins in our analysis of ICO dates.
        if 'value' in coin_data.columns:
            pass
        else:
            # Record the ICO date of the coin
            ico_dates[ticker] = coin_data.iloc[0]['time']
    return coins_data, ico_dates


def build_price_and_volume_dataframe(coins_data: pd.DataFrame):
    '''
    Builds a price dataframe and a volume dataframe
    '''
    prices = pd.concat([df["close"].rename(df.name) for df in coins_data], axis = 1)
    volumes = pd.concat([df["volumeto"].rename(df.name) for df in coins_data], axis = 1)
    # INSERT column 'Active' into dataframe to count active coins at each date
    prices['Active'] = prices.count(axis=1)
    return prices, volumes
    # # INSERT column 'Active' into dataframe to count active coins at each date
    # df['Active'] = df.count(axis=1)
    # # if tl_plt:
    # #      plot_timeline(dates,symbols,'2013-2019 Cryptocurrency ICO Dates')
    # # if len(coins_plt) > 0:
    # #     # Plot the prices of 4 coins
    # #     plot_cryptos(coins,coins_plt[0],coins_plt[1],coins_plt[2],coins_plt[3])
    # # if peryear_plt:
    # #     plot_launches_per_year(years, df)
    # return df, vol_df




# --------------------------Make eigenportfolios------------------------------------------
# def do_pca(date, dataframe, pcawindow, PCS, active, plot = False):
#     '''
#     Do pca on the dataframe inputted. Return the eigenvectors corresponding to the first 'Number_factors' principal
#     components
#     :param date: date from which to conduct PCA
#     :param dataframe: ENTIRE returns dataframe
#     :param pcawindow: length of PCA window
#     :param PCS: number of eigenvectors to return
#     :param active: list of active cryptocurrencies in the SAMPLE
#     :return: A dataframe of eigenvectors
#     '''
#     pcawindow_start = date - timedelta(days=pcawindow)
#     pca_data = dataframe.loc[(dataframe.index >= pcawindow_start)&(dataframe.index <= date), active].copy()
#     s = pca_data.std(ddof=1, axis=0)
#     sample2 = StandardScaler().fit_transform(pca_data)
#     # Call PCA function to do PCA
#     pcs = ['PC{}'.format(i) for i in range(1,PCS+1)]
#     pca = PCA(n_components=PCS)
#     pca2 = pca.fit_transform(sample2)
#     pcdf = pd.DataFrame(pca.components_.T, columns=pcs, index=sample.columns)
#     eigvals = pca.explained_variance_
#     exp_var = pca.explained_variance_ratio_
#     cum_expvar = np.cumsum(exp_var)
#     # plot variance explained graph
#     if plot:
#         print ('First principal component explains {} of the variance'.format(round(exp_var[0],4)))
#         print ('Var explained by the 1st {} components:{}'.format(PCS,round(cum_expvar[-1],4)))
#         plt.figure(5)
#         plt.bar(['PC {}'.format(i) for i in range(1,PCS+1)], exp_var, color='lightsteelblue', edgecolor='k', width=0.6)
#         plt.xticks(['PC {}'.format(i) for i in range(1,PCS+1)], rotation= 0)
#         plt.plot(['PC {}'.format(i) for i in range(1,PCS+1)], cum_expvar, ls='--', color='cornflowerblue')
#         plt.legend(['Cumulative variance explained','Proportion of variance explained'],fontsize=7)
#         plt.title('Variance explained by PC 1 - PC {}'.format(PCS),fontsize = 13)
#     # Multiply rows by 1/STDEV of each coin return to get eigen portfolio weights
#     eig_portfolios = pcdf.div(s, axis=0)
#     # Multiply colums by 1/sum of each column weights
#     # total_weights = eig_portfolios.sum(axis=0)
#     # eig_portfolios = eig_portfolios.div(total_weights, axis = 1)
#     return eig_portfolios, eigvals

# def get_eigenportfolio_returns(eig_portfolios, eigvals, dataframe, active, date):
#     '''
#     Retrieve the historical trialing window eigenportfolio returns
#     :param eig_portfolios: eigen portfolio weights
#     :param eigvals: eigenvalues of eigenportoflios
#     :param dataframe: entire returns dataframe
#     :param active: list of coins in the sample
#     :param date:  date from which to retrieve data from
#     :return: series of eigenportfolio returns
#     '''
#     pca_data = dataframe.loc[(dataframe.index <= date), active].copy()
#     # Eigen portfolio's are the columns. Add returns for each eigen portfolio of interest to pca_data
#     eig_ports = ['EP{}'.format(i) for i in range(1,PCS+1)]
#     for i in range(1, PCS + 1):
#         pc = 'PC{}'.format(i)
#         ep = 'EP{}'.format(i)
#         pca_data[ep] = ((np.sum(pca_data[active].multiply(eig_portfolios[pc].to_list()), axis=1)) * -1) / (eigvals[i - 1])
#     return pca_data[eig_ports]



# ----------- Regress each coin's returns on eigenportfolio returns
def regress_pcafactors(x, y, date, window):
    '''
    :param x: independent variables - eigenportfolio returns
    :param y: dependent variable - coin return
    :param date: regress on all data up till this date
    :param window: length of time to do regression
    :return: residuals, alpha
    '''
    start_date = date - timedelta(days=window)
    x = x.iloc[(x.index <= date) & (x.index > start_date)]
    y = y.iloc[(y.index <= date) & (y.index > start_date)]
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    alpha = results.params[0]
    resids = results.resid
    Xt = np.cumsum(resids)
    return alpha, Xt

def regress_OUprocess(Xt):
    '''
    :param resids:
    :return: s-score
    '''
    ARmodel = sm.tsa.AR(Xt,freq='D').fit(maxlag=1)
    params = ARmodel.params
    resids = ARmodel.resid
    resid_var = np.var(resids)
    return params[0], params[1], resid_var


def display_trade(y, s_scores, df, startdate, coin, open_long = -1.25, open_short = 1.25, close_long = -0.5, close_short = 0.75):
    '''
    Displays a graphic of trading signals against actual price
    :param y: a series of trading signals for the coin
    :param s_scores: a series of s_scores for the coin
    :param df: entire PRICE df
    :param startdate: start of sample
    :param coin: coin to display
    :param open_long:
    :param open_short:
    :param close_long:
    :param close_short:
    :return:
    '''
    y.name = 'Signals'
    actual_return = scores_to_returns(s_scores)
    cum_ret = np.cumprod(actual_return)
    cum_ret = cum_ret - 1
    coin_ret = df.loc[(df.index >= startdate) & (df.index <= startdate + timedelta(days=365)), coin]
    joined = pd.concat([y,coin_ret],axis=1)
    long_signals = joined.loc[joined['Signals'] == 'open long', coin]
    short_signals = joined.loc[joined['Signals'] == 'open short', coin]
    long_exits = joined.loc[joined['Signals'] == 'close long', coin]
    short_exits = joined.loc[joined['Signals'] == 'close short', coin]
    fig, ax = plt.subplots(figsize=(10, 12))
    L_ax = plt.subplot2grid((3, 1), (0, 0))
    R_ax = plt.subplot2grid((3, 1), (1, 0))
    B_ax = plt.subplot2grid((3, 1), (2, 0))
    cum_ret.plot(title='Cumulative return of {} over the sample period'.format(coin),ax=B_ax)
    vals = B_ax.get_yticks()
    B_ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    s_scores.plot(ax=R_ax,title='{} S-score over time'.format(coin),grid=True)
    coin_ret.plot(ax=L_ax,title='{}:USD over time'.format(coin))
    L_ax.set_xlabel('')
    R_ax.set_xlabel('')
    R_ax.axhline(y=open_long,color='g',linestyle='--')
    R_ax.axhline(y=open_short,color='r',linestyle='--')
    R_ax.axhline(y=close_long,color='g',linestyle='-')
    R_ax.axhline(y=close_short, color='r', linestyle='-')
    L_ax.scatter(long_signals.index, long_signals, c = 'g',marker='o')
    L_ax.scatter(long_exits.index, long_exits, c='g', marker='x')
    L_ax.scatter(short_signals.index, short_signals, c = 'r', marker='o')
    L_ax.scatter(short_exits.index, short_exits, c='r', marker='x')
    L_ax.annotate('Enter long', xy=(long_signals.index[0], long_signals[0]),fontsize=8)
    L_ax.annotate('Exit long', xy=(long_exits.index[0],long_exits[0]),fontsize=8)
    L_ax.annotate('Enter short', xy=(short_signals.index[0],short_signals[0]),fontsize=8)
    L_ax.annotate('Exit short', xy=(short_exits.index[0],short_exits[0]),fontsize=8)
    return

def parse_s_scores(y, open_long, open_short, close_long, close_short):
    '''
    :param y: list of s-scores
    :return: a series of long/short signals with corresponding dates
    '''

    dates = []
    signals = []
    major_signals = []
    for i in range(len(y)):
        score = y.iloc[i]
        date = y.index[i]
        signal = 'hold'
        try:
            last_signal = major_signals[-1]
        except IndexError:
            # print ('No signals yet')
            last_signal = None
        if score > open_short and last_signal != 'open short': #First check if s > 1.25
            signal = 'open short'
            major_signals.append(signal)
        elif score < open_long and last_signal != 'open long': #Check if s < - 1.25
            signal = 'open long'
            major_signals.append(signal)
        elif score < close_short and last_signal == 'open short': #Check if S < 0.75
            signal = 'close short'
            major_signals.append(signal)
        elif score > close_long and last_signal == 'open long':
            signal = 'close long'
            major_signals.append(signal)
        elif score >= close_short and last_signal == 'open short':
            signal = 'hold short'
        elif score <= close_long and last_signal == 'open long':
            signal = 'hold long'
        dates.append(date)
        signals.append(signal)
    signal_series = pd.Series(signals, index = dates, name=y.name)
    return signal_series

def scores_to_returns(y, open_long = -1.4, open_short = 1.05, close_long = -0.5, close_short = 0.75, annual_ir=0.015, slippage = 0.0005):
    '''
    series of s_scores to returns
    :param y: series of s_scores
    :param annual_ir: annual interest rate
    :return: a series of returns corresponding to the input s-scores
    '''
    signals = parse_s_scores(y,open_long, open_short, close_long, close_short)
    coin = signals.name
    signals.name = 'Signals'
    daily_ir = annual_ir/365
    combined = pd.concat([signals,returns[coin]],axis=1)
    combined = combined.dropna()
    # neutral positions
    combined['Return'] = 1 + daily_ir
    # short positions
    combined.loc[(combined['Signals'] == 'open short'), 'Return'] = 1 + daily_ir - slippage
    combined.loc[(combined['Signals'] == 'hold short') | (combined['Signals'] == 'close short'), 'Return'] = 1 - combined[coin]
    combined.loc[(combined['Signals'] == 'close short'), 'Return'] = 1 - combined[coin] - slippage
    # long positions
    combined.loc[(combined['Signals'] == 'open long'), 'Return'] = 1 + daily_ir - slippage
    combined.loc[(combined['Signals'] == 'hold long') | (combined['Signals'] == 'close long'), 'Return'] = 1 + combined[coin]
    combined.loc[(combined['Signals'] == 'close long'), 'Return'] = 1 + combined[coin] - slippage
    x = combined['Return']
    x.name = coin
    return x

def build_s_score(a, b, resid_var, PCA_window):
    '''
    computes the s-score for a coin on a given day
    :param a: 1st param - constant term from OU regression
    :param b: 2nd param - coefficient on AR term from OU regression
    :param resid_var: variance of residuals from OU regression
    :return: s_score
    '''
    if b > 0.9672:
        # print ('Mean reversion too slow relative to estimation window')
        pass
    k = -np.log(b)*365

    m = a/(1-b)
    sigma = np.sqrt((resid_var*2*k)/(1-b**2))
    sigma_v2 = sigma/np.sqrt(2*k)
    sigma_eq = np.sqrt(resid_var/(1-b**2))
    s = -m/sigma_eq
    return m, sigma_eq

def retrieve_s_score(x):
    '''

    :param x: a iterable containing [eig_returns, coin_ret, date, regression_window, PCA_window]
    :return: corresponding s-score
    '''
    eig_returns = x[0]
    coin_ret = x[1]
    date = x[2]
    regression_window = x[3]
    PCA_window = x[4]
    alpha, Xt = regress_pcafactors(eig_returns, coin_ret, date, regression_window)
    a, b, resid_var = regress_OUprocess(Xt)
    m, sigma_eq = build_s_score(a, b, resid_var, PCA_window)
    return m, sigma_eq

def backtest(PCA_window, regression_window, sample_window, PCS, returns, active, startdate, pc_interval):
    storage = {}
    pool = Pool(os.cpu_count())
    # Iterate over the sample period and get s scores:
    for i in range(sample_window+1):
        date = startdate + timedelta(days=i)
        # Recompute eigenportfolios every 60 days
        if i%pc_interval == 0:
            eigportfolios, eigvals= do_pca(date, returns, PCA_window, PCS, active) #Recompute eigenportfolios every 60 days
        eig_returns = get_eigenportfolio_returns(eigportfolios, eigvals, returns, active, date)
        pool_input = []
        for coin in active:
            if i == 0:
                storage[coin] = []
            pool_input.append((eig_returns, returns[coin],date,regression_window,PCA_window))
        out = pool.map(retrieve_s_score,pool_input) #a list of tuples, with m and sigma_eq
        all_m = [i[0] for i in out if i!=0]
        all_sigmas = [y[1] for y in out]
        avg_m = np.sum(all_m)/len(all_m)
        modified_m = [s_score[0] - avg_m for s_score in out]
        adjusted_s_scores = [-m/sigma for m,sigma in zip(modified_m, all_sigmas)]
        for count, coin in enumerate(active):
            storage[coin].append(adjusted_s_scores[count])
    return storage

def get_PnL(results, plot = False):
    ''''''
    portfolio_returns = results.apply(scores_to_returns, axis=0)
    portfolio_returns['Mean'] = np.mean(portfolio_returns, axis=1)
    cum_return = np.cumprod((portfolio_returns['Mean']))
    if plot:
        plt.figure(2)
        cum_return.plot()
        plt.title('Cumulative portfolio return over time')
    return cum_return

def save_regression_results(storage, sample, pca_window, reg_window, pcs, save=False):
    '''
    :param storage: the output from a backtest, s-score dictionary
    :param sample: the sample used
    :param pca_window: length of window used for PCA
    :param reg_window: length of window used for regressions
    :param pcs:  number of pcs
    :return: saved csv file
    '''
    start = datetime.strftime(sample.iloc[0].name.date(),'%Y-%m-%d')
    end = datetime.strftime(sample.iloc[-1].name.date(),'%Y-%m-%d')
    s_results = pd.DataFrame.from_dict(storage, orient='index').transpose()
    s_results.index = sample.index
    s_results.index.name = 'Date'
    if save:
        s_results.to_csv(r'C:\Users\Bing\Documents\NumTech Ass 2\S-score results\{} - {} {} {} {} S_scores.csv'.format(start, end, pca_window, reg_window, pcs))
    return s_results

def plot_portfolio_composition(path):
    '''

    :param path: file path of the backtest to be plotted
    :return: plot
    '''
    results = pd.read_csv(path, index_col=0, parse_dates=True)
    q = results.apply(parse_s_scores, axis=0)
    new = q.T.apply(pd.value_counts).fillna(0)
    new = new.T
    new['long'] = new['hold long'] + new['close long']
    new['short'] = new['hold short'] + new['close short']
    new['bonds'] = new['hold'] + new['open long'] + new['open short']
    new = new.drop(['hold long','close long', 'hold short', 'close short', 'hold', 'open long', 'open short'],axis=1)
    new.index = new.index.strftime('%b %Y')
    new.iloc[::31].plot(kind='bar', stacked=True, title='Composition of portfolio over sample period', rot=0)
    return

def returns_to_profits(series, value=100):
    '''

    :param series: the returns series of a security
    :return: price at each point
    '''
    series.iloc[0] = value
    series = np.cumprod(series)
    return series


# # #  read in index-price data - comparing index with first eigen portfolio
# bt40 = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\Coinmetrics data\b40.csv", index_col=0, parse_dates=True)
# bt40 = bt40.rename(columns={'value': 'BT40'})
# bt40 = bt40.pct_change()
# bt40.name = 'BT40'
# # compare_plot(bt40, eig_returns['EP1'])

# eth = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\Bletchley Indexes\bletchley_ethereum_even.csv", index_col=0, parse_dates=True)
# eth = eth.rename(columns={'value': 'ETH'})
# eth = eth.pct_change()
# eth.name = 'ETH'


def plot(prices_df, ico_dates):
    '''
    Plots some graphs related to the passed data.
    '''
    to_plot = ['BTC','DOGE','DASH','ETH']
    plotting.plot_cryptos(prices_df, to_plot)
    plotting.plot_launches_per_year(ico_dates, prices_df)
    plotting.plot_timeline(ico_dates)

if __name__ == "__main__":
    starttime = time.time()
    coins_data, ico_dates = load_coin_data(PATH_TO_COINS_DATA)
    prices_df, volumes_df = build_price_and_volume_dataframe(coins_data)

    # Generate a dataframe with returns
    returns = prices_df.pct_change()
    returns.index.name = 'Date'
    print(returns.shape)
    # ========= PARAMETERS ============
    PCA_window = 160
    regression_window = 50
    pc_interval = 60
    PCS = 5
    volume = 1000000
    sample_window = 365 #147
    base_capital = 10000
    startdate = datetime(2018,1,1) # Choose the time period of the sample
    sample = make_sample(returns, volumes_df, startdate, volume, sample_window, PCA_window)
    print(sample)                 
    # capital_portion = base_capital / len(active)

    # plot_portfolio_composition(r"C:\Users\Bing\Documents\NumTech Ass 2\S-score results\In sample\160 50 5 S_scores.csv")
    eigen_portfolios, eigen_vals = do_pca(startdate, sample, PCA_window, PCS)
    print(eigen_portfolios)
    eig_returns = get_eigenportfolio_returns(datetime(2019,1,1), sample, eigen_portfolios, eigen_vals, PCS)
    print(eig_returns)
    # (eigen_portfolios, eigen_vals, returns, active, datetime(2019,1,1))
    
    # a = eigportfolios['PC2'].sort_values(ascending=False)
    # plt.figure(3)
    # plt.bar(a[:20].index, a[:20], width=0.5, color='slategrey', edgecolor='k')
    # plt.title('Second Eigenvector')
    # plt.show()

    # storage = backtest(PCA_window, regression_window, sample_window, PCS, returns, active, startdate, pc_interval)
    # results = save_regression_results(storage, sample, PCA_window, regression_window, PCS, save=False)
    # PnL = get_PnL(results)
    # portfolio_returns = results.apply(scores_to_returns, axis=0)
    # portfolio_returns.iloc[0] = capital_portion
    # portfolio_returns = portfolio_returns.apply(np.cumprod, axis=0)
    # portfolio_value_evolution = np.sum(portfolio_returns,axis=1)
    # plt.figure(10)
    # portfolio_value_evolution.plot(title='Portfolio value over time')
    # print('That took {} seconds'.format(time.time() - starttime))
    # sigs = parse_s_scores(results['XRP'])
    # display_trade(sigs, results['XRP'], df, startdate, 'XRP')
