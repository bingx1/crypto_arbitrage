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
from strategy import make_sample, do_pca, get_eigenportfolio_returns, backtest

PATH_TO_COINS_DATA = "../data/coinmarketcap_data/"


def clean_coin_data(coin_data: pd.DataFrame) -> None:
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
    plt.show()

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

def scores_to_returns(y, open_long = -1.25, open_short = 1.25, close_long = -0.5, close_short = 0.75, annual_ir=0.015, slippage = 0.0005):
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


def get_PnL(results):
    ''''''
    portfolio_returns = results.apply(scores_to_returns, axis=0)
    portfolio_returns['Mean'] = np.mean(portfolio_returns, axis=1)
    cum_return = np.cumprod((portfolio_returns['Mean']))    
    return cum_return

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


def plot(prices_df, ico_dates):
    '''
    Plots some graphs related to the passed data.
    '''
    to_plot = ['BTC','DOGE','DASH','ETH']
    plotting.plot_cryptos(prices_df, to_plot)
    plotting.plot_launches_per_year(ico_dates, prices_df)
    plotting.plot_timeline(ico_dates)

def get_portfolio_value_evolution(initial_capital: int, signals: pd.DataFrame, n_coins: int):
    capital_portion = initial_capital / n_coins
    portfolio_returns = signals.apply(scores_to_returns, axis=0)
    portfolio_returns.iloc[0] = capital_portion
    portfolio_returns = portfolio_returns.apply(np.cumprod, axis=0)
    portfolio_value_evolution = np.sum(portfolio_returns,axis=1)
    return portfolio_value_evolution

if __name__ == "__main__":
    starttime = time.time()
    coins_data, ico_dates = load_coin_data(PATH_TO_COINS_DATA)
    prices_df, volumes_df = build_price_and_volume_dataframe(coins_data)

    # Generate a dataframe with returns
    returns = prices_df.pct_change()
    returns.index.name = 'Date'
    
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

    eigen_portfolios, eigen_vals = do_pca(startdate, sample, PCA_window, PCS)
    eig_returns = get_eigenportfolio_returns(datetime(2019,1,1), sample, eigen_portfolios, eigen_vals, PCS)

    output = backtest(PCA_window, regression_window, sample_window, PCS, sample, startdate, pc_interval)
    output.to_csv("output.csv")
    print(output)
    # plot_portfolio_composition(r"C:\Users\Bing\Documents\NumTech Ass 2\S-score results\In sample\160 50 5 S_scores.csv")
    # PnL = get_PnL(output)
    # plotting.plot_pnl(PnL)
    # portfolio_value_evolution = get_portfolio_value_evolution(base_capital, output, len(sample.columns))
    # plt.plot(portfolio_value_evolution)
    # Demonstrate the trading signals for LTC
    sigs = parse_s_scores(output['LTC'],  open_long = -1.25, open_short = 1.25, close_long = -0.5, close_short = 0.75)
    display_trade(sigs, output['LTC'], prices_df, startdate, 'LTC')