import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import os
import glob
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import time as time

def read_price_data(path, tl_plt = False, peryear_plt = False, coins_plt = []):
    '''
    :param path: path that contains all the csv files
    :param tl_pt: whether to plot the timeline or not
    :param peryear_plt: where to plot coins per year or not
    :param coins_plt: 4 coins to plot
    :return: dataframe
    '''
    coins = {}
    dates = []
    years = {2013: 0, 2014: 0, 2015: 0, 2016: 0, 2017: 0, 2018: 0, 2019: 0}
    last_vol = {}
    symbols = []
    # read in the price data
    for filename in glob.glob(os.path.join(path, '*.csv')):
        tick = filename.split('\\')[-1][:-4]
        # print (tick)
        coin_data = pd.read_csv(filename)
    #     clean up the data
        coin_data['time'] = pd.to_datetime(coin_data['time'])
        coin_data.index = pd.to_datetime(coin_data['time'])
    #     store the data into the dict
        coins[tick] = coin_data
    #     DO not include coins
        if 'value' in coin_data.columns:
            pass
        else:
    #     store the ticker and ICO date into coins and dates and VOLUME
            dates.append(coin_data.iloc[0]['time'])
            years[coin_data.iloc[0]['time'].year] += 1
            symbols.append(tick)
            last_vol[tick] = coin_data.iloc[-1]['volumeto']
    # build price and volume dataframe for all coins n
    df = coins['BTC'][['close']].copy()
    df = df.rename(columns={'close': 'BTC'})
    vol_df = coins['BTC'][['volumeto']].copy()
    vol_df = vol_df.rename(columns={'volumeto': 'BTC'})
    for coin in coins:
        if coin == 'BTC':
            pass
        # put all other coins into dataframe
        else:
            df[coin] = coins[coin]['close']
            vol_df[coin] = coins[coin]['volumeto']
    # INSERT column 'Active' into dataframe to count active coins at each date
    df['Active'] = df.count(axis=1)
    return df, vol_df


def get_sample(startdate, df, vol_df, volume, length, pcawindow):
    '''
    :param startdate: first day of sample
    :param df: entire returns dataframe
    :param vol_df:  entire volume dataframe
    :param liquidity: volume/liquidity requirement throughout the sample
    :param length: length of the sample
    :param pcawindow: trailing window to conduct PCA
    :return: the sample returns data
    '''
    enddate = startdate + timedelta(days=length)
    earliest_date = startdate - timedelta(days=pcawindow)
    # Create the sample, dropping the 'active' column
    sample = df.loc[(df.index >= startdate) & (df.index <= enddate)].dropna(axis=1).copy().drop('Active',axis=1)
    print ('Dataset contains price data for {} coins between {} and {}'.format(len(sample.columns),startdate.date(), enddate.date()))
    # Check which coins in the sample have traded for 1 > year since the beginning of the sample period
    valid = df.loc[df.index == earliest_date].dropna(axis=1).columns.to_list()
    print ('Of these, {} have traded since {}'.format(len(valid), earliest_date.date()))
    # Drop coins that haven't traded for 365 days prior to start of sample
    sample = sample.drop([coin for coin in sample.columns if coin not in valid],axis=1)
    # Report coins that satisfy some volume constraint throughout the whole sample period
    store = []
    for coin in sample.columns:
        validdata = np.sum(vol_df.loc[(vol_df.index >= startdate) & (vol_df.index <= enddate),coin] > volume)
        store.append(validdata)
    vol_check = pd.DataFrame(store, index=sample.columns,columns = ['Valid'])
    active = [tick for tick in sample.columns if vol_check.loc[tick]['Valid'] == vol_check.max()['Valid']]
    print ('Out of {} coins, daily volume is > {} between {} and {} for {} of them.'.format
           (len(sample.columns), volume,startdate.date(),enddate.date(),len(active)))
    sample = sample.drop([coin for coin in sample.columns if coin not in active],axis=1)
    return sample, active

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
    signals = parse_s_scores(y, open_long, open_short, close_long, close_short)
    coin = signals.name
    signals.name = 'Signals'
    daily_ir = annual_ir/365
    combined = pd.concat([signals,sample[coin]],axis=1)
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

def get_PnL(results, capital_portion = 10000):
    ''''''
    capital_portion = capital_portion / len(results.columns)
    portfolio_returns = results.apply(scores_to_returns, axis=0)
    portfolio_returns.iloc[0] = capital_portion
    portfolio_returns = portfolio_returns.apply(np.cumprod, axis=0)
    portfolio_value_evolution = np.sum(portfolio_returns,axis=1)

    return portfolio_value_evolution

df, vol_df = read_price_data(r'C:\Users\Bing\Documents\NumTech Ass 2\Coinmarketcap data')
returns = df.pct_change()
returns.index.name = 'Date'
sample, active = get_sample(datetime(2019,1,1), returns, vol_df, 10000, 147, 160)
enter_long = {}
capital_portion = 10000/len(active)
results = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\S-score results\Out of sample\2019-01-01 - 2019-05-28 160 50 5 60 S_scores.csv", index_col=0, parse_dates=True)
for i in range(20,80,5):
    i = i/100
    portfolio_returns = results.apply(scores_to_returns, axis=0, args=[-1.4,1.05,-0.5,i])
    portfolio_returns.iloc[0] = capital_portion
    portfolio_returns = portfolio_returns.apply(np.cumprod, axis=0)
    portfolio_value_evolution = np.sum(portfolio_returns, axis=1)
    max_drawdown = np.min(portfolio_value_evolution.diff(1))
    std = np.std(portfolio_value_evolution.pct_change())*np.sqrt(365)
    ret = (portfolio_value_evolution.iloc[-1]-portfolio_value_evolution.iloc[0])/portfolio_value_evolution.iloc[0]
    sharpe = (ret - 0.015)/ std
    print (ret,std,max_drawdown,sharpe)
    enter_long[i] = (ret,std,max_drawdown, sharpe)

x = pd.DataFrame.from_dict(enter_long).T
x.index.name = 's_exit_short'
x.columns= ['Annual Return','Standard deviation of returns','Maximum Drawdown','Sharpe Ratio']
# x.to_csv(r'C:\Users\Bing\Documents\NumTech Ass 2\exit_short.csv')
# dic = {}
# for filename in glob.glob(os.path.join(r'C:\Users\Bing\Documents\NumTech Ass 2\S-score results\In sample\Regression windows for PCA window of 160 and 5 PCS', '*.csv')):
#     tick = filename.split('\\')[-1][:-4]
#     params = tick.split(' ')
#     pca_window = params[3]
#     reg_window = params[4]
#     pcs = params[5]
#     # pca_interval = params[6]
#     results = pd.read_csv(filename, index_col=0, parse_dates=True)
#     i = get_PnL(results)
#     b = i[-1]
#     dic['{} {} {}'.format(pca_window,reg_window,pcs)] = i
#
# for i in dic:
#     print (i, dic[i].iloc[-1])
