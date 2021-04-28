import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_timeline(dates, symbols, title):
 # Choose some nice levels
 levels = np.tile([-11,11,-9,9,-6, 6, -3, 3, -1, 1],
                  int(np.ceil(len(dates) / 10)))[:len(dates)]
 # Create figure and plot a stem plot with the date
 fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
 # ax.set(title=title,fontsize=25)
 fig.suptitle(t=title,fontsize=15)
 markerline, stemline, baseline = ax.stem(dates, levels, linefmt="g-", basefmt="k-")
 plt.setp(markerline, mec="k", mfc="w", zorder=3)
 # Shift the markers to the baseline by replacing the y-data by zeros.
 markerline.set_ydata(np.zeros(len(dates)))
 # annotate lines
 current = "left"
 vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
 for d, l, r, va in zip(dates, levels, symbols, vert):
  ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l) * 3),
              textcoords="offset points", va=va, ha=current, fontsize=8)
  if current == "left":
   current = "right"
  else:
   current = "left"
 # format xaxis with 4 month intervals
 ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=4))
 ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
 plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
 # remove y axis and spines
 ax.get_yaxis().set_visible(False)
 for spine in ["left", "top", "right"]:
  ax.spines[spine].set_visible(False)
 ax.margins(y=0.1)
 plt.show()
 return

def plot_cryptos(coins, c1,c2,c3,c4):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax3 = plt.subplot2grid((2,2),(1,0))
    ax4 = plt.subplot2grid((2,2),(1,1))
    ax1.plot(coins[c1]['date'],coins[c1]['price(USD)'],'g-')
    ax1.set_title('{}:USD'.format(c1),fontsize=10)
    dates = [datetime(2013,1,1),datetime(2014,1,1),datetime(2015,1,1), datetime(2016,1,1),datetime(2017,1,1), datetime(2018,1,1), datetime(2019,1,1)]
    ax1.set_xticks(dates)
    ax2.plot(coins[c2]['date'],coins[c2]['price(USD)'],'g-')
    ax2.set_title('{}:USD'.format(c2),fontsize=10)
    ax2.set_xticks(dates)
    ax3.plot(coins[c3]['date'],coins[c3]['price(USD)'],'g-')
    ax3.set_title('{}:USD'.format(c3),fontsize=10)
    ax3.set_xticks(dates)
    ax4.plot(coins[c4]['date'],coins[c4]['price(USD)'],'g-')
    ax4.set_title('{}:USD'.format(c4),fontsize=10)
    ax4.set_xticks(dates)
    fig.suptitle('Ripple, Bitcoin, Dash, Ethereum USD Pairs 2013-2019',fontsize=14)
    return

def plot_launches_per_year(years, df):
    '''

    :param years:     # Plot the number of currencies launched in each year
    :param df: dataframe of all the price data
    :return: nothing, just a diagram
    '''
    fig, ax = plt.subplots(figsize=(10, 4))
    line_ax = plt.subplot2grid((1, 2), (0, 0))
    bar_ax = plt.subplot2grid((1, 2), (0, 1))
    bar_ax.bar(years.keys(),years.values(),color='g',edgecolor='k')
    plt.title('# of Cryptocurrency launches by year', fontsize=14)
    df['Active'].plot(ax=line_ax, color='g')
    line_ax.set_title('Active Cryptocurrencies',fontsize=14)
    line_ax.set_xlabel('')
    return

def plot_coin(df, coin):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax1 = plt.subplot2grid((1,2),(0,0))
    ax2 = plt.subplot2grid((1,2),(0,1))
    df[coin].dropna().plot(ax=ax1, title = '{}:USD Price'.format(coin))
    df[coin].dropna().pct_change().plot(ax=ax2, title = '{} Returns'.format(coin))
    return

def compare_plot(ind,ep):
    '''
    :param ind: Index (Bletchley crypto index) returns [series]
    :param ep: Eigenportfolio returns [series]
    :return: Subplots, index returns on the left, eigenportfolio on the right
    '''
    combined = pd.concat([ind,ep],axis=1)
    combined = combined.dropna()
    # Plotting the return of the eigen portfolio vs the return of the BT40 Crypto index
    fig, ax = plt.subplots(figsize=(10, 4))
    return_ax = plt.subplot2grid((1, 2), (0, 0))
    eig_ax = plt.subplot2grid((1, 2), (0, 1))
    combined[ind.name].plot(ax=return_ax, title='Returns on {} Cryto Index'.format(ind.name),color='cornflowerblue')
    combined[ep.name].plot(ax=eig_ax, title='Returns on Eigenportfolio {}'.format(ep.name),color='cornflowerblue')
    return