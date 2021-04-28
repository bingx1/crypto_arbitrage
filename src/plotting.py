import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import Counter
import sklearn

def plot_timeline(ico_dates: dict):
 
 dates = [np.datetime64(date) for date in ico_dates.values()]
 symbols = ico_dates.keys()
 
 # Choose some nice levels
 levels = np.tile([-11,11,-9,9,-6, 6, -3, 3, -1, 1],
                  int(np.ceil(len(dates) / 10)))[:len(dates)]
 # Create figure and plot a stem plot with the date
 fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
 # ax.set(title=title,fontsize=25)
 fig.suptitle(t="Cryptocurrency ICO Timeline",fontsize=15)
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



def plot_cryptos(coins: pd.DataFrame, coins_to_plot: list):
    '''
    Constructs a plot of 4 subplots, where each subplot is the price chart of one of the
    passed cryptocurrencies.
    '''
    fig, ax = plt.subplots(figsize=(10, 8))
    axes = [plt.subplot2grid((2,2),(0,0)), plt.subplot2grid((2,2),(0,1)), 
            plt.subplot2grid((2,2),(1,0)), plt.subplot2grid((2,2),(1,1))]

    for axis, coin in zip(axes, coins_to_plot):
        coin_data = coins.loc[coins[coin].isna()==False, coin]
        axis.plot(coin_data)
        axis.set_title('{}:USD'.format(coin),fontsize=10)
        axis.get_xaxis().set_major_formatter(mdates.DateFormatter("%Y"))
        axis.set_xlim(coins.index[0], coins.index[-1])
    fig.suptitle(str(coins_to_plot).strip('[]')  + ' USD Pairs ' + str(coins.index[0].year) + '-' + str(coins.index[-1].year),fontsize=14)
    plt.show()


def plot_launches_per_year(ico_dates: dict, df: pd.DataFrame):
    '''
    Plots the number of cryptocurrencies launched in each year, and the number of cryptocurrencies
    in active circulation (per this dataset) in each year.
    '''
    counter = Counter([date[:4] for date in ico_dates.values()])
    in_order = sorted(counter.items())
    fig, ax = plt.subplots(figsize=(10, 4))
    line_ax = plt.subplot2grid((1, 2), (0, 0))
    bar_ax = plt.subplot2grid((1, 2), (0, 1))
    bar_ax.bar([i[0] for i in in_order],[i[1] for i in in_order],color='g',edgecolor='k')
    plt.title('# of Cryptocurrency launches by year', fontsize=14)
    df['Active'].plot(ax=line_ax, color='g')
    line_ax.set_title('Active Cryptocurrencies',fontsize=14)
    line_ax.set_xlabel('')
    plt.show()


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

def plot_pca(pca: sklearn.decomposition.PCA, PCS: int):
    exp_var = pca.explained_variance_ratio_
    cum_expvar = np.cumsum(exp_var)
    # plot variance explained graph
    print ('First principal component explains {} of the variance'.format(round(exp_var[0],4)))
    print ('Var explained by the 1st {} components:{}'.format(PCS,round(cum_expvar[-1],4)))
    plt.figure(5)
    plt.bar(['PC {}'.format(i) for i in range(1,PCS+1)], exp_var, color='lightsteelblue', edgecolor='k', width=0.6)
    plt.xticks(['PC {}'.format(i) for i in range(1,PCS+1)], rotation= 0)
    plt.plot(['PC {}'.format(i) for i in range(1,PCS+1)], cum_expvar, ls='--', color='cornflowerblue')
    plt.legend(['Cumulative variance explained','Proportion of variance explained'],fontsize=7)
    plt.title('Variance explained by PC 1 - PC {}'.format(PCS),fontsize = 13)
    plt.show()

def plot_eigenportfolio(eigen_portfolios: pd.DataFrame, to_plot: int):
    a = eigen_portfolios[f'PC{to_plot}'].sort_values(ascending=False)
    plt.figure(3)
    plt.bar(a[:20].index, a[:20], width=0.5, color='slategrey', edgecolor='k')
    plt.title(f'Eigenvector {to_plot}')
    plt.show()

def plot_pnl(cumulative_portfolio_return):
    plt.figure(2)
    cumulative_portfolio_return.plot()
    plt.title('Cumulative portfolio return over time')
    plt.show()

def plot_portfoliovalue(portfolio_value):
    plt.figure(10)
    portfolio_value.plot(title='Portfolio value over time')
    plt.show()