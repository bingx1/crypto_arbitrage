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

 def plot_cryptos(btc,ltc,dash,eth):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax3 = plt.subplot2grid((2,2),(1,0))
    ax4 = plt.subplot2grid((2,2),(1,1))
    ax1.plot(btc['date'],btc['price(USD)'],'g-')
    ax1.set_title('BTC:USD',fontsize=10)
    ax1.set_xticks([datetime(2013,1,1),datetime(2014,1,1),datetime(2015,1,1)
                    ,datetime(2016,1,1),datetime(2017,1,1),
                    datetime(2018,1,1),datetime(2019,1,1),datetime(2019,1,1)])
    ax2.plot(ltc['date'],ltc['price(USD)'],'g-')
    ax2.set_title('LTC:USD',fontsize=10)
    ax2.set_xticks([datetime(2013,1,1),datetime(2014,1,1),datetime(2015,1,1)
                    ,datetime(2016,1,1),datetime(2017,1,1),
                    datetime(2018,1,1),datetime(2019,1,1),datetime(2019,1,1)])
    ax3.plot(dash['date'],dash['price(USD)'],'g-')
    ax3.set_title('DASH:USD',fontsize=10)
    ax3.set_xticks([datetime(2013,1,1),datetime(2014,1,1),datetime(2015,1,1)
                    ,datetime(2016,1,1),datetime(2017,1,1),
                    datetime(2018,1,1),datetime(2019,1,1)],datetime(2019,1,1))
    ax4.plot(eth['date'],eth['price(USD)'],'g-')
    ax4.set_title('ETH:USD',fontsize=10)
    ax4.set_xticks([datetime(2013,1,1),datetime(2014,1,1),datetime(2015,1,1)
                    ,datetime(2016,1,1),datetime(2017,1,1),
                    datetime(2018,1,1),datetime(2019,1,1),datetime(2019,1,1)])
    fig.suptitle('Ripple, Bitcoin, Dash, Ethereum USD Pairs 2013-2019',fontsize=14)
    return

def plot_coin(df, coin):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax1 = plt.subplot2grid((1,2),(0,0))
    ax2 = plt.subplot2grid((1,2),(0,1))
    df[coin].dropna().plot(ax=ax1, title = '{}:USD Price'.format(coin))
    df[coin].dropna().pct_change().plot(ax=ax2, title = '{} Returns'.format(coin))
    return
# plot_timeline(dates2017,symbols2017,'Cryptocurrency ICOs in 2017')
# plot_timeline(dates,symbols,'2013-2019 Cryptocurrency ICO Dates')
# set up storage variables
coins = {}
dates = []
years = {2013:0,2014:0,2015:0,2016:0,2017:0,2018:0,2019:0}
last_vol = {}
symbols = []

# read in the price data
for filename in glob.glob(os.path.join(r'C:\Users\Bing\Documents\NumTech Ass 2\Coinmarketcap data', '*.csv')):
    tick = filename.split('\\')[-1][:-4]
    # print (tick)
    coin_data = pd.read_csv(filename)
#     clean up the data
#     coin_data = coin_data.dropna()
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
#  read in index-price data
bt40 = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\Coinmetrics data\b40.csv")
bt40.index = pd.to_datetime(bt40['date'])
bt40.pop('date')
bt40 = bt40.rename(columns={'value':'BT40'})
bt40 = bt40.pct_change()

# build price and volume dataframe for all coins n
df = coins['BTC'][['close']].copy()
df = df.rename(columns={'close':'BTC'})
vol_df = coins['BTC'][['volumeto']].copy()
vol_df = vol_df.rename(columns={'volumeto':'BTC'})
for coin in coins:
    if coin == 'BTC':
        pass
    # put all other coins into dataframe
    else:
        df[coin] = coins[coin]['close']
        vol_df[coin] = coins[coin]['volumeto']
# INSERT column 'Active' into dataframe to count active coins at each date
df['Active'] = df.count(axis=1)

# Plot the number of currencies launched in each year
# fig, ax = plt.subplots(figsize=(10, 4))
# line_ax = plt.subplot2grid((1, 2), (0, 0))
# bar_ax = plt.subplot2grid((1, 2), (0, 1))
# bar_ax.bar(years.keys(),years.values(),color='g',edgecolor='k')
# plt.title('# of Cryptocurrency launches by year', fontsize=14)
# df['Active'].plot(ax=line_ax, color='g')
# line_ax.set_title('Active Cryptocurrencies',fontsize=14)
# line_ax.set_xlabel('')
#  Plot the prices of 4 coins and also the timeline of crypto ICOs
# plot_cryptos(coins['btc'],coins['ltc'],coins['dash'],coins['eth'])
# plot_timeline(dates,symbols,'2013-2019 Cryptocurrency ICO Dates')

# ========= PARAMETERS ============
PCA_window = 250
regression_window = 60
PCS = 15
# Generate a dataframe with returns
returns = df.pct_change()
# Choose the time period of the sample
startdate = datetime(2018,6,1)
enddate = startdate + timedelta(days=365)
earliest_date = startdate - timedelta(days = PCA_window)
# ================= ANALYSIS OF THE SAMPLE ============================
x = df.loc[df.index == startdate].dropna(axis=1).columns.to_list()
# Remove 'Active' column
x.pop()
print ('Dataset contains price data for {} coins between {} and {}'.format(len(x),startdate.date(), enddate.date()))
# Create the sample, dropping the 'active' column
sample = returns.loc[(returns.index >= startdate)].dropna(axis=1).copy().drop('Active',axis=1)
# Check which coins in the sample have traded for 1 > year since the beginning of the sample period
valid = returns.loc[returns.index == earliest_date].dropna(axis=1).columns.to_list()
print ('Of these, {} have traded since {}'.format(len(valid),earliest_date))
# Drop coins that haven't traded for 365 days prior to start of sample
sample = sample.drop([coin for coin in sample.columns if coin not in valid],axis=1)
# Report coins that satisfy some volume constraint throughout the whole sample period
volume = 10000
store = []
for coin in sample.columns:
    validdata = np.sum(vol_df.loc[(vol_df.index >= startdate) & (vol_df.index <= enddate),coin] > volume)
    store.append(validdata)
vol_check = pd.DataFrame(store, index=sample.columns,columns = ['Valid'])
active = [tick for tick in sample.columns if vol_check.loc[tick]['Valid'] == vol_check.max()['Valid']]
print ('Out of {} coins, daily volume is > {} between {} and {} for {} of them.'.format
       (len(sample.columns),volume,startdate.date(),enddate.date(),len(active)))
sample = sample.drop([coin for coin in sample.columns if coin not in active],axis=1)
# -------------Manually calculate PCA------------
# Use the 1-year rolling window to calculate PCA
date = startdate
pcawindow_start = startdate - timedelta(days=PCA_window)
pca_data = returns.loc[(returns.index >= pcawindow_start)&(returns.index < date), active].copy()
m = pca_data.mean(axis=0)
s = pca_data.std(ddof=1, axis=0)
stdevs = s.sort_values()
outlier_sd = 5
check = stdevs.loc[stdevs > outlier_sd]
# normalised time-series as an input for PCA
dfPort = (pca_data - m) / s
# ----- USE THE Standardised returns ------------
c = np.cov(dfPort.values.T)  # covariance matrix
# co = np.corrcoef(sample.values.T)  # correlation matrix
co = np.corrcoef(dfPort.values.T)
# perform PCA
eigvals, eigvecs = np.linalg.eig(c)
pcs = ['PC{}'.format(i) for i in range(1,len(sample.columns)+1)]
manual_results = pd.DataFrame(eigvecs,index=sample.columns,columns=pcs)
var_exp = [i/np.sum(eigvals) for i in eigvals]
plt.figure(4)
plt.bar(pcs[:PCS], var_exp[:PCS],color='lightsteelblue',edgecolor='k')
plt.xticks(pcs[:PCS],rotation=-90)
plt.plot(pcs[:PCS],np.cumsum(var_exp)[:PCS],ls='--',color='cornflowerblue')
plt.legend(['Cumulative variance explained','Proportion of variance explained'],fontsize=7)
plt.title('{} - {}: Variance explained by PC1-PC{}'.format(pcawindow_start.date(), date.date(), PCS),fontsize = 13)
print ('Var explained by the 1st principal component: {}'.format(var_exp[0]))
print ('Manual: Var explained by the 1st {} components:{}'.format(PCS, np.sum(var_exp[:PCS])))
# # ---------Use SciKit learn--------------
# standardise the returns
sample2 = StandardScaler().fit_transform(pca_data)
# Call PCA function to do PCA
pca = PCA(n_components=PCS)
pca2 = pca.fit_transform(sample2)
pcdf = pd.DataFrame(pca.components_.T, columns=pcs[:PCS], index=sample.columns)
# plot variance explained graph
plt.figure(5)
exp_var = pca.explained_variance_ratio_
cum_expvar = np.cumsum(exp_var)
print ('Sklearn: Var explained by the 1st {} components:{}'.format(PCS,cum_expvar[PCS-1]))
plt.bar(pcs[:PCS],exp_var[:PCS],color='green',edgecolor='k')
plt.xticks(pcs[:PCS],rotation=-90)
plt.plot(pcs[:PCS],cum_expvar[:PCS],ls='--',color='olivedrab')
plt.legend(['Cumulative variance explained','Proportion of variance explained'],fontsize=7)
plt.title('{} - {}: Variance explained by PC1-PC{}'.format(pcawindow_start.date(), date.date(), PCS),fontsize = 13)
# --------------------------Make eigenportfolios------------------------------------------
def do_pca(date, dataframe, pcawindow, number_factors):
    '''
    Do pca on the dataframe inputted. Return the eigenvectors corresponding to the first 'Number_factors' principal
    components
    '''
# Multiply rows by 1/STDEV of each coin return to get eigen portfolio weights
eig_portfolios = pcdf.div(s,axis=0)
# Eigen portfolio's are the columns. Add returns for each eigen portfolio of interest to pca_data
for i in range (1,PCS+1):
    pc = 'PC{}'.format(i)
    pca_data[pc] = ((np.sum(pca_data[active].multiply(eig_portfolios[pc].to_list()),axis=1))*-1)/np.sqrt(eigvals[i-1])
# Plotting the return of the first eigen portfolio vs the return of the BT40 Crypto index
fig, ax = plt.subplots(figsize=(10, 4))
return_ax = plt.subplot2grid((1, 2), (0, 0))
eig_ax = plt.subplot2grid((1, 2), (0, 1))
bt40.loc[(bt40.index >= pcawindow_start)&(bt40.index < date), 'BT40'].plot(ax=return_ax, title='Returns on BT40 Cryto Index')
pca_data.loc[(pca_data.index >= datetime(2017,11,19)) &(pca_data.index < date), 'PC1' ].plot(ax=eig_ax, title='Returns on 1st Eigenportfolio')
# ----------- Regress each coin's returns on eigenportfolio returns
def regress(x, y):
    '''x: dependent variables, Eigenportfolio returns, Y: coin returns'''
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()


