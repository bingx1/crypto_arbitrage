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

all_data = {}
for filename in glob.glob(os.path.join(r'C:\Users\Bing\Documents\NumTech Ass 2\Stocks\individual_stocks_5yr', '*.csv')):
    tick = filename.split('\\')[-1][:-9]
    print (tick)
    data = pd.read_csv(filename)
#     clean up the data
    data.index = pd.to_datetime(data['date'])
#     store the data into the dict#
    all_data[tick]=data

df = all_data['A'][['close']].copy()
df = df.rename(columns={'close':'A'})
for stock in all_data:
    if stock == 'A':
        pass
    # put all other coins into dataframe
    else:
        df[stock] = all_data[stock]['close']
returns = df.pct_change()
for stock in all_data:
    if np.sum(np.isnan(returns[stock])) > 1:
        returns.pop(stock)
returns = returns.dropna()
valid = returns.columns
s = returns.std(ddof=1, axis=0)
# PCA
sample2 = StandardScaler().fit_transform(returns)
# Call PCA function to do PCA
pcs = ['PC{}'.format(i) for i in range(1,len(returns.columns)+1)]
pca = PCA(n_components=len(returns.columns))
pca2 = pca.fit_transform(sample2)
eigenvalues = pca.explained_variance_
pcdf = pd.DataFrame(pca.components_, columns=pcs, index=returns.columns)
# Divide rows by STDEV of each coin return to get eigen portfolio weights
eig_portfolios = pcdf.div(s,axis=0)
# Eigen portfolio's are the columns. Add returns for each eigen portfolio of interest to pca_data
for i in range (1,15+1):
    pc = 'PC{}'.format(i)
    returns[pc] = np.sum(returns[valid].multiply(eig_portfolios[pc].to_list()),axis=1)/np.sqrt(eigenvalues[i-1])
sp500 = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\Coinmetrics data\sp500.csv")
sp500.index = pd.to_datetime(sp500['date'])
sp500.pop('date')
join = pd.concat([returns['PC1'],sp500.pct_change()],axis=0)
join = join.rename(columns={0:'PC1','value':'SP500'})
fig, ax = plt.subplots(figsize=(10, 4))
return_ax = plt.subplot2grid((1, 2), (0, 0))
eig_ax = plt.subplot2grid((1, 2), (0, 1))
sp500 = sp500.pct_change()
sp500.loc[(sp500.index >= datetime(2013,2,11)) & (sp500.index <= datetime(2018,2,7))].plot(ax=return_ax, title='Returns on Index')
returns['PC1'].plot(ax=eig_ax, title='Returns on PC1')