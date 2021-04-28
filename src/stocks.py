import pandas as pd
from pandas.core.indexes.base import Index
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import os
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functools import reduce

PATH_TO_STOCKS_DATA = "../data/stocks/individual_stocks_5yr"


def load_stock_data(fpath: str) -> list:
    '''
    Returns a list of pandas series - index is the date, values are the stock price data and name is the stock's ticker
    '''
    price_data = []
    for filename in glob.glob(os.path.join(fpath, '*.csv')):
        fname = filename.split('/')[-1]
        ticker = fname.split('_')[0]
        # print(fname, ticker)
        data = pd.read_csv(filename)
        data.index = pd.to_datetime(data['date'])
        price_data.append(data['close'].rename(ticker))
    return price_data

def build_returns_dataframe(stock_prices: list) -> pd.DataFrame:
    '''
    Converts the list of stock prices into a single pandas dataframe containing returns.
    Also cleans the pandas dataframe by removing missing values.
    '''
    df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), stock_prices)
    returns = df.pct_change()
    returns = returns.dropna(axis = 0)
    return returns



# valid = returns.columns
# s = returns.std(ddof=1, axis=0)
# # PCA
# sample2 = StandardScaler().fit_transform(returns)
# # Call PCA function to do PCA
# pcs = ['PC{}'.format(i) for i in range(1,len(returns.columns)+1)]
# pca = PCA(n_components=len(returns.columns))
# pca2 = pca.fit_transform(sample2)
# eigenvalues = pca.explained_variance_
# pcdf = pd.DataFrame(pca.components_, columns=pcs, index=returns.columns)
# # Divide rows by STDEV of each coin return to get eigen portfolio weights
# eig_portfolios = pcdf.div(s,axis=0)
# # Eigen portfolio's are the columns. Add returns for each eigen portfolio of interest to pca_data
# for i in range (1,15+1):
#     pc = 'PC{}'.format(i)
#     returns[pc] = np.sum(returns[valid].multiply(eig_portfolios[pc].to_list()),axis=1)/np.sqrt(eigenvalues[i-1])
# sp500 = pd.read_csv(r"C:\Users\Bing\Documents\NumTech Ass 2\Coinmetrics data\sp500.csv")
# sp500.index = pd.to_datetime(sp500['date'])
# sp500.pop('date')
# join = pd.concat([returns['PC1'],sp500.pct_change()],axis=0)
# join = join.rename(columns={0:'PC1','value':'SP500'})
# fig, ax = plt.subplots(figsize=(10, 4))
# return_ax = plt.subplot2grid((1, 2), (0, 0))
# eig_ax = plt.subplot2grid((1, 2), (0, 1))
# sp500 = sp500.pct_change()
# sp500.loc[(sp500.index >= datetime(2013,2,11)) & (sp500.index <= datetime(2018,2,7))].plot(ax=return_ax, title='Returns on Index')
# returns['PC1'].plot(ax=eig_ax, title='Returns on PC1')


if __name__ == "__main__":
    stock_data = load_stock_data(PATH_TO_STOCKS_DATA)
    returns_df = build_returns_dataframe(stock_data)
    print(returns_df.head())