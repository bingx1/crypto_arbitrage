import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functools import reduce

matplotlib.use('TkAgg')

PATH_TO_STOCKS_DATA = "../data/stocks/individual_stocks_5yr"
PATH_TO_SP500_DATA = "../data/coinmetrics_data/sp500.csv"

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
    # df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), stock_prices)
    df = pd.concat(stock_prices, axis = 1)
    returns = df.pct_change()
    returns.drop(df.head(1).index, inplace=True)
    before = returns.shape
    returns = returns.dropna(axis = 1)
    print("READING IN STOCK DATA - Dropped columns with NaNs, ", before, " -> ", returns.shape)
    return returns


def perform_pca(returns_df: pd.DataFrame, n):
    '''
    Uses the sklearn library to perform principle components analysis on the stocks.
    '''
    stocks = returns_df.columns
    stdev_returns = returns_df.std(ddof=1, axis=0)
    # Standardise the data
    scaled = StandardScaler().fit_transform(returns_df)
    # Conduct principle components analysis and project onto principle components
    pca = PCA(n_components=len(returns_df.columns))
    print(scaled.shape)
    transformed = pca.fit_transform(scaled)

    eigenvalues = pca.explained_variance_
    pc_df = pd.DataFrame(pca.components_, columns = ['PC{}'.format(i) for i in range(1,len(returns_df.columns)+1)], index = returns_df.columns)
    # Divide rows by STDEV of each coin return to get eigen portfolio weights
    eigenportfolios = pc_df.div(stdev_returns, axis=0)
    # The columns of 'eigenportfolios' are the relevant eigenportfolios. 
    # Lets add the returns of the first n eigenportfolios to the returns dataframe.
    for i in range(1, n+1):
        pc = 'PC{}'.format(i)
        returns_df[pc] = np.sum(returns_df[stocks].multiply(eigenportfolios[pc].to_list()),axis=1)/np.sqrt(eigenvalues[i-1])
    return returns_df

def load_sp500data(fpath: str) -> pd.DataFrame:
    '''
    Reads in the S&P500 daily price data and returns a pandas dataframe.
    '''
    sp500 = pd.read_csv(fpath)
    sp500.index = pd.to_datetime(sp500['date'])
    sp500.pop('date')
    sp500 = sp500.pct_change()
    return sp500


# Not used.
def add_sp500data(returns_df: pd.DataFrame, fpath: str) -> pd.DataFrame:
    sp500_df = load_sp500data(fpath)
    join = pd.concat([returns_df['PC1'], sp500_df],axis=0)
    join = join.rename(columns={0:'PC1','value':'SP500'})
    return join



def plot_and_compare(returns_df: pd.DataFrame, sp500_df: pd.DataFrame):
    '''
    Plots the daily returns from the first eigenportfolio next to the returns of the S&P500.
    '''
    # fig, ax = plt.subplots(figsize=(10, 4))
    return_ax = plt.subplot2grid((1, 2), (0, 0))
    eig_ax = plt.subplot2grid((1, 2), (0, 1))
    first_date = np.datetime64(returns_df.index[0].date())
    last_date = np.datetime64(returns_df.index[-1].date())
    print("PLOTTING - Comparing returns for the period from ", first_date, " to ", last_date)
    sp500_df = sp500_df.loc[(sp500_df.index >= first_date) & (sp500_df.index <= last_date)].plot(ax=return_ax, title='Returns on Index')
    returns_df['PC1'].plot(ax=eig_ax, title='Returns on PC1')
    plt.show(block=True)
    


def main():
    '''
    Entrypoint into the application
    '''
    stock_data = load_stock_data(PATH_TO_STOCKS_DATA)
    returns_df = build_returns_dataframe(stock_data)
    returns_df = perform_pca(returns_df, 15)
    print(returns_df.head())
    sp500_df = load_sp500data(PATH_TO_SP500_DATA)
    plot_and_compare(returns_df, sp500_df)


if __name__ == "__main__":
    main()