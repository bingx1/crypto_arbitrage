from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from logger import Logger

logger = Logger("master")
# ================= ANALYSIS OF THE SAMPLE ============================
def make_sample(returns_df, volume_df, sample_start_date, min_volume, sample_length, pca_window):
    '''
    Retrieves a portion of the data to use to test the trading strategy.
    Function parameters determine the characteristics of the sample.
    '''
    # We need data between the below first and last dates. Data prior to the start of the sample is needed for PCA.
    last_date = sample_start_date + timedelta(days=sample_length)
    first_date = sample_start_date - timedelta(days=pca_window)

    # Make sample
    sample = returns_df.loc[(returns_df.index >= first_date) & (returns_df.index <= last_date)].copy().drop('Active',axis=1)
    # Remove coins for which there is not data for the whole period.
    sample = sample.dropna(axis=1)
    logger.log('BUILDING SAMPLE - Sample contains price data for {} coins between {} and {}'.format(len(sample.columns), sample_start_date.date(), last_date.date()))

    # Keep coins that meet the specified trading volume requirement
    volumes_sample = volume_df.loc[(volume_df.index >= first_date) & (volume_df.index <= last_date), sample.columns]
    total_volumes = volumes_sample.sum(axis = 0)

    sample = sample.drop([coin for coin in sample.columns if total_volumes[coin] < min_volume], axis=1)
    logger.log('BUILDING SAMPLE - After dropping coins that did not meet the volume requirement of {} there are now {} coins'.format(min_volume, len(sample.columns)))
    
    return sample



def do_pca(date, returns_df, pcawindow, PCS, active):
    '''
    Do pca on the returns_df inputted. Return the eigenvectors corresponding to the first 'Number_factors' principal
    components
    :param date: date from which to conduct PCA
    :param returns_df: ENTIRE returns returns_df
    :param pcawindow: length of PCA window
    :param PCS: number of eigenvectors to return
    :param active: list of active cryptocurrencies in the SAMPLE
    :return: A returns_df of eigenvectors
    '''
    pcawindow_start = date - timedelta(days=pcawindow)
    # Might be incorrect making the date inclusive at both ends.
    pca_data = returns_df.loc[(returns_df.index >= pcawindow_start) & (returns_df.index <= date), active]
    stdev_returns = pca_data.std(ddof=1, axis=0)
    # Standardise the data
    sample = StandardScaler().fit_transform(pca_data)
    # Call PCA function to do PCA
    pca = PCA(n_components=PCS)
    transformed = pca.fit_transform(sample)
    
    pcdf = pd.returns_df(pca.components_.T, columns=['PC{}'.format(i) for i in range(1,PCS+1)], index=pca_data.columns)
    eigen_values = pca.explained_variance_
    # Multiply rows by 1/ STDEV of each coin return to get eigen portfolio weights
    eigen_portfolios = pcdf.div(stdev_returns, axis=0)

    return eigen_portfolios, eigen_values

def get_eigenportfolio_returns(eigen_portfolios, eigen_values, returns_df, active, date, n_pcs):
    '''
    Retrieve the historical trailling window eigenportfolio returns
    :param eigen_portfolios: eigen portfolio weights
    :param eigen_values: eigenvalues of eigenportoflios
    :param returns_df: entire returns_df
    :param active: list of coins in the sample
    :param date:  date from which to retrieve data from
    :return: series of eigenportfolio returns
    '''
    pca_data = returns_df.loc[(returns_df.index <= date), active]
    # Eigen portfolio's are the columns. Add returns for each eigen portfolio of interest to pca_data
    eig_ports = ['EP{}'.format(i) for i in range(1,n_pcs+1)]
    for i in range(1, n_pcs + 1):
        pc = 'PC{}'.format(i)
        ep = 'EP{}'.format(i)
        pca_data[ep] = ((np.sum(pca_data[active].multiply(eigen_portfolios[pc].to_list()), axis=1)) * -1) / (eigen_values[i - 1])
    return pca_data[eig_ports]


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