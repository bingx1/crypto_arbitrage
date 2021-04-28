from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from logger import Logger
import statsmodels.api as sm
from multiprocessing import Pool
import os, time

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
    sample = returns_df.loc[(returns_df.index >= first_date) & (
        returns_df.index <= last_date)].copy().drop('Active', axis=1)
    # Remove coins for which there is not data for the whole period.
    sample = sample.dropna(axis=1)
    logger.log('BUILDING SAMPLE - Sample contains price data for {} coins between {} and {}'.format(
        len(sample.columns), sample_start_date.date(), last_date.date()))

    # Keep coins that meet the specified trading volume requirement
    volumes_sample = volume_df.loc[(volume_df.index >= first_date) & (
        volume_df.index <= last_date), sample.columns]
    total_volumes = volumes_sample.sum(axis=0)

    sample = sample.drop(
        [coin for coin in sample.columns if total_volumes[coin] < min_volume], axis=1)
    logger.log('BUILDING SAMPLE - After dropping coins that did not meet the volume requirement of {} there are now {} coins'.format(min_volume, len(sample.columns)))

    return sample


def do_pca(date, sample, pca_window, n_pcs):
    '''
    Perform PCA using sklearn with the given parameters. 
    Returns the resulting eigenportfolios and eigenvalues.
    '''
    window_start = date - timedelta(days=pca_window)
    logger.log('PERFORMING PCA - Using {} principle components and data from {} to {}, inclusive.'.format(
        n_pcs, window_start.date(), date.date()))
    # Might be incorrect making the date inclusive at both ends.
    pca_data = sample.loc[(sample.index >= window_start)
                          & (sample.index <= date)]
    stdev_returns = pca_data.std(ddof=1, axis=0)

    # Standardise the data
    sample = StandardScaler().fit_transform(pca_data)

    # Call PCA function to do PCA
    pca = PCA(n_components=n_pcs)
    transformed = pca.fit_transform(sample)

    pcdf = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(
        i) for i in range(1, n_pcs+1)], index=pca_data.columns)
    # Multiply rows by 1/ STDEV of each coin return to get eigenportfolio weights
    eigen_portfolios = pcdf.div(stdev_returns, axis=0)
    eigen_values = pca.explained_variance_
    return eigen_portfolios, eigen_values


def get_eigenportfolio_returns(date, sample, eigen_portfolios, eigen_values, n_pcs):
    '''
    Returns the historical returns of the passed eigenportfolios. 
    '''
    returns_data = sample.loc[(sample.index <= date)]
    logger.log(
        f'CALCULATING - Calculating eigenportfolio returns to {date.date()}')
    eigenportfolio_returns = []
    for i in range(1, n_pcs + 1):
        ep_returns = ((np.sum(returns_data.multiply(
            eigen_portfolios[f'PC{i}']), axis=1)) * -1) / (eigen_values[i - 1])
        ep_returns.name = f'EP{i}'
        eigenportfolio_returns.append(ep_returns)
    return pd.DataFrame(eigenportfolio_returns).T


def calculate_s_score(x):
    '''

    :param x: a iterable containing [eig_returns, coin_ret, date, regression_window, PCA_window]
    :return: corresponding s-score
    '''
    eig_returns = x[0]
    coin_ret = x[1]
    date = x[2]
    regression_window = x[3]
    PCA_window = x[4]
    alpha, Xt = regress_pcafactors(
        eig_returns, coin_ret, date, regression_window)
    a, b, resid_var = regress_OUprocess(Xt)
    m, sigma_eq = build_s_score(a, b, resid_var)
    return m, sigma_eq

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
    ARmodel = sm.tsa.AR(Xt, freq='D').fit(maxlag=1)
    params = ARmodel.params
    resids = ARmodel.resid
    resid_var = np.var(resids)
    return params[0], params[1], resid_var


def build_s_score(a, b, resid_var):
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


def backtest(pca_window, regression_window, window_size, n_pcs, sample, startdate, recompute_interval):
    '''
    Called to run the strategy on a specific window of data with the passed parameters.
    Returns a dataframe containing the s-scores (to be used as trading signals) for each cryptocurrency over the window 
    '''
    output = {}
    pool = Pool(os.cpu_count())
    start_time = time.time()
    logger.log(f"BACKTESTING - Starting backtest at {time.asctime()} on {window_size}-day period beginning from {startdate.date()}.")
    logger.log(f"BACKTESTING - [Parameters] PCA Window: {pca_window}, Regression Window: {regression_window}.")
    logger.log(f"BACKTESTING - Recomputing eigenportfolios every {recompute_interval} days.")

    # Iterate over the sample period and get s scores:
    for i in range(window_size+1):
        date = startdate + timedelta(days=i)
        # Recompute eigenportfolios every 60 days
        if i % recompute_interval == 0:
            eigportfolios, eigvals = do_pca(date, sample, pca_window, n_pcs)

        eig_returns = get_eigenportfolio_returns(
            date, sample, eigportfolios, eigvals, n_pcs)

        # Use Python multiprocessing to calculate s-scores:
        pool_input = []
        for coin in sample.columns:
            if i == 0:
                output[coin] = []
            pool_input.append(
                (eig_returns, sample[coin], date, regression_window, pca_window))
        # Call processors to work on the input
        # a list of tuples, with m and sigma_eq
        out = pool.map(calculate_s_score, pool_input)
        all_m = [i[0] for i in out if i != 0]
        all_sigmas = [y[1] for y in out]
        avg_m = np.sum(all_m)/len(all_m)
        modified_m = [s_score[0] - avg_m for s_score in out]
        adjusted_s_scores = [-m/sigma for m,
                             sigma in zip(modified_m, all_sigmas)]
        for count, coin in enumerate(sample.columns):
            output[coin].append(adjusted_s_scores[count])

    logger.log(f"BACKTESTING - Completed back test. Took {time.time() - start_time}")
    end_date = startdate + timedelta(days=window_size)
    dates = pd.date_range(start=startdate, end=end_date)
    output_df = pd.DataFrame.from_dict(output)
    output_df.index = dates
    return output_df
