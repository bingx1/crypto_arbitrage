
import numpy as np
import pandas as pd
import cryptocompare as cc
# list of coins

coin_list = cc.get_coin_list()
coins = sorted(list(coin_list.keys()))
# get data for all available coins
coin_data = {}
for i in range(len(coins)//50 + 1):
    # limited to a list containing at most 300 characters #
    coins_to_get = coins[(50*i):(50*i+50)]
    message = cc.get_price(coins_to_get, curr='USD', full=True)
    coin_data.update(message['RAW'])
# remove 'USD' level
for k in coin_data.keys():
    coin_data[k] = coin_data[k]['USD']
# coin_data = pd.DataFrame.from_dict(coin_data, orient='index')
# coin_data = coin_data.sort_values('MKTCAP', ascending=False)
# # exclude coins that haven't traded in last 24 hours
# # TOTALVOLUME24H is the amount the coin has been traded
# # in 24 hours against ALL its trading pairs
# coin_data = coin_data[coin_data['TOTALVOLUME24H'] != 0]
# top_coins = coin_data[:100].index
# df_dict = {}
# for coin in top_coins:
#     hist = cc.get_historical_price_day(coin, curr='USD', limit=2000)
#     if hist:
#         hist_df = pd.DataFrame(hist['Data'])
#         hist_df['time'] = pd.to_datetime(hist_df['time'], unit='s')
# #         hist_df.index = hist_df['time']
#         del hist_df['time']
#         df_dict[coin] = hist_df