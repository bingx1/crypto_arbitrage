from cryptocmd import CmcScraper
import os
s = os.listdir(r"C:\Users\Bing\Documents\NumTech Ass 2\Fulldata")
have = os.listdir(r'C:\Users\Bing\Documents\NumTech Ass 2\Coinmarketcap data')
to_get = [coin for coin in s if coin not in have]
# needed = ['GEO', 'RBIES', 'MNE', 'GRAM', 'AEC', 'ATX', 'ELP', 'ZRC', 'BTX', 'VRS', 'OPT', 'SSC', 'TTC', 'PRO', 'BNT', 'POLY', 'NAS', 'LUX', 'KNC']
needed = ['NANO']
unavail = []
for file in to_get:
    coin = file.split('.csv')[0]
    try:
        path = r'C:\Users\Bing\Documents\NumTech Ass 2\Coinmarketcap data\{}'.format(coin+'.csv')
        scraper = CmcScraper(coin)
        df = scraper.get_dataframe()
        df = df.rename({'Date':'time','Close':'close','Volume':'volumeto'},axis=1)
        df.index = df['time']
        df = df.sort_index()
        df.pop('time')
        df.to_csv(path)
    except:
        unavail.append(coin)
        pass