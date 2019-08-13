from pandas_datareader import data as pdr
import requests
import pandas as pd
import numpy as np
import math

# Download Stock data from Yahoo Finance
def download_stock_data(start_date, end_date, asset_list, filename):
    """
    asset_list: list of asset to download
    filename: name of the csv file where the data go to
    """
    for i, asset in enumerate(asset_list):
        if i == 0:
            data = pdr.get_data_yahoo(asset_list[i], start_date,end_date)[['Adj Close']]
            all_prices = data.rename(columns={"Adj Close": asset})
            
        else:
            new_asset = pdr.get_data_yahoo(asset_list[i], start_date,end_date)[['Adj Close']]
            all_prices = all_prices.join(new_asset.rename(columns={"Adj Close": asset}))
    
    all_prices.to_csv('./00_Data/' + filename + '.csv')
    return

# Extract Return and accumulated price data (starting value 1) from raw stock prices
def process_stock_data(all_prices):
    """
    input:
    all_prices: data frame, price of a list of stock from start date to the end date
    
    return:
    all_price_norm: normalized asset price with the price on the first day as 1
    all_returns: relative price
    """
    n_time, n_assets = all_prices.shape
    # Calculate returns
    all_returns = all_prices.pct_change(1).apply(lambda a : a + 1)
    
    # Calculate relative price time series
    all_prices_norm = all_returns.copy()
    for i in range(n_time):
        if i == 0:
            all_prices_norm.iloc[0] = np.ones(n_assets)
        else:
            for j in range(n_assets):
                if math.isnan(all_prices_norm.iloc[i,j]):
                    all_prices_norm.iloc[i,j] = all_prices_norm.iloc[i-1,j]
                else:
                    all_prices_norm.iloc[i,j] = all_prices_norm.iloc[i,j] * all_prices_norm.iloc[i-1,j]
                    
    return all_prices_norm, all_returns

#1/fluc, 1, 1/fluc, 1, ...
def teststock1(T, fluc):
    stock = np.ones(T+1)
    for i in range(T+1):
        if i%2 == 1:
            stock[i] = 1.0/fluc
    return stock

#Brownian Motion
def teststock2(T, delta, sigma):
    stock1 = np.zeros(T+1)
    stock1[0] = 3.0
    for i in range(T):
        stock1[i+1] = stock1[i] + sigma + norm.rvs(scale=delta**2)
        if (stock1[i+1] <= 0):
            print('Negative Warning!')
    return stock1

#Garbage 1, 1/fluc, 1/pow(fluc,2) ,...
def teststock3(T, discount):
    stock = np.ones(T+1, dtype=float)
    for i in range(T):
        stock[i+1] = stock[i]*discount
    return stock