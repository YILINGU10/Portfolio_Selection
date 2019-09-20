import pandas as pd
import numpy as np
import cvxpy as cvx

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt

from datafunctions import *

# Initialize plotly offline
plotly.offline.init_notebook_mode(connected=True)

#####################
# For visualization #
#####################

def visualize_spy(other, SPY):
    spy_and_original = pd.DataFrame(other)
    spy_and_original.insert(1, 'Original', SPY.values)
    spy_and_original.columns = ['Prediction', 'Original']
    spy_and_original.index = SPY.index
    iplot(spy_and_original.iplot(asFigure=True, kind='scatter',xTitle='Dates',yTitle='Relative Price',title='Predicted SPY vs SPY'))
    return spy_and_original

#####################
# Get wealth factor #
#####################

def get_wealth_factor(portfolio, returns):
    T, m = returns.shape
    wealth_factors = np.ones(T)
    for i in range(1, T):
        wealth_factors[i] = wealth_factors[i - 1] * np.dot(
            portfolio[i - 1], returns[i - 1, :])
    return wealth_factors
    

###################################################
# For benchmarking portfolio selection algorithms #
###################################################

def benchmark_portfolio(portfolio, algorithm_name, SPY_benchmark = None, stock_prices = None, stock_prices_norm = None, stock_returns = None):  
    """
    Feed portfolio in with stocks' names as columns and dates as index!
    """
    
    # For faster processing if norm and returns are provided already
    if stock_prices_norm is None and stock_returns is None:
        stock_prices_norm, stock_returns = process_stock_data(stock_prices)

    # NaN in first row screws up cvxpy
    if all(stock_returns.iloc[0].isna()):
        stock_returns = stock_returns.iloc[1:]
        
    all_cumu_wealth = pd.DataFrame(np.zeros((stock_returns.shape[0], 1)))
    
    if SPY_benchmark:
        all_cumu_wealth['SPY'] = SPY_benchmark
        
    # Solves the convex optimization problem of the Best constant rebalanced portfolio in hindsight
    _, all_cumu_wealth['BCRP_reward'] = bcrp_wealth_factors(stock_returns)

    # Uniform Constant Rebalanced Portfolio
    all_cumu_wealth['UCRP_reward'] = ucrp_wealth_factors(stock_returns)
    
    # User's algorithm
    all_cumu_wealth[algorithm_name] = get_wealth_factor(portfolio, stock_returns.values)

    # plot
    iplot(all_cumu_wealth.iplot(asFigure=True, kind='scatter',xTitle='Dates',yTitle='Wealth Factor',title='Relative Wealth over time'))
    
def benchmark_portfolios(portfolios, algorithm_names, SPY_benchmark, stock_prices = None, stock_prices_norm = None, stock_returns = None):  
    """
    Feed portfolio in with stocks' names as columns and dates as index!
    """

    # For faster processing if norm and returns are provided already
    if stock_prices_norm is None and stock_returns is None:
        stock_prices_norm, stock_returns = process_stock_data(stock_prices)

    # NaN in first row screws up cvxpy
    if all(stock_returns.iloc[0].isna()):
        stock_returns = stock_returns.iloc[1:]
        
    all_cumu_wealth = pd.DataFrame(np.zeros((stock_returns.shape[0], 1)))

    # Solves the convex optimization problem of the Best constant rebalanced portfolio in hindsight
    _, all_cumu_wealth['BCRP_reward'] = bcrp_wealth_factors(stock_returns)

    # Uniform Constant Rebalanced Portfolio
    all_cumu_wealth['UCRP_reward'] = ucrp_wealth_factors(stock_returns)

    for portfolio, algorithm_name in zip(portfolios, algorithm_names):
        # User's algorithm
        all_cumu_wealth[algorithm_name] = get_wealth_factor(portfolio, stock_returns.values)

    # plot
    iplot(all_cumu_wealth.iplot(asFigure=True, kind='scatter',xTitle='Dates',yTitle='Wealth Factor',title='Relative Wealth over time'))

def ucrp_wealth_factors(stock_returns):
    T, m = stock_returns.shape

    #ratio for each asset
    r = 1.0/m

    #Benchmarks uniform constant portfolio weighting
    W = np.repeat(r,m)
    portfolio = np.tile(W, (T, 1))
    
    # Calculate the wealth factors array over time
    return get_wealth_factor(portfolio, stock_returns.values)

def bcrp_wealth_factors(stock_returns):
    asset_returns = stock_returns.values
    T, m = asset_returns.shape

    w = cvx.Variable(m)
    S = 0
    for i in range(T-1):
        S += cvx.log(asset_returns[i,:]*w)
    objective = cvx.Maximize(S)
    constraints = [cvx.sum(w) == 1, w >= 0]

    prob = cvx.Problem(objective, constraints)
    prob.solve(solver = 'SCS')  # Returns the optimal value.

    # get the optimal constant weight vector
    w_nom = w.value
    portfolio = np.tile(w_nom, (T, 1))
    
    #calculate the development of the CRP
    wealth_factors = get_wealth_factor(portfolio, asset_returns)
    
    return w_nom, wealth_factors