import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PSenv(gym.Env):
    """
    Observation: 
        Type: Box(2)
        Num	Observation                                                   Min         Max
        i	Previous closing price = current opening price of asset i    -Inf         Inf
        
    Actions:
        Description: A portfolio vector for one trading period.
        Type: Box(m) where m is the amount of assets.
        Num	Action
        i = number from 1...t	Proportion of capital to invest in the ith asset
        
    Reward:
        sum of a_(t-1, i) * y_(t-1, i), where a is the portfolio matrix and y is the asset returns matrix
        
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
        
    Episode Termination:b
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_assets, t_prices):
        high = np.repeat(np.finfo(np.float32).max, n_assets)
        self.t = 0
        self.t_prices = t_prices

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Box(np.zeros(n_assets),
                                       np.ones(n_assets),
                                       dtype=np.float32)

        self.state = t_prices[0]

    def step(self, action):
#         assert all(np.sum(action, axis = 0) == 1), "Portfolio doesn't sum up to one"
        t_price = self.t_prices[self.t]
    
        t_minus_1_prices = self.state
        
        t_returns = t_price / t_minus_1_prices # element wise division
        self.state = t_price # update the state
        
        self.t += 1
        
        reward = np.dot(action.T, t_returns)
        
        done = self.t >= len(self.t_prices)

        return np.array(self.state), done, reward, {}

    def reset(self):
        self.t = 0
        return self.t_prices[0]