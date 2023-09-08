'''
Author: dkl
Description: BS公式
Date: 2023-04-18 15:24:01
'''
import numpy as np
from scipy.stats import norm


def euro_bs_call(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def euro_bs_put(S0, K, r, T, sigma):
    call = euro_bs_call(S0, K, r, T, sigma)
    put = call + K*np.exp(-r*T) - S0
    return put
