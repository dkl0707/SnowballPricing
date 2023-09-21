'''
Author: dkl
Description: 欧式看跌期权theta
Date: 2023-09-20 11:54:57
'''
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号


def theta_option(S, K, sigma, r, T, optype):
    import numpy as np
    from scipy.stats import norm
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    theta_call = -(S*sigma*np.exp(-d1**2/2)) / \
        (2*np.sqrt(2*np.pi*T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    if optype == 'call':
        theta = theta_call
    else:
        theta = theta_call + r*K*np.exp(-r*T)
    return theta


# 标的价格与Theta的关系
S_list = np.linspace(0.5, 1.5, 100)
theta_list = theta_option(S=S_list, K=1, sigma=0.13, r=0.03, T=1, optype='put')


plt.figure(figsize=(12, 6))
plt.plot(S_list, -theta_list, label='看跌期权')
plt.grid('True')
plt.savefig('欧式看跌期权空头theta.png')
plt.show()
