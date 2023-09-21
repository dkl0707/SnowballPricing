'''
Author: dkl
Description: MC雪球定价
Date: 2023-08-29 08:34:49
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
mpl.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
np.random.seed(0)                              # fix random seed


def stock_monte_carlo(start_price, T, r, q, sigma, steps, simulations):
    '''
    Desctription
    ----------
    蒙特卡洛模拟股票价格

    Parameters
    ----------
    start_price : float
        模拟起始股价.
    T : float
        模拟时间长度，如T=1表示持续1年
    r : float
        股票收益率（年化）.
    sigma : float
        股票波动率（年化）.
    steps : float
        往后预测步数.
    simulations : float
        蒙特卡洛模拟次数.

    Returns
    -------
    price : numpy.array
        模拟股价记录矩阵.

    '''

    dt = T/steps
    # Define a price array
    price = np.zeros((steps + 1, simulations))
    price[0] = start_price
    for i in range(1, steps + 1):
        z = np.random.standard_normal(simulations)
        price[i] = price[i - 1] * \
            np.exp(((r-q) - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return price.T


def snowball_valuation(prices, B1, B2, strike_out_payment, div_payment, rf):
    """
    Description
    ----------
    雪球结构定价

    Parameters
    ----------
    prices: numpy.ndarray
        蒙特卡洛所得的一条模拟价格路径.
    B1: float
        雪球的敲入价格.
    B2: float
        雪球的敲出价格.
    strike_out_payment: float
        敲出票息.
    div_payment: float
        红利票息.
    rf: float
        无风险收益率（年化）.

    Returns
    -------
    value: float
        结构定价.
    ret: float
        期间收益.
    down_flag: bool
        是否敲入.
    up_flag: bool
        是否敲出.
    end_day: int
        存续天数.

    """
    value = None
    down_flag = False                        # 是否已经触发敲入
    up_flag = False                          # 是否触发敲出
    days = np.arange(0, prices.shape[0], 1)  # 包括起始日一共有253天
    up_B2 = (prices >= B2)                   # 高于敲出价格的日期
    KO_observe = (days % 21 == 0) & (days > 0)          # 敲出观察日
    KO_days = days[up_B2*KO_observe]         # 敲出日
    KI_days = days[prices < B1]              # 敲入日
    up_flag = (len(KO_days) > 0)
    down_flag = (len(KI_days) > 0)
    if len(KO_days) > 0:                     # 曾敲出
        end_day = KO_days[0]                 # 敲出日期
        value = strike_out_payment*(end_day/252)*np.exp(-rf*end_day/252)
    elif len(KO_days) == 0 and len(KI_days) > 0:  # 曾敲入
        end_day = 252                        # 持有至到期
        if prices[-1] < prices[0]:             # 如果到期标的小于期初价格
            value = (prices[-1]-prices[0])/prices[0] * np.exp(-rf*end_day/252)
        else:                                # 如果到期标的高于期初价格，获益为0
            value = 0
    elif len(KO_days) == 0 and len(KI_days) == 0:  # 未敲入也未敲出
        end_day = 252
        value = div_payment*(end_day/252)*np.exp(-rf*end_day/252)
    else:
        raise ValueError
    return value


def main(S0=1):
    # 用于MC的常数
    steps = 252
    T = 1
    r = 0.03
    q = 0
    sigma = 0.13
    simulation_time = 5000
    # 雪球结构参数
    B2 = 1.03  # 敲出界限：103% 每个月观察敲出
    B1 = 0.85  # 敲入界限：85% 每天观察敲入
    strike_out_payment = 0.2  # 敲出票息=20%（年化）
    div_payment = 0.2  # 红利票息=20%（年化）
    rf = 0.03
    record = stock_monte_carlo(S0, T, r, q, sigma, steps, simulation_time)
    value_lst = []
    for i in tqdm(range(len(record)), leave=False):
        prices = record[i]
        value = snowball_valuation(
            prices, B1, B2, strike_out_payment, div_payment, rf)
        value_lst.append(value)
    # 组合价值
    return value_lst

if __name__ == '__main__':
    # 雪球定价
    value_lst = main()
    print('MC下的雪球结构定价:', np.mean(value_lst))
    # 雪球收益分布
    plt.figure(figsize=(10, 5))
    plt.hist(value_lst, bins=30)
    plt.title('雪球收益分布')
    plt.savefig('MC下的雪球收益分布.png')
    plt.show()
    # 雪球delta计算
    start_price = 1.03*1.5
    end_price = 0.85*0.7
    S_arr = np.linspace(start_price, end_price, 250)
    fvalue_arr = np.zeros(250)
    for i in tqdm(range(len(S_arr))):
        S0 = S_arr[i]
        value_lst = main(S0)
        fvalue_arr[i] = np.mean(value_lst)
    delta_arr = (fvalue_arr[2:] - fvalue_arr[:-2])/(S_arr[2:]-S_arr[:-2])
    S_arr = S_arr[1:-1]
    plt.figure(figsize=(10, 5))
    plt.plot(S_arr, delta_arr)
    plt.title('MC下的雪球delta')
    plt.savefig('MC下的雪球delta.png')
    plt.show()
