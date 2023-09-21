'''
Author: dkl
Description: MC情景分析
Date: 2023-08-29 08:32:53
'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import wraps
from datetime import datetime
from tqdm import tqdm
mpl.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
np.random.seed(0)                              # fix random seed


def timer(func):
    # 计时器，在函数上一行加上@timer即可
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Started running function: %s" % func.__name__)
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        cost_time = (end_time - start_time).total_seconds()
        print("Finished running function: %s" % func.__name__)
        print("----------Time cost: {}s----------".format(cost_time))
        print("")
        return res
    return wrapper


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
        price[i] = price[i - 1] * np.exp(((r-q) - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return price.T


def snowball_valuation(prices, B1, B2, strike_out_payment, div_payment, rf):
    """
    Description
    ----------
    雪球结构定价

    Parameters
    ----------
    prices : numpy.ndarray
        蒙特卡洛所得的一条模拟价格路径.
    B1 : float
        雪球的敲入价格.
    B2 : float
        雪球的敲出价格.
    strike_out_payment : float
        敲出票息.
    div_payment : float
        红利票息.
    rf : float
        无风险收益率（年化）.

    Returns
    -------
    value : float
        结构定价.
    ret : float
        期间收益.
    down_flag : bool
        是否敲入.
    up_flag : bool
        是否敲出.
    end_day : int
        存续天数.

    """

    value = None
    down_flag = False                        # 是否已经触发敲入
    up_flag = False                          # 是否触发敲出
    days = np.arange(0, prices.shape[0], 1)  # 包括起始日一共有253天
    up_B2 = (prices >= B2)                   # 高于敲出价格的日期
    KO_observe = (days % 21 == 0)            # 敲出观察日
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
            value = (prices[-1]-prices[0]) * np.exp(-rf*end_day/252)
        else:                                # 如果到期标的高于期初价格，获益为0
            value = 0
    else:  # 未敲入也未敲出
        end_day = 252
        value = div_payment*(end_day/252)*np.exp(-rf*end_day/252)
    return value, down_flag, up_flag, end_day


def main():
    # 用于MC的常数
    steps = 252
    T = 1
    r = 0.03
    q = 0
    sigma = 0.13
    simulation_time = 300000
    # 雪球结构参数
    S0 = 1.0
    B2 = 1.03*S0  # 敲出界限：103% 每个月观察敲出
    B1 = 0.85*S0  # 敲入界限：85% 每天观察敲入
    strike_out_payment = 0.2  # 敲出票息=20%（年化）
    div_payment = 0.2  # 红利票息=20%（年化）
    rf = 0.03
    record = stock_monte_carlo(S0, T, r, q, sigma, steps, simulation_time)
    result_df = pd.DataFrame(index=range(len(record)), columns=["value", "ret", "down_KI", "up_KO", "exist_days"])
    for i in tqdm(range(len(record))):
        prices = record[i]
        value, down_flag, up_flag, exist_days = snowball_valuation(prices, B1, B2, strike_out_payment, div_payment, rf)
        result_df.loc[i]["value"] = value
        result_df.loc[i]["down_KI"] = down_flag
        result_df.loc[i]["up_KO"] = up_flag
        result_df.loc[i]["exist_days"] = exist_days

    snowball_df = pd.DataFrame(index=["价值", "最大损失", "存续交易日数", "概率"],
                               columns=["雪球结构整体", "敲出", "敲入未敲出", "未敲入也未敲出"])
    # 组合价值
    snowball_df.loc["价值"]["雪球结构整体"] = result_df["value"].mean()
    snowball_df.loc["最大损失"]["雪球结构整体"] = result_df["value"].min()
    snowball_df.loc["存续交易日数"]["雪球结构整体"] = result_df["exist_days"].mean()

    # 敲出
    snowball_df.loc["价值"]["敲出"] = result_df.query("up_KO==True")["value"].mean()
    snowball_df.loc["最大损失"]["敲出"] = result_df.query("up_KO==True")["value"].min()
    snowball_df.loc["存续交易日数"]["敲出"] = result_df.query("up_KO==True")["exist_days"].mean()
    snowball_df.loc["概率"]["敲出"] = result_df.query("up_KO==True").shape[0]/result_df.shape[0]
    # 敲入
    snowball_df.loc["价值"]["敲入未敲出"] = result_df.query("down_KI==True and up_KO==False")["value"].mean()
    snowball_df.loc["最大损失"]["敲入未敲出"] = result_df.query("down_KI==True and up_KO==False")["value"].min()
    snowball_df.loc["存续交易日数"]["敲入未敲出"] = result_df.query("down_KI==True and up_KO==False")["exist_days"].mean()
    snowball_df.loc["概率"]["敲入未敲出"] = result_df.query("down_KI==True and up_KO==False").shape[0]/result_df.shape[0]
    # 未敲入也未敲出
    snowball_df.loc["价值"]["未敲入也未敲出"] = result_df.query("down_KI==False and up_KO==False")["value"].mean()
    snowball_df.loc["最大损失"]["未敲入也未敲出"] = result_df.query("down_KI==False and up_KO==False")["value"].min()
    snowball_df.loc["存续交易日数"]["未敲入也未敲出"] = result_df.query("down_KI==False and up_KO==False")["exist_days"].mean()
    snowball_df.loc["概率"]["未敲入也未敲出"] = result_df.query("down_KI==False and up_KO==False").shape[0]/result_df.shape[0]
    snowball_df.to_excel("MC下的雪球情景分析.xlsx")


if __name__ == '__main__':
    main()
