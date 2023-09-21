'''
Author: dkl
Description: FDM雪球定价
Date: 2023-08-28 22:54:39
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from basic_fdm import BasicFDM
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号


class AutocallFDM(BasicFDM):
    '''
    Autocall期权定价
    '''

    def __init__(self,
                 S0,
                 barrier_ko,
                 c,
                 r,
                 q,
                 T,
                 sigma,
                 S_min,
                 S_max,
                 M,
                 N,
                 method
                 ):
        '''
        Description
        ----------
        构造函数，输入期权参数

        Parameters
        ----------
        S0: float. 当前标的价格
        barrier_ko: float. 敲出价格
        c: float. 票面利率
        r: float. 无风险利率
        q: float. 红利率
        T: float. 到期时间
        sigma: float. 标的波动率
        S_min: float. 网格中最小标的价格
        S_max: float. 网格中最大标的价格
        M: int. 价格维度上网格段数
        N: int. 时间维度上网格段数
        method: str. 定价方法，有Explicit, Implicit, Crank-Nicolson
        '''
        super().__init__(S0, r, q, T, sigma, S_max, S_min, M, N)
        self.c = c
        self.barrier_ko = barrier_ko

    def set_boundary(self):
        # 边界条件1： 股票价格为S_min时，期权价格为0
        self.f_mat[:, 0] = 0
        # 边界条件2：到期时间T时，在敲出部分获得收益
        idx = int((self.barrier_ko - self.S_min)/self.delta_S)
        self.f_mat[self.N, 0:idx+1] = 0
        self.f_mat[self.N, idx+1:] = self.c * self.T
        # 边界条件3：敲出时才会获得收益
        for i in range(21, self.N+1, 21):
            # 敲出日
            self.f_mat[i, idx:self.M+1] = self.c*i*self.delta_T
            # 问题
            t_arr = np.arange(i-21, i)*self.delta_T
            # 敲出部分: 为下期收益的贴现
            self.f_mat[i-21:i, self.M] = self.c * (i * self.delta_T) * np.exp(-self.r*(i*self.delta_T-t_arr))
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算相应的系数，按照时间进行倒推
        self.calc_coef()
        for i in tqdm(range(self.N, 0, -1), leave=False):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            if ((i - 1) % 21 == 0) & (i != 1):
                idx = int((self.barrier_ko - self.S_min)/self.delta_S)
                self.f_mat[i - 1, 1:idx+1] = fi_1[0:idx]
            else:
                # 赋值给下个fi
                self.f_mat[i - 1, 1:self.M] = fi_1
        return


class DNTFDM(BasicFDM):
    '''
    双边触碰失效期权有限差分代码
    '''

    def __init__(self,
                 S0,
                 barrier_ko,
                 barrier_ki,
                 c,
                 r,
                 q,
                 T,
                 sigma,
                 S_min,
                 S_max,
                 M,
                 N,
                 method
                 ):
        '''
        Description
        ----------
        构造函数，输入期权参数

        Parameters
        ----------
        S0: float. 当前标的价格
        barrier_ko: float. 敲出价格
        barrier_ki: float. 敲入价格
        c: float. 票面利率
        r: float. 无风险利率
        q: float. 红利率
        T: float. 到期时间
        sigma: float. 标的波动率
        S_min: float. 网格中最小标的价格
        S_max: float. 网格中最大标的价格
        M: int. 价格维度上网格段数
        N: int. 时间维度上网格段数
        method: str. 定价方法，有Explicit, Implicit, Crank-Nicolson
        '''
        super().__init__(S0, r, q, T, sigma, S_max, S_min, M, N)
        self.c = c
        self.barrier_ki = barrier_ki
        self.barrier_ko = barrier_ko

    def set_boundary(self):
        # 边界条件1： 股票价格为S_min或者敲入时，期权价格为0
        idx_in = int((self.barrier_ki - self.S_min)/self.delta_S)
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        self.f_mat[:, 0] = 0
        self.f_mat[:, 0:idx_in+1] = 0
        # 边界条件2：到期时间T时，仅在非敲入敲出部分获得全部票息收益
        self.f_mat[self.N, idx_in+1:idx_out+1] = self.c * self.T
        self.f_mat[self.N, 0:idx_in+1] = 0
        self.f_mat[self.N, idx_out+1:] = 0
        # 边界条件3: 股票价格为Smax或者敲出时，期权价格为0
        for i in range(21, self.N+1, 21):
            # 股票价格为Smax
            self.f_mat[i-21:i, self.M] = 0
            # 敲出部分
            self.f_mat[i, idx_out+1:self.M+1] = 0
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        # 初始化网格矩阵
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算系数矩阵
        self.calc_coef()
        idx_ki = int((self.barrier_ki - self.S_min)/self.delta_S)
        idx_ko = int((self.barrier_ko - self.S_min)/self.delta_S)
        for i in tqdm(range(self.N, 0, -1), leave=False):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            # 如果是敲出日，范围应该是敲入到敲出
            if ((i - 1) % 21 == 0) & (i != 1):
                self.f_mat[i - 1, idx_ki+1:idx_ko+1] = fi_1[idx_ki:idx_ko]
            else:
                # 非敲出日范围应该是敲入到最后
                self.f_mat[i - 1, idx_ki+1:self.M] = fi_1[idx_ki:self.M-1]
        return


class UOPFDM(BasicFDM):
    '''
    上涨失效看跌期权有限差分代码
    '''

    def __init__(self,
                 S0,
                 barrier_ko,
                 c,
                 r,
                 q,
                 T,
                 sigma,
                 S_min,
                 S_max,
                 M,
                 N,
                 method
                 ):
        '''
        Description
        ----------
        构造函数，输入期权参数

        Parameters
        ----------
        S0: float. 当前标的价格
        barrier_ko: float. 敲出价格
        c: float. 票面利率
        r: float. 无风险利率
        q: float. 红利率
        T: float. 到期时间
        sigma: float. 标的波动率
        S_min: float. 网格中最小标的价格
        S_max: float. 网格中最大标的价格
        M: int. 价格维度上网格段数
        N: int. 时间维度上网格段数
        method: str. 定价方法，有Explicit, Implicit, Crank-Nicolson
        '''
        super().__init__(S0, r, q, T, sigma, S_max, S_min, M, N)
        self.c = c
        self.barrier_ko = barrier_ko

    def set_boundary(self):
        # 边界条件1： 股票价格为S_min时，不会敲出，且看跌期权必定执行
        ti = np.linspace(0, self.T, self.N+1)
        self.f_mat[:, 0] = self.S0*np.exp(-self.r*(self.T-ti)) - self.S_min
        # 边界条件2：到期时间T时，不敲出收益为max((S0-ST)/S0, 0),敲出收益为0
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        si = np.linspace(self.S_min, self.S_max, self.M+1)
        self.f_mat[self.N, 0:idx_out + 1] = \
            np.maximum((self.S0 - si)/self.S0, 0)[0:idx_out+1]
        self.f_mat[self.N, idx_out+1:] = 0
        # 边界条件3: 股票价格为Smax或者敲出时，收益为0
        for i in range(21, self.N+1, 21):
            self.f_mat[i, idx_out+1:] = 0
            self.f_mat[i-21:i, self.M] = 0
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        # 初始化网格矩阵
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算系数矩阵
        self.calc_coef()
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        for i in tqdm(range(self.N, 0, -1), leave=False):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            if ((i - 1) % 21 == 0) & (i != 1):
                self.f_mat[i - 1, 1:idx_out+1] = fi_1.reshape(-1)[0:idx_out]
            else:
                self.f_mat[i - 1, 1:self.M] = fi_1.reshape(-1)
        return


class DKOPFDM(BasicFDM):
    '''
    双边失效看跌期权有限差分代码
    '''

    def __init__(self,
                 S0,
                 barrier_ko,
                 barrier_ki,
                 c,
                 r,
                 q,
                 T,
                 sigma,
                 S_min,
                 S_max,
                 M,
                 N,
                 method
                 ):
        '''
        Description
        ----------
        构造函数，输入期权参数

        Parameters
        ----------
        S0: float. 当前标的价格
        barrier_ko: float. 敲出价格
        barrier_ki: float. 敲入价格
        c: float. 票面利率
        r: float. 无风险利率
        q: float. 红利率
        T: float. 到期时间
        sigma: float. 标的波动率
        S_min: float. 网格中最小标的价格
        S_max: float. 网格中最大标的价格
        M: int. 价格维度上网格段数
        N: int. 时间维度上网格段数
        method: str. 定价方法，有Explicit, Implicit, Crank-Nicolson
        '''
        super().__init__(S0, r, q, T, sigma, S_max, S_min, M, N)
        self.c = c
        self.barrier_ki = barrier_ki
        self.barrier_ko = barrier_ko

    def set_boundary(self):
        idx_in = int((self.barrier_ki - self.S_min)/self.delta_S)
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        # 边界条件1： 股票价格为S_min或者向下敲出时，期权价格为0
        self.f_mat[:, 0] = 0
        self.f_mat[:, 0:idx_in+1] = 0
        # 边界条件2：到期时间T时，仅在非敲入敲出部分获得看跌期权收益
        si = np.linspace(self.S_min, self.S_max, self.M+1)
        vanillaT = np.maximum((self.S0 - si)/self.S0, 0)
        self.f_mat[self.N, idx_in+1:idx_out+1] = vanillaT[idx_in+1:idx_out+1]
        self.f_mat[self.N, 0:idx_in+1] = 0
        self.f_mat[self.N, idx_out+1:] = 0
        # 边界条件3: 股票价格为Smax或者向上敲出时，期权价格为0
        for i in range(21, self.N+1, 21):
            # 股票价格为Smax
            self.f_mat[i-21:i, self.M] = 0
            # 敲出部分
            self.f_mat[i, idx_out+1:] = 0
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        # 初始化网格矩阵
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算系数矩阵
        self.calc_coef()
        idx_ki = int((self.barrier_ki - self.S_min)/self.delta_S)
        idx_ko = int((self.barrier_ko - self.S_min)/self.delta_S)
        for i in tqdm(range(self.N, 0, -1), leave=False):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            # 如果是敲出日，范围应该是敲入到敲出
            if ((i - 1) % 21 == 0) & (i != 1):
                self.f_mat[i - 1, idx_ki+1:idx_ko+1] = fi_1[idx_ki:idx_ko]
            else:
                # 非敲出日范围应该是敲入到最后
                self.f_mat[i - 1, idx_ki+1:self.M] = fi_1[idx_ki:self.M-1]
        return


class SnowballFDM(object):
    def __init__(self,
                 S0=1,
                 barrier_ko=1.03,
                 barrier_ki=0.85,
                 c=0.2,
                 r=0.03,
                 q=0,
                 T=1,
                 sigma=0.13,
                 S_min=0,
                 S_max=4,
                 M=1000,
                 N=252,
                 method='Crank-Nicolson'
                 ):
        self.S0 = S0
        self.barrier_ko = barrier_ko
        self.barrier_ki = barrier_ki
        self.c = c
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.S_min = S_min
        self.S_max = S_max
        self.M = M
        self.N = N
        self.delta_S = (S_max - S_min) / M
        self.delta_T = T / N
        self.method = method

    def get_snowball_fvalue(self):
        fdm1 = AutocallFDM(self.S0, self.barrier_ko, self.c, self.r, self.q,
                           self.T, self.sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm2 = DNTFDM(self.S0, self.barrier_ko, self.barrier_ki, self.c, self.r, self.q,
                      self.T, self.sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm3 = UOPFDM(self.S0, self.barrier_ko, self.c, self.r, self.q, self.T,
                      self.sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm4 = DKOPFDM(self.S0, self.barrier_ko, self.barrier_ki, self.c, self.r, self.q,
                       self.T, self.sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        f1 = fdm1.get_f_value()
        f2 = fdm2.get_f_value()
        f3 = fdm3.get_f_value()
        f4 = fdm4.get_f_value()
        fvalue = f1 + f2 - f3 + f4
        return fvalue

    def get_snowball_f_mat(self, sigma=None, r=None):
        if sigma is None:
            sigma = self.sigma
        if r is None:
            r = self.r
        fdm1 = AutocallFDM(self.S0, self.barrier_ko, self.c, r, self.q,
                           self.T, sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm2 = DNTFDM(self.S0, self.barrier_ko, self.barrier_ki, self.c, r, self.q,
                      self.T, sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm3 = UOPFDM(self.S0, self.barrier_ko, self.c, r, self.q, self.T,
                      sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm4 = DKOPFDM(self.S0, self.barrier_ko, self.barrier_ki, self.c, r, self.q,
                       self.T, sigma, self.S_min, self.S_max, self.M, self.N, self.method)
        fdm1.set_boundary()
        fdm2.set_boundary()
        fdm3.set_boundary()
        fdm4.set_boundary()
        fdm1.calc_f_mat()
        fdm2.calc_f_mat()
        fdm3.calc_f_mat()
        fdm4.calc_f_mat()
        f_mat = fdm1.f_mat + fdm2.f_mat - fdm3.f_mat + fdm4.f_mat
        return f_mat

    def plot_snowball_delta(self, plot_3d=False):
        start_price = round(self.barrier_ki*0.7, 1)
        end_price = round(self.barrier_ko*1.5, 1)
        # 计算期权矩阵
        f_mat = self.get_snowball_f_mat()
        if plot_3d:
            # 计算delta矩阵
            T_arr = np.linspace(0, self.T, self.N+1)
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            delta_mat = (f_mat[:, 2:] - f_mat[:, :-2]) / np.tile((S_arr[2:] - S_arr[:-2]).reshape(1, -1), (self.N+1, 1))
            S_arr = S_arr[1:-1]
            # 选取Smin到Smax的值
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))[0]
            delta_mat = delta_mat[:, cond]
            S_arr = S_arr[cond]
            # 绘制曲面
            T_mat = np.tile(T_arr.reshape(-1, 1), (1, delta_mat.shape[1]))
            S_mat = np.tile(S_arr.reshape(1, -1), (delta_mat.shape[0], 1))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(T_mat, S_mat, delta_mat, cmap='rainbow', linewidth=0, antialiased=True)
            fig.colorbar(surf)
            ax.set_title('delta(FDM)')
            ax.set_xlabel('T')
            ax.set_ylabel('S')
            plt.savefig('FDM下的雪球delta_3D.png')
            plt.show()
        else:
            fvalue_arr = f_mat[0, :]
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            # delta_arr对应的是Smin+deltaS到Smax-deltaS之间的值
            delta_arr = (fvalue_arr[2:] - fvalue_arr[:-2])/(S_arr[2:]-S_arr[:-2])
            S_arr = S_arr[1:-1]
            # 选出从start_price到end_price之间的delta
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))
            delta_arr = delta_arr[cond]
            S_arr = S_arr[cond]
            # 绘制曲线
            plt.figure(figsize=(10, 5))
            plt.title('delta(FDM)')
            plt.plot(S_arr, delta_arr)
            plt.xticks(np.arange(start_price, end_price, 0.1))
            plt.savefig('FDM下的雪球delta.png')
            plt.show()
        return

    def plot_snowball_gamma(self, plot_3d=False):
        start_price = round(self.barrier_ki*0.7, 1)
        end_price = round(self.barrier_ko*1.5, 1)
        # 计算期权矩阵
        f_mat = self.get_snowball_f_mat()
        if plot_3d:
            # 计算gamma矩阵
            T_arr = np.linspace(0, self.T, self.N+1)
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            d2f = f_mat[:, 2:] + f_mat[:, :-2] - 2*f_mat[:, 1:-1]
            ds2 = ((S_arr[2:]-S_arr[:-2])/2)**2
            ds2 = np.tile((ds2).reshape(1, -1), (self.N+1, 1))
            gamma_mat = d2f/ds2
            S_arr = S_arr[1:-1]
            # 选取Smin到Smax的值
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))[0]
            gamma_mat = gamma_mat[:, cond]
            S_arr = S_arr[cond]
            # 绘制曲面
            T_mat = np.tile(T_arr.reshape(-1, 1), (1, gamma_mat.shape[1]))
            S_mat = np.tile(S_arr.reshape(1, -1), (gamma_mat.shape[0], 1))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(T_mat, S_mat, gamma_mat, cmap='rainbow', linewidth=0, antialiased=True)
            fig.colorbar(surf)
            ax.set_title('gamma(FDM)')
            ax.set_xlabel('T')
            ax.set_ylabel('S')
            plt.savefig('FDM下的雪球gamma_3D.png')
            plt.show()
        else:
            # 绘制gamma折线图
            fvalue_arr = f_mat[0, :]
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            d2f = fvalue_arr[2:] + fvalue_arr[:-2] - 2*fvalue_arr[1:-1]
            ds2 = ((S_arr[2:]-S_arr[:-2])/2)**2
            gamma_arr = d2f/ds2
            S_arr = S_arr[1:-1]
            # 选出从start_price到end_price之间的delta
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))
            gamma_arr = gamma_arr[cond]
            S_arr = S_arr[cond]
            # 绘制曲线
            plt.figure(figsize=(10, 5))
            plt.title('gamma(FDM)')
            plt.plot(S_arr, gamma_arr)
            plt.xticks(np.arange(start_price, end_price, 0.1))
            plt.yticks(np.arange(min(gamma_arr)//50*50, (max(gamma_arr)//50+1)*50, 50))
            plt.savefig('FDM下的雪球gamma.png')
            plt.show()
        return

    def plot_snowball_theta(self, plot_3d=False):
        start_price = round(self.barrier_ki*0.7, 1)
        end_price = round(self.barrier_ko*1.5, 1)
        # 计算期权矩阵
        f_mat = self.get_snowball_f_mat()
        if plot_3d:
            # 计算theta矩阵
            T_arr = np.linspace(0, self.T, self.N+1)
            theta_mat = (f_mat[1:, :] - f_mat[:-1, :]) / np.tile((T_arr[1:] - T_arr[:-1]).reshape(-1, 1), (1, self.M+1))
            T_arr = T_arr[1:]
            # 选取Smin到Smax的值
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))[0]
            theta_mat = theta_mat[:, cond]
            S_arr = S_arr[cond]
            # 绘制曲面
            T_mat = np.tile(T_arr.reshape(-1, 1), (1, theta_mat.shape[1]))
            S_mat = np.tile(S_arr.reshape(1, -1), (theta_mat.shape[0], 1))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(T_mat, S_mat, theta_mat, cmap='rainbow', linewidth=0, antialiased=True)
            fig.colorbar(surf)
            ax.set_title('theta(FDM)')
            ax.set_xlabel('T')
            ax.set_ylabel('S')
            ax.set_zlim3d(np.quantile(theta_mat, 0.01), np.quantile(theta_mat, 0.99))
            plt.savefig('FDM下的雪球theta_3D.png')
            plt.show()
        else:
            # 绘制theta折线图
            theta_arr = (f_mat[1, :] - f_mat[0, :]) / self.delta_T
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            # 选出从start_price到end_price之间的delta
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))
            theta_arr = theta_arr[cond]
            S_arr = S_arr[cond]
            # 绘制曲线
            plt.figure(figsize=(10, 5))
            plt.title('theta(FDM)')
            plt.plot(S_arr, theta_arr)
            plt.xticks(np.arange(start_price, end_price, 0.1))
            plt.savefig('FDM下的雪球theta.png')
            plt.show()
        return

    def plot_snowball_vega(self, plot_3d=False):
        start_sigma = 0.01
        end_sigma = 0.4
        delta_sigma = 0.04
        start_price = round(self.barrier_ki*0.7, 1)
        end_price = round(self.barrier_ko*1.5, 1)
        sigma_arr = np.arange(start_sigma, end_sigma, delta_sigma)
        f_mat = np.zeros((len(sigma_arr), self.M+1))
        for i in tqdm(range(len(sigma_arr))):
            temp_f_mat = self.get_snowball_f_mat(sigma=sigma_arr[i])
            f_mat[i, :] = temp_f_mat[0, :]
        if plot_3d:
            # 计算vega矩阵
            vega_mat = (f_mat[2:, :] - f_mat[:-2, :]) / np.tile((sigma_arr[2:] - sigma_arr[:-2]).reshape(-1, 1), (1, self.M+1))
            sigma_arr = sigma_arr[1:-1]
            # 选取Smin到Smax的值
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))[0]
            vega_mat = vega_mat[:, cond]
            S_arr = S_arr[cond]
            # 绘制曲面
            sigma_mat = np.tile(sigma_arr.reshape(-1, 1), (1, vega_mat.shape[1]))
            S_mat = np.tile(S_arr.reshape(1, -1), (vega_mat.shape[0], 1))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(sigma_mat, S_mat, vega_mat, cmap='rainbow', linewidth=0, antialiased=True)
            fig.colorbar(surf)
            ax.set_title('vega(FDM)')
            ax.set_xlabel('sigma')
            ax.set_ylabel('S')
            plt.savefig('FDM下的雪球vega_3D.png')
            plt.show()
        else:
            # 绘制vega折线图
            idx = int((self.sigma - start_sigma)/delta_sigma)
            vega_arr = (f_mat[idx+1, :] - f_mat[idx-1, :]) / (2*delta_sigma)
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            # 选出从start_price到end_price之间的delta
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))
            vega_arr = vega_arr[cond]
            S_arr = S_arr[cond]
            # 绘制曲线
            plt.figure(figsize=(10, 5))
            plt.title('vega(FDM)')
            plt.plot(S_arr, vega_arr)
            plt.xticks(np.arange(start_price, end_price, 0.1))
            plt.savefig('FDM下的雪球vega.png')
            plt.show()
        return

    def plot_snowball_rho(self, plot_3d=False):
        start_r = 0.005
        end_r = 0.05
        delta_r = 0.005
        start_price = round(self.barrier_ki*0.7, 1)
        end_price = round(self.barrier_ko*1.5, 1)
        r_arr = np.arange(start_r, end_r, delta_r)
        f_mat = np.zeros((len(r_arr), self.M+1))
        for i in tqdm(range(len(r_arr))):
            temp_f_mat = self.get_snowball_f_mat(r=r_arr[i])
            f_mat[i, :] = temp_f_mat[0, :]
        if plot_3d:
            # 计算rho矩阵
            rho_mat = (f_mat[2:, :] - f_mat[:-2, :]) / np.tile((r_arr[2:] - r_arr[:-2]).reshape(-1, 1), (1, self.M+1))
            r_arr = r_arr[1:-1]
            # 选取Smin到Smax的值
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))[0]
            rho_mat = rho_mat[:, cond]
            S_arr = S_arr[cond]
            # 绘制曲面
            r_mat = np.tile(r_arr.reshape(-1, 1), (1, rho_mat.shape[1]))
            S_mat = np.tile(S_arr.reshape(1, -1), (rho_mat.shape[0], 1))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(r_mat, S_mat, rho_mat, cmap='rainbow', linewidth=0, antialiased=True)
            fig.colorbar(surf)
            ax.set_title('rho(FDM)')
            ax.set_xlabel('r')
            ax.set_ylabel('S')
            plt.savefig('FDM下的雪球rho_3D.png')
            plt.show()
        else:
            # 绘制rho折线图
            idx = int((self.r - start_r)/delta_r)
            rho_arr = (f_mat[idx+1, :] - f_mat[idx-1, :]) / (2*delta_r)
            S_arr = np.linspace(self.S_min, self.S_max, self.M + 1)
            # 选出从start_price到end_price之间的delta
            cond = np.where((S_arr >= start_price) & (S_arr <= end_price))
            rho_arr = rho_arr[cond]
            S_arr = S_arr[cond]
            # 绘制曲线
            plt.figure(figsize=(10, 5))
            plt.title('rho(FDM)')
            plt.plot(S_arr, rho_arr)
            plt.xticks(np.arange(start_price, end_price, 0.1))
            plt.savefig('FDM下的雪球rho.png')
            plt.show()
        return


if __name__ == '__main__':
    df = pd.DataFrame({'q': np.arange(0, 0.2, 0.01)})
    for idx in tqdm(range(len(df))):
        q = df.loc[idx, 'q']
        fdm = SnowballFDM(q=q)
        df.loc[idx, 'price'] = fdm.get_snowball_fvalue()
    df.to_excel('红利率变化下的雪球价格.xlsx', index=False)
