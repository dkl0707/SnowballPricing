'''
Author: dkl
Description: 雪球有限差分定价
Date: 2023-08-28 22:54:39
'''
import numpy as np
from tqdm import tqdm
from basic_fdm import BasicFDM


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
        for i in range(0, self.N+1, 21):
            self.f_mat[i, idx:self.M+1] = self.c*i*self.delta_T
            t_arr = np.arange(i, i+21)*self.delta_T
            t_arr = t_arr[t_arr<=self.T]
            self.f_mat[i:i+21, self.M] = self.c * \
                (i * self.delta_T) * np.exp(self.r*(t_arr-i*self.delta_T))
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算相应的系数，按照时间进行倒推
        self.calc_coef()
        for i in tqdm(range(self.N, 0, -1)):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            if (i-1) % 21 == 0:
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
        for i in range(0, self.N+1, 21):
            # 股票价格为Smax
            self.f_mat[i:i+21, self.M] = 0
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
        for i in tqdm(range(self.N, 0, -1)):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            # 如果是敲出日，范围应该是敲入到敲出
            if (i - 1) % 21 == 0:
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
        # 边界条件2：到期时间T时，不敲出收益为max(S0-ST, 0),敲出收益为0
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        si = np.linspace(self.S_min, self.S_max, self.M+1)
        self.f_mat[self.N, 0:idx_out + 1] = \
            np.maximum(self.S0 - si, 0)[0:idx_out+1]
        self.f_mat[self.N, idx_out+1:] = 0
        # 边界条件3: 股票价格为Smax或者敲出时，收益为0
        for i in range(0, self.N+1, 21):
            self.f_mat[i, idx_out+1:] = 0
            self.f_mat[i:i+21, self.M] = 0
        self._flag_set_boundary = True
        return

    def calc_f_mat(self):
        # 初始化网格矩阵
        if not self._flag_set_boundary:
            raise ValueError('please set f_mat boundary first')
        # 计算系数矩阵
        self.calc_coef()
        idx_out = int((self.barrier_ko - self.S_min)/self.delta_S)
        for i in tqdm(range(self.N, 0, -1)):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            if (i-1) % 21 == 0:
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
        vanillaT = np.maximum(self.S0 - si, 0)
        self.f_mat[self.N, idx_in+1:idx_out+1] = vanillaT[idx_in+1:idx_out+1]
        self.f_mat[self.N, 0:idx_in+1] = 0
        self.f_mat[self.N, idx_out+1:] = 0
        # 边界条件3: 股票价格为Smax或者向上敲出时，期权价格为0
        for i in range(0, self.N+1, 21):
            # 股票价格为Smax
            self.f_mat[i:i+21, self.M] = 0
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
        for i in tqdm(range(self.N, 0, -1)):
            fi = (self.f_mat[i, 1:self.M]).reshape(-1, 1)
            # 计算fi-1
            fi_1 = self.get_fi_1(i)
            # 如果是敲出日，范围应该是敲入到敲出
            if (i - 1) % 21 == 0:
                self.f_mat[i - 1, idx_ki+1:idx_ko+1] = fi_1[idx_ki:idx_ko]
            else:
                # 非敲出日范围应该是敲入到最后
                self.f_mat[i - 1, idx_ki+1:self.M] = fi_1[idx_ki:self.M-1]
        return


def get_snowball_fdm():
    S0 = 1
    barrier_ko = 1.03
    barrier_ki = 0.85
    c = 0.2
    r = 0.03
    q = 0
    T = 1
    sigma = 0.13
    S_min = 0
    S_max = 4
    M = 4000
    N = 252
    method = 'Crank-Nicolson'
    fdm1 = AutocallFDM(S0, barrier_ko, c, r, q, T,
                       sigma, S_min, S_max, M, N, method)
    fdm2 = DNTFDM(S0, barrier_ko, barrier_ki, c, r, q,
                  T, sigma, S_min, S_max, M, N, method)
    fdm3 = UOPFDM(S0, barrier_ko, c, r, q, T,
                  sigma, S_min, S_max, M, N, method)
    fdm4 = DKOPFDM(S0, barrier_ko, barrier_ki, c, r, q,
                   T, sigma, S_min, S_max, M, N, method)
    v1 = fdm1.get_f_value()
    print('Autocall期权定价:', v1)
    v2 = fdm2.get_f_value()
    print('DNT期权定价:', v2)
    v3 = fdm3.get_f_value()
    print('UOP期权定价:', v3)
    v4 = fdm4.get_f_value()
    print('DKOP期权定价:', v4)
    return v1+v2-v3+v4


if __name__ == '__main__':
    fvalue = get_snowball_fdm()
    print('有限差分计算雪球期权价值', fvalue)
