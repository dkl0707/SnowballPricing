'''
Author: dkl
Description: FDM欧式香草期权定价
Date: 2023-04-18 15:23:33
'''
import numpy as np
from basic_fdm import BasicFDM
from bs import euro_bs_call, euro_bs_put


class VanillaFDM(BasicFDM):
    '''
    欧式期权有限差分代码
    '''

    def __init__(self,
                 S0=1,
                 K=1,
                 r=0.03,
                 q=0,
                 T=1,
                 sigma=0.13,
                 S_min=0,
                 S_max=2,
                 M=100,
                 N=252,
                 method='Crank-Nicolson',
                 opt_type='call'):
        '''
        Description
        ----------
        构造函数，输入欧式期权参数

        Parameters
        ----------
        S0: float. 当前标的价格
        K: float. 执行价格-----用于测试欧式期权定价，后面删除
        r: float. 无风险收益率
        T: float. 到期时间
        sigma: float. 标的波动率
        S_min: float. 网格中最小标的价格
        S_max: float. 网格中最大标的价格
        M: int. 价格维度上网格段数
        N: int. 时间维度上网格段数
        method: str. 定价方法，有Explicit, Implicit, Crank-Nicolson，默认为Crank-Nicolson
        opt_type: str. 是'call'或者'put'，默认为'call'
        '''
        # 期权参数
        self.K = K  # 期权执行价
        super().__init__(S0, r, q, T, sigma, S_max, S_min, M, N, method)
        if opt_type not in ['call', 'put']:
            raise ValueError('opt_type must be call or put!')
        self.opt_type = opt_type

    def set_boundary(self):
        if self.opt_type == 'call':
            self._call_set_boundary()
        else:
            self._put_set_boundary()
        self._flag_set_boundary = True

    def _call_set_boundary(self):
        # 边界条件1： 股票价格为S_min时，期权价格为0
        self.f_mat[:, 0] = 0
        # 边界条件2：到期时间T时，期权价格f=max(ST-K, 0)
        for j in range(self.M + 1):
            ST = self.S_min + j * self.delta_S
            self.f_mat[self.N, j] = np.maximum(ST - self.K, 0)
        # 边界条件3：S=S_max时，call=S_max-Kexp(-r(T-t))
        for i in range(self.N + 1):
            ti = self.T - i * self.delta_T
            self.f_mat[i, self.M] = self.S_max - self.K * np.exp(-self.r * ti)
        self._init_f_mat_flag = True

    def _put_set_boundary(self):
        # 边界条件1： 初始价格为S_min时，期权价格为f=Kexp(-r(T-t))-Smin
        for i in range(self.N + 1):
            ti = self.T - i * self.delta_T
            self.f_mat[i, 0] = self.K * np.exp(-self.r * ti) - self.S_min
        # 边界条件2：到期时间T时，期权价格f=max(K-ST, 0)
        for j in range(self.M + 1):
            ST = self.S_min + j * self.delta_S
            self.f_mat[self.N, j] = np.maximum(self.K - ST, 0)
        # 边界条件3：S=S_max时，put=0
        self.f_mat[:, self.M] = 0
        self._init_f_mat_flag = True


if __name__ == '__main__':
    # 欧式期权参数
    S0 = 1
    K = 1
    r = 0.03
    q = 0
    T = 1
    sigma = 0.2
    S_max = 4
    S_min = 0
    M = 1000
    N = 252
    fdm_call = VanillaFDM(S0=S0,
                          K=K,
                          r=r,
                          q=q,
                          T=T,
                          sigma=sigma,
                          S_max=S_max,
                          S_min=S_min,
                          M=M,
                          N=N,
                          method='Crank-Nicolson',
                          opt_type='call')
    vanilla_call = fdm_call.get_f_value()
    print('有限差分计算call价值', vanilla_call)
    print('BS计算call价值', euro_bs_call(S0, K, r, T, sigma))
    fdm_put = VanillaFDM(S0=S0,
                         K=K,
                         r=r,
                         T=T,
                         sigma=sigma,
                         S_max=S_max,
                         S_min=S_min,
                         M=M,
                         N=N,
                         opt_type='put')
    vanilla_put = fdm_put.get_f_value()
    print('有限差分计算put价值', vanilla_put)
    print('BS计算put价值', euro_bs_put(S0, K, r, T, sigma))
