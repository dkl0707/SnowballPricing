<!--
 * @Author: dkl
 * @Description: 用期货对冲的BS方程推导
 * @Date: 2023-09-25 09:58:46
-->
# 用期货对冲的BS方程推导
假设标的资产价格是$S_t$，期货价格是$F_t$，雪球价格是$f_t$。使用无套利定价法，资产组合
$$\Pi_t=-f+\Delta_t F_t$$
同时，因为期货自带贴水属性，所以我们会发现：
$$
F_t=S_te^{-q(T-t)}
$$
其中,q是连续复利下的红利率。对于中证500期货，这个红利率一般来说都有8%左右，不能随意忽略。资产组合可以写成
$$\Pi_t=-f+\Delta_t S_te^{-q(T-t)}$$
继续写偏微分方程。因为无套利定价，所以我们有:
$$d\Pi_t=r\Pi_tdt$$
而前者
$$
\begin{aligned}
d\Pi_t
&=-df+\Delta_t d(S_t e^{-q(T-t)})\\
&=-(\frac{\partial f}{\partial t}+rS\frac{\partial f}{\partial S}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 f}{\partial S^2})dt-\frac{\partial f}{\partial S}\sigma SdW_t+\Delta_t (e^{-q(T-t)}dS_t+qS_te^{-q(T-t)}dt)\\
&=-(\frac{\partial f}{\partial t}+rS\frac{\partial f}{\partial S}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 f}{\partial S^2})dt-\frac{\partial f}{\partial S}\sigma SdW_t+\Delta_t (e^{-q(T-t)}(rSdt+\sigma SdW_t)+qS_te^{-q(T-t)}dt)\\
&=-(\frac{\partial f}{\partial t}+rS\frac{\partial f}{\partial S}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 f}{\partial S^2}-\Delta_t e^{-q(T-t)}rS-qS_te^{-q(T-t)}
)dt+(\Delta_t e^{-q(T-t)}-\frac{\partial f}{\partial S})\sigma SdW_t\\
\end{aligned}
$$
所以波动项
$$
\Delta_t e^{-q(T-t)}-\frac{\partial f}{\partial S}=0
$$
也就是
$$
\Delta_t=\frac{\partial f}{\partial S}e^{q(T-t)}
$$
这和我们直觉上是相符的，在做delta对冲的时候，若用期货对冲，因为期货价格有贴水，所以在份额上就要有所增加，增加的份额正好是$e^{q(T-t)}$倍.

我们再考虑均值项，有
$$
(\frac{\partial f}{\partial t}+rS\frac{\partial f}{\partial S}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 f}{\partial S^2}-\Delta_t e^{-q(T-t)}rS-qS_te^{-q(T-t)}
)+r(-f+\Delta_t S_te^{-q(T-t)})=0
$$
将之前的波动项的结论带入,整理有:
$$
\frac{\partial f}{\partial t}+(r-q) S\frac{\partial f}{\partial S}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 f}{\partial S^2}=rf
$$
这就相当于一个带红利率的BS方程。利用风险中性测度求解，那么等价于假设股票有红利率q的情况下，其随机过程为
$$
dS_t=(r-q)S_tdt+\sigma S_t dW_t
$$
期权定价结果为到期收益$f_T$在Q测度下的贴现均值，也就是
$$
f_t=E^Q[e^{-r(T-t)}f_T]
$$