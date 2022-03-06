
## 中值积分

$$
\begin{aligned}
\text{}^{I_{k+1}}_{G}\hat{\bar{q}}
&= \exp\bigg(\frac{1}{2}\boldsymbol{\Omega}\big({\boldsymbol{\omega}}_{m,k}-\hat{\mathbf{b}}_{g,k}\big)\Delta t\bigg)
\text{}^{I_{k}}_{G}\hat{\bar{q}} \\
^G\hat{\mathbf{v}}_{k+1} &= \text{}^G\hat{\mathbf{v}}_{I_k} - {}^G\mathbf{g}\Delta t
+\text{}^{I_k}_G\hat{\mathbf{R}}^\top(\mathbf{a}_{m,k} - \hat{\mathbf{b}}_{\mathbf{a},k})\Delta t\\
^G\hat{\mathbf{p}}_{I_{k+1}}
&= \text{}^G\hat{\mathbf{p}}_{I_k} + {}^G\hat{\mathbf{v}}_{I_k} \Delta t
- \frac{1}{2}{}^G\mathbf{g}\Delta t^2
+ \frac{1}{2} \text{}^{I_k}_{G}\hat{\mathbf{R}}^\top(\mathbf{a}_{m,k} - \hat{\mathbf{b}}_{\mathbf{a},k})\Delta t^2
\end{aligned}
$$

## RK4积分

* See this wikipedia page on [Runge-Kutta Methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).
* We are doing a RK4 method, [this wolframe page](http://mathworld.wolfram.com/Runge-KuttaMethod.html) has the forth order equation
* defined below. We define function \f$ f(t,y) \f$ where y is a function of time t, see @ref imu_kinematic for the definition of the
* continous-time functions.
$$
\begin{aligned}
{k_1} &= f({t_0}, y_0) \Delta t  \\
{k_2} &= f( {t_0}+{\Delta t \over 2}, y_0 + {1 \over 2}{k_1} ) \Delta t \\
{k_3} &= f( {t_0}+{\Delta t \over 2}, y_0 + {1 \over 2}{k_2} ) \Delta t \\
{k_4} &= f( {t_0} + {\Delta t}, y_0 + {k_3} ) \Delta t \\
y_{0+\Delta t} &= y_0 + \left( {
    {1 \over 6} {k_1} + {1 \over 3} {k_2} + {1 \over 3} {k_3} + {1 \over 6} {
        k_4
    }} \right)
\end{aligned}
$$

