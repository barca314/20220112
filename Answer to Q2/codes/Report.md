# Question2:Enzyme Kinetics
## 8.1 Using the law of mass action, write down four equations for the rate of changes of the four species, E, S, ES, and P.
### Solution:
&emsp;The four equations are below:
$$\frac{d[S]}{dt} = k_2[ES]-k_1[E][S]$$
$$\frac{d[E]}{dt} = (k_3+k_2)[ES] - k_1[E][S]$$
$$\frac{d[ES]}{dt} = k_1[E][S] - (k_3 + k_2)[ES]$$
$$\frac{d[P]}{dt} = k_3[ES]$$

## 8.2 Write a code to numerically solve these four equations using the fourth-order Runge-Kutta method. For this exercise, assume that the initial concentration of E is 1 μM, the initial concentration of S is 10 μM, and the initial concentrations of ES and P are both 0. The rate constants are: k1=100/μM/min, k2=600/min, k3=150/min.
### Solution:
&emsp;For simplicity,let $[S]$ equal s,let $[ES]$ equal c, let $[E]$ equal e, let $[P]$ equal p. And the four equations above in code like below:
$$fs(t,s,c,e,p) = k_2c-k_1es$$
$$fe(t,s,c,e,p) = (k_3+k_2)c - k_1es$$
$$fc(t,s,c,e,p) = k_1es - (k_3 + k_2)c$$
$$fp(t,s,c,e,p) = k_3c$$
We define the function in the code like below:<br>
```python
def fs(t, s, c, e, p):
    """ds/dt"""
    return k2*c - k1*s*e

def fe(t, s, c, e, p):
    """de/dt"""
    return (k3 + k2)*c - k1*s*e

def fc(t, s, c, e, p):
    """dc/dt"""
    return k1*s*e - (k2+k3)*c

def fp(t, s, c, e, p):
    """dp/dt"""
    return k3*c    
```
&emsp;For $s$,use $S_-1,S_-2,S_-3,S_-4$ represent the four parameters $K_1,K_2,K_3,K_4$ in the fourth-order Runge-Kutta method. Similiarly, for e, the parameters are $E_-1,E_-2,$$E_-3,E_-4$, for c, they are $C_-1,C_-2,C_-3,C_-4$, and $P_-1,P_-2,P_-3,P_-4$. After many attempts, set the step size $h$ to $0.00002$ and set the range of t as $(0,0.01)$. In my solution, I use the classic fourth-order Runge-Kutta method, and calculate the parameters like below:<br>
```python
S_1 = fs(t, s, c, e, p)
E_1 = fe(t, s, c, e, p)
C_1 = fc(t, s, c, e, p)
P_1 = fp(t, s, c, e, p)

S_2 = fs(t + h/2, s + h*S_1/2, c + h*C_1/2, e + h*E_1/2, p + h*P_1/2)
E_2 = fe(t + h/2, s + h*S_1/2, c + h*C_1/2, e + h*E_1/2, p + h*P_1/2)
C_2 = fc(t + h/2, s + h*S_1/2, c + h*C_1/2, e + h*E_1/2, p + h*P_1/2)
P_2 = fp(t + h/2, s + h*S_1/2, c + h*C_1/2, e + h*E_1/2, p + h*P_1/2)

S_3 = fs(t + h/2, s + h*S_2/2, c + h*C_2/2, e + h*E_2/2, p + h*P_2/2)
E_3 = fe(t + h/2, s + h*S_2/2, c + h*C_2/2, e + h*E_2/2, p + h*P_2/2)
C_3 = fc(t + h/2, s + h*S_2/2, c + h*C_2/2, e + h*E_2/2, p + h*P_2/2)
P_3 = fp(t + h/2, s + h*S_2/2, c + h*C_2/2, e + h*E_2/2, p + h*P_2/2)

S_4 = fs(t + h, s + h*S_3, c + h*C_3, e + h*E_3, p + h*P_3)
E_4 = fe(t + h, s + h*S_3, c + h*C_3, e + h*E_3, p + h*P_3)
C_4 = fc(t + h, s + h*S_3, c + h*C_3, e + h*E_3, p + h*P_3)
P_4 = fp(t + h, s + h*S_3, c + h*C_3, e + h*E_3, p + h*P_3)
```
Then get the iteration value like below:<br>
```python
s = s + h*(S_1 + 2*S_2 + 2*S_3 + S_4)/6
c = c + h*(C_1 + 2*C_2 + 2*C_3 + C_4)/6
e = e + h*(E_1 + 2*E_2 + 2*E_3 + E_4)/6
p = p + h*(P_1 + 2*P_2 + 2*P_3 + P_4)/6
```
Part of the code running results like below<br>
```
Results of fourth-order Runge-Kutta
----------------------------------------
t               s               c               e               p
0.00000000      10.00000000     0.00000000      1.00000000      0.00000000
0.00002000      9.98033561      0.01963476      0.98036524      0.00002963
0.00004000      9.96132523      0.03855767      0.96144233      0.00011710
0.00006000      9.94294388      0.05679582      0.94320418      0.00026029
0.00008000      9.92516761      0.07437517      0.92562483      0.00045721
0.00010000      9.90797347      0.09132062      0.90867938      0.00070591
0.00012000      9.89133944      0.10765603      0.89234397      0.00100453
0.00014000      9.87524441      0.12340432      0.87659568      0.00135126
0.00016000      9.85966815      0.13858747      0.86141253      0.00174439
0.00018000      9.84459120      0.15322655      0.84677345      0.00218224
......
```
Collect all the data every time, and use the matplotlib python library to plot the concentration of s, c, e and p as funtions of the time t, and we get the image below:<br>
<div align="center"><img src=./Figure_1.svg width=60% align=center/></div><br>

## 8.3 We define the velocity, V, of the enzymatic reaction to be the rate of change of the product P. Plot the velocity V as a function of the concentration of the substrate S. You should find that, when the concentrations of S are small, the velocity V increases approximately linearly. At large concentrations of S, however, the velocity V saturates to a maximum value, Vm. Find this value Vm from your plot.
### Solution:
&emsp;To get the rate of change of the profuct P, I use part of the process of the RK method. As we all konw, to a certain extent, the four parameters are the slope of the curve which is the function between the concertain and the time. And the slope is just the velocity, so I calculate the average of leading ten times parameter of the RK method as what we should get. The code is just similar to the second question. Then use the matplotlib to plot the curve of the concentration and the velocity. When the concentration is small, we can see that the velocity V increases approximately linearly, the image like below:<br>
<div align="center"><img src=./Figure_2.svg width=60% align=center/></div><br>
When the concentration is large, we can see that the velocity V saturates to a maximum value about 160, the image like below:<br>
<div align="center"><img src=./Figure_3.svg width=60% align=center/></div><br>