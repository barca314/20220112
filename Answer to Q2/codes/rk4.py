# let [ES] = c,[E] = e,[S] = s,[P] = p, t
# k1 = 100,k2 = 600,k3 = 150
# fs(t, s, c, e, p) = k2*c - k1*s*e
# fe(t, s, c, e, p) = (k3 + k2)*c - k1*s*e
# fc(t, s, c, e, p) = k1*s*e - (k2+k3)*c
# fp(t, s, c, e, p) = k3*c
# e0 = 1; s0 = 10; c0 = 0; p0 = 0
# let h = 0.02

import math
import numpy as np
import matplotlib.pyplot as plt

k1 = 100 # miuM/min
k2 = 600 # /min
k3 = 150 # /min
e0 = 1 # miuM
s0 = 10 # miuM
c0 = 0
p0 = 0
h = 0.00002


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

def RK4(t, s, c, e, p,h):
    """
    t, s, c, e, p is the initial value
    """
    tarry, sarry, carry, earry, parry = [], [], [], [], []
    while t <= 0.01 :
        tarry.append(t)
        sarry.append(s)
        carry.append(c)
        earry.append(e)
        parry.append(p)
        t += h

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

        s = s + h*(S_1 + 2*S_2 + 2*S_3 + S_4)/6
        c = c + h*(C_1 + 2*C_2 + 2*C_3 + C_4)/6
        e = e + h*(E_1 + 2*E_2 + 2*E_3 + E_4)/6
        p = p + h*(P_1 + 2*P_2 + 2*P_3 + P_4)/6

    return tarry, sarry, carry, earry, parry

def initial_RK4(t, s, c, e, p,h):
    """
    t, s, c, e, p is the initial value
    """
    tarry, sarry, carry, earry, parry = [], [], [], [], []
    parray = 0
    while t <= 0.0002 :
        tarry.append(t)
        sarry.append(s)
        carry.append(c)
        earry.append(e)
        parry.append(p)
        t += h

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

        s = s + h*(S_1 + 2*S_2 + 2*S_3 + S_4)/6
        c = c + h*(C_1 + 2*C_2 + 2*C_3 + C_4)/6
        e = e + h*(E_1 + 2*E_2 + 2*E_3 + E_4)/6
        p = p + h*(P_1 + 2*P_2 + 2*P_3 + P_4)/6

        parray += ((P_1 + 2*P_2 + 2*P_3 + P_4)/6)
    parray /= 10

    return parray

def main():
    tarry, sarry, carry, earry, parry= RK4(0, s0, c0, e0, p0, h)
    print("Results of fourth-order Runge-Kutta")
    print('-'*40)
    print("t\t\ts\t\tc\t\te\t\tp\t\t")
    for i in range(len(tarry)):
        print("%.8f\t%.8f\t%.8f\t%.8f\t%.8f" % (tarry[i], sarry[i], carry[i], earry[i], parry[i]), end="")
        print("\n", end="")

    parray_s = []
    sarray = []

    iteration = 200000
    s_max =1300
    for i in range(iteration):
        curr = s_max / iteration * i
        parray_s.append(initial_RK4(0, curr, c0, e0, p0, h))
        sarray.append(curr)
    
    plt.xlabel("concentration",loc='right')
    plt.ylabel("velocity",loc='top')
    plt.plot(sarray,parray_s)
    plt.show()

    # plt.xlabel("time",loc='right')
    # plt.ylabel("concentration",loc='top')
    # plt.plot(sarry,label='s')
    # plt.plot(carry,label='c')
    # plt.plot(earry,label='e')
    # plt.plot(parry,label='p')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
