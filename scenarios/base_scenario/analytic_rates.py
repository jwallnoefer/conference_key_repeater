from __future__ import division
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import scipy.special


def y_N(p, N):
    return p**N


def y_2(p, N):
    return p**2 / (N - 1)


def y_N_a(p, N):
    return p


def y_2_a(p, N):
    return p / (N - 1)


def p(d):
    return np.exp(-d / 22)


def d(p):
    return np.log10(p) / (-0.02)


def h(x):
    if x == 0 or x == 1:
        return 0
    elif x < 0:
        return 0
    else:
        entr = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    return entr


# key rate in dependence of qber
def r(x, N):
    return 1 - 2 * h(Q_X(x, N))


def Q_X(x, N):
    return 0.5 * (1 - (1 - x) ** N)


## multipartite key rate
def rnc(f, N, p_a, p_b):
    return 1 - h(Q_AB_new(f, N, p_a, p_b)) - h(Q_X_new(f, N, p_a, p_b))


##  bipartite with memories
def r2(f, p_a, p_b):
    return 1 - h(Q_AB_2_mem(f)) - h(Q_X_2_mem(f, p_a, p_b))


## expected dephasing at Bob's site
def E_b(p_a, p_b):
    # Bob waits for teleportation measurement + correction
    return (
        np.exp(2 * d(p_b) / c)
        * p_a
        * p_b
        / (
            (np.exp((T_p + d(p_a) / c) / T_2) + p_a - 1)
            * (np.exp((T_p + 2 * d(p_b) / c) / T_2) + p_b - 1)
        )
    )

    # Bobs wait, no correction
    # return (
    #     np.exp(d(p_b) / c)
    #     * p_a
    #     * p_b
    #     / (
    #         (np.exp((T_p + d(p_a) / c) / T_2) + p_a - 1)
    #         * (np.exp((T_p + 2 * d(p_b) / c) / T_2) + p_b - 1)
    #     )
    # )

    # Bob measures directly
    # return 1


## expected dephasing at the central station
def E_c(p_a, p_b):
    return (
        np.exp(2 * d(p_b) / c)
        * p_a
        * p_b
        / (
            (np.exp((T_p + d(p_a) / c) / T_2) + p_a - 1)
            * (np.exp((T_p + 2 * d(p_b) / c) / T_2) + p_b - 1)
        )
    )


## intermediate steps
def A(p_a, p_b):
    return 0.5 * (1 + E_b(p_a, p_b) * E_c(p_a, p_b))


def B(p_a, p_b):
    return 0.5 * (1 - E_b(p_a, p_b) * E_c(p_a, p_b))


## bipartite memory qbers
def Q_AB_2_mem(f):
    return f * (1 - f / 2)


def Q_X_2_mem(f, p_a, p_b):
    return (
        (1 - 3 * f / 4) * (1 - f) * B(p_a, p_b)
        + (f / 4) * (1 - f) * A(p_a, p_b)
        + 0.25 * f
        + 0.5 * f * (1 - 0.5 * f)
    )


## multipartite qbers and intermediate definitions
def theta(f, p_a, p_b):
    return (1 - f) * A(p_a, p_b) + f / 4


def phi(f, p_a, p_b):
    return (1 - f) * B(p_a, p_b) + f / 4


def Theta_Phi(f, N, p_a, p_b):
    einszwei = []
    for k in range(0, N):
        sum1 = (
            scipy.special.binom(N - 1, k)
            * (theta(f, p_a, p_b) ** (N - 1 - k))
            * phi(f, p_a, p_b) ** (k)
        )
        einszwei.append(sum1)
    return sum(einszwei)


def Theta_Phi_minus(f, N, p_a, p_b):
    einszwei2 = []
    for k in range(0, N):
        sum2 = (
            ((-1) ** k)
            * scipy.special.binom(N - 1, k)
            * (theta(f, p_a, p_b) ** (N - 1 - k))
            * phi(f, p_a, p_b) ** (k)
        )
        einszwei2.append(sum2)
    return sum(einszwei2)


def only_fd(f, N):
    einsdrei = []
    for k in range(0, N):
        sum2 = scipy.special.binom(N - 1, k) * ((0.25 * f) ** (N - 1))
        einsdrei.append(sum2)
    return sum(einsdrei)


def Q_AB_new(f, N, p_a, p_b):
    return 1 - ((1 - 0.5 * f) * Theta_Phi(f, N, p_a, p_b) + 0.5 * f * only_fd(f, N))


def Q_X_new(f, N, p_a, p_b):
    return 0.5 * (
        1
        - (1 - 0.75 * f) * Theta_Phi_minus(f, N, p_a, p_b)
        + 0.25 * f * Theta_Phi_minus(f, N, p_a, p_b)
    )


## yield per time interval for long link
def yield_per_time_multi(d_A):
    return 1 / (p(d_A) ** (-1) * d_A / c)


def yield_per_time_bi(d_A, N):
    return 1 / ((N - 1) * p(d_A) ** (-1) * d_A / c)


## new multi rate
def multi_rate_per_time(N, d_A, d_B, f):
    return yield_per_time_multi(d_A) * rnc(f, N, p(d_A), p(d_B))


def bi_rate_per_time(N, d_A, d_B, fd):
    return yield_per_time_bi(d_A, N) * r2(fd, p(d_A), p(d_B))


def bi_rate_per_time_nomem(N, d_A, fd):
    return yield_per_time_bi(d_A, N) * r(fd, 2)


## no depolarization in simulation
# speed of light in optical fibre
c = 2 * 10**5
# dephasing time of the QMs
T_2 = 1
# preparation time of a bipartite entangled state
T_p = 2 * 10 ** (-6)
# channel depolarization
fd = 0
# number of participants
N = 4
# length of short links
d_B = 4
dis = np.arange(1, 200, 1)


# Q_X = []
evaluation = []


for i in dis:
    print(i, multi_rate_per_time(N, i, d_B, fd))
    # Q_X.append(Q_X_new(fd, N, p(i), p(d_B)))
    evaluation.append(multi_rate_per_time(N, i, d_B, fd))


np.savez("results/ana_results_original_protocol", array1=dis, array2=evaluation)
