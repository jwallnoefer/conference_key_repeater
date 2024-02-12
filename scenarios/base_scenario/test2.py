import numpy as np

p_A = 0.001
tau_A = 1
p_B = 0.1
tau_B = 1


def analytic_formula(p_A, p_B, tau_A, tau_B):
    return p_A * p_B / ((np.exp(tau_A) + p_A - 1) * (np.exp(tau_B) + p_B - 1))


def aux_1(p, tau):
    return (np.exp(tau) * p * (1 - p) - (np.exp(tau) + p - 1)) / (np.exp(tau) + p - 1)


def aux_2(p, tau):
    return np.exp(tau) / (p * np.exp(tau) - np.exp(tau) + 1)


def my_formula(p_A, p_B, tau_A, tau_B):
    return aux_1(p_A, tau_A) / aux_2(p_B, tau_B)


def sampled_formula(p_A, p_B, tau_A, tau_B, num_samples=1000000):
    t_Bs = np.abs(
        np.random.geometric(p_A, size=num_samples) * tau_A
        + np.random.geometric(p_B, size=num_samples) * tau_B
    )
    # t_Bs = np.delete(t_Bs, np.where(t_Bs < 0))
    return np.average(np.exp(-t_Bs))


print(analytic_formula(p_A, p_B, tau_A, tau_B), sampled_formula(p_A, p_B, tau_A, tau_B))
# def analytic_formula(p, tau):
#     return p / (np.exp(tau) + p - 1) / ( np.exp(1/p_B * tau_B))
x = np.logspace(-1, -4, base=10)
y1 = [analytic_formula(a, p_B, tau_A, tau_B) for a in x]
# y1 = [analytic_formula(a, tau_A) for a in x]
y2 = [sampled_formula(a, p_B, tau_A, tau_B) for a in x]
y3 = [my_formula(a, p_B, tau_A, tau_B) for a in x]
import matplotlib.pyplot as plt

plt.plot(x, y1, label="analytic")
plt.plot(x, y2, label="sampled")
plt.plot(x, y3, label="my")
plt.xscale("log")
plt.grid()
plt.legend()
plt.show()

# def analytic_formula(p, tau):
#     return (p / (np.exp(tau) + p - 1))**-1
#
# def sampled_formula(p, tau, num_samples=int(1e6)):
#     ts = np.random.geometric(p, size=num_samples) * tau
#     return np.average(np.exp(-ts))
#
# x = np.logspace(0, -4, base=10)
# y1 = [analytic_formula(p, tau=1) for p in x]
# y2 = [sampled_formula(p, tau=1) for p in x]
# import matplotlib.pyplot as plt
# plt.plot(x, y1, label="analytic")
# plt.plot(x, y2, label="sampled")
# plt.xscale("log")
# plt.grid()
# plt.legend()
# plt.show()
