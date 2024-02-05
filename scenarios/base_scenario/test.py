import re

import numpy as np
from requsim.libs.aux_functions import apply_m_qubit_map
from requsim.tools.noise_channels import z_noise_channel
import requsim.libs.matrix as mat

from multipartite_requsim.event import _generate_GHZ_proj_function
from requsim.tools.evaluation import binary_entropy


N_BOBS = 3
p_A = 0.001
tau_A = 1
p_B = 0.1
tau_B = 1


def dephasing_param(t):
    return np.exp(-t / 100)


def average_separate(num_samples=10):
    t_Bs = (
        np.random.geometric(p_A, size=num_samples) * tau_A
        - np.random.geometric(p_B, size=num_samples) * tau_B
    )
    return np.average(dephasing_param(t_Bs)) ** N_BOBS


def average_together(num_samples=10):
    N_As = np.random.geometric(p_A, size=num_samples)
    aux = []
    for i in range(N_BOBS):
        N_B = np.random.geometric(p_B, size=num_samples)
        t_Bs = N_As * tau_A - N_B * tau_B
        aux.append(dephasing_param(t_Bs))
    return np.average(np.prod(aux, axis=0))


samples = int(1e6)
print(average_together(samples), average_separate(samples))

# def sample_separate():
#     res = 1
#     for i in range(N_BOBS):
#         t_B = np.random.geometric(p_A) * tau_A - np.random.geometric(p_B) * tau_B
#         res = res * dephasing_param(t_B)
#     return res


def project_to_ghz(pair_states):
    num_pairs = len(pair_states)
    proj_func = _generate_GHZ_proj_function(num_pairs)
    total_state = mat.tensor(*[state for state in pair_states])
    combining_indices = list(np.arange(0, 2 * num_pairs, 2))
    new_state = apply_m_qubit_map(
        map_func=proj_func, qubit_indices=combining_indices, rho=total_state
    )
    new_state = mat.ptrace(rho=new_state, sys=combining_indices)
    new_state = new_state / np.trace(new_state)
    return new_state


def lambda_plus(state, num_parties: int):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    lambda_plus = np.real_if_close(np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0])
    return lambda_plus


def lambda_minus(state, num_parties: int):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s - z1s)

    lambda_minus = np.real_if_close(
        np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0]
    )
    return lambda_minus


def extract_lengths_and_keyrates(line):
    match = re.search(r"Task \d+: Length = ([\d.]+), Rate = ([\d.-]+)", line)
    if match:
        length = float(match.group(1))
        keyrate = float(match.group(2))
        return length, keyrate
    return None


def get_with_average(
    d_A, d_B, F_INIT, P_LINK, T_P, T_DP, num_parties, comm_speed=2e8, L_ATT=22e3
):
    state = F_INIT * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - F_INIT) / 3 * (
        mat.psiplus @ mat.H(mat.psiplus)
        + mat.phiminus @ mat.H(mat.phiminus)
        + mat.psiminus @ mat.H(mat.psiminus)
    )
    state_A = state
    f_D = 4 / 3 * (1 - F_INIT)
    tau_A = T_P + d_A / comm_speed
    tau_B = T_P + 2 * d_B / comm_speed
    p_A = P_LINK * np.exp(-d_A / L_ATT)
    p_B = P_LINK * np.exp(-d_B / L_ATT)

    special_exp = (
        np.exp(2 * d_B / comm_speed)
        * p_A
        * p_B
        / ((np.exp(tau_A / T_DP) + p_A - 1) * (np.exp(tau_B / T_DP) + p_B - 1))
    )

    lambda_dp = (1 - special_exp) / 2

    state = z_noise_channel.apply_to(rho=state, qubit_indices=[0], epsilon=lambda_dp)
    state = z_noise_channel.apply_to(rho=state, qubit_indices=[1], epsilon=lambda_dp)

    pair_states = [state_A] + [state] * (num_parties - 1)
    final_state = project_to_ghz(pair_states)
    a = lambda_plus(final_state, num_parties)
    b = lambda_minus(final_state, num_parties)
    e_z = 1 - a - b
    e_x = 0.5 * (1 - a + b)
    fraction = 1 - binary_entropy(e_x) - binary_entropy(e_z)
    rate = 1 / (p_A ** (-1) * d_A / comm_speed)
    return rate * fraction


def global_average(
    d_A,
    d_B,
    F_INIT,
    P_LINK,
    T_P,
    T_DP,
    num_parties,
    comm_speed=2e8,
    L_ATT=22e3,
    num_samples=100,
):
    state = F_INIT * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - F_INIT) / 3 * (
        mat.psiplus @ mat.H(mat.psiplus)
        + mat.phiminus @ mat.H(mat.phiminus)
        + mat.psiminus @ mat.H(mat.psiminus)
    )
    state_A = state
    f_D = 4 / 3 * (1 - F_INIT)
    tau_A = T_P + d_A / comm_speed
    tau_B = T_P + 2 * d_B / comm_speed
    p_A = P_LINK * np.exp(-d_A / L_ATT)
    p_B = P_LINK * np.exp(-d_B / L_ATT)

    N_As = np.random.geometric(p_A, size=num_samples)
    aux = []
    for i in range(num_parties - 1):
        N_B = np.random.geometric(p_B, size=num_samples)
        t_Bs = N_As * tau_A - N_B * tau_B + 2 * d_B / comm_speed
        epsilons = (1 - np.exp(-t_Bs / T_DP)) / 2
        states = [
            z_noise_channel.apply_to(
                z_noise_channel.apply_to(state, [0], epsilon=epsilon),
                [1],
                epsilon=epsilon,
            )
            for epsilon in epsilons
        ]
        aux.append(states)
        # aux.append(np.average(states, axis=0))

    # pair_states = [state_A] + aux
    # final_state = project_to_ghz(pair_states)
    pair_states_by_trial = [[state_A] + list(x) for x in zip(*aux)]
    final_state = np.average(
        [project_to_ghz(pair_states) for pair_states in pair_states_by_trial], axis=0
    )
    a = lambda_plus(final_state, num_parties)
    b = lambda_minus(final_state, num_parties)
    e_z = 1 - a - b
    e_x = 0.5 * (1 - a + b)
    fraction = 1 - binary_entropy(e_x) - binary_entropy(e_z)
    rate = 1 / (p_A ** (-1) * d_A / comm_speed)
    return rate * fraction


T_2 = 1
# preparation time of a bipartite entangled state
T_p = 2 * 10 ** (-6)
# channel depolarization
F_INIT = 1.0
# number of participants
N = 4
# length of short links
d_B = 4000
dis = np.arange(1e3, 200e3, 1e3)

res = [
    get_with_average(
        d_A=length, d_B=d_B, F_INIT=F_INIT, P_LINK=1, T_P=T_p, T_DP=T_2, num_parties=N
    )
    for length in dis
]
res2 = [
    global_average(
        d_A=length,
        d_B=d_B,
        F_INIT=F_INIT,
        P_LINK=1,
        T_P=T_p,
        T_DP=T_2,
        num_parties=N,
        num_samples=100,
    )
    for length in dis
]
print(res2)
l1 = np.load("results/ana_results_bobs_measure_directly.npz")
print(l1.files)
x1, y1 = l1["array1"], l1["array2"]
l2 = np.load("results/ana_results_bobs_wait_no_correction.npz")
x2, y2 = l2["array1"], l2["array2"]
l3 = np.load("results/ana_results_original_protocol.npz")
x3, y3 = l3["array1"], l3["array2"]

with open("results/num_keyrates.rtf", "r") as file:
    lines = file.readlines()

lengths = []
keyrates = []

for line in lines:
    info = extract_lengths_and_keyrates(line)
    if info:
        length, keyrate = info
        lengths.append(length)
        keyrates.append(keyrate)

import matplotlib.pyplot as plt

plt.plot(x1 * 1000, y1, label="direct")
plt.plot(x2 * 1000, y2, label="wait")
plt.plot(x3 * 1000, y3, label="original")
plt.plot(dis, res, label="new")
plt.plot(dis, res2, label="aaaaaah")
plt.scatter(lengths, keyrates, label="sim")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()
