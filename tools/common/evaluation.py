"""A collection of evaluation files for multipartite rates."""
from warnings import warn

import numpy as np
import pandas as pd
import requsim.libs.matrix as mat


def binary_entropy(p):
    """Calculate the binary entropy.

    Parameters
    ----------
    p : scalar
        Must be in interval [0, 1]. Usually an error rate.

    Returns
    -------
    scalar
        The binary entropy of `p`.

    """
    if p == 1 or p == 0:
        return 0
    elif p < 0:  # can happen numerically, when it should be 0
        return 0
    else:
        res = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        if np.isnan(res):
            warn(f"binary_entropy was called with p={p} and returned nan")
        return res


def calculate_qber_x(data: pd.DataFrame, num_parties: int):
    # span measurement operator
    Xs = [mat.X] * num_parties
    X_all = mat.tensor(*Xs)

    states = data["state"]
    qx_array = []
    for state in states:
        qx = 0.5 * (1 - np.trace(X_all @ state))
        qx_array.append(np.real_if_close(qx))

    return qx_array


def calculate_qber_z(data: pd.DataFrame, num_parties: int):
    states = data["state"]

    # compute all Z_0 Z_i operators and store them
    operators = []

    for i in range(1, num_parties):
        ops = []
        for j in range(num_parties):
            if j == 0 or j == i:
                ops.append(mat.Z)
            else:
                ops.append(mat.I(2))
        operators.append(mat.tensor(*ops))

    qz_array = []

    for state in states:
        qz_values = [0.5 * (1 - np.trace(op @ state)) for op in operators]
        qz_values = np.real_if_close(qz_values)
        qz_array.append(np.max(qz_values))

    return qz_array


def calculate_keyrate_time(ez_array, ex_array, time_interval, return_std_err=False):
    e_z = np.mean(ez_array)
    e_x = np.mean(ex_array)
    num_ghz = len(ez_array)
    ghz_per_time = num_ghz / time_interval
    keyrate = ghz_per_time * (1 - binary_entropy(e_x) - binary_entropy(e_z))
    if not return_std_err:
        return keyrate

    variance_z = np.std(ez_array)
    variance_x = np.std(ex_array)

    if e_z == 0:
        keyrate_std = ghz_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * variance_x
        )
    else:
        keyrate_std = ghz_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * variance_x
            + (-np.log2(e_z) + np.log2(1 - e_z)) ** 2 * variance_z
        )
    keyrate_std_err = keyrate_std / np.sqrt(num_ghz)
    return keyrate, keyrate_std_err


# def calculate_keyrate_time(
#     lambda_plus, lambda_minus, time_interval, return_std_err=False
# ):
#     e_z = 1 - np.mean(lambda_plus) - np.mean(lambda_minus)
#     e_x = 0.5 * (1 - np.mean(lambda_plus) + np.mean(lambda_minus))
#     num_ghz = len(lambda_plus)
#     ghz_per_time = num_ghz / time_interval
#     keyrate = ghz_per_time * (1 - binary_entropy(e_x) - binary_entropy(e_z))
#     if not return_std_err:
#         return keyrate
#     variance_z = np.std(lambda_plus) ** 2 + np.std(lambda_minus) ** 2
#     variance_x = 0.5**2 * (np.std(lambda_plus) ** 2 + np.std(lambda_minus) ** 2)

#     if e_z == 0:
#         keyrate_std = ghz_per_time * np.sqrt(
#             (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * variance_x
#         )
#     else:
#         keyrate_std = ghz_per_time * np.sqrt(
#             (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * variance_x
#             + (-np.log2(e_z) + np.log2(1 - e_z)) ** 2 * variance_z
#         )
#     keyrate_std_err = keyrate_std / np.sqrt(num_ghz)
#     return keyrate, keyrate_std_err


def ghz_fidelity(data: pd.DataFrame, num_parties):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    states = data["state"]
    fidelity_list = np.real_if_close(
        [np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0] for state in states]
    )
    fidelity = np.mean(fidelity_list)
    fidelity_std_err = np.std(fidelity_list) / np.sqrt(len(fidelity_list))
    return fidelity, fidelity_std_err


def standard_ghz_evaluation(data_frame, num_parties=None):
    """Evaluate quality and speed of GHZ state distribution.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame with columns "time" and "state", representing
        when each state distribtuion was made and the density matrix
        associated with it.
    num_parties : int or None
        The number of parties sharing the GHZ state. If None, will be
        inferred from first state in data_frame["state"]. Default: None

    Returns
    -------
    list of scalars
        contains: raw rate,
                  average fidelity,
                  standard error of the mean of fidelity,
                  average asymptotic key rate per time,
                  standard error of the mean of key rate per time
    """
    if num_parties is None:  # infer num_parties
        num_parties = int(np.log2(data_frame["state"][0].shape[0]))

    time_interval = data_frame["time"].iloc[-1]
    raw_rate = len(data_frame["time"]) / time_interval
    fidelity, fidelity_std_err = ghz_fidelity(data=data_frame, num_parties=num_parties)

    ex_array = calculate_qber_x(data=data_frame, num_parties=num_parties)
    ez_array = calculate_qber_z(data=data_frame, num_parties=num_parties)

    e_z = np.mean(ez_array)
    e_x = np.mean(ex_array)

    key_per_time, key_per_time_std_err = calculate_keyrate_time(
        ez_array=ez_array,
        ex_array=ex_array,
        time_interval=time_interval,
        return_std_err=True,
    )
    return [
        raw_rate,
        fidelity,
        fidelity_std_err,
        key_per_time,
        key_per_time_std_err,
        e_x,
        e_z,
    ]
