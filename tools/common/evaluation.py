"""A collection of evaluation files for multipartite rates."""


def lambda_plus(data: pd.DataFrame, num_parties: int):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    states = data["state"]
    lambda_plus = np.real_if_close(
        [np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0] for state in states]
    )
    return lambda_plus


def lambda_minus(data: pd.DataFrame, num_parties: int):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s - z1s)

    states = data["state"]
    lambda_minus = np.real_if_close(
        [np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0] for state in states]
    )
    return lambda_minus


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


def calculate_keyrate_time(
    lambda_plus, lambda_minus, time_interval, return_std_err=False
):
    e_z = 1 - np.mean(lambda_plus) - np.mean(lambda_minus)
    e_x = 0.5 * (1 - np.mean(lambda_plus) + np.mean(lambda_minus))
    num_pairs = len(lambda_plus)
    pair_per_time = num_pairs / time_interval
    keyrate = pair_per_time * (1 - binary_entropy(e_x) - binary_entropy(e_z))
    if not return_std_err:
        return keyrate
    # WIP CONTINUE HERE


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

    raw_rate = len(data_frame["time"]) / data_frame["time"].iloc[-1]
    fidelity, fidelity_std_err = ghz_fidelity(data=data_frame, num_parties=num_parties)

    lambda_p = lambda_plus(data=data_frame, num_parties=num_parties)
    lambda_m = lambda_minus(data=data_frame, num_parties=num_parties)
