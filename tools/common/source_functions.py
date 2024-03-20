from functools import lru_cache

import numpy as np
import requsim.libs.matrix as mat
from requsim.libs.aux_functions import distance, apply_single_qubit_map
from requsim.tools.noise_channels import w_noise_channel
from .consts import SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C
from .consts import ATTENUATION_LENGTH_IN_OPTICAL_FIBER as L_ATT


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d) ** 2)


def generate_round_based_notify_both(f_init, p_link, t_p, p_d, comm_speed=C):
    @lru_cache()  # caching only makes sense with stationary stations and sources
    def state_generation(source):
        state = f_init * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - f_init) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        station_distance = distance(
            source.target_stations[0], source.target_stations[1]
        )
        comm_distance = (
            np.max(
                [
                    distance(source, source.target_stations[0]),
                    distance(source, source.target_stations[1]),
                ]
            )
            + station_distance
        )
        time_until_ready = comm_distance / comm_speed
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                # while qubit and classical information are travelling dephasing already occurs
                # the amount of time depends on where the station is located
                storage_time = time_until_ready - distance(source, station) / C
                state = apply_single_qubit_map(
                    map_func=station.memory_noise,
                    qubit_index=idx,
                    rho=state,
                    t=storage_time,
                )
            if station.dark_count_probability is not None:
                # dark counts are handled here because the information about eta is needed for that
                eta = p_link * np.exp(-station_distance / L_ATT)
                state = apply_single_qubit_map(
                    map_func=w_noise_channel,
                    qubit_index=idx,
                    rho=state,
                    alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability),
                )
        return state

    def time_distribution(source):
        # this is specifically for a source that is housed directly at a station, otherwise what
        # constitutes one trial is more involved
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source, source.target_stations[1]),
            ]
        )
        comm_time = 2 * comm_distance / C
        eta = p_link * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - p_d) ** 2
        trial_time = t_p + comm_time
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time

    def state_generation_distribute_outer(source):
        state = f_init * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - f_init) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        station_distance = distance(
            source.target_stations[0], source.target_stations[1]
        )
        comm_distance = (
            np.max(
                [
                    distance(source, source.target_stations[0]),
                    distance(source, source.target_stations[1]),
                ]
            )
            + station_distance
        )
        time_until_ready = station_distance / comm_speed
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                # while qubit and classical information are travelling dephasing already occurs
                # the amount of time depends on where the station is located
                storage_time = time_until_ready - distance(source, station) / C
                state = apply_single_qubit_map(
                    map_func=station.memory_noise,
                    qubit_index=idx,
                    rho=state,
                    t=storage_time,
                )
            if station.dark_count_probability is not None:
                # dark counts are handled here because the information about eta is needed for that
                eta = p_link * np.exp(-station_distance / L_ATT)
                state = apply_single_qubit_map(
                    map_func=w_noise_channel,
                    qubit_index=idx,
                    rho=state,
                    alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability),
                )
        return state

    def time_distribution_distribute_outer(source):
        # this is specifically for a source that is housed directly at a station, otherwise what
        # constitutes one trial is more involved
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source, source.target_stations[1]),
            ]
        )
        comm_time = 2 * comm_distance / C
        eta = p_link * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - p_d) ** 2
        trial_time = t_p + comm_time
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time - comm_time / 2

    # specifically for mode="measure" and source_position="outer"
    def state_generation_measure_outer(source):
        state = f_init * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - f_init) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        station_distance = distance(
            source.target_stations[0], source.target_stations[1]
        )
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                # while qubit and classical information are travelling dephasing already occurs
                # the amount of time depends on where the station is located
                storage_time = 0
                state = apply_single_qubit_map(
                    map_func=station.memory_noise,
                    qubit_index=idx,
                    rho=state,
                    t=storage_time,
                )
            if station.dark_count_probability is not None:
                # dark counts are handled here because the information about eta is needed for that
                eta = p_link * np.exp(-station_distance / L_ATT)
                state = apply_single_qubit_map(
                    map_func=w_noise_channel,
                    qubit_index=idx,
                    rho=state,
                    alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability),
                )
        return state

    def time_distribution_measure_outer(source):
        # this is specifically for a source that is housed directly at a station, otherwise what
        # constitutes one trial is more involved
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source, source.target_stations[1]),
            ]
        )
        eta = p_link * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - p_d) ** 2
        random_num = np.random.geometric(eta_effective)
        return random_num * t_p

    return {
        "state_generation": state_generation,
        "time_distribution": time_distribution,
        "state_generation_distribute_outer": state_generation_distribute_outer,
        "time_distribution_distribute_outer": time_distribution_distribute_outer,
        "state_generation_measure_outer": state_generation_measure_outer,
        "time_distribution_measure_outer": time_distribution_measure_outer,
    }


def generate_round_based_notify_one(f_init, p_link, t_p, p_d, comm_speed=C):
    @lru_cache()  # caching only makes sense with stationary stations and sources
    def state_generation(source):
        state = f_init * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - f_init) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        station_distance = distance(
            source.target_stations[0], source.target_stations[1]
        )
        comm_distance = (
            np.min(
                [
                    distance(source, source.target_stations[0]),
                    distance(source, source.target_stations[1]),
                ]
            )
            + station_distance
        )
        time_until_ready = comm_distance / comm_speed
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                # while qubit and classical information are travelling dephasing already occurs
                # the amount of time depends on where the station is located
                storage_time = time_until_ready - distance(source, station) / C
                state = apply_single_qubit_map(
                    map_func=station.memory_noise,
                    qubit_index=idx,
                    rho=state,
                    t=storage_time,
                )
            if station.dark_count_probability is not None:
                # dark counts are handled here because the information about eta is needed for that
                eta = p_link * np.exp(-station_distance / L_ATT)
                state = apply_single_qubit_map(
                    map_func=w_noise_channel,
                    qubit_index=idx,
                    rho=state,
                    alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability),
                )
        return state

    def time_distribution(source):
        # this is specifically for a source that is housed directly at a station, otherwise what
        # constitutes one trial is more involved
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source, source.target_stations[1]),
            ]
        )
        comm_time = 2 * comm_distance / C
        eta = p_link * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - p_d) ** 2
        trial_time = t_p + comm_time
        random_num = np.random.geometric(eta_effective)
        return (
            random_num * trial_time - comm_distance / C
        )  # last communication to outer station is not counted

    return {
        "state_generation": state_generation,
        "time_distribution": time_distribution,
    }


def generate_continuous_send_notify_one(f_init, p_link, t_p, p_d, comm_speed=C):
    @lru_cache()  # caching only makes sense with stationary stations and sources
    def state_generation(source):
        state = f_init * (mat.phiplus @ mat.H(mat.phiplus)) + (1 - f_init) / 3 * (
            mat.psiplus @ mat.H(mat.psiplus)
            + mat.phiminus @ mat.H(mat.phiminus)
            + mat.psiminus @ mat.H(mat.psiminus)
        )
        station_distance = distance(
            source.target_stations[0], source.target_stations[1]
        )
        for idx, station in enumerate(source.target_stations):
            if station.dark_count_probability is not None:
                # dark counts are handled here because the information about eta is needed for that
                eta = p_link * np.exp(-station_distance / L_ATT)
                state = apply_single_qubit_map(
                    map_func=w_noise_channel,
                    qubit_index=idx,
                    rho=state,
                    alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability),
                )
        return state

    def time_distribution(source):
        # this is specifically for a source that is housed directly at one of the stations
        # this is for continuous generation of states with clock_rate 1/t_p
        # where receiver station can simply close input once successful arrival is detected
        comm_distance = np.max(
            [
                distance(source, source.target_stations[0]),
                distance(source, source.target_stations[1]),
            ]
        )
        eta = p_link * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - p_d) ** 2
        trial_time = t_p
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time

    return {
        "state_generation": state_generation,
        "time_distribution": time_distribution,
    }
