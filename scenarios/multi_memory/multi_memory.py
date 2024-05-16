from functools import lru_cache
from warnings import warn

import requsim.libs.matrix as mat
import numpy as np
import pandas as pd
from requsim.events import SourceEvent, EntanglementSwappingEvent
from requsim.libs.aux_functions import distance, apply_single_qubit_map
from requsim.noise import NoiseChannel
from requsim.quantum_objects import MultiQubit, Station, SchedulingSource
from requsim.tools.evaluation import standard_bipartite_evaluation
from requsim.tools.noise_channels import w_noise_channel, z_noise_channel
from requsim.tools.protocol import Protocol
from requsim.world import World
from common.source_functions import (
    generate_round_based_notify_both,
    generate_round_based_notify_one,
    generate_continuous_send_notify_one,
)


from common.consts import (
    SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C,
    ATTENUATION_LENGTH_IN_OPTICAL_FIBER,
)
from multipartite_requsim.event import ConnectBellsToGHZEvent


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_channel)


@lru_cache(maxsize=1000)
def is_source_event_between_stations(event, station1, station2):
    return (
        isinstance(event, SourceEvent)
        and (station1 in event.source.target_stations)
        and (station2 in event.source.target_stations)
    )


@lru_cache(maxsize=1000)
def is_event_connecting_pairs(event, pairs):
    return isinstance(event, ConnectBellsToGHZEvent) and np.all(
        [pair in pairs for pair in event.pairs]
    )


class CentralMultipartyProtocol(Protocol):
    """A GHZ distribution protocol with a single central station.

    Parameters
    ----------
    world : World
    central_station : Station
        The central station that will make the measurement to connect the Bell pairs.
    stations : list[Station]
        The stations of the participating parties, which will end up with the final GHZ state.
    sources : list[Source]
        Sources to create the elementary links between the central station and the others.
        Should be one source per participating party in the same order as stations`.
    communication_speed : scalar
        To calculate delays correctly.
        Default: speed of light in optical fibre 2e8
    """

    def __init__(
        self,
        world,
        central_station,
        stations,
        sources,
        num_memories,
        communication_speed=C,
    ):
        self.central_station = central_station
        self.stations = stations
        self._N = len(stations)
        self.sources = sources
        assert len(self.stations) == len(self.sources)
        self.num_parties = len(self.stations)
        self.communication_speed = communication_speed
        self.num_memories = num_memories
        self.time_list = []
        self.state_list = []
        super(CentralMultipartyProtocol, self).__init__(world=world)

    def setup(self):
        pass

    @property
    def data(self):
        return pd.DataFrame(
            {
                "time": self.time_list,
                "state": self.state_list,
            }
        )

    def _get_pairs_between_stations(self, station1, station2):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(lambda pair: pair.is_between_stations(station1, station2), pairs)
        )

    def _get_pairs_scheduled(self, station1, station2):
        return list(
            filter(
                lambda event: is_source_event_between_stations(
                    event, station1, station2
                ),
                self.world.event_queue.queue,
            )
        )

    def _eval_ghz(self, long_range_ghz: MultiQubit):
        comm_distance = np.max(
            [distance(self.central_station, station) for station in self.stations]
        )
        # comm_distance is simple upper limit for swapping communication
        comm_time = comm_distance / self.communication_speed

        for qubit in long_range_ghz.qubits:
            for time_dependent_noise in qubit._time_dependent_noises:
                qubit.apply_noise(time_dependent_noise, t=comm_time)
        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_ghz.state]
        return

    def check(self, message=None):
        pairs_by_station = [
            self._get_pairs_between_stations(station, self.central_station)
            for station in self.stations
        ]
        num_pairs_by_station = [len(pair_list) for pair_list in pairs_by_station]
        num_pairs_scheduled_by_station = [
            len(self._get_pairs_scheduled(station, self.central_station))
            for station in self.stations
        ]

        try:
            long_distance_ghzs = self.world.world_objects[f"{self._N}-qubit MultiQubit"]
        except KeyError:
            long_distance_ghzs = []

        # STEP 1: For each link, if there are no pairs established and
        #         no pairs scheduled: Schedule a pair.
        for i, station in enumerate(self.stations):
            num_to_schedule = self.num_memories - (
                num_pairs_by_station[i] + num_pairs_scheduled_by_station[i]
            )
            for j in range(num_to_schedule):
                self.sources[i].schedule_event()

        # STEP 2: If all links are present, merge them into a GHZ state.
        num_to_merge = np.min(num_pairs_by_station)
        if num_to_merge:
            # merge the first only since this will be checked again anyways
            pairs = [
                pair_list[-1] for pair_list in pairs_by_station
            ]  # merge newest pairs if possible
            try:
                next(
                    filter(
                        lambda event: is_event_connecting_pairs(event, tuple(pairs)),
                        self.world.event_queue.queue,
                    )
                )
                is_already_scheduled = True
            except StopIteration:
                is_already_scheduled = False
            if not is_already_scheduled:
                connect_event = ConnectBellsToGHZEvent(
                    time=self.world.event_queue.current_time,
                    pairs=pairs,
                    station=self.central_station,
                )
                self.world.event_queue.add_event(connect_event)

        # STEP 3: If a long range GHZ is present, save its data and delete
        #         the associated objects.
        if long_distance_ghzs:
            for ghz in long_distance_ghzs:
                self._eval_ghz(ghz)
                for qubit in ghz.qubits:
                    qubit.destroy()
                ghz.destroy()


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d) ** 2)


def run(
    distance_from_central,
    distance_A,
    num_parties,
    max_iter,
    params,
    num_memories=1,
    mode="distribute",
    source_position="central",
):
    allowed_params = [
        "P_LINK",
        "T_P",
        "P_D",
        "F_INIT",
        "COMMUNICATION_SPEED",
        "T_DP",
        "T_CUT",
    ]
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack parameters
    P_LINK = params.get("P_LINK", 1.0)
    T_P = params.get("T_P", 0)  # preparation time
    P_D = params.get("P_D", 0)  # dark count probability
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    CS = params.get("COMMUNICATION_SPEED", C)  # communication_speed
    T_CUT = params.get("T_CUT", None)  # cutoff time
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(
            e.__traceback__
        )
    L_ATT = ATTENUATION_LENGTH_IN_OPTICAL_FIBER

    N = num_parties

    source_funcs = generate_round_based_notify_both(
        f_init=F_INIT, p_link=P_LINK, t_p=T_P, p_d=P_D, comm_speed=CS
    )
    state_generation = source_funcs["state_generation"]
    time_distribution = source_funcs["time_distribution"]

    # for source_position="outer" and mode="measure"
    state_generation_measure_outer = source_funcs["state_generation_measure_outer"]
    time_distribution_measure_outer = source_funcs["time_distribution_measure_outer"]

    # for source_position="outer" and mode="distribute"
    state_generation_distribute_outer = source_funcs[
        "state_generation_distribute_outer"
    ]
    time_distribution_distribute_outer = source_funcs[
        "time_distribution_distribute_outer"
    ]

    # setup scenario
    world = World()

    central_station = Station(
        world=world,
        position=np.array([0, 0]),
        memory_noise=None,
        memory_cutoff_time=T_CUT,
        dark_count_probability=P_D,
    )

    # asymmetric setup, other_stations[0] is placed distance_A away from central station, other_stations[1:] are placed distance_from_central away from central station
    angles = np.linspace(0, 2 * np.pi, num=N - 1, endpoint=False)
    other_stations = [
        Station(
            world,
            position=np.array([0, distance_A]),
            memory_noise=None,
            memory_cutoff_time=T_CUT,
            dark_count_probability=P_D,
        )
    ] + [
        Station(
            world,
            position=np.array(
                [
                    distance_from_central * np.cos(phi),
                    distance_from_central * np.sin(phi),
                ]
            ),
            memory_noise=None,
            memory_cutoff_time=T_CUT,
            dark_count_probability=P_D,
        )
        for phi in angles
    ]

    if mode == "distribute":
        for station in other_stations + [central_station]:
            station.memory_noise = construct_dephasing_noise_channel(
                dephasing_time=T_DP
            )

        if source_position == "central":
            sources = [
                SchedulingSource(
                    world=world,
                    position=central_station.position,
                    target_stations=[central_station, station],
                    time_distribution=time_distribution,
                    state_generation=state_generation,
                )
                for station in other_stations
            ]

        elif source_position == "outer":
            sources = [
                SchedulingSource(
                    world=world,
                    position=station.position,
                    target_stations=[station, central_station],
                    time_distribution=time_distribution_distribute_outer,
                    state_generation=state_generation_distribute_outer,
                )
                for station in other_stations
            ]

        else:
            raise ValueError(
                f"Unsupported source_position: {source_position}. Supported values are 'central' and 'outer'."
            )

    elif mode == "measure":
        central_station.memory_noise = construct_dephasing_noise_channel(
            dephasing_time=T_DP
        )

        if source_position == "central":
            sources = [
                SchedulingSource(
                    world=world,
                    position=central_station.position,
                    target_stations=[central_station, station],
                    time_distribution=time_distribution,
                    state_generation=state_generation,
                )
                for station in other_stations
            ]

        elif source_position == "outer":
            sources = [
                SchedulingSource(
                    world=world,
                    position=station.position,
                    target_stations=[station, central_station],
                    time_distribution=time_distribution_measure_outer,
                    state_generation=state_generation_measure_outer,
                )
                for station in other_stations
            ]

        else:
            raise ValueError(
                f"Unsupported source_position: {source_position}. Supported values are 'central' and 'outer'."
            )

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Supported modes are 'distribute' and 'measure'."
        )

    protocol = CentralMultipartyProtocol(
        world=world,
        central_station=central_station,
        stations=other_stations,
        sources=sources,
        num_memories=num_memories,
        communication_speed=CS,
    )
    protocol.setup()

    # main loop
    current_message = None
    while len(protocol.time_list) < max_iter:
        # world.print_status()
        # input()
        protocol.check(current_message)
        # world.print_status()
        # input()
        current_message = world.event_queue.resolve_next_event()

    return protocol


def ghz_fidelity(data: pd.DataFrame, num_parties: int):
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


# only correct for one single memory per party
def calculate_keyrate_time(lambda_plus, lambda_minus, time_interval):
    e_z = 1 - np.mean(lambda_plus) - np.mean(lambda_minus)
    e_x = 0.5 * (1 - np.mean(lambda_plus) + np.mean(lambda_minus))
    num_pairs = len(lambda_plus)
    pair_per_time = num_pairs / time_interval
    keyrate = pair_per_time * (1 - binary_entropy(e_x) - binary_entropy(e_z))
    return keyrate


def kilo(list1):
    list2 = []
    for x in list1:
        list2.append(x / 1000)

    return list2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    max_iter = 1
    num_parties = 4
    lengths = np.linspace(1e3, 30e3, num=1)
    fidelities = []
    fidelity_std_err = []
    for length in lengths:
        print(f"{length/1000:.2f}")
        res = run(
            distance_from_central=length,
            distance_A=length,
            num_parties=num_parties,
            max_iter=max_iter,
            params={"P_LINK": 0.01, "T_DP": 100e-3, "F_INIT": 0.99, "T_CUT": None},
            num_memories=1,
            mode="distribute",
            source_position="outer",
        )
        evaluation = ghz_fidelity(data=res.data, num_parties=num_parties)
        fidelities.append(evaluation[0])
        fidelity_std_err.append(evaluation[1])
    fidelities_2 = []
    fidelity_std_err_2 = []
    for length in lengths:
        print(f"{length/1000:.2f}")
        res = run(
            distance_from_central=length,
            distance_A=length,
            num_parties=num_parties,
            max_iter=max_iter,
            params={"P_LINK": 0.01, "T_DP": 100e-3, "F_INIT": 0.99},
            num_memories=5,
            mode="distribute",
            source_position="outer",
        )
        evaluation = ghz_fidelity(data=res.data, num_parties=num_parties)
        fidelities_2.append(evaluation[0])
        fidelity_std_err_2.append(evaluation[1])
    print(f"One memory: {fidelities}")
    print(f"Multiple memories: {fidelities_2}")
    plt.errorbar(lengths / 1000, fidelities, yerr=fidelity_std_err, fmt="o", ms=3)
    plt.errorbar(lengths / 1000, fidelities_2, yerr=fidelity_std_err_2, fmt="o", ms=3)
    plt.xlabel("distance to central station [km]")
    plt.ylabel("average fidelity F")
    plt.grid()
    plt.show()
