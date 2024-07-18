import numpy as np
from requsim.libs.aux_functions import apply_m_qubit_map

from multipartite_requsim.event import (
    ConnectBellsToGHZEvent,
    _generate_GHZ_proj_function,
)
from requsim.world import World
from requsim.quantum_objects import Station, Pair, MultiQubit
from requsim.events import EntanglementSwappingEvent
import requsim.libs.matrix as mat

rho_phiplus = mat.phiplus @ mat.H(mat.phiplus)


def n_qubit_ghz_dm(num_qubits):
    z0s = [mat.z0] * num_qubits
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_qubits
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)
    return ghz_psi @ mat.H(ghz_psi)


def test_produces_ghz_in_ideal():
    for num_parties in range(2, 7):
        world = World()
        central_station = Station(world=world, position=0)
        stations = [Station(world=world, position=100) for i in range(num_parties)]
        pairs = []
        for station in stations:
            qubits = [central_station.create_qubit(), station.create_qubit()]
            new_pair = Pair(world=world, qubits=qubits, initial_state=rho_phiplus)
            pairs += [new_pair]
        event = ConnectBellsToGHZEvent(time=0, pairs=pairs, station=central_station)
        world.event_queue.add_event(event)
        return_value = world.event_queue.resolve_next_event()
        assert len(world.world_objects["Pair"]) == 0
        assert len(world.world_objects[f"{num_parties}-qubit MultiQubit"]) == 1
        output = return_value["output_state"]
        assert np.allclose(output.state, n_qubit_ghz_dm(num_parties))


def test_qubit_ordering_does_not_matter():
    pass


def test_two_pairs_equivalent_to_ent_swap():
    first_state = np.random.random((4, 4))
    first_state = first_state + mat.H(first_state)
    first_state = first_state / np.trace(first_state)
    second_state = np.random.random((4, 4))
    second_state = second_state + mat.H(second_state)
    second_state = second_state / np.trace(second_state)
    world = World()
    left_station = Station(world=world, position=0)
    central_station = Station(world=world, position=100)
    right_station = Station(world=world, position=200)
    left_pair = Pair(
        world=world,
        qubits=[left_station.create_qubit(), central_station.create_qubit()],
        initial_state=first_state,
    )
    right_pair = Pair(
        world=world,
        qubits=[central_station.create_qubit(), right_station.create_qubit()],
        initial_state=second_state,
    )
    event = ConnectBellsToGHZEvent(
        time=0, pairs=[left_pair, right_pair], station=central_station
    )
    world.event_queue.add_event(event)
    return_value = world.event_queue.resolve_next_event()
    output = return_value["output_state"]
    output_state = output.state
    left_pair = Pair(
        world=world,
        qubits=[left_station.create_qubit(), central_station.create_qubit()],
        initial_state=first_state,
    )
    right_pair = Pair(
        world=world,
        qubits=[central_station.create_qubit(), right_station.create_qubit()],
        initial_state=second_state,
    )
    event = EntanglementSwappingEvent(
        time=0, pairs=[left_pair, right_pair], station=central_station
    )
    world.event_queue.add_event(event)
    return_value = world.event_queue.resolve_next_event()
    new_pair = return_value["output_pair"]
    expected_state = new_pair.state
    assert np.allclose(output_state, expected_state)


def test_equivalent_to_previous_implementation():
    class OldImplementation(ConnectBellsToGHZEvent):
        def _main_effect(self):
            """Resolve the event.

            Performs a measurement to connect Bell pairs to a GHZ state.

            Returns
            -------
            dict
                The return_dict of this event is updated with this.
            """
            # some stuff here is written so in theory it should also work with MultiQubit and not only Pair objects
            all_qubits = [qubit for pair in self.pairs for qubit in pair.qubits]
            # assert every pair has exactly one qubit at the station
            for pair in self.pairs:
                assert [(qubit in self.station.qubits) for qubit in pair.qubits].count(
                    True
                ) == 1
            # find qubits at station
            combining_indices = []
            combining_qubits = []
            leftover_qubits = []
            for idx, qubit in enumerate(all_qubits):
                if qubit in self.station.qubits:
                    combining_indices += [idx]
                    combining_qubits += [qubit]
                else:
                    leftover_qubits += [qubit]
            # do state transformation
            num_pairs = len(self.pairs)
            proj_func = _generate_GHZ_proj_function(num_pairs)
            for pair in self.pairs:
                pair.update_time()
            total_state = mat.tensor(*[pair.state for pair in self.pairs])
            new_state = apply_m_qubit_map(
                map_func=proj_func, qubit_indices=combining_indices, rho=total_state
            )
            new_state = mat.ptrace(rho=new_state, sys=combining_indices)
            new_state = new_state / np.trace(new_state)
            output = MultiQubit(
                world=self.pairs[0].world,
                qubits=leftover_qubits,
                initial_state=new_state,
            )
            # cleanup
            for qubit in combining_qubits:
                qubit.destroy()
            for pair in self.pairs:
                pair.destroy()
            return {"output_state": output, "combining_station": self.station}

    for num_parties in range(
        2, 7
    ):  # old implementation fails at higher number of parties
        world = World()
        central_station = Station(world=world, position=0)
        stations = [Station(world=world, position=100) for i in range(num_parties)]
        pairs = []
        states_list = []
        for station in stations:
            qubits = [central_station.create_qubit(), station.create_qubit()]
            initial_state = np.random.random((4, 4))
            initial_state = initial_state + mat.H(initial_state)
            initial_state = initial_state / np.trace(initial_state)
            states_list.append(initial_state)
            new_pair = Pair(world=world, qubits=qubits, initial_state=initial_state)
            pairs += [new_pair]
        event = ConnectBellsToGHZEvent(time=0, pairs=pairs, station=central_station)
        world.event_queue.add_event(event)
        return_value = world.event_queue.resolve_next_event()
        assert len(world.world_objects["Pair"]) == 0
        assert len(world.world_objects[f"{num_parties}-qubit MultiQubit"]) == 1
        output = return_value["output_state"]
        # now do the same with old implementation
        pairs = []
        for station, initial_state in zip(stations, states_list):
            qubits = [central_station.create_qubit(), station.create_qubit()]
            states_list.append(initial_state)
            new_pair = Pair(world=world, qubits=qubits, initial_state=initial_state)
            pairs += [new_pair]
        event = OldImplementation(time=0, pairs=pairs, station=central_station)
        world.event_queue.add_event(event)
        return_value = world.event_queue.resolve_next_event()
        output_compare = return_value["output_state"]
        assert np.allclose(output.state, output_compare.state)
