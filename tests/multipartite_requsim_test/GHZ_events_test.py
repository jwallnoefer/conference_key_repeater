import numpy as np
from multipartite_requsim.event import ConnectBellsToGHZEvent, MergeToGHZEvent
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


def test_merging_produces_ghz_in_ideal():
    for num_parties in range(2, 9):
        world = World()
        central_station = Station(world=world, position=0)
        stations = [Station(world=world, position=100) for i in range(num_parties)]
        pairs = []
        for station in stations:
            qubits = [central_station.create_qubit(), station.create_qubit()]
            new_pair = Pair(world=world, qubits=qubits, initial_state=rho_phiplus)
            pairs += [new_pair]
        event = MergeToGHZEvent(time=0, inputs=pairs, station=central_station)
        world.event_queue.add_event(event)
        return_value = world.event_queue.resolve_next_event()
        assert len(world.world_objects["Pair"]) == 0
        assert len(world.world_objects[f"{num_parties+1}-qubit MultiQubit"]) == 1
        output = return_value["output_state"]
        assert np.allclose(output.state, n_qubit_ghz_dm(num_parties + 1))
