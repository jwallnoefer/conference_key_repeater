import numpy as np
from multipartite_requsim.event import ConnectBellsToGHZEvent
from requsim.world import World
from requsim.quantum_objects import Station, Pair, MultiQubit
from requsim.events import EntanglementSwappingEvent
import requsim.libs.matrix as mat


def n_qubit_ghz_dm(num_qubits):
    z0s = [mat.z0] * num_qubits
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_qubits
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)
    return ghz_psi @ mat.H(ghz_psi)


def test_produces_ghz_in_ideal():
    rho_phiplus = mat.phiplus @ mat.H(mat.phiplus)
    for num_parties in range(2, 7):
        print(num_parties)
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
    pass
