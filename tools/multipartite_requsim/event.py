import numpy as np
from requsim.events import Event
import requsim.libs.matrix as mat
from requsim.libs.aux_functions import apply_m_qubit_map
from requsim.quantum_objects import MultiQubit


def _generate_GHZ_proj_function(num_qubits):
    z0s = [mat.z0] * num_qubits
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_qubits
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)
    proj = ghz_psi @ mat.H(ghz_psi)

    def proj_func(rho):
        return proj @ rho @ proj

    return proj_func


def _generate_merge_proj_func(num_qubits):
    z0s = [mat.z0] * num_qubits
    z0s = mat.tensor(*z0s)
    z0s_minus_1 = [mat.z0] * (num_qubits - 1)
    z0s_minus_1 = mat.tensor(*z0s_minus_1)
    z1s = [mat.z1] * num_qubits
    z1s = mat.tensor(*z1s)
    proj = z0s @ mat.H(z0s) + mat.tensor(mat.z1, z0s_minus_1) @ mat.H(z1s)

    def proj_func(rho):
        return proj @ rho @ mat.H(proj)

    return proj_func


def _project_on_x0(rho):
    proj = mat.x0 @ mat.H(mat.x0)
    return proj @ rho @ mat.H(proj)


class ConnectBellsToGHZEvent(Event):
    """Connect multiple Bell pairs meeting at a station to a GHZ state.

    Note: if only two Bell pairs are specified this is equivalent to entanglement swapping.

    Additional information in return dict of resolve method:

    "output_state" : MultiQubit
        The final GHZ state generated by the connection operation.
    "connecting_station" : Station
        The station where the connection operation was performed.

    Parameters
    ----------
    time : scalar
        Time at which the event will be resolved.
    pairs : list[Pair or MultiQubit]
        The left pair and the right pair.
    station : Station
        The station where the entanglement swapping is performed.
    callback_functions : list of callables, or None
        these will be called in order, after the event has been resolved.
        Callbacks can also be added with the add_callback method.
        Default: None

    """

    def __init__(self, time, pairs, station, callback_functions=None):
        self.pairs = pairs
        self.station = station
        super(ConnectBellsToGHZEvent, self).__init__(
            time=time,
            required_objects=pairs + [qubit for pair in pairs for qubit in pair.qubits],
            callback_functions=callback_functions,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(time={self.time}, pairs={self.pairs}), "
            + f"station={self.station}, "
            + f"callback_functions={self._callback_functions}"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} at time={self.time} using pairs "
            + ", ".join([x.label for x in self.pairs])
            + "."
        )

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
        for pair in self.pairs:
            pair.update_time()
        first_pair = self.pairs[0]
        for idx, qubit in enumerate(first_pair.qubits):
            if qubit in self.station.qubits:
                first_qubit = qubit
                first_index = idx
                break
        state = first_pair.state
        combining_qubits = []
        leftover_qubits = list(first_pair.qubits)
        # maybe this whole thing could be transformed into some tensor transformation
        proj_func = _generate_merge_proj_func(num_qubits=2)
        # combine states one by one to avoid building a very large density matrix
        for current_input in self.pairs[1:]:
            current_length = len(leftover_qubits)
            for idx, qubit in enumerate(current_input.qubits):
                if qubit in self.station.qubits:
                    additional_index = idx
                    combining_qubits += [qubit]
                else:
                    leftover_qubits += [qubit]
            second_index = current_length + additional_index
            state = mat.tensor(state, current_input.state)
            state = apply_m_qubit_map(
                map_func=proj_func,
                qubit_indices=[first_index, second_index],
                rho=state,
            )
            state = mat.ptrace(rho=state, sys=[second_index])
        # then also measure the remaining qubit for the first pair
        state = apply_m_qubit_map(
            map_func=_project_on_x0, qubit_indices=[first_index], rho=state
        )
        state = mat.ptrace(rho=state, sys=[first_index])
        leftover_qubits.remove(first_qubit)
        combining_qubits.append(first_qubit)
        state = state / np.trace(state)
        output = MultiQubit(
            world=self.pairs[0].world,
            qubits=leftover_qubits,
            initial_state=state,
        )
        # cleanup
        for qubit in combining_qubits:
            qubit.destroy()
        for pair in self.pairs:
            pair.destroy()
        return {"output_state": output, "connecting_station": self.station}


class MergeBellToGHZEvent(Event):
    pass
