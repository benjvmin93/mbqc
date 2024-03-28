import numpy as np
from .clifford import X, Z, get, TABLE
from . import pauli
from .command import Plane
from .logger import logger
from .state import plus

import random

# rng = np.random.default_rng()

CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)

SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


def meas_op(vec: tuple[float, float, float], measurement: int):
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        cliff = get(i + 1).matrix
        sign = (-1) ** measurement
        op_mat += sign * vec[i] * cliff / 2
    return op_mat


class StateVec:
    def __init__(self, input_nodes=[]):
        """
        Initalize a new state vector in the |+> state.
        """
        nQubits = len(input_nodes)
        self.psi = np.ones((2,) * nQubits) / 2 ** (
            nQubits / 2
        )  # Initialize statevector in |+> ⊗^n
        self.node_index = input_nodes

    def __repr__(self) -> str:
        return str(f"{self.psi.flatten()}")

    def __eq__(self, __value: object) -> bool:
        return np.array_equal(self.psi.flatten(), np.array(__value))

    @property
    def norm(self) -> float:
        """
        Returns the norm of the state vector.
        """
        return _norm(self.psi)

    def tensor(self, other: np.ndarray):
        """
        Performs self ⊗ other.
        """
        new_shape = len(self.psi.shape) + len(other.shape)
        self.psi = np.kron(self.psi.flatten(), other.flatten()).reshape(
            (2,) * new_shape
        )

    def get_state_vector(self) -> np.ndarray:
        return self.psi

    def prepare_state(self, target: int) -> None:
        """
        Append new qubit to the end of self.psi and update self.node_index
        We assume that 'target' doesn't exist in the state vector because we
        shouldn't prepare inputs qubits nor qubits that have already been prepared
        """
        new_qubit = plus
        self.tensor(new_qubit)
        self.node_index.append(target)
        # logger.debug(
        #     "[N]({index}): statevec={flat}, shape={shape}".format(
        #         index=self.node_index.index(target),
        #         flat=self.psi.flatten(),
        #         shape=self.psi.flatten().shape,
        #     )
        # )

    def entangle(self, control: int, target: int) -> None:
        """
        Entangles the two qubits by applying CZ on target according to control.
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        control = self.node_index.index(control)
        target = self.node_index.index(target)
        self.psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), (control, target)))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))
        # logger.info(f"Entangling qubit {control} with qubit {target}")
        # logger.debug(f"[E]({control},{target}): statevec={self.psi.flatten()}")

    def swap(self, qubits: tuple[int, int]) -> None:
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (i, j) qubit indices
        """
        # logger.info(f"Swap qubits {qubits[0]} with {qubits[1]}")

        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)
        # logger.debug(
        #     f"[SWAP]({qubits[0]},{qubits[1]}): statevec={self.psi.flatten()}, shape={self.psi.shape}"
        # )

    def measure(
        self,
        index: int,
        plane: Plane,
        angle: int,
        s_domain: list[int],
        t_domain: list[int],
        measurements: list[int],
        vop: int = 0,
    ) -> int:
        """
        Measure the qubit at index.
        Returns:
            list[int]: The updated measurements list.
        """
        # logger.info(f"Measuring qubit {index} in plane {plane} and angle {angle}.")

        # Get projected states
        s_signal = sum([measurements[i] for i in s_domain])
        t_signal = sum([measurements[i] for i in t_domain])

        measure_update = pauli.MeasureUpdate.compute(
            pauli.Plane[plane], s_signal % 2 == 1, t_signal % 2 == 1, TABLE[vop]
        )
        angle *= np.pi
        angle = angle * measure_update.coeff + measure_update.add_term
        vec = measure_update.new_plane.polar(angle)

        op = meas_op(vec, measurement=0)  # Assume measurement == 0
        loc = self.node_index.index(index)  # Get right index within state vector
        proba_Plus = self.expectation_single(op, loc)
        proba_Minus = 1 - proba_Plus
        measurement = random.choices([0, 1], weights=[proba_Plus, proba_Minus]).pop()
        if measurement == 1:
            op = meas_op(vec, measurement=1)  # Re-compute measurement operator

        # logger.debug(
        #     "[M]({index}): projected_state={projstate}, shape={shape}".format(
        #         index=index, projstate=flat, shape=flat.shape
        #     )
        # )

        # Project state
        self.psi = self.single_qubit_evolution(op, loc)

        # Remove measured qubit from state vector
        self.remove_qubit(loc)
        # Remove qubit index from node list
        self.node_index.remove(index)

        return measurement

    def apply_correction(
        self, type: str, index: int, domain: list[int], measurement_results: list[int]
    ) -> None:
        """
        Applies correction 'X' or 'Z' to the qubit at 'index' according to the signal domain measurements.
        """
        if sum([measurement_results[i] for i in domain]) % 2 == 1:
            # Get right index within self.node_index
            sv_index = self.node_index.index(index)
            cliff_gate = X if type == "X" else Z
            self.psi = self.single_qubit_evolution(cliff_gate.matrix, sv_index)
            # logger.info(f"[{type}]({index}): new_psi={self.psi.flatten()}")

    def single_qubit_evolution(self, op: np.ndarray, index: int):
        """
        Apply one qubit operator to |psi> at right index.
        """
        psi = np.tensordot(op, self.psi, (1, index))
        psi = np.moveaxis(psi, 0, index)
        return psi

    def multi_qubit_evolution(self, op: np.ndarray, qargs: tuple[int, int]) -> None:
        """
        Apply multi qubit operator to |psi> with (control, target) in qargs.
        """
        op_dim = int(np.log2(len(op)))
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        self.psi = np.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = np.moveaxis(self.psi, [i for i in range(len(qargs))], qargs)

    def normalize(self) -> None:
        """
        Normalize vector state (ie. divides it by its norm).
        """
        norm = _norm(self.psi)
        self.psi /= norm

    def remove_qubit(self, index: int) -> None:
        """
        Remove qubit at 'index' from the state vector.
        """
        norm = _norm(self.psi)
        assert not np.isclose(norm, 0)
        psi = self.psi.take(indices=0, axis=index)
        psi_norm = _norm(psi)
        self.psi = (
            psi if not np.isclose(psi_norm, 0) else self.psi.take(indices=1, axis=index)
        )
        self.normalize()

    def expectation_single(self, op: np.ndarray, index: int) -> np.complex128:
        self.normalize()
        evolved = self.single_qubit_evolution(op, index)
        return np.dot(self.psi.flatten().conjugate(), evolved.flatten())


def _norm(psi: np.ndarray) -> float:
    """
    Computes the norm of a state vector.
    """
    return np.sqrt(abs(sum(psi.flatten().conj() * psi.flatten())))
