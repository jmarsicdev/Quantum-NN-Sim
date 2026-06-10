"""Circuit representation and a simulator that tracks the hardware execution budget."""
from dataclasses import dataclass, field

import torch

from . import gates
from .state import apply_gate, probabilities, zero_state


@dataclass(frozen=True)
class Op:
    gate: str
    qubits: tuple[int, ...]
    # int: index into the trainable parameter vector; float: fixed angle; None: non-parameterized
    param: int | float | None = None


@dataclass
class Circuit:
    n_qubits: int
    ops: list[Op] = field(default_factory=list)

    def add(self, gate: str, *qubits: int, param: int | float | None = None) -> "Circuit":
        self.ops.append(Op(gate, qubits, param))
        return self

    @property
    def n_params(self) -> int:
        return 1 + max((op.param for op in self.ops if isinstance(op.param, int)), default=-1)


class Simulator:
    """Runs circuits; counts executions and shots the way a hardware budget would.

    One "execution" = one prepare-run-measure pass (at whatever shot count),
    the unit the source paper's 1,792-vs-28 comparison is measured in.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.executions = 0
        self.shots_used = 0

    def reset_budget(self):
        self.executions = 0
        self.shots_used = 0

    def run(self, circuit: Circuit, params: torch.Tensor | None = None,
            state: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass -> final state. Does not touch the budget (measurement does)."""
        psi = state if state is not None else zero_state(circuit.n_qubits, self.device)
        for op in circuit.ops:
            if op.param is None:
                u = gates.FIXED[op.gate](device=self.device)
            else:
                theta = params[op.param] if isinstance(op.param, int) else torch.tensor(
                    op.param, device=self.device)
                u = gates.PARAMETERIZED[op.gate](theta)
            psi = apply_gate(psi, u, op.qubits)
        return psi

    def expect_z(self, psi: torch.Tensor, shots: int | None = None) -> torch.Tensor:
        """⟨Z_q⟩ for every qubit. shots=None gives the exact value (impossible on hardware);
        a finite shot count samples bitstrings and estimates, like real hardware."""
        n = psi.dim()
        self.executions += 1
        if shots is None:
            probs = psi.abs().square()
            return torch.stack(
                [probs.sum(dim=[ax for ax in range(n) if ax != q]) @ torch.tensor(
                    [1.0, -1.0], device=psi.device) for q in range(n)])
        self.shots_used += shots
        samples = torch.multinomial(probabilities(psi), shots, replacement=True)
        bits = (samples.unsqueeze(1) >> torch.arange(n - 1, -1, -1, device=psi.device)) & 1
        return 1.0 - 2.0 * bits.float().mean(dim=0)
