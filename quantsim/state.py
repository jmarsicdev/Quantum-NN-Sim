"""State-vector engine: states are complex tensors of shape (2,)*n_qubits."""
import torch

from .gates import CDTYPE


def zero_state(n_qubits: int, device=None) -> torch.Tensor:
    psi = torch.zeros(2**n_qubits, dtype=CDTYPE, device=device)
    psi[0] = 1.0
    return psi.reshape((2,) * n_qubits)


def apply_gate(psi: torch.Tensor, u: torch.Tensor, qubits: tuple[int, ...]) -> torch.Tensor:
    """Apply a 2^k x 2^k unitary to the given qubit axes via tensor contraction."""
    n = psi.dim()
    k = len(qubits)
    rest = [ax for ax in range(n) if ax not in qubits]
    psi = psi.permute(*qubits, *rest).reshape(2**k, -1)
    psi = u.to(psi.device) @ psi
    psi = psi.reshape((2,) * n)
    inverse = [0] * n
    for i, ax in enumerate((*qubits, *rest)):
        inverse[ax] = i
    return psi.permute(*inverse)


def probabilities(psi: torch.Tensor) -> torch.Tensor:
    return psi.abs().square().reshape(-1)


def reduced_density_matrix(psi: torch.Tensor, qubit: int) -> torch.Tensor:
    """2x2 density matrix of one qubit (partial trace over the rest). For Bloch-sphere viz."""
    n = psi.dim()
    rest = [ax for ax in range(n) if ax != qubit]
    m = psi.permute(qubit, *rest).reshape(2, -1)
    return m @ m.conj().T


def bloch_vector(psi: torch.Tensor, qubit: int) -> torch.Tensor:
    """(x, y, z) Bloch coordinates of one qubit. Length < 1 means entangled with the rest."""
    rho = reduced_density_matrix(psi, qubit)
    bx = 2 * rho[0, 1].real
    by = 2 * rho[1, 0].imag
    bz = (rho[0, 0] - rho[1, 1]).real
    return torch.stack([bx, by, bz])


def entanglement_entropy(psi: torch.Tensor, cut: int) -> torch.Tensor:
    """Von Neumann entropy across a bipartition [0:cut) vs [cut:n). 0 = unentangled."""
    m = psi.reshape(2**cut, -1)
    svals = torch.linalg.svdvals(m)
    p = svals.square()
    p = p[p > 1e-12]
    return -(p * torch.log2(p)).sum()
