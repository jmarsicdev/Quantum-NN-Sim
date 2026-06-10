"""Gate matrices. Parameterized gates are differentiable functions of theta."""
import torch

CDTYPE = torch.complex64


def _c(x: torch.Tensor) -> torch.Tensor:
    return x.to(CDTYPE)


def h(device=None) -> torch.Tensor:
    return torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE, device=device) / 2**0.5


def x(device=None) -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=device)


def cnot(device=None) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=CDTYPE, device=device,
    )


def rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = _c(torch.cos(theta / 2)), _c(torch.sin(theta / 2))
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = _c(torch.cos(theta / 2)), _c(torch.sin(theta / 2))
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rz(theta: torch.Tensor) -> torch.Tensor:
    e_neg = torch.polar(torch.ones_like(theta), -theta / 2).to(CDTYPE)
    e_pos = torch.polar(torch.ones_like(theta), theta / 2).to(CDTYPE)
    zero = torch.zeros_like(e_neg)
    return torch.stack([torch.stack([e_neg, zero]), torch.stack([zero, e_pos])])


def rxx(theta: torch.Tensor) -> torch.Tensor:
    """exp(-i theta/2 X⊗X). Generator has eigenvalues ±1 → parameter-shift applies."""
    c, s = _c(torch.cos(theta / 2)), _c(torch.sin(theta / 2))
    eye = torch.eye(4, dtype=CDTYPE, device=theta.device)
    xx = torch.zeros(4, 4, dtype=CDTYPE, device=theta.device)
    xx[0, 3] = xx[1, 2] = xx[2, 1] = xx[3, 0] = 1
    return c * eye - 1j * s * xx


PARAMETERIZED = {"rx": rx, "ry": ry, "rz": rz, "rxx": rxx}
FIXED = {"h": h, "x": x, "cnot": cnot}
