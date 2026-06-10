"""Three ways to get gradients of a loss over per-qubit ⟨Z⟩ expectations.

- autodiff:      exact backprop through the state vector. Impossible on real
                 hardware (intermediate states are unobservable) — our ground truth.
- naive shift:   parameter-shift rule, 2 circuit executions per parameter.
- parallel shift: the paper's trick — within one butterfly layer, blocks act on
                 disjoint qubit pairs, so shifting the same role of every block
                 simultaneously yields all gradients from local ⟨Z⟩ readouts in
                 2 × ROLES_PER_BLOCK executions, independent of qubit count.
                 Only valid when the trained layer is the last one in the circuit
                 (nothing downstream mixes the lightcones), which is exactly the
                 layer-wise training setting.
"""
import math
from typing import Callable

import torch

from .butterfly import ROLES_PER_BLOCK, ButterflyQNN
from .circuit import Simulator

LossFn = Callable[[torch.Tensor], torch.Tensor]  # z_exps (n_qubits,) -> scalar loss

SHIFT = math.pi / 2


def _loss_and_dldz(z: torch.Tensor, loss_fn: LossFn) -> tuple[float, torch.Tensor]:
    z = z.detach().requires_grad_()
    loss = loss_fn(z)
    (dldz,) = torch.autograd.grad(loss, z)
    return loss.item(), dldz


def grads_autodiff(sim: Simulator, qnn: ButterflyQNN, params: torch.Tensor,
                   x: torch.Tensor, loss_fn: LossFn,
                   upto_layer: int | None = None) -> tuple[float, torch.Tensor]:
    p = params.detach().clone().requires_grad_()
    z = sim.expect_z(sim.run(qnn.circuit(x, upto_layer), p))
    loss = loss_fn(z)
    (grad,) = torch.autograd.grad(loss, p)
    return loss.item(), grad


def grads_naive_shift(sim: Simulator, qnn: ButterflyQNN, params: torch.Tensor,
                      x: torch.Tensor, loss_fn: LossFn, layer: int,
                      shots: int | None = None) -> tuple[float, torch.Tensor]:
    """2 executions per parameter of the trained layer."""
    circuit = qnn.circuit(x, upto_layer=layer)
    with torch.no_grad():
        z = sim.expect_z(sim.run(circuit, params), shots)
    loss, dldz = _loss_and_dldz(z, loss_fn)
    grad = torch.zeros_like(params)
    sl = qnn.layer_param_slice(layer)
    with torch.no_grad():
        for p_idx in range(sl.start, sl.stop):
            dz = torch.zeros_like(z)
            for sign in (1.0, -1.0):
                shifted = params.clone()
                shifted[p_idx] += sign * SHIFT
                dz += sign * sim.expect_z(sim.run(circuit, shifted), shots)
            grad[p_idx] = dldz @ (dz / 2)
    return loss, grad


def grads_parallel_shift(sim: Simulator, qnn: ButterflyQNN, params: torch.Tensor,
                         x: torch.Tensor, loss_fn: LossFn, layer: int,
                         shots: int | None = None) -> tuple[float, torch.Tensor]:
    """2 × ROLES_PER_BLOCK executions total, regardless of qubit count.

    Requires `layer` to be the last layer in the executed circuit (layer-wise
    training): each block's params then only influence ⟨Z⟩ on its own two qubits.
    """
    circuit = qnn.circuit(x, upto_layer=layer)
    blocks = qnn.layers[layer]
    with torch.no_grad():
        z = sim.expect_z(sim.run(circuit, params), shots)
    loss, dldz = _loss_and_dldz(z, loss_fn)
    grad = torch.zeros_like(params)
    with torch.no_grad():
        for role in range(ROLES_PER_BLOCK):
            dz = torch.zeros_like(z)
            for sign in (1.0, -1.0):
                shifted = params.clone()
                for blk in blocks:  # disjoint supports -> shift them all at once
                    shifted[blk.params[role]] += sign * SHIFT
                dz += sign * sim.expect_z(sim.run(circuit, shifted), shots)
            dz /= 2
            for blk in blocks:
                grad[blk.params[role]] = (
                    dldz[blk.a] * dz[blk.a] + dldz[blk.b] * dz[blk.b])
    return loss, grad
