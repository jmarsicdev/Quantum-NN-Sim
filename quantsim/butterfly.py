"""Butterfly QNN from arXiv:2606.03517: FFT-patterned pairing, O(n log n) params, log depth.

Layer with stride s pairs qubits (i, i+s). Each block on pair (a, b) is
RY(θ0) on a, RY(θ1) on b, RXX(θ2) on (a, b) — 3 params per block.
"""
from dataclasses import dataclass

import torch

from .circuit import Circuit, Op


@dataclass(frozen=True)
class Block:
    a: int
    b: int
    params: tuple[int, int, int]  # (ry_a, ry_b, rxx) indices into the global param vector

ROLES_PER_BLOCK = 3


def butterfly_pairs(n_qubits: int, stride: int) -> list[tuple[int, int]]:
    return [(i, i + stride) for i in range(n_qubits)
            if (i // stride) % 2 == 0 and i + stride < n_qubits]


class ButterflyQNN:
    def __init__(self, n_qubits: int, n_layers: int):
        if n_qubits & (n_qubits - 1):
            raise ValueError("butterfly pattern needs a power-of-2 qubit count")
        self.n_qubits = n_qubits
        self.layers: list[list[Block]] = []
        p = 0
        n_strides = max(n_qubits.bit_length() - 1, 1)  # log2(n) distinct strides, cycled
        for layer_idx in range(n_layers):
            stride = 2 ** (layer_idx % n_strides)
            blocks = []
            for a, b in butterfly_pairs(n_qubits, stride):
                blocks.append(Block(a, b, (p, p + 1, p + 2)))
                p += ROLES_PER_BLOCK
            self.layers.append(blocks)
        self.n_params = p

    def layer_param_slice(self, layer: int) -> slice:
        first = self.layers[layer][0].params[0]
        last = self.layers[layer][-1].params[-1]
        return slice(first, last + 1)

    def circuit(self, x: torch.Tensor, upto_layer: int | None = None) -> Circuit:
        """Angle-encode features x (one per qubit), then butterfly layers [0, upto_layer]."""
        c = Circuit(self.n_qubits)
        for q in range(self.n_qubits):
            c.ops.append(Op("ry", (q,), param=float(x[q])))
        n_layers = len(self.layers) if upto_layer is None else upto_layer + 1
        for blocks in self.layers[:n_layers]:
            for blk in blocks:
                c.ops.append(Op("ry", (blk.a,), param=blk.params[0]))
                c.ops.append(Op("ry", (blk.b,), param=blk.params[1]))
                c.ops.append(Op("rxx", (blk.a, blk.b), param=blk.params[2]))
        return c

    def init_params(self, device=None, seed: int | None = None) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        return (torch.rand(self.n_params, generator=g) * 0.2 - 0.1).to(device)
