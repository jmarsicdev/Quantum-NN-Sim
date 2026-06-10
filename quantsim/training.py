"""Layer-wise training: optimize one butterfly layer at a time, freezing the rest.

The quantum circuit feeds per-qubit ⟨Z⟩ into a classical logistic head
(hybrid model, same shape as the source paper). The head trains by ordinary
backprop — it lives on a classical computer even in the hardware setting.
"""
from dataclasses import dataclass, field

import torch

from .butterfly import ButterflyQNN
from .circuit import Simulator
from .gradients import grads_autodiff, grads_naive_shift, grads_parallel_shift

MODES = ("autodiff", "naive-shift", "parallel-shift")


@dataclass
class TrainLog:
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)  # per-step, trained layer only
    layer_boundaries: list[int] = field(default_factory=list)  # step index where each layer starts
    executions: int = 0
    shots_used: int = 0


class HybridModel(torch.nn.Module):
    def __init__(self, qnn: ButterflyQNN, device: str | None = None):
        super().__init__()
        self.qnn = qnn
        self.sim = Simulator(device)
        self.params = qnn.init_params(device=self.sim.device, seed=0)
        self.head = torch.nn.Linear(qnn.n_qubits, 1, device=self.sim.device)

    def head_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logit = self.head(z).squeeze(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logit, y)

    @torch.no_grad()
    def predict(self, xs: torch.Tensor, shots: int | None = None) -> torch.Tensor:
        probs = []
        for x in xs:
            z = self.sim.expect_z(self.sim.run(self.qnn.circuit(x), self.params), shots)
            probs.append(torch.sigmoid(self.head(z).squeeze(-1)))
        return torch.stack(probs)


def train_layerwise(model: HybridModel, xs: torch.Tensor, ys: torch.Tensor,
                    mode: str = "parallel-shift", epochs_per_layer: int = 8,
                    lr: float = 0.1, shots: int | None = None) -> TrainLog:
    assert mode in MODES, f"mode must be one of {MODES}"
    qnn, sim = model.qnn, model.sim
    sim.reset_budget()
    log = TrainLog()
    step = 0
    for layer in range(len(qnn.layers)):
        log.layer_boundaries.append(step)
        sl = qnn.layer_param_slice(layer)
        head_opt = torch.optim.Adam(model.head.parameters(), lr=lr)
        layer_params = model.params[sl].clone()
        quantum_opt = torch.optim.Adam([layer_params.requires_grad_()], lr=lr)
        for _ in range(epochs_per_layer):
            total_loss, grad_sum = 0.0, torch.zeros_like(model.params)
            head_opt.zero_grad()
            for x, y in zip(xs, ys):
                loss_fn = lambda z: model.head_loss(z, y)
                if mode == "autodiff":
                    loss, g = grads_autodiff(sim, qnn, model.params, x, loss_fn,
                                             upto_layer=layer)
                elif mode == "naive-shift":
                    loss, g = grads_naive_shift(sim, qnn, model.params, x, loss_fn,
                                                layer, shots)
                else:
                    loss, g = grads_parallel_shift(sim, qnn, model.params, x, loss_fn,
                                                   layer, shots)
                total_loss += loss
                grad_sum += g
                # head gradient: ordinary backprop through the classical part
                with torch.no_grad():
                    z = sim.expect_z(sim.run(qnn.circuit(x, upto_layer=layer),
                                             model.params), shots)
                model.head_loss(z, y).div(len(xs)).backward()
            head_opt.step()
            quantum_opt.zero_grad()
            layer_params.grad = grad_sum[sl] / len(xs)
            quantum_opt.step()
            with torch.no_grad():
                model.params[sl] = layer_params
            log.losses.append(total_loss / len(xs))
            log.grad_norms.append((grad_sum[sl] / len(xs)).norm().item())
            step += 1
    log.executions = sim.executions
    log.shots_used = sim.shots_used
    return log
