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


class LayerwiseStepper:
    """One epoch per step() call, advancing through layers — drives live streaming."""

    def __init__(self, model: HybridModel, xs: torch.Tensor, ys: torch.Tensor,
                 mode: str = "parallel-shift", epochs_per_layer: int = 8,
                 lr: float = 0.1, shots: int | None = None):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.model, self.xs, self.ys = model, xs, ys
        self.mode, self.epochs_per_layer, self.lr, self.shots = mode, epochs_per_layer, lr, shots
        self.layer = 0
        self.epoch_in_layer = 0
        self.done = False
        self._enter_layer()

    def _enter_layer(self):
        model = self.model
        self._slice = model.qnn.layer_param_slice(self.layer)
        self._head_opt = torch.optim.Adam(model.head.parameters(), lr=self.lr)
        self._layer_params = model.params[self._slice].clone().requires_grad_()
        self._quantum_opt = torch.optim.Adam([self._layer_params], lr=self.lr)

    def step(self) -> tuple[float, float]:
        """Run one full-batch epoch on the current layer -> (mean loss, grad norm)."""
        assert not self.done
        model, qnn, sim = self.model, self.model.qnn, self.model.sim
        total_loss, grad_sum = 0.0, torch.zeros_like(model.params)
        self._head_opt.zero_grad()
        for x, y in zip(self.xs, self.ys):
            loss_fn = lambda z: model.head_loss(z, y)
            if self.mode == "autodiff":
                loss, g = grads_autodiff(sim, qnn, model.params, x, loss_fn,
                                         upto_layer=self.layer)
            elif self.mode == "naive-shift":
                loss, g = grads_naive_shift(sim, qnn, model.params, x, loss_fn,
                                            self.layer, self.shots)
            else:
                loss, g = grads_parallel_shift(sim, qnn, model.params, x, loss_fn,
                                               self.layer, self.shots)
            total_loss += loss
            grad_sum += g
            # head gradient: ordinary backprop through the classical part
            with torch.no_grad():
                z = sim.expect_z(sim.run(qnn.circuit(x, upto_layer=self.layer),
                                         model.params), self.shots)
            model.head_loss(z, y).div(len(self.xs)).backward()
        self._head_opt.step()
        self._quantum_opt.zero_grad()
        self._layer_params.grad = grad_sum[self._slice] / len(self.xs)
        self._quantum_opt.step()
        with torch.no_grad():
            model.params[self._slice] = self._layer_params
        grad_norm = (grad_sum[self._slice] / len(self.xs)).norm().item()

        self.epoch_in_layer += 1
        if self.epoch_in_layer >= self.epochs_per_layer:
            self.layer += 1
            self.epoch_in_layer = 0
            if self.layer >= len(qnn.layers):
                self.done = True
            else:
                self._enter_layer()
        return total_loss / len(self.xs), grad_norm


def train_layerwise(model: HybridModel, xs: torch.Tensor, ys: torch.Tensor,
                    mode: str = "parallel-shift", epochs_per_layer: int = 8,
                    lr: float = 0.1, shots: int | None = None) -> TrainLog:
    model.sim.reset_budget()
    stepper = LayerwiseStepper(model, xs, ys, mode, epochs_per_layer, lr, shots)
    log = TrainLog()
    step = 0
    while not stepper.done:
        if stepper.epoch_in_layer == 0:
            log.layer_boundaries.append(step)
        loss, grad_norm = stepper.step()
        log.losses.append(loss)
        log.grad_norms.append(grad_norm)
        step += 1
    log.executions = model.sim.executions
    log.shots_used = model.sim.shots_used
    return log
