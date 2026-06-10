"""FastAPI backend for the quantSim dashboard.

Speaks the WebSocket protocol the frontend's mock (SimSocket in
frontend/js/engine.js) defines: server sends hello / config / tick /
run_started / run_paused / run_complete / run_reset; client sends
{cmd: configure|start|pause|step|reset|speed}. Tick frames stream *real*
training data: three models (exact autodiff, naive parameter-shift,
parallelized parameter-shift) trained in lockstep, one epoch per tick.

Run:  uv run uvicorn quantsim.server:app --port 8000
Open: http://localhost:8000/
"""
import asyncio
import math
import time
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .butterfly import ButterflyQNN
from .state import bloch_vector
from .training import HybridModel, LayerwiseStepper

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# protocol key -> training mode
MODES = {"exact": "autodiff", "naive": "naive-shift", "par": "parallel-shift"}


def make_dataset(n_samples: int, n_features: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(n_samples, n_features, generator=g) * math.pi
    ys = ((xs[:, 0] > math.pi / 2) ^ (xs[:, 1] > math.pi / 2)).float()
    return xs, ys


def _bloch_spherical(psi: torch.Tensor, qubit: int) -> dict:
    v = bloch_vector(psi, qubit)
    r = v.norm().item()
    theta = math.acos(max(-1.0, min(1.0, v[2].item() / r))) if r > 1e-9 else math.pi / 2
    phi = math.atan2(v[1].item(), v[0].item()) % (2 * math.pi)
    return {"theta": theta, "phi": phi, "r": r}


class Session:
    def __init__(self, n_qubits: int = 8, n_layers: int = 3,
                 shots: int | None = 2048, epochs_per_layer: int = 12,
                 n_samples: int = 16, device: str | None = None):
        n_qubits = 2 ** max(1, min(int(math.log2(max(2, n_qubits))), 5))
        self.n_qubits, self.n_layers = n_qubits, max(1, n_layers)
        self.shots, self.epochs_per_layer = shots, epochs_per_layer
        self.n_samples, self.device = n_samples, device
        self.xs, self.ys = make_dataset(n_samples, n_qubits)
        self.steppers: dict[str, LayerwiseStepper] = {}
        for key, mode in MODES.items():
            torch.manual_seed(0)  # identical head init across modes
            model = HybridModel(ButterflyQNN(n_qubits, self.n_layers), device=device)
            self.steppers[key] = LayerwiseStepper(
                model, self.xs, self.ys, mode=mode,
                epochs_per_layer=epochs_per_layer,
                shots=None if mode == "autodiff" else shots)
        self.epoch = 0
        self.history = {key: [] for key in MODES}
        self.grads_history: list[list[float | None]] = []
        self._last_layer_grad: list[float | None] = [None] * self.n_layers

    @property
    def total_epochs(self) -> int:
        return self.n_layers * self.epochs_per_layer

    @property
    def done(self) -> bool:
        return self.steppers["par"].done

    @property
    def active_layer(self) -> int:
        return min(self.epoch // self.epochs_per_layer, self.n_layers - 1)

    def config_msg(self) -> dict:
        return {
            "type": "config",
            "nQubits": self.n_qubits,
            "nLayers": self.n_layers,
            "shots": "inf" if self.shots is None else self.shots,
            "epochsPerLayer": self.epochs_per_layer,
            "totalEpochs": self.total_epochs,
            "layerStarts": [l * self.epochs_per_layer for l in range(self.n_layers)],
            "bloch": self._bloch(),
        }

    def _bloch(self) -> list[dict]:
        st = self.steppers["par"]
        with torch.no_grad():
            psi = st.model.sim.run(
                st.model.qnn.circuit(self.xs[0], upto_layer=self.active_layer),
                st.model.params)
        return [_bloch_spherical(psi, q) for q in range(self.n_qubits)]

    def tick(self) -> dict:
        layer = self.steppers["par"].layer
        losses, grad_norm = {}, 0.0
        for key, stepper in self.steppers.items():
            loss, gnorm = stepper.step()
            losses[key] = loss
            if key == "par":
                grad_norm = gnorm
        self._last_layer_grad[layer] = grad_norm
        grads = [self._last_layer_grad[j] if j <= layer else None
                 for j in range(self.n_layers)]
        self.epoch += 1
        for key in MODES:
            self.history[key].append(losses[key])
        self.grads_history.append(grads)
        return self.frame(losses, grads)

    def frame(self, losses: dict | None = None, grads: list | None = None) -> dict:
        if losses is None and self.epoch > 0:
            losses = {key: self.history[key][-1] for key in MODES}
            grads = self.grads_history[-1]
        return {
            "type": "tick",
            "epoch": self.epoch,
            "totalEpochs": self.total_epochs,
            "activeLayer": self.active_layer,
            "done": self.done,
            "loss": losses,
            "grads": grads,
            "bloch": self._bloch(),
            "budget": {key: st.model.sim.executions
                       for key, st in self.steppers.items()},
        }


app = FastAPI(title="quantSim")


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "quantSim.html")


app.mount("/js", StaticFiles(directory=FRONTEND / "js"), name="js")


@app.websocket("/quantsim/v1")
async def quantsim_ws(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(obj: dict):
        async with send_lock:
            await ws.send_json(obj)

    state = {"session": Session(), "running": False, "speed": 3.0, "task": None}
    await send({"type": "hello", "server": "quantsim/0.1.0", "protocol": 1})
    await send(state["session"].config_msg())

    async def run_loop():
        loop = asyncio.get_running_loop()
        while state["running"] and not state["session"].done:
            t0 = time.monotonic()
            frame = await loop.run_in_executor(None, state["session"].tick)
            await send(frame)
            if state["session"].done:
                state["running"] = False
                await send({"type": "run_complete"})
                return
            await asyncio.sleep(max(0.0, 1.0 / state["speed"] - (time.monotonic() - t0)))

    def stop_loop():
        state["running"] = False
        if state["task"] is not None:
            state["task"].cancel()
            state["task"] = None

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "configure":
                stop_loop()
                shots = msg.get("shots")
                state["session"] = Session(
                    n_qubits=int(msg.get("nQubits", 8)),
                    n_layers=int(msg.get("nLayers", 3)),
                    shots=None if shots in ("inf", None) else int(shots))
                await send(state["session"].config_msg())
            elif cmd == "start":
                if state["session"].done:
                    s = state["session"]
                    state["session"] = Session(s.n_qubits, s.n_layers, s.shots,
                                               s.epochs_per_layer, s.n_samples)
                    await send(state["session"].config_msg())
                if not state["running"]:
                    state["running"] = True
                    state["task"] = asyncio.create_task(run_loop())
                await send({"type": "run_started"})
            elif cmd == "pause":
                state["running"] = False
                await send({"type": "run_paused"})
            elif cmd == "step":
                state["running"] = False
                if not state["session"].done:
                    loop = asyncio.get_running_loop()
                    frame = await loop.run_in_executor(None, state["session"].tick)
                    await send(frame)
                if state["session"].done:
                    await send({"type": "run_complete"})
            elif cmd == "reset":
                stop_loop()
                s = state["session"]
                state["session"] = Session(s.n_qubits, s.n_layers, s.shots,
                                           s.epochs_per_layer, s.n_samples)
                await send(state["session"].config_msg())
                await send({"type": "run_reset"})
            elif cmd == "speed":
                state["speed"] = max(0.25, min(float(msg.get("value", 3)), 12.0))
    except WebSocketDisconnect:
        stop_loop()
