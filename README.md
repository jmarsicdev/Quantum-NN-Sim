# quantSim

A from-scratch PyTorch quantum circuit simulator + quantum neural network
trainer, replicating the training method of
[arXiv:2606.03517](https://arxiv.org/abs/2606.03517) (Butterfly circuits,
layer-wise training, parallelized parameter-shift gradients), with an
educational visualization layer planned on top.

New to quantum computing? Start with [RESEARCH.md](RESEARCH.md).

## Setup

```bash
uv sync --extra dev --extra server   # CPU wheels (dev VM)
uv run pytest                        # verify the physics
uv run python examples/train_demo.py

# live training dashboard:
uv run uvicorn quantsim.server:app --port 8000   # then open http://localhost:8000/
```

**On the desktop (RTX 5080):** the 5080 is Blackwell (sm_120) and needs the
CUDA 12.8 PyTorch build. Edit `pyproject.toml`, change the index URL
`whl/cpu` → `whl/cu128`, then `uv sync --extra dev`. All code is
device-agnostic — pass `device="cuda"` to `Simulator`/`HybridModel` (it
auto-detects by default). Practical ceiling ≈ 28 qubits at complex64 on 16GB.

## Layout

- `quantsim/state.py` — state vectors, gate application (einsum), Bloch
  vectors, entanglement entropy (the visualization hooks)
- `quantsim/gates.py` — differentiable parameterized gate matrices
- `quantsim/circuit.py` — circuit repr + `Simulator` with a hardware
  execution-budget counter and shot-noise measurement
- `quantsim/butterfly.py` — the paper's Butterfly architecture
- `quantsim/gradients.py` — autodiff (ground truth) vs naive parameter-shift
  vs the paper's parallelized parameter-shift
- `quantsim/training.py` — layer-wise trainer, hybrid quantum-classical model
- `quantsim/server.py` — FastAPI + WebSocket backend streaming live training
  data to the dashboard (protocol defined by `frontend/js/engine.js`)
- `frontend/` — dashboard from Claude design; runs against its built-in mock
  when opened as a file, and against the live backend when served over http
  (`?live=0` to force the mock)
- `tests/` — physics correctness + gradient-equivalence proofs + WS protocol

## Status / roadmap

- [x] Phase 1: simulator core
- [x] Phase 2: Butterfly QNN, three gradient modes, layer-wise training
- [x] Phase 3: live web dashboard (FastAPI + WebSocket; Bloch spheres,
      per-layer gradient norms / barren plateaus, budget comparison)
      — pending: drop in refined design, verify in a browser
- [ ] Phase 4: real-dataset task (UCI heart disease)
- [ ] Phase 5: publish as educational tool (in-browser small circuits +
      recorded GPU training replays)
