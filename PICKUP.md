# Pick-up point

Last touched 2026-06-10 on the WSL2 VM (CPU build). Everything is committed and
pushed through `f05592a`. This file is the to-do list for resuming on the
desktop (Arch + RTX 5080).

## Where things stand

- [x] Simulator core — state vectors, differentiable gates, shot sampling,
      execution-budget counter
- [x] Butterfly QNN, layer-wise trainer, hybrid model
- [x] Three gradient modes, verified equivalent (autodiff / naive shift /
      parallel shift) — 22 tests passing
- [x] Live dashboard: FastAPI/WebSocket backend + frontend with circuit
      diagram, Bloch spheres, loss curves, plateau heatmap, budget odometers
- [x] Model analysis feature: live commentary feed, timeline replay scrubber,
      end-of-run training report (`quantsim/analysis.py`, commit `f05592a`)
- [x] Educational README
- [ ] **GPU build on the desktop ← start here**
- [ ] Phase 4: real dataset (UCI heart disease)
- [ ] Phase 5: publish as a hosted educational tool

## First task on the desktop: GPU build

1. `pyproject.toml` line 27: change `whl/cpu` → `whl/cu128`
   (Blackwell needs CUDA 12.8 wheels; comment at line 23 says the same).
2. `uv sync --extra dev --extra server`
3. Sanity check:
   `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
4. `uv run pytest` — should still be 22 passed (tests run on CPU tensors but
   confirm nothing broke in the swap).
5. Start the dashboard and crank qubits past what the VM could handle —
   the simulator auto-detects cuda. Practical ceiling ≈ 28 qubits at
   complex64 on 16 GB. Watch the barren-plateau monitor dim as n grows;
   that's experiment #2 from the README actually working at scale.

## Phase 4: UCI heart disease

Goal: replace the synthetic two-moons-style data with a real task so the
dashboard tells a real story.

- Dataset: UCI Heart Disease (Cleveland), 303 rows, 13 features, binary label.
  Public, no credentialing.
- Plan: small data module (`quantsim/data.py`) — load CSV, scale features to
  rotation-angle range, reduce 13 → n_qubits features (top-k by variance or
  PCA), train/test split.
- Server: add a dataset option to the config message; report held-out test
  accuracy per mode in the training report (the report builder already takes
  `accuracy` per mode — just feed it test-set numbers instead of train).
- Frontend: dataset picker in Run control; otherwise everything (curves,
  analysis feed, report) works unchanged.

## Phase 5: publishing as an educational tool

Two-lane idea sketched earlier:

- **In-browser**: the mock engine (`frontend/js/engine.js`, `QuantSimEngine`)
  already has full parity with the live protocol — small-qubit runs can stay
  fully client-side, zero backend cost.
- **Recorded replays**: the replay machinery added in `f05592a` stores
  per-epoch frames client-side; recording real GPU runs to a JSON frame
  stream and replaying them through the same scrubber is the natural
  implementation. Big-qubit runs become static files.
- Then: static hosting (GitHub Pages or similar), no live server needed.

## Commands

```bash
uv sync --extra dev --extra server      # install (flip index for GPU first)
uv run pytest                           # 22 tests
uv run python examples/train_demo.py    # terminal 3-mode comparison
uv run uvicorn quantsim.server:app --port 8000    # dashboard
```

## Workflow notes

- Design iterations went through Claude's design tool (prompt → design link →
  integrate). If new design files are dropped into `frontend/`, re-apply the
  live-socket patches (LiveSocket in `engine.js`, socket pick in `app.jsx`) —
  see git history of commit `7e03906`.
- Browser verification: headless Playwright driving system chromium
  (`p.chromium.launch(executable_path="/usr/bin/chromium")`). The script used
  for the analysis feature lived at `/tmp/verify_analysis.py` (not committed);
  key checks were: feed populates with tagged items, scrubbing rewinds every
  panel + shows the replay badge, report auto-opens with 3 mode rows and
  takeaways, mock parity via `?live=0`.
- `?live=0` query param forces the mock SimSocket over http — that's how the
  frontend is previewed without a backend (file:// does NOT work: Babel can't
  XHR the split jsx files cross-origin).
