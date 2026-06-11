# quantSim

**Watch a neural network train on a (simulated) quantum computer — and see
exactly why that's hard.**

quantSim is a from-scratch quantum circuit simulator with a live dashboard
that visualizes quantum neural network (QNN) training in real time. It
replicates the training method from
[arXiv:2606.03517](https://arxiv.org/abs/2606.03517), a 2026 paper that
demonstrated practical QNN training on real trapped-ion quantum hardware.

No quantum background needed — start here, then go deeper with
[RESEARCH.md](RESEARCH.md).

---

## The big idea in one paragraph

A quantum neural network is a machine learning model where the "network" is a
quantum circuit: a sequence of operations applied to qubits, with tunable
knobs (rotation angles) instead of weights. You feed data in, measure the
qubits to get numbers out, and adjust the knobs to reduce a loss — same recipe
as any neural network. The catch: on a real quantum computer **you cannot use
backpropagation**, because looking inside a quantum computer mid-computation
destroys the state. Every gradient must instead be estimated by *re-running
the whole circuit many times*, which gets expensive fast. This project is a
playground for seeing that problem, and the paper's clever fix for it,
with your own eyes.

## A crash course (5 minutes)

### 1. A qubit register is just a big vector

A classical n-bit register holds *one* of 2ⁿ values. An n-qubit quantum state
holds a **complex number for every one of the 2ⁿ values simultaneously** —
called *amplitudes*. So the state of 8 qubits is a vector of 256 complex
numbers. If you know machine learning: it's an embedding vector, except the
entries are complex and their squared magnitudes must sum to 1.

That 2ⁿ growth is the whole story of quantum computing: 30 qubits already
means a billion amplitudes, which is why simulating them classically hits a
wall, and why real quantum hardware is interesting.

### 2. Gates are matrix multiplies

Every quantum operation ("gate") is a matrix multiplied against that state
vector. Like a neural network layer — but with no nonlinearity, no bias, and
the matrix must be *unitary* (it preserves the vector's length, so nothing is
ever lost or amplified). Two kinds matter here:

- **Rotation gates** like `Ry(θ)` act on one qubit. The angle θ is the
  *learnable parameter* — the quantum analogue of a weight.
- **Entangling gates** like `Rxx(θ)` act on *two* qubits at once and tie
  their fates together ("entanglement"). This is what lets n qubits represent
  things that n independent qubits can't — the resource classical networks
  don't have.

### 3. Getting an answer out = sampling

You can't read the amplitudes. The only way information leaves a quantum
computer is **measurement**: the register collapses to one classical
bitstring, with probability equal to the squared amplitude (like sampling a
token from a softmax). One run = one sample. To estimate anything useful —
e.g. "how often is qubit 3 a zero?" — you re-run the circuit hundreds or
thousands of times (**"shots"**) and average. Fewer shots = noisier numbers.
The dashboard's shot slider controls exactly this.

The quantity we use is each qubit's **⟨Z⟩ expectation**: +1 if the qubit is
always 0, −1 if always 1, in between otherwise. Those n numbers are the
"output layer" of our quantum network.

### 4. The full model, end to end

```
features x  ──►  Ry(x_i) per qubit  ──►  trainable layers  ──►  measure ⟨Z⟩  ──►  tiny classical
 (classical)       "encoding"            (Ry + Rxx knobs)       (n numbers)      logistic head ──► prediction
```

This "hybrid quantum-classical" shape is exactly what the paper ran on IonQ's
hardware: quantum circuit in the middle, ordinary classical layer at the end.

## Why training is the hard part

In PyTorch, backprop gives you every gradient in one backward pass — because
the framework can see every intermediate activation. A quantum computer hides
its intermediate state by physics, not by API design. The workaround is the
**parameter-shift rule**: to get the gradient for knob θ, run the whole
circuit twice — once with θ nudged up by π/2, once down — and take half the
difference. Remarkably, this gives the *exact* gradient, not an approximation.

But: 2 circuit runs *per parameter per training step*, each "run" itself
being thousands of shots. A circuit with 1,000 parameters costs 2,000
executions per step. Quantum computers bill by execution. This cost is the
single biggest obstacle to quantum machine learning, and it's what the paper
attacks with three ideas working together:

1. **Butterfly architecture** — wire the entangling gates in the same pattern
   as the FFT (pair qubits whose index differs in one bit). Information from
   every qubit reaches every other in only log₂(n) layers, with O(n log n)
   parameters instead of O(n²).
2. **Layer-wise training** — train one layer at a time, freezing the rest.
   Smaller optimization problems, more stable, and it dodges the "barren
   plateau" (see below).
3. **Parallel parameter-shift** — within one butterfly layer, every gate
   touches a *different* pair of qubits. So you can nudge *all* of them at
   once and read each gate's gradient from its own qubits' measurements.
   Cost per layer: **constant** — 6 executions instead of 2-per-parameter.
   At the paper's 128-qubit scale: 28 executions instead of 1,792.

One more villain worth knowing: **barren plateaus**. In big random circuits,
gradients shrink *exponentially* as you add qubits — the quantum version of
vanishing gradients, but much worse. Structured circuits and layer-wise
training are currently the main defenses. The dashboard has a panel that
shows this happening.

## Reading the dashboard

Run it (below), press **play**, and:

| Panel | What it shows | What to look for |
|---|---|---|
| **Circuit topology** | The actual model: encoding column, then butterfly-wired layers | The glowing layer is the one being trained right now; dimmed layers are frozen. Note the FFT-style crossing pattern. |
| **Qubit states** (Bloch spheres) | Each qubit's individual state as an arrow in a ball | Arrow direction = the state; arrow **length** = purity. Arrows *shrink* as qubits entangle with each other — a shrinking arrow literally is entanglement building up. |
| **Training loss** | The same model trained 3 ways (see below) | All three should descend together. Drop the shot count and watch the hardware-realistic curves get noisy while autodiff stays smooth. |
| **Barren plateau monitor** | Gradient magnitude per layer per epoch | Healthy training = bright cells. Crank the qubit count up and watch the whole map fade — gradients vanishing in real time. |
| **Hardware budget** | Cumulative circuit executions per mode | The whole point of the paper in one number: naive parameter-shift explodes, the parallelized version barely moves. |
| **Model analysis** | A rule engine narrating the run: it watches the same numbers you do and explains them as they happen | Each note is tagged with the panel it describes. Amber = milestone, red = warning, cyan = insight. |

### Replay and the training report

The **timeline** slider in Run control replays any earlier epoch — every panel
(curves, heatmap, Bloch arrows, odometers, commentary) rewinds to exactly what
it showed at that moment, so you can step through a run after the fact and read
the explanation for each piece at each epoch. When a run finishes, a written
**training report** opens: final loss/accuracy/execution-count per mode, a
layer-by-layer verdict (healthy gradients vs barren plateau), and the
takeaways the run actually demonstrated. From there, one click steps you back
through the whole run.

### The three training modes (same model, same data, same loss)

- **Exact autodiff** — ordinary backprop through the simulation. *Physically
  impossible on real hardware*; shown as ground truth.
- **Naive parameter-shift** — hardware-legal: 2 circuit runs per parameter,
  with shot noise. Correct but ruinously expensive.
- **Parallelized parameter-shift** — the paper's method: same gradients,
  constant runs per layer. The gap between this row and the naive row in the
  budget panel *is the paper's contribution.*

### Experiments to try

1. **See shot noise become gradient noise**: set shots to 32, watch the loss
   curves jitter; set ∞ and they snap to the autodiff curve.
2. **Find a barren plateau**: raise qubits step by step and watch the plateau
   monitor dim and training stall.
3. **Verify the headline claim**: at each qubit count, compare the budget
   odometers — the naive/parallel ratio grows with n.

## Run it

```bash
uv sync --extra dev --extra server   # CPU build (any machine)
uv run pytest                        # 22 tests: physics, gradient equivalence, analysis
uv run python examples/train_demo.py # terminal comparison of the 3 modes

uv run uvicorn quantsim.server:app --port 8000   # dashboard at http://localhost:8000/
```

**GPU (NVIDIA Blackwell, e.g. RTX 5080):** needs CUDA 12.8 wheels — in
`pyproject.toml` change the index URL `whl/cpu` → `whl/cu128`, re-run
`uv sync`, and pass `device="cuda"` (auto-detected by default). Practical
ceiling ≈ 28 qubits at complex64 on 16 GB.

## Code layout

- `quantsim/state.py` — state vectors, gate application, Bloch vectors,
  entanglement entropy
- `quantsim/gates.py` — differentiable gate matrices
- `quantsim/circuit.py` — circuits + simulator with execution-budget counter
  and shot sampling
- `quantsim/butterfly.py` — the paper's Butterfly architecture
- `quantsim/gradients.py` — the three gradient methods, side by side (start
  reading here: each function is one of the dashboard's three curves)
- `quantsim/training.py` — layer-wise trainer + hybrid model
- `quantsim/analysis.py` — rule-based commentary engine + training-report builder
- `quantsim/server.py` — FastAPI/WebSocket backend streaming live training
- `frontend/` — the dashboard (runs standalone on a built-in mock when opened
  as a file; live data when served over http)
- `tests/` — the receipts: parameter-shift gradients match autodiff to 1e-4,
  and the parallel trick's execution counts are what the paper claims

## Roadmap

- [x] Simulator core, Butterfly QNN, three gradient modes, layer-wise training
- [x] Live dashboard
- [ ] Real-dataset task (UCI heart disease)
- [ ] Publish as a hosted educational tool (in-browser circuits + recorded
      GPU training replays)
