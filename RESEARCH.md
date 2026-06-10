# quantSim Research Guide

A study path for building a quantum neural network simulator, written for someone
coming from NLP/LLMs with no quantum background. Topics are ordered so that each
project phase only needs the tiers before it.

**Source paper:** [Scalable On-Hardware Training of Quantum Neural Networks and
Application to Clinical Data Imputation (arXiv:2606.03517)](https://arxiv.org/abs/2606.03517)
— covered by [The Quantum Insider](https://thequantuminsider.com/2026/06/09/researchers-demonstrate-scalable-quantum-neural-network-training-on-quantum-hardware/).

---

## Tier 1 — Math you mostly already know, relabeled

*Needed for: Phase 1 (simulator core)*

### 1. Qubits & state vectors
An n-qubit quantum state is a unit vector in C^(2^n) — complex numbers called
**amplitudes**, one per classical bitstring. You already know embeddings; this is
an embedding with complex entries and a hard normalization constraint
(sum of |amplitude|² = 1).

- The exponential size (2^n) is why simulation hits a memory wall (~28 qubits on
  a 16GB GPU at complex64) and why quantum hardware is interesting at all.
- Search terms: *quantum state vector*, *probability amplitude*, *computational basis*.

### 2. Quantum gates as unitary matrices
Every quantum operation is a matrix multiply against the state vector — like a
linear layer, but constrained to be **unitary** (U†U = I: reversible,
norm-preserving). No nonlinearities, no biases, no information loss until
measurement.

- Single-qubit gates are 2×2 matrices applied to one tensor axis; two-qubit
  gates are 4×4 applied to a pair of axes. In PyTorch this is an `einsum`.
- Key gates for this project: rotations RX(θ), RY(θ), RZ(θ) (the *learnable*
  gates — θ is the parameter), Hadamard H, CNOT, and two-qubit rotations like
  RXX/RZZ (the *entangling* gates).
- Search terms: *unitary matrix*, *Pauli matrices*, *rotation gates*, *universal gate set*.

### 3. Measurement & the Born rule
Squared amplitude magnitudes form a probability distribution over bitstrings —
structurally like a softmax output. Sampling a measurement = drawing one
bitstring from it, and it **destroys the state**. This is the only way
information exits a quantum computer, and it's why you can't "just backprop"
through hardware: intermediate activations are unobservable.

- **Expectation values** (e.g., ⟨Z⟩ of a qubit ∈ [−1, 1]) are what QML losses
  are built from; on hardware they're estimated by averaging many **shots**
  (repeated runs), giving noisy estimates — like estimating a probability from
  N samples.
- Search terms: *Born rule*, *quantum measurement*, *expectation value of an
  observable*, *Pauli-Z measurement*.

### 4. Tensor products & entanglement
n qubits live in the tensor product of n 2-dim spaces — that's where 2^n comes
from. Most states **cannot** be factored into independent per-qubit states; the
non-factorizable part is **entanglement**. It's the resource that classical
networks don't have, and the reason quantum-advantage claims exist.

- Bell state (H then CNOT from |00⟩) is the "hello world" of entanglement —
  it's one of our unit tests.
- Per-qubit views of an entangled state require **reduced density matrices**
  (partial trace) — this is how the Bloch-sphere visualization will work.
- Search terms: *tensor product*, *Bell state*, *entanglement entropy*,
  *reduced density matrix*, *Bloch sphere*.

---

## Tier 2 — Quantum machine learning

*Needed for: Phase 2 (Butterfly QNN + training)*

### 5. Parameterized quantum circuits (PQCs)
The "model architecture" of QML: a fixed circuit topology whose rotation angles
are learnable parameters, trained by a *classical* optimizer (Adam, SGD). Also
called variational quantum circuits / variational quantum algorithms (VQA).

- Mental model: a forward pass is `encode(x) → U(θ) → measure`, where U(θ) is a
  deep stack of unitary "layers."
- Search terms: *parameterized quantum circuit*, *variational quantum algorithm*,
  *quantum neural network*, *hybrid quantum-classical model*.

### 6. Data encoding
How classical features enter a quantum state — the quantum analogue of
tokenization + embedding, and a surprisingly deep design choice that affects
model expressivity.

- **Angle encoding** (what we use): feature x_i sets a rotation angle RY(x_i) on
  qubit i. Simple, n features for n qubits.
- **Amplitude encoding**: 2^n features packed into the amplitudes directly —
  exponentially compact but expensive to prepare.
- Search terms: *quantum data encoding*, *angle encoding*, *amplitude encoding*,
  *quantum feature map*.

### 7. The parameter-shift rule  ← the conceptual heart of this project
You can't backprop through physical hardware (no access to intermediate states),
so gradients come from running the circuit at shifted parameter values: for a
gate exp(−iθP/2) with Pauli generator P,

    ∂⟨E⟩/∂θ = [ ⟨E(θ + π/2)⟩ − ⟨E(θ − π/2)⟩ ] / 2

This is **exact** (not a finite-difference approximation), but costs 2 circuit
executions *per parameter* — O(n²) total for typical architectures. That cost is
the obstacle the source paper attacks.

- Search terms: *parameter-shift rule*, *quantum gradients*, *gradient-based
  training of quantum circuits*.

### 8. Barren plateaus
For unstructured (random) circuits, gradient magnitudes vanish **exponentially**
in qubit count — the quantum version of vanishing gradients, and the central
obstacle in QNN training. Structure is the cure: the paper's Butterfly topology +
layer-wise training exist precisely to dodge this.

- Our per-layer gradient-magnitude visualization makes this directly visible.
- Search terms: *barren plateau*, *trainability of variational circuits*,
  *layer-wise training quantum*.

---

## Tier 3 — What makes the source paper tick

*Needed for: Phase 2 details, Phase 3 visualization, desktop scaling*

### 9. Butterfly circuit topology
The FFT wiring diagram applied to qubit pairing: layer k connects qubits whose
indices differ in bit k. Gives O(n log n) parameters and log(n) depth while
still letting information from every qubit reach every other. If you've seen
butterfly matrices in sparse-attention / structured-linear-layer literature,
it's the same picture.

- Search terms: *butterfly circuit quantum*, *FFT butterfly diagram*,
  *quantum circuit ansatz design*.

### 10. The paper's parallelized parameter-shift
Within one butterfly layer, gates act on **disjoint qubit pairs** and commute.
When training only the last layer (layer-wise scheme) against **local
observables** (per-qubit ⟨Z⟩), each parameter only influences the qubits its
gate touches — so you can shift *all* the layer's parameters simultaneously and
read every gradient from the corresponding local measurement. Cost per layer:
constant in n (their numbers: 1,792 → 28 executions at 128 qubits).

- This is what we replicate and verify against autodiff ground truth.

### 11. Shot noise & NISQ hardware
Real hardware estimates every expectation from finite samples (shots), so every
loss and gradient is noisy — SGD where the minibatch noise comes from physics.
Add gate errors and decoherence, and you get the "NISQ" (Noisy
Intermediate-Scale Quantum) era. Our simulator's shot-count knob interpolates
between idealized math (∞ shots) and hardware realism.

- Search terms: *shot noise quantum*, *NISQ*, *quantum hardware noise models*,
  *trapped-ion vs superconducting qubits* (the paper used IonQ trapped ions).

### 12. State-vector simulation & tensor networks
How classical simulators work: state-vector (what we build — exact, memory-bound
at 2^n) vs **tensor networks** (what the paper used for 32-qubit simulation —
compress low-entanglement states, scale further). Tensor networks are a deep
follow-on topic if you want to push past ~28 qubits on the 5080.

- Search terms: *state vector simulator*, *tensor network quantum simulation*,
  *matrix product states*, *NVIDIA cuQuantum* (for comparison/validation).

---

## Resources, in recommended order

1. **[Quantum Country](https://quantum.country)** — Michael Nielsen & Andy
   Matuschak's spaced-repetition essays. The best zero-to-one intro in
   existence. Read "Quantum computing for the very curious" first.
2. **[Quirk](https://algassert.com/quirk)** — drag-and-drop in-browser circuit
   simulator. Build gate intuition by playing; also prior art for our
   educational-tool ambitions.
3. **[PennyLane QML demos](https://pennylane.ai/qml)** — closest to your ML
   background; we don't use the library, but their explanations of PQCs,
   parameter-shift, and barren plateaus are excellent.
4. **The source paper** ([arXiv:2606.03517](https://arxiv.org/abs/2606.03517)) —
   readable after Tiers 1–2.
5. *Skip for now:* Nielsen & Chuang ("Mike & Ike", the standard textbook) —
   authoritative but overkill for this project.

## Phase → topic map

| Phase | Build | Topics needed |
|---|---|---|
| 1 | State-vector simulator core | 1–4 |
| 2 | Butterfly QNN, three gradient modes | 5–10 |
| 3 | Visualization dashboard | 4 (density matrices), 8 (plateaus) |
| 4 | Real-data training task | 6 (encoding) |
| 5 | Publish as educational tool | 11–12 for the "how real hardware differs" story |
