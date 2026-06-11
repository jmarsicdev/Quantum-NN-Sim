"""Rule-based commentary: turns raw training telemetry into per-epoch
observations and an end-of-run report.

The Analyzer sees exactly what the dashboard panels see — losses, gradient
norms, Bloch purities, execution budgets — and emits short explanations tagged
with the panel they belong to. Rules are stateful: one-time lessons fire once
per run, recurring checks are rate-limited, layer milestones re-arm per layer.
Pure Python (no torch) so it is cheap and trivially testable.
"""
import math

PANELS = ("circuit", "bloch", "loss", "heat", "budget")
TONES = ("info", "insight", "warning", "milestone")
TONE_RANK = {tone: i for i, tone in enumerate(TONES)}

COINFLIP = math.log(2)  # BCE of always predicting p=0.5

MAX_OBSERVATIONS = 4


def _obs(panel: str, text: str, tone: str = "info") -> dict:
    assert panel in PANELS and tone in TONES
    return {"panel": panel, "text": text, "tone": tone}


class Analyzer:
    def __init__(self, n_qubits: int, n_layers: int, shots: int | None,
                 epochs_per_layer: int, n_blocks: int, n_samples: int):
        self.n = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.epochs_per_layer = epochs_per_layer
        self.n_blocks = n_blocks
        self.n_samples = n_samples
        self.params_per_layer = 3 * n_blocks
        self.total_params = self.params_per_layer * n_layers
        # per-epoch circuit executions (see gradients.py + LayerwiseStepper)
        self.per_epoch = {
            "exact": 2 * n_samples,                    # 1 autodiff + 1 head eval
            "naive": n_samples * (2 + 6 * n_blocks),   # 2 runs per parameter
            "par": 8 * n_samples,                      # 2 x 3 roles, any width
        }
        self._fired: set[str] = set()
        self._cool: dict[str, int] = {}
        self._exact_hist: list[float] = []
        self._layer_start_loss: float | None = None
        self._layer_start_grad: float | None = None
        self._dev_ema = {"naive": 0.0, "par": 0.0}

    # —— repetition control ——————————————————————————————————————
    def _once(self, key: str) -> bool:
        if key in self._fired:
            return False
        self._fired.add(key)
        return True

    def _every(self, key: str, epoch: int, period: int) -> bool:
        if epoch < self._cool.get(key, 0):
            return False
        self._cool[key] = epoch + period
        return True

    # —— main entry ———————————————————————————————————————————————
    def observe(self, *, epoch: int, layer: int, layer_epoch: int,
                losses: dict, grad_norm: float, bloch: list[dict],
                budget: dict) -> dict:
        """One call per completed epoch (epoch is 1-based; layer/layer_epoch
        describe the layer that just trained). Returns
        {"headline": str|None, "observations": [{panel,text,tone}, ...]}."""
        out: list[dict] = []
        self._exact_hist.append(losses["exact"])
        if layer_epoch == 0:
            prev = self._exact_hist[-2] if len(self._exact_hist) > 1 else None
            self._layer_start_loss = prev if prev is not None else losses["exact"]
            self._layer_start_grad = grad_norm

        out += self._run_start(epoch, losses)
        out += self._layer_events(epoch, layer, layer_epoch, losses)
        out += self._loss_rules(epoch, layer, layer_epoch, losses)
        out += self._grad_rules(epoch, layer, layer_epoch, grad_norm)
        out += self._bloch_rules(epoch, bloch)
        out += self._budget_rules(epoch, layer, layer_epoch, budget)

        out.sort(key=lambda o: -TONE_RANK[o["tone"]])
        out = out[:MAX_OBSERVATIONS]
        return {"headline": out[0]["text"] if out else None, "observations": out}

    # —— rule groups ——————————————————————————————————————————————
    def _run_start(self, epoch: int, losses: dict) -> list[dict]:
        if epoch != 1:
            return []
        out = [_obs("circuit",
                    f"Training begins. Features are angle-encoded as Ry rotations, then layer 1 of "
                    f"{self.n_layers} trains while the others wait. Only {self.params_per_layer} of "
                    f"{self.total_params} parameters are learning right now — that is layer-wise training.",
                    "milestone"),
               _obs("budget",
                    f"Cost per epoch ({self.n_samples} samples): autodiff {self.per_epoch['exact']} runs, "
                    f"naive shift {self.per_epoch['naive']} (2 per parameter), parallel shift "
                    f"{self.per_epoch['par']} — a constant, no matter how many qubits.")]
        if self.shots is None:
            out.append(_obs("loss",
                            "Shots = ∞: parameter-shift gradients are mathematically exact, so all three "
                            "curves should lie on top of each other. The difference on real hardware is "
                            "noise and cost, not correctness.", "insight"))
        else:
            out.append(_obs("loss",
                            f"Every ⟨Z⟩ is estimated from {self.shots} measurements, so the two "
                            f"hardware-style curves carry sampling noise that autodiff never sees."))
        return out

    def _layer_events(self, epoch: int, layer: int, layer_epoch: int,
                      losses: dict) -> list[dict]:
        if layer_epoch != 0 or layer == 0:
            return []
        start = self._layer_start_loss
        pct = (1 - losses["exact"] / start) * 100 if start else 0.0
        n_strides = max(self.n.bit_length() - 1, 1)
        stride = 2 ** (layer % n_strides)
        out = [_obs("circuit",
                    f"Layer {layer} is frozen at its trained values; layer {layer + 1} now trains. "
                    f"Its entanglers pair qubit i with qubit i±{stride} (butterfly stride {stride}), "
                    f"mixing information across a new distance.", "milestone")]
        if self._once(f"freeze-explained"):
            out.append(_obs("heat",
                            "Frozen layers keep their rows in the heatmap dim — their parameters get no "
                            "more updates, which keeps each optimization small and dodges barren plateaus."))
        return out

    def _loss_rules(self, epoch: int, layer: int, layer_epoch: int,
                    losses: dict) -> list[dict]:
        out = []
        exact, naive, par = losses["exact"], losses["naive"], losses["par"]
        spread = max(abs(naive - exact), abs(par - exact))

        if self.shots is None and epoch >= 3 and spread < 1e-4 and self._once("identical"):
            out.append(_obs("loss",
                            f"All three losses agree to within {spread:.0e}. Parameter shift is not an "
                            f"approximation — run the circuit at θ±π/2 and you get the exact gradient.",
                            "insight"))
        if self.shots is not None:
            a = 0.7 * self._dev_ema["naive"] + 0.3 * abs(naive - exact)
            b = 0.7 * self._dev_ema["par"] + 0.3 * abs(par - exact)
            self._dev_ema = {"naive": a, "par": b}
            if epoch >= 4 and max(a, b) > 0.02 and self._every("shotnoise", epoch, 10):
                out.append(_obs("loss",
                                f"Shot noise in action: at {self.shots} shots the hardware-style losses are "
                                f"wandering ±{max(a, b):.3f} around the exact curve. More shots would tighten "
                                f"them — at a price.", "warning" if max(a, b) > 0.06 else "info"))

        if exact < COINFLIP and self._once("beat-coinflip"):
            out.append(_obs("loss",
                            f"Loss dropped below ln 2 ≈ {COINFLIP:.3f} — the cross-entropy of pure guessing. "
                            f"The model is now genuinely better than a coin flip.", "milestone"))
        if exact < 0.35 and self._once("loss-035"):
            out.append(_obs("loss",
                            "Loss under 0.35: the circuit's measured ⟨Z⟩ values have become features the "
                            "classical head can separate cleanly.", "insight"))

        if layer_epoch >= 4 and len(self._exact_hist) >= 4:
            recent = self._exact_hist[-4:]
            improvement = recent[0] - recent[-1]
            if 0 <= improvement < 0.002 and self._every(f"stall-L{layer}", epoch, self.epochs_per_layer):
                out.append(_obs("loss",
                                f"Loss has flattened mid-layer (Δ{improvement:.4f} over 4 epochs). This layer "
                                f"may have learned all it can — the layer-wise schedule moves on soon.",
                                "info"))
        return out

    def _grad_rules(self, epoch: int, layer: int, layer_epoch: int,
                    grad_norm: float) -> list[dict]:
        out = []
        if grad_norm < 1e-3 and self._once(f"plateau-L{layer}"):
            out.append(_obs("heat",
                            f"‖∇θ‖ = {grad_norm:.1e} on the training layer — barren-plateau territory. "
                            f"Gradients this small barely move the parameters; at {self.n} qubits this is "
                            f"the exponential vanishing the paper is designed to avoid.", "warning"))
        elif grad_norm >= 1e-2 and epoch == 2 and self._once("grads-healthy"):
            out.append(_obs("heat",
                            f"‖∇θ‖ = {grad_norm:.2f} — healthy and bright. Raise the qubit count and watch "
                            f"this cell dim: that fading is the barren plateau forming."))
        start = self._layer_start_grad
        if (start and start > 1e-3 and grad_norm < 0.25 * start
                and layer_epoch >= 2 and self._once("grad-fade-converge")):
            out.append(_obs("heat",
                            f"The active layer's gradient fell from {start:.2f} to {grad_norm:.2f} as its "
                            f"loss bottomed out. Fading near an optimum is healthy convergence — a plateau "
                            f"is when gradients start near zero.", "insight"))
        return out

    def _bloch_rules(self, epoch: int, bloch: list[dict]) -> list[dict]:
        if not bloch:
            return []
        out = []
        rs = [b["r"] for b in bloch]
        min_r, mean_r = min(rs), sum(rs) / len(rs)
        q_min = rs.index(min_r)
        if min_r < 0.9 and self._once("ent-onset"):
            out.append(_obs("bloch",
                            f"q{q_min}'s arrow has shrunk to |r| = {min_r:.2f}. That is entanglement: the "
                            f"qubit's state is no longer fully its own, so its individual arrow loses "
                            f"length. A classical bit could never do this.", "insight"))
        if mean_r < 0.6 and self._once("ent-strong"):
            out.append(_obs("bloch",
                            f"Average purity is down to {mean_r:.2f} — the register is now strongly "
                            f"entangled. The information lives in correlations between qubits, which is "
                            f"the resource a quantum model has and a classical one doesn't.", "milestone"))
        if epoch == self.epochs_per_layer and min_r > 0.97 and self._once("ent-none"):
            out.append(_obs("bloch",
                            "Arrows still near full length after a whole layer: the Rxx entanglers have "
                            "stayed close to identity, so qubits remain nearly independent so far."))
        return out

    def _budget_rules(self, epoch: int, layer: int, layer_epoch: int,
                      budget: dict) -> list[dict]:
        out = []
        ratio = budget["naive"] / budget["par"] if budget.get("par") else 0.0
        if layer_epoch == self.epochs_per_layer - 1 and self._every("budget-layer", epoch, 2):
            out.append(_obs("budget",
                            f"Layer {layer + 1} done: naive shift has used {budget['naive']:,} circuit "
                            f"executions vs {budget['par']:,} parallelized — ×{ratio:.1f} for identical "
                            f"gradients. That gap is the paper's headline result.",
                            "insight" if layer == 0 else "info"))
        if budget["naive"] - budget["par"] > 10_000 and self._once("budget-10k"):
            out.append(_obs("budget",
                            f"The naive method is now {budget['naive'] - budget['par']:,} executions ahead "
                            f"of the parallel trick — pure overhead. On paid hardware this is the bill "
                            f"that makes naive training impractical.", "warning"))
        return out


# —— end-of-run report ————————————————————————————————————————————
def build_report(*, n_qubits: int, n_layers: int, shots: int | None,
                 epochs_per_layer: int, n_blocks: int, n_samples: int,
                 history: dict[str, list[float]],
                 grads_history: list[list[float | None]],
                 budget: dict[str, int], accuracy: dict[str, float],
                 bloch: list[dict], entropy: float | None) -> dict:
    """Assemble the final report from full-run telemetry. Pure data in/out."""
    exact = history["exact"]
    total_epochs = len(exact)
    params_per_layer = 3 * n_blocks

    layers = []
    for l in range(n_layers):
        seg = exact[l * epochs_per_layer:(l + 1) * epochs_per_layer]
        if not seg:
            continue
        g = [row[l] for row in grads_history[l * epochs_per_layer:(l + 1) * epochs_per_layer]
             if row[l] is not None]
        layers.append({
            "layer": l,
            "startLoss": seg[0],
            "endLoss": seg[-1],
            "deltaPct": (1 - seg[-1] / seg[0]) * 100 if seg[0] else 0.0,
            "peakGrad": max(g) if g else None,
            "endGrad": g[-1] if g else None,
            "plateau": bool(g) and max(g) < 1e-3,
        })

    ratio = budget["naive"] / budget["par"] if budget.get("par") else 0.0
    rs = [b["r"] for b in bloch] or [1.0]
    mean_r, min_r = sum(rs) / len(rs), min(rs)
    spread = (sum(abs(history["naive"][i] - exact[i]) + abs(history["par"][i] - exact[i])
                  for i in range(total_epochs)) / (2 * total_epochs)) if total_epochs else 0.0

    takeaways = []
    if shots is None:
        takeaways.append(
            f"With infinite shots, all three trainers were numerically identical (mean deviation "
            f"{spread:.1e}) — parameter shift gives the *exact* gradient. Hardware's real costs are "
            f"shot noise and circuit executions, not gradient quality.")
    else:
        takeaways.append(
            f"At {shots} shots the hardware-style losses wandered ±{spread:.3f} around the exact "
            f"curve on average. Every gradient on a real device is a statistical estimate.")
    takeaways.append(
        f"Identical gradients, wildly different bills: naive parameter-shift spent "
        f"{budget['naive']:,} circuit executions, the parallelized rule {budget['par']:,} "
        f"(×{ratio:.1f}). Naive grows with parameter count ({params_per_layer}/layer here); "
        f"parallel stays at 8 runs per sample per epoch at any width.")
    plateaued = [c["layer"] + 1 for c in layers if c["plateau"]]
    if plateaued:
        takeaways.append(
            f"Layer(s) {', '.join('L' + str(i) for i in plateaued)} trained with gradients under "
            f"10⁻³ — barren-plateau symptoms. Expect this to worsen as qubit count grows.")
    else:
        takeaways.append(
            f"No barren plateau at {n_qubits} qubits: every layer kept usable gradients. "
            f"Re-run wider to watch them shrink — that is the scaling wall.")
    if min_r < 0.9:
        takeaways.append(
            f"Final state is entangled: mean qubit purity {mean_r:.2f} (lowest {min_r:.2f})"
            + (f", cut entropy {entropy:.2f} bits" if entropy is not None else "")
            + ". Shrunken Bloch arrows are that entanglement made visible.")
    else:
        takeaways.append(
            "Qubits ended nearly pure — this run barely used entanglement. More layers or larger "
            "initial angles would push the register into genuinely quantum territory.")
    best = max(accuracy, key=lambda k: accuracy[k])
    takeaways.append(
        f"Train accuracy — autodiff {accuracy['exact']:.0%}, naive shift {accuracy['naive']:.0%}, "
        f"parallel shift {accuracy['par']:.0%}. The hardware-legal methods "
        + ("match the impossible ground truth." if abs(accuracy[best] - accuracy["exact"]) < 0.07
           else "trail ground truth — noise in gradients becomes noise in the model."))

    return {
        "headline": (
            f"Trained {n_layers} layer(s) × {epochs_per_layer} epochs at {n_qubits} qubits — "
            f"parallel parameter-shift reached {accuracy['par']:.0%} accuracy for {budget['par']:,} "
            f"circuit runs; naive needed {budget['naive']:,} for the same gradients."),
        "config": {"nQubits": n_qubits, "nLayers": n_layers,
                   "shots": "inf" if shots is None else shots,
                   "epochsPerLayer": epochs_per_layer, "totalEpochs": total_epochs,
                   "paramsPerLayer": params_per_layer,
                   "totalParams": params_per_layer * n_layers, "samples": n_samples},
        "modes": {k: {"finalLoss": history[k][-1] if history[k] else None,
                      "accuracy": accuracy[k], "executions": budget[k]}
                  for k in ("exact", "naive", "par")},
        "layers": layers,
        "budget": {"naiveOverPar": ratio,
                   "perEpoch": {"exact": 2 * n_samples,
                                "naive": n_samples * (2 + 6 * n_blocks),
                                "par": 8 * n_samples}},
        "bloch": {"meanPurity": mean_r, "minPurity": min_r, "entropy": entropy},
        "takeaways": takeaways,
    }
