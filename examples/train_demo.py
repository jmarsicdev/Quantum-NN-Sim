"""Train the Butterfly QNN three ways on a synthetic task and compare:
final accuracy, gradient quality, and — the paper's headline — the hardware
execution budget. Run: uv run python examples/train_demo.py
"""
import math
import time

import torch

from quantsim import ButterflyQNN, HybridModel, train_layerwise

N_QUBITS = 8
N_LAYERS = 3
N_SAMPLES = 32
EPOCHS_PER_LAYER = 6
SHOTS = 2048  # per measurement for the hardware-realistic modes


def make_dataset(n: int, n_features: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(n, n_features, generator=g) * math.pi
    ys = ((xs[:, 0] > math.pi / 2) ^ (xs[:, 1] > math.pi / 2)).float()  # XOR task
    return xs, ys


def main():
    xs, ys = make_dataset(N_SAMPLES, N_QUBITS)
    print(f"{N_QUBITS} qubits, {N_LAYERS} butterfly layers, "
          f"{N_SAMPLES} samples, {EPOCHS_PER_LAYER} epochs/layer\n")
    header = f"{'mode':<16} {'loss':>7} {'acc':>6} {'executions':>11} {'shots':>12} {'time':>7}"
    print(header)
    print("-" * len(header))

    for mode, shots in [("autodiff", None),
                        ("naive-shift", SHOTS),
                        ("parallel-shift", SHOTS)]:
        torch.manual_seed(0)
        model = HybridModel(ButterflyQNN(N_QUBITS, N_LAYERS), device="cpu")
        t0 = time.perf_counter()
        log = train_layerwise(model, xs, ys, mode=mode,
                              epochs_per_layer=EPOCHS_PER_LAYER, shots=shots)
        dt = time.perf_counter() - t0
        acc = ((model.predict(xs) > 0.5).float() == ys).float().mean().item()
        note = " (ground truth, impossible on hardware)" if mode == "autodiff" else ""
        print(f"{mode:<16} {log.losses[-1]:>7.4f} {acc:>6.2f} "
              f"{log.executions:>11,} {log.shots_used:>12,} {dt:>6.1f}s{note}")

    print("\nThe budget gap between naive-shift and parallel-shift is the paper's")
    print("contribution; it widens with qubit count (1,792 vs 28 at 128 qubits).")


if __name__ == "__main__":
    main()
