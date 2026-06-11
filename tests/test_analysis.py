"""The commentary engine: shape, repetition control, milestones, report."""
import math

from quantsim.analysis import MAX_OBSERVATIONS, PANELS, TONES, Analyzer, build_report


def make_analyzer(shots=None, n_qubits=8, n_layers=3, epochs_per_layer=12):
    return Analyzer(n_qubits=n_qubits, n_layers=n_layers, shots=shots,
                    epochs_per_layer=epochs_per_layer, n_blocks=n_qubits // 2,
                    n_samples=16)


def telemetry(epoch, layer, layer_epoch, *, loss=0.7, dev=0.0, grad=0.3, r=1.0,
              budget=(100, 1000, 300)):
    return dict(
        epoch=epoch, layer=layer, layer_epoch=layer_epoch,
        losses={"exact": loss, "naive": loss + dev, "par": loss + dev / 3},
        grad_norm=grad,
        bloch=[{"theta": 0.4, "phi": 0.1, "r": r}] * 8,
        budget={"exact": budget[0], "naive": budget[1], "par": budget[2]})


def drive(an, epochs, epochs_per_layer=12, **kw):
    """Run `epochs` ticks through the analyzer, returning every observation."""
    seen = []
    for e in range(1, epochs + 1):
        layer, layer_epoch = (e - 1) // epochs_per_layer, (e - 1) % epochs_per_layer
        out = an.observe(**telemetry(e, layer, layer_epoch, **kw))
        assert out["headline"] is None or isinstance(out["headline"], str)
        assert len(out["observations"]) <= MAX_OBSERVATIONS
        for o in out["observations"]:
            assert o["panel"] in PANELS and o["tone"] in TONES and o["text"]
        seen.extend(out["observations"])
    return seen


def test_run_start_fires_once():
    an = make_analyzer()
    obs = drive(an, 3)
    intros = [o for o in obs if "Training begins" in o["text"]]
    assert len(intros) == 1 and intros[0]["tone"] == "milestone"


def test_layer_transition_milestone():
    an = make_analyzer()
    out_boundary = None
    for e in range(1, 14):  # epochs_per_layer=12 -> layer 1 starts at epoch 13
        layer, layer_epoch = (e - 1) // 12, (e - 1) % 12
        out = an.observe(**telemetry(e, layer, layer_epoch))
        if e == 13:
            out_boundary = out
    texts = [o["text"] for o in out_boundary["observations"]]
    assert any("layer 2 now trains" in t for t in texts)
    assert any("frozen" in t.lower() for t in texts)


def test_plateau_warning_once_per_layer():
    an = make_analyzer()
    obs = drive(an, 24, grad=1e-4)  # two layers of vanishing gradients
    plateaus = [o for o in obs if "barren-plateau territory" in o["text"]]
    assert len(plateaus) == 2  # once for layer 0, once for layer 1
    assert all(o["tone"] == "warning" for o in plateaus)


def test_exactness_insight_only_with_infinite_shots():
    inf = drive(make_analyzer(shots=None), 6, dev=0.0)
    assert sum("agree to within" in o["text"] for o in inf) == 1
    finite = drive(make_analyzer(shots=128), 6, dev=0.0)
    assert not any("agree to within" in o["text"] for o in finite)


def test_shot_noise_observation_rate_limited():
    an = make_analyzer(shots=32)
    obs = drive(an, 20, dev=0.08)
    noisy = [o for o in obs if "Shot noise in action" in o["text"]]
    assert 1 <= len(noisy) <= 2  # cooldown of 10 epochs


def test_entanglement_thresholds_fire_once():
    an = make_analyzer()
    obs = drive(an, 10, r=0.5)
    assert sum("That is entanglement" in o["text"] for o in obs) == 1
    assert sum("strongly" in o["text"] for o in obs) == 1


def test_coinflip_milestone():
    an = make_analyzer()
    drive(an, 2, loss=0.8)
    out = an.observe(**telemetry(3, 0, 2, loss=0.6))
    assert any("coin flip" in o["text"] for o in out["observations"])


def test_build_report_shape_and_math():
    E, L = 12, 2
    history = {
        "exact": [0.8 - 0.02 * i for i in range(E * L)],
        "naive": [0.8 - 0.02 * i + 0.01 for i in range(E * L)],
        "par": [0.8 - 0.02 * i - 0.01 for i in range(E * L)],
    }
    grads = [[0.3 if j <= (i // E) else None for j in range(L)] for i in range(E * L)]
    rep = build_report(
        n_qubits=4, n_layers=L, shots=512, epochs_per_layer=E, n_blocks=2,
        n_samples=16, history=history, grads_history=grads,
        budget={"exact": 768, "naive": 5376, "par": 3072},
        accuracy={"exact": 0.94, "naive": 0.88, "par": 0.91},
        bloch=[{"theta": 0.3, "phi": 0.2, "r": 0.7}] * 4, entropy=0.9)
    assert isinstance(rep["headline"], str) and "4 qubits" in rep["headline"]
    assert rep["config"]["totalEpochs"] == E * L
    assert len(rep["layers"]) == L
    assert math.isclose(rep["budget"]["naiveOverPar"], 5376 / 3072)
    assert rep["modes"]["par"]["accuracy"] == 0.91
    assert not rep["layers"][0]["plateau"]
    assert len(rep["takeaways"]) >= 4
    assert all(isinstance(t, str) for t in rep["takeaways"])
