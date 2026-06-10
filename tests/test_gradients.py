"""The core scientific claims: parameter-shift == autodiff, and the parallel
trick gives identical gradients in O(1) executions per layer."""
import torch

from quantsim import ButterflyQNN, Simulator
from quantsim.butterfly import ROLES_PER_BLOCK
from quantsim.gradients import grads_autodiff, grads_naive_shift, grads_parallel_shift


def _setup(n_qubits=8, n_layers=3):
    qnn = ButterflyQNN(n_qubits=n_qubits, n_layers=n_layers)
    sim = Simulator(device="cpu")
    params = qnn.init_params(seed=3) + 0.3  # away from zero so gradients are non-trivial
    x = torch.linspace(0.2, 2.8, n_qubits)
    w = torch.linspace(-1, 1, n_qubits)
    loss_fn = lambda z: torch.sigmoid(z @ w).square().mean()
    return qnn, sim, params, x, loss_fn


def test_naive_shift_matches_autodiff_on_last_layer():
    qnn, sim, params, x, loss_fn = _setup()
    last = len(qnn.layers) - 1
    _, g_auto = grads_autodiff(sim, qnn, params, x, loss_fn, upto_layer=last)
    _, g_shift = grads_naive_shift(sim, qnn, params, x, loss_fn, layer=last)
    sl = qnn.layer_param_slice(last)
    assert torch.allclose(g_auto[sl], g_shift[sl], atol=1e-4)


def test_parallel_shift_matches_naive_shift():
    qnn, sim, params, x, loss_fn = _setup()
    last = len(qnn.layers) - 1
    _, g_naive = grads_naive_shift(sim, qnn, params, x, loss_fn, layer=last)
    _, g_par = grads_parallel_shift(sim, qnn, params, x, loss_fn, layer=last)
    sl = qnn.layer_param_slice(last)
    assert torch.allclose(g_naive[sl], g_par[sl], atol=1e-4)


def test_parallel_shift_matches_on_every_layer_during_layerwise_training():
    """Validity condition: trained layer is last in the *executed* circuit."""
    qnn, sim, params, x, loss_fn = _setup(n_qubits=8, n_layers=3)
    for layer in range(len(qnn.layers)):
        _, g_auto = grads_autodiff(sim, qnn, params, x, loss_fn, upto_layer=layer)
        _, g_par = grads_parallel_shift(sim, qnn, params, x, loss_fn, layer=layer)
        sl = qnn.layer_param_slice(layer)
        assert torch.allclose(g_auto[sl], g_par[sl], atol=1e-4), f"layer {layer}"


def test_execution_budget_savings():
    """The paper's headline: constant executions per layer vs 2-per-parameter."""
    qnn, sim, params, x, loss_fn = _setup(n_qubits=16, n_layers=1)
    n_blocks = len(qnn.layers[0])

    sim.reset_budget()
    grads_naive_shift(sim, qnn, params, x, loss_fn, layer=0)
    naive_execs = sim.executions
    assert naive_execs == 1 + 2 * n_blocks * ROLES_PER_BLOCK  # 1 forward + 2/param

    sim.reset_budget()
    grads_parallel_shift(sim, qnn, params, x, loss_fn, layer=0)
    par_execs = sim.executions
    assert par_execs == 1 + 2 * ROLES_PER_BLOCK  # 1 forward + 2/role, independent of n

    assert par_execs < naive_execs


def test_shot_noise_converges_to_exact_gradient():
    torch.manual_seed(0)
    qnn, sim, params, x, loss_fn = _setup(n_qubits=4, n_layers=1)
    _, g_exact = grads_parallel_shift(sim, qnn, params, x, loss_fn, layer=0)
    _, g_noisy = grads_parallel_shift(sim, qnn, params, x, loss_fn, layer=0, shots=100_000)
    sl = qnn.layer_param_slice(0)
    assert torch.allclose(g_exact[sl], g_noisy[sl], atol=0.05)
