import math

import pytest
import torch

from quantsim import (ButterflyQNN, Circuit, Simulator, apply_gate, bloch_vector,
                      entanglement_entropy, probabilities, zero_state)
from quantsim.gates import PARAMETERIZED, cnot, h


def test_hadamard_makes_plus_state():
    psi = apply_gate(zero_state(1), h(), (0,))
    assert torch.allclose(probabilities(psi), torch.tensor([0.5, 0.5]))


def test_bell_state():
    """H then CNOT from |00> -> (|00> + |11>)/sqrt(2), the hello-world of entanglement."""
    psi = zero_state(2)
    psi = apply_gate(psi, h(), (0,))
    psi = apply_gate(psi, cnot(), (0, 1))
    assert torch.allclose(probabilities(psi), torch.tensor([0.5, 0.0, 0.0, 0.5]), atol=1e-6)
    # maximally entangled: 1 bit of entanglement entropy, Bloch vector at origin
    assert entanglement_entropy(psi, 1).item() == pytest.approx(1.0, abs=1e-5)
    assert bloch_vector(psi, 0).norm().item() == pytest.approx(0.0, abs=1e-6)


def test_parameterized_gates_are_unitary():
    theta = torch.tensor(0.7331)
    for name, fn in PARAMETERIZED.items():
        u = fn(theta)
        eye = torch.eye(u.shape[0], dtype=u.dtype)
        assert torch.allclose(u.conj().T @ u, eye, atol=1e-6), name


def test_norm_preserved_through_circuit():
    qnn = ButterflyQNN(n_qubits=4, n_layers=3)
    sim = Simulator(device="cpu")
    psi = sim.run(qnn.circuit(torch.rand(4) * math.pi), qnn.init_params(seed=1))
    assert probabilities(psi).sum().item() == pytest.approx(1.0, abs=1e-5)


def test_exact_expect_z_matches_shot_estimate():
    torch.manual_seed(0)
    qnn = ButterflyQNN(n_qubits=4, n_layers=2)
    sim = Simulator(device="cpu")
    psi = sim.run(qnn.circuit(torch.rand(4)), qnn.init_params(seed=2))
    exact = sim.expect_z(psi)
    sampled = sim.expect_z(psi, shots=200_000)
    assert torch.allclose(exact, sampled, atol=0.02)


def test_ry_rotates_bloch_vector():
    """RY(pi/2)|0> should sit on the +x axis of the Bloch sphere."""
    c = Circuit(1).add("ry", 0, param=math.pi / 2)
    psi = Simulator(device="cpu").run(c)
    assert torch.allclose(bloch_vector(psi, 0), torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
