"""Compilation and simulation tests for the ml subpackage."""

from concrete_fhe_toolkit.ml import (
    compile_decision_tree_node,
    compile_hinge_loss,
    compile_leaky_relu,
    compile_relu,
    compile_threshold_activation,
    compile_unit_step,
)


def test_relu_compiles_and_simulates():
    circuit = compile_relu(-5, 5)

    assert int(circuit.simulate(-3)) == 0
    assert int(circuit.simulate(0)) == 0
    assert int(circuit.simulate(4)) == 4


def test_leaky_relu_compiles_and_simulates():
    circuit = compile_leaky_relu(-15, 15, alpha=0.1)

    assert int(circuit.simulate(7)) == 7
    assert int(circuit.simulate(-12)) == -1
    assert int(circuit.simulate(-5)) == 0


def test_unit_step_compiles_and_simulates():
    circuit = compile_unit_step(-5, 5)

    assert int(circuit.simulate(-2)) == 0
    assert int(circuit.simulate(0)) == 1
    assert int(circuit.simulate(3)) == 1


def test_threshold_activation_compiles_and_simulates():
    circuit = compile_threshold_activation(-5, 5)

    assert int(circuit.simulate(3, 1)) == 1
    assert int(circuit.simulate(1, 3)) == 0
    assert int(circuit.simulate(2, 2)) == 1


def test_decision_tree_node_compiles_and_simulates():
    circuit = compile_decision_tree_node(-5, 5)

    assert int(circuit.simulate(4, 2, 3, -3)) == 3
    assert int(circuit.simulate(1, 2, 3, -3)) == -3


def test_hinge_loss_compiles_and_simulates():
    circuit = compile_hinge_loss(-3, 3)

    assert int(circuit.simulate(1, 1)) == 0
    assert int(circuit.simulate(-1, 1)) == 2
    assert int(circuit.simulate(0, -1)) == 1
