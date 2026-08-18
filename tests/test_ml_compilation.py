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


def test_decision_tree_inference_compiles_and_simulates():
    import numpy as np
    from concrete import fhe

    from concrete_fhe_toolkit.ml import decision_tree_inference

    tree = {
        "feature": 0,
        "threshold": 5,
        "left": {"feature": 1, "threshold": 3, "left": 30, "right": 20},
        "right": 10,
    }

    def program(x):
        return decision_tree_inference([x[0], x[1]], tree)

    compiler = fhe.Compiler(program, {"x": "encrypted"})
    circuit = compiler.compile(
        [
            np.array([0, 0], dtype=np.int64),
            np.array([9, 5], dtype=np.int64),
            np.array([0, 5], dtype=np.int64),
            np.array([9, 0], dtype=np.int64),
        ]
    )

    assert int(circuit.simulate(np.array([7, 4], dtype=np.int64))) == 30
    assert int(circuit.simulate(np.array([7, 1], dtype=np.int64))) == 20
    assert int(circuit.simulate(np.array([2, 4], dtype=np.int64))) == 10
