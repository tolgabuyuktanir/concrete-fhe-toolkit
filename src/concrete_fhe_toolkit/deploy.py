"""Client/server deployment helpers for compiled circuits.

Concrete splits a compiled circuit into two artifacts:

- **server.zip** — the evaluation logic. Ships to the untrusted server; it
  contains no key material and can only compute on ciphertexts.
- **client.zip** — the client specs. Stays with the data owner, who
  generates keys locally, encrypts inputs, and decrypts results.

These helpers wrap that split into three calls, so a trained toolkit model
becomes a deployable encrypted service:

Example:
    ```python
    from concrete_fhe_toolkit import deploy
    from concrete_fhe_toolkit.ml import FHELogisticRegression

    # --- offline: train/compile and export the two artifacts ---
    model = FHELogisticRegression(weights=[3, 2], bias=-7)
    model.compile(inputset)
    deploy.save_deployment(model.circuit, "deployment/")

    # --- server side (untrusted): load logic only ---
    server = deploy.load_server("deployment/")

    # --- client side (data owner): keys, encrypt, decrypt ---
    client = deploy.load_client("deployment/")
    client.keys.generate()
    args = client.encrypt([4, 1])
    result = server.run(args, evaluation_keys=client.evaluation_keys)
    print(client.decrypt(result))  # 1
    ```
"""

from __future__ import annotations

import os

from ._compat import fhe

SERVER_FILENAME = "server.zip"
CLIENT_FILENAME = "client.zip"


def save_deployment(circuit: "fhe.Circuit", directory: str) -> None:
    """Export a compiled circuit as server.zip + client.zip artifacts.

    The directory is created when missing. No key material is written:
    ``server.zip`` holds only evaluation logic and ``client.zip`` holds the
    specs the data owner needs to generate keys locally.

    Args:
        circuit: A compiled ``fhe.Circuit`` (for a toolkit model, compile
            it first and pass ``model.circuit``).
        directory: Destination directory for the two artifacts.

    Example:
        ```python
        model.compile(inputset)
        deploy.save_deployment(model.circuit, "deployment/")
        ```
    """
    if circuit is None:
        raise ValueError("circuit is not compiled; call compile(...) first")
    os.makedirs(directory, exist_ok=True)
    circuit.server.save(os.path.join(directory, SERVER_FILENAME))
    circuit.client.save(os.path.join(directory, CLIENT_FILENAME))


def load_server(directory: str) -> "fhe.Server":
    """Load the evaluation-only server artifact from a deployment directory.

    Safe to run on untrusted infrastructure — the server can only compute
    on ciphertexts and never sees keys or plaintexts.

    Example:
        ```python
        server = deploy.load_server("deployment/")
        result = server.run(encrypted_args, evaluation_keys=keys)
        ```
    """
    path = os.path.join(directory, SERVER_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {SERVER_FILENAME} in {directory!r}")
    return fhe.Server.load(path)


def load_client(directory: str) -> "fhe.Client":
    """Load the client specs from a deployment directory (data-owner side).

    Generate keys locally with ``client.keys.generate()`` before the first
    ``encrypt``; keys never leave the client except the public evaluation
    keys passed to ``server.run``.

    Example:
        ```python
        client = deploy.load_client("deployment/")
        client.keys.generate()
        args = client.encrypt(sample)
        ```
    """
    path = os.path.join(directory, CLIENT_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {CLIENT_FILENAME} in {directory!r}")
    return fhe.Client.load(path)


__all__ = [
    "CLIENT_FILENAME",
    "SERVER_FILENAME",
    "load_client",
    "load_server",
    "save_deployment",
]
