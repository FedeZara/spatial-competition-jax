"""JAX device resolution helpers."""

from __future__ import annotations

import jax


def resolve_device(device_str: str | None) -> jax.Device:
    """Resolve a CLI device string to a :class:`jax.Device`.

    Accepted formats:
        ``None``  – JAX default (first available accelerator)
        ``'cpu'`` – first CPU device
        ``'gpu'`` – first GPU device
        ``'gpu:1'`` – GPU device with id 1
        ``'tpu'`` – first TPU device
        ``'tpu:0'`` – TPU device with id 0

    Returns:
        A :class:`jax.Device` instance.

    Raises:
        ValueError: If the requested device index is out of range.
    """
    if device_str is None:
        return jax.devices()[0]

    device_str = device_str.strip().lower()

    # Parse optional index: "gpu:1" → backend="gpu", idx=1
    if ":" in device_str:
        backend, idx_str = device_str.split(":", 1)
        idx = int(idx_str)
    else:
        backend = device_str
        idx = 0

    devices = jax.devices(backend)
    if idx >= len(devices):
        raise ValueError(
            f"Requested {backend}:{idx} but only {len(devices)} "
            f"{backend} device(s) available: {devices}"
        )
    return devices[idx]
