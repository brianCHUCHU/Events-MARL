from __future__ import annotations

import random
from contextlib import contextmanager
from copy import deepcopy

import numpy as np


def seed_before_vmas_import(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_globals(seed: int) -> np.random.Generator:
    import torch
    from vmas.simulator.environment import Environment

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # VMAS keeps this list outside Torch's global RNG. Its decorators retain
    # the list identity, so update its contents rather than replacing it.
    Environment.vmas_random_state[:] = [
        torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate(),
    ]
    return np.random.default_rng(seed)


@contextmanager
def isolated_vmas_rng():
    """Preserve VMAS and host RNG streams, including when evaluation fails."""
    import torch
    from vmas.simulator.environment import Environment

    vmas_state = deepcopy(Environment.vmas_random_state)
    numpy_state, python_state = deepcopy(np.random.get_state()), random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            Environment.vmas_random_state[:] = vmas_state
            np.random.set_state(numpy_state)
            random.setstate(python_state)

