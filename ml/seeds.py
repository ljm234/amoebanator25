"""
Phase 9.5 - single source of truth for random-seed configuration.

Every training and evaluation entry point should call `set_global_seeds()` at
the top of `main()`. This pins the three RNGs that the pipeline can touch:

  * Python's `random` module
  * NumPy
  * PyTorch (CPU + MPS / CUDA when available)

It also disables PyTorch's cuDNN benchmark mode and enables deterministic
algorithms where supported, so two runs on the same hardware produce
bit-identical model.pt files.

The default seed is 42 (the same value used throughout `ml.training` and
`ml.training_calib_dca` for `train_test_split(..., random_state=42)`).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_SEED: int = 42
SEED_ENV: str = "AMOEBANATOR_SEED"


@dataclass(frozen=True)
class SeedReport:
    """Returned by set_global_seeds so callers can log the values they pinned."""
    seed: int
    python_random: bool
    numpy: bool
    torch_cpu: bool
    torch_cuda: bool
    torch_mps: bool
    deterministic: bool


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    env = os.environ.get(SEED_ENV)
    if env is not None and env.strip():
        try:
            return int(env)
        except ValueError as e:
            raise ValueError(
                f"{SEED_ENV} must be an integer; got {env!r}"
            ) from e
    return DEFAULT_SEED


def set_global_seeds(
    seed: int | None = None,
    *,
    deterministic: bool = True,
) -> SeedReport:
    """
    Pin all RNGs to the given seed (or AMOEBANATOR_SEED, or 42).

    Parameters
    ----------
    seed:
        Explicit seed; falls back to env var, then DEFAULT_SEED.
    deterministic:
        When True, also disables cuDNN benchmark and asks PyTorch to use
        deterministic algorithms. Set False if you need a small speed boost
        on training-only code paths.
    """
    actual = _resolve_seed(seed)

    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(actual)

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps_available:
        # PyTorch's MPS backend uses the same RNG as torch.manual_seed; this
        # branch is here for symmetry / reporting.
        pass

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some kernels (e.g. older MPS ops) raise NotImplementedError under
            # use_deterministic_algorithms; the pin is still effective for the
            # parts that are deterministic.
            pass

    return SeedReport(
        seed=actual,
        python_random=True,
        numpy=True,
        torch_cpu=True,
        torch_cuda=cuda_available,
        torch_mps=mps_available,
        deterministic=deterministic,
    )
