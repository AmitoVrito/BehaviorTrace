"""Low-rank gradient embeddings (TRAK / LoGra lineage).

("compact per-rollout gradient embeddings - low-rank projections in the
TRAK/LoGra lineage").

The projection dimension `d` is the central cost/fidelity knob:
- Drives `behaviortrace.eval.metrics` LDS quality.
- Drives the an earlier experiment overhead measurement (< 5% target).
- Supports cross-checkpoint comparison.

This module implements the projection layer in pure NumPy. The TRL hook
that actually *captures* per-rollout gradients lives in
`backends.trl_backend`. The two halves are deliberately split so this
projection layer can be unit-tested without a model.

Projection choice: sparse Achlioptas-style ±1/0 matrix scaled by
1/sqrt(d). This preserves inner products in expectation (Johnson-
Lindenstrauss) at memory cost much lower than dense Gaussian. Seeded for
reproducibility.
"""

from __future__ import annotations

import numpy as np

from .interfaces import GradientEmbedding


class RandomProjectionEmbedder:
    """Sparse JL random projection over per-rollout gradient vectors.

    Usage:
        emb = RandomProjectionEmbedder(input_dim=D, dim=d, seed=42)
        e = emb.embed_vector("rollout-0001", step=12, grad=flat_grad)

    The projection matrix is built lazily on first call; afterwards it is
    cached on the instance. Re-instantiating with the same `seed` and
    `input_dim` reproduces the same projection - required for
    cross-checkpoint comparability.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        seed: int = 0,
        density: float = 1.0 / 3.0,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim > input_dim:
            raise ValueError(
                f"projection dim ({dim}) cannot exceed input_dim ({input_dim}); "
                "JL only reduces dimension"
            )
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density}")
        self._input_dim = input_dim
        self._dim = dim
        self._seed = seed
        self._density = density
        self._projection: np.ndarray | None = None  # shape (dim, input_dim)

    # -- API ----------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def embed(self, rollout_id: str, step: int) -> GradientEmbedding:
        """Protocol method - not callable without a backend supplying the gradient.

        Use `embed_vector` when you already have the gradient in hand
        (tests, post-hoc replay).
        """
        raise NotImplementedError(
            "RandomProjectionEmbedder.embed requires a backend gradient source; "
            "use embed_vector(rollout_id, step, grad) directly or wire this "
            "embedder into a Backend implementation."
        )

    def embed_vector(self, rollout_id: str, step: int, grad: np.ndarray) -> GradientEmbedding:
        """Project a flat gradient vector to the low-rank embedding."""
        if grad.ndim != 1:
            raise ValueError(f"grad must be 1-D, got shape {grad.shape}")
        if grad.shape[0] != self._input_dim:
            raise ValueError(
                f"grad input_dim mismatch: expected {self._input_dim}, got {grad.shape[0]}"
            )
        proj = self._get_projection()
        vec = proj @ grad.astype(np.float32, copy=False)
        return GradientEmbedding(rollout_id=rollout_id, step=step, vector=vec)

    def embed_batch(self, rollout_ids: list[str], step: int, grads: np.ndarray) -> list[GradientEmbedding]:
        """Project a batch of gradients (shape (N, input_dim)) at once."""
        if grads.ndim != 2:
            raise ValueError(f"grads must be 2-D (N, D), got shape {grads.shape}")
        if grads.shape[1] != self._input_dim:
            raise ValueError(
                f"grads input_dim mismatch: expected {self._input_dim}, got {grads.shape[1]}"
            )
        if len(rollout_ids) != grads.shape[0]:
            raise ValueError(
                f"rollout_ids ({len(rollout_ids)}) and grads ({grads.shape[0]}) length mismatch"
            )
        proj = self._get_projection()
        out = grads.astype(np.float32, copy=False) @ proj.T  # (N, dim)
        return [
            GradientEmbedding(rollout_id=rid, step=step, vector=out[i])
            for i, rid in enumerate(rollout_ids)
        ]

    # -- internals ----------------------------------------------------------

    def _get_projection(self) -> np.ndarray:
        if self._projection is None:
            rng = np.random.default_rng(self._seed)
            # Sparse Achlioptas projection scaled by 1/sqrt(d)
            # Each entry is ±sqrt(1/density)/sqrt(d) w.p. density/2 each, else 0.
            if self._density == 1.0:
                # Dense ±1: simpler, slightly higher memory but fastest matmul.
                signs = rng.choice([-1.0, 1.0], size=(self._dim, self._input_dim))
                proj = signs.astype(np.float32) / np.sqrt(self._dim)
            else:
                u = rng.random(size=(self._dim, self._input_dim))
                sign = rng.choice([-1.0, 1.0], size=(self._dim, self._input_dim))
                mask = u < self._density
                scale = np.float32(1.0 / np.sqrt(self._dim * self._density))
                proj = (mask * sign).astype(np.float32) * scale
            self._projection = proj
        return self._projection


# ---------------------------------------------------------------------------
# Full-gradient coordinate sampling
# ---------------------------------------------------------------------------
# The original capture took the FIRST input_dim flattened entries, which in a HF model
# is just the first rows of embed_tokens.weight (token ids 0..~42) - an unrepresentative
# corner that does NOT contain the behavior-token direction. Instead, sample a fixed set
# of k coordinates at random from across the WHOLE flattened gradient. The coordinate set
# depends only on the per-parameter sizes and a fixed seed, so it is identical between the
# backend capture and the notebook target (required for the dot/cosine to be meaningful).

def build_coord_plan(param_numels: list[int], k: int, seed: int):
    """Return (k_eff, plan). plan is a list of (param_index, local_offsets, out_positions)
    so a length-k vector can be gathered: out[out_positions] = param_grad[param_index][local_offsets]."""
    from collections import defaultdict

    total = int(sum(int(n) for n in param_numels))
    k_eff = int(min(k, total))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=k_eff, replace=False))
    bounds = np.cumsum([0] + [int(n) for n in param_numels])
    which = np.searchsorted(bounds, idx, side="right") - 1
    off = (idx - bounds[which]).astype(np.int64)
    groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for j in range(k_eff):
        groups[int(which[j])][0].append(int(off[j]))
        groups[int(which[j])][1].append(j)
    plan = [(w, np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
            for w, (a, b) in groups.items()]
    return k_eff, plan


def gather_coords_np(flat_grads: list, k_eff: int, plan) -> np.ndarray:
    """Gather the sampled coordinates from a list of flat (numpy) per-parameter gradients."""
    out = np.zeros(k_eff, dtype=np.float32)
    for w, loff, opos in plan:
        out[opos] = np.asarray(flat_grads[w]).reshape(-1)[loff]
    return out


# ---------------------------------------------------------------------------
# CountSketch of the full gradient
# ---------------------------------------------------------------------------
# Random coordinate sampling reads ~0.004% of a 1.5B model and can miss a signal
# localized in a few parameters (e.g. the ~4600 params of the 3 QZXBT embedding rows).
# A CountSketch uses EVERY coordinate: entry i is added with a +/-1 sign into bucket
# hash(i) of `input_dim` buckets (scatter_add). Inner products (hence dot and cosine)
# are preserved in expectation, so the behavior DIRECTION survives. Deterministic from
# the global index + seed, so the backend capture and the notebook target produce
# compatible sketches as long as they iterate the same parameters in the same order.
#
# The bucket and sign come from a splitmix64 finalizer of the index. A plain
# multiplicative-shift hash left consecutive indices (e.g. the rows of one weight matrix)
# correlated, so the sketch error ran ~1.4x over the 1/sqrt(buckets) theory with a small
# bias; splitmix mixes every bit, and separate salts make the bucket and sign independent.

# splitmix64 constants, pre-reduced to signed int64 (torch int64 multiply wraps = hashing).
_SM_M1 = -4658895280553007687   # 0xBF58476D1CE4E5B9 - 2**64
_SM_M2 = -7723592293110705685   # 0x94D049BB133111EB - 2**64
_SM_BUCKET_SALT = -7046029254386353131   # 0x9E3779B97F4A7C15 - 2**64
_SM_SIGN_SALT = 6971419160186945601      # an unrelated odd constant


def _splitmix64(x):
    # Unsigned right shift on torch int64 = arithmetic shift masked to the low (64-k) bits.
    z = (x ^ ((x >> 30) & ((1 << 34) - 1))) * _SM_M1
    z = (z ^ ((z >> 27) & ((1 << 37) - 1))) * _SM_M2
    return z ^ ((z >> 31) & ((1 << 33) - 1))


def countsketch_gradient(grad_tensors, input_dim: int, seed: int = 0, chunk: int = 8_000_000):
    """Return a length-`input_dim` CountSketch (numpy float32) of the concatenation of the
    given per-parameter gradient tensors (ordered). `input_dim` must be a power of two."""
    import torch

    if input_dim & (input_dim - 1):
        raise ValueError(f"input_dim must be a power of two, got {input_dim}")
    tensors = [g for g in grad_tensors if g is not None]
    if not tensors:
        return np.zeros(input_dim, dtype=np.float32)
    dev = tensors[0].device
    sketch = torch.zeros(input_dim, dtype=torch.float32, device=dev)
    mask = input_dim - 1
    offset = 0
    for g in tensors:
        gf = g.detach().reshape(-1).float()
        n = int(gf.shape[0])
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            gi = torch.arange(s, e, device=dev, dtype=torch.int64) + (offset + seed)
            bucket = _splitmix64(gi + _SM_BUCKET_SALT) & mask
            sign = (_splitmix64(gi + _SM_SIGN_SALT) & 1).to(torch.float32) * 2.0 - 1.0
            sketch.index_add_(0, bucket, sign * gf[s:e])
        offset += n
    out = sketch.detach().cpu()
    try:
        return out.numpy().astype(np.float32)
    except RuntimeError:
        # torch<->numpy bridge unavailable (e.g. mismatched numpy build); go via a list.
        return np.asarray(out.tolist(), dtype=np.float32)
