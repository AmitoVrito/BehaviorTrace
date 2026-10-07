"""RandomProjectionEmbedder - JL guarantees, reproducibility, batch."""

import numpy as np
import pytest

from behaviortrace.instrumentation.gradient_embed import RandomProjectionEmbedder


def test_invalid_dims():
    with pytest.raises(ValueError):
        RandomProjectionEmbedder(input_dim=0, dim=4)
    with pytest.raises(ValueError):
        RandomProjectionEmbedder(input_dim=10, dim=0)
    with pytest.raises(ValueError):
        RandomProjectionEmbedder(input_dim=4, dim=8)  # JL only reduces


def test_invalid_density():
    with pytest.raises(ValueError):
        RandomProjectionEmbedder(input_dim=10, dim=4, density=0)
    with pytest.raises(ValueError):
        RandomProjectionEmbedder(input_dim=10, dim=4, density=1.5)


def test_embed_vector_shape():
    emb = RandomProjectionEmbedder(input_dim=100, dim=8, seed=0)
    g = np.random.default_rng(0).standard_normal(100)
    e = emb.embed_vector("r1", step=3, grad=g)
    assert e.vector.shape == (8,)
    assert e.rollout_id == "r1"
    assert e.step == 3


def test_embed_vector_wrong_shape_raises():
    emb = RandomProjectionEmbedder(input_dim=100, dim=8, seed=0)
    with pytest.raises(ValueError):
        emb.embed_vector("r1", step=0, grad=np.zeros((100, 1)))
    with pytest.raises(ValueError):
        emb.embed_vector("r1", step=0, grad=np.zeros(50))


def test_reproducibility_same_seed():
    g = np.random.default_rng(0).standard_normal(200)
    a = RandomProjectionEmbedder(input_dim=200, dim=16, seed=42).embed_vector("r", 0, g).vector
    b = RandomProjectionEmbedder(input_dim=200, dim=16, seed=42).embed_vector("r", 0, g).vector
    assert np.allclose(a, b)


def test_different_seeds_differ():
    g = np.random.default_rng(0).standard_normal(200)
    a = RandomProjectionEmbedder(input_dim=200, dim=16, seed=1).embed_vector("r", 0, g).vector
    b = RandomProjectionEmbedder(input_dim=200, dim=16, seed=2).embed_vector("r", 0, g).vector
    assert not np.allclose(a, b)


def test_batch_matches_per_vector():
    rng = np.random.default_rng(7)
    grads = rng.standard_normal((5, 64))
    emb = RandomProjectionEmbedder(input_dim=64, dim=16, seed=11)
    batch = emb.embed_batch([f"r{i}" for i in range(5)], step=3, grads=grads)
    for i, e in enumerate(batch):
        single = emb.embed_vector(f"r{i}", step=3, grad=grads[i])
        assert np.allclose(e.vector, single.vector)


def test_jl_preserves_inner_products_in_expectation():
    """Heuristic JL check: with d=512 the projected inner products track
    the original ones to within a few percent on random unit vectors."""
    rng = np.random.default_rng(0)
    D, d, n = 1024, 512, 64
    X = rng.standard_normal((n, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    emb = RandomProjectionEmbedder(input_dim=D, dim=d, seed=0, density=1.0)
    projected = emb.embed_batch([f"r{i}" for i in range(n)], step=0, grads=X)
    P = np.stack([e.vector for e in projected])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    orig_sim = X @ X.T
    proj_sim = P @ P.T
    # Mean abs error on a random sample of off-diagonal entries
    iu = np.triu_indices(n, k=1)
    err = np.abs(orig_sim[iu] - proj_sim[iu]).mean()
    assert err < 0.1, f"JL distortion too large: {err}"


def test_embed_protocol_method_unavailable_without_backend():
    emb = RandomProjectionEmbedder(input_dim=10, dim=4)
    with pytest.raises(NotImplementedError):
        emb.embed("r0", 0)


def test_countsketch_preserves_inner_product_and_sign():
    """CountSketch of the full gradient preserves dot/cosine: two vectors that
    share a localized spike stay correlated, one that lacks it stays near-orthogonal."""
    torch = pytest.importorskip("torch")
    from behaviortrace.instrumentation.gradient_embed import countsketch_gradient

    torch.manual_seed(0)

    def mk(spike):
        parts = [torch.randn(1000), torch.zeros(50000), torch.randn(3000)]
        parts[1][1234:1334] = spike  # localized signal in an otherwise-zero region
        return parts

    a, b, c = mk(2.0), mk(2.0), mk(0.0)
    k = 1 << 16
    sa = countsketch_gradient(a, k, 0)
    sb = countsketch_gradient(b, k, 0)
    sc = countsketch_gradient(c, k, 0)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    assert cos(sa, sb) > 0.05          # shared spike -> correlated
    assert abs(cos(sa, sc)) < 0.02     # no shared spike -> near-orthogonal
    assert np.allclose(sa, countsketch_gradient(a, k, 0))              # deterministic
    assert not np.allclose(sa, countsketch_gradient(a, k, 7))         # seed changes the sketch


def test_countsketch_hash_spread_at_theory():
    """The sketch's cosine-estimate spread over seeds must stay near the 1/sqrt(k) theory,
    so a later hash edit cannot quietly reintroduce the correlated-index bias.
    The old multiplicative-shift hash sat at ~1.4x theory with a ~-0.004 bias; splitmix64 is
    ~0.8x with ~0 bias."""
    torch = pytest.importorskip("torch")
    from behaviortrace.instrumentation.gradient_embed import countsketch_gradient

    torch.manual_seed(2)
    n, k, n_seeds = 2_000_000, 1 << 14, 16
    theory = 1.0 / np.sqrt(k)
    u, w = torch.randn(n), torch.randn(n)
    # a=u ; b=u+scale*w chosen so the true cosine is ~0.05 (a small, realistic signal).
    scale = float(np.sqrt((1 / 0.05**2 - 1) * (u.norm() ** 2 / w.norm() ** 2).item()))
    a, b = u, u + scale * w
    true_cos = ((a @ b) / (a.norm() * b.norm())).item()
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    errs = np.array([cos(countsketch_gradient([a], k, s), countsketch_gradient([b], k, s)) - true_cos
                     for s in range(n_seeds)])
    assert errs.std() < 1.2 * theory, f"sketch spread {errs.std():.4f} exceeds 1.2x theory {theory:.4f}"
    assert abs(errs.mean()) < 0.4 * theory, f"sketch bias {errs.mean():+.4f} too large"
