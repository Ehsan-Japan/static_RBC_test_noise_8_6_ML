"""
ray_dataset.py — build LABELED 1D RAY TRACES for training the transition
detector.

The unit of data is one ray: the charge-sensor signal sampled at N points
along a straight line in (P1, P2) voltage space, plus a 0/1 label per point
("is there a charge transition here?").

Two sources, same output format:

  1. rays_from_sample()  — the real thing.  Cuts random rays through the
     sensor grid of a FINISHED sample folder and labels every ray point
     from the ground-truth transition map the simulator saved
     (double_dot_data.npy).  Training data therefore costs nothing new: every
     run you already generated becomes ray training data.

  2. synthetic_rays()    — a fast standalone generator (logistic steps at
     random positions + drift).  Used for pretraining and for
     testing this module without any sample folders.

Both return
    X : (n_rays, n_points) float32   per-ray z-scored sensor traces
    Y : (n_rays, n_points) float32   1.0 within `label_radius` of a transition
"""
import os
from typing import Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------

def _load_grid(npy_path: str):
    """(ux, uy, Z) from a flattened (Vx, Vy, z) .npy file."""
    data = np.load(npy_path)
    ux, uy = np.unique(data[:, 0]), np.unique(data[:, 1])
    Z = data[:, 2].reshape(len(uy), len(ux))
    return ux, uy, Z


def _bilinear(ux, uy, Z, pts):
    """Bilinear interpolation of grid Z at (N,2) voltage points."""
    x = np.clip(pts[:, 0], ux[0], ux[-1])
    y = np.clip(pts[:, 1], uy[0], uy[-1])
    ix = np.clip(np.searchsorted(ux, x) - 1, 0, len(ux) - 2)
    iy = np.clip(np.searchsorted(uy, y) - 1, 0, len(uy) - 2)
    tx = (x - ux[ix]) / (ux[ix + 1] - ux[ix])
    ty = (y - uy[iy]) / (uy[iy + 1] - uy[iy])
    return ((1 - tx) * (1 - ty) * Z[iy, ix] + tx * (1 - ty) * Z[iy, ix + 1]
            + (1 - tx) * ty * Z[iy + 1, ix] + tx * ty * Z[iy + 1, ix + 1])


# Default polynomial order removed from every trace before z-scoring.  See
# normalize_trace() for why this is not zero.
DETREND_DEGREE = 3


def normalize_trace(X, detrend_degree: int = DETREND_DEGREE):
    """
    The ONE preprocessing the detector sees, at training and at inference.

    Z-scoring alone is not enough.  Measured on the 300-device training run:
    a degree-1 ramp explains 64% of the variance of a typical ray and a
    degree-2 fit explains 92% — the charge sensor's own smooth response to
    the gate voltage, which is identical from ray to ray and carries no
    information about where the transitions are.  With that background left
    in, the traces are nearly indistinguishable: effective rank 3.6 over 2400
    rays, and half of them have a near-duplicate (cosine > 0.99).  Removing a
    low-order polynomial first lifts the rank to 10.5 and drops duplicates to
    28% on the same data, because what survives is the step structure.

    Both callers MUST go through here.  detector.py used to carry its own copy
    of the z-score, which is exactly the kind of thing that silently drifts
    away from the training preprocessing and costs a re-training to find.

    detrend_degree=0 restores the old plain z-score.
    """
    X = np.asarray(X, dtype=np.float32)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[None, :]
    if detrend_degree > 0 and X.shape[1] > detrend_degree + 1:
        t = np.linspace(-1.0, 1.0, X.shape[1])
        B = np.vander(t, detrend_degree + 1)
        # lstsq over all traces at once: B (n_points, deg+1), X.T (n_points, n)
        coef = np.linalg.lstsq(B, X.T, rcond=None)[0]
        X = X - (B @ coef).T
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-9
    out = ((X - mu) / sd).astype(np.float32)
    return out[0] if squeeze else out


def _zscore(X):
    """Backwards-compatible alias — plain z-score, no detrending."""
    return normalize_trace(X, detrend_degree=0)


# ----------------------------------------------------------------------
# Source 1: rays cut from a finished sample folder
# ----------------------------------------------------------------------

def rays_from_sample(sample_dir: str,
                     n_rays: int = 200,
                     n_points: int = 128,
                     ray_length_frac: float = 0.7,
                     label_radius_frac: float = 1.5,
                     rng: Optional[np.random.Generator] = None,
                     detrend_degree: int = DETREND_DEGREE,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random rays through one sample's sensor grid, labeled from the
    simulator's ground-truth transition map.

    Parameters
    ----------
    n_rays           : rays to cut from this sample
    n_points         : samples per ray
    ray_length_frac  : ray length as a fraction of the voltage-window side
    label_radius_frac: labeling tolerance in units of one GROUND-TRUTH grid
                       cell (1.5 = the transition line plus its neighbours,
                       matching the band thickness the sweeps produce)
    detrend_degree   : polynomial order removed before z-scoring, see
                       normalize_trace().  0 = the old plain z-score.
    """
    rng = rng or np.random.default_rng()
    sim = os.path.join(sample_dir, "numpy", "simulation")
    ux, uy, Z = _load_grid(os.path.join(sim, "charge_sensing_data.npy"))
    gx, gy, G = _load_grid(os.path.join(sim, "double_dot_data.npy"))

    # ground-truth transition cell centres and labeling radius in volts
    cell = float(min(np.diff(gx).mean(), np.diff(gy).mean()))
    radius = label_radius_frac * cell
    ys, xs = np.nonzero(G > 0.5)
    truth_pts = np.stack([gx[xs], gy[ys]], axis=1)          # (M, 2)

    span = min(ux[-1] - ux[0], uy[-1] - uy[0])
    L = ray_length_frac * span

    X = np.empty((n_rays, n_points), dtype=np.float32)
    Y = np.zeros((n_rays, n_points), dtype=np.float32)
    for k in range(n_rays):
        theta = rng.uniform(0, np.pi)                       # any orientation
        d = np.array([np.cos(theta), np.sin(theta)])
        # start so the whole ray stays inside the window
        lo = np.array([ux[0], uy[0]]) + np.abs(d) * 0     # corner
        p0 = np.array([rng.uniform(ux[0], ux[-1] - abs(d[0]) * L),
                       rng.uniform(uy[0], uy[-1] - abs(d[1]) * L)])
        ts = np.linspace(0.0, L, n_points)
        pts = p0[None, :] + ts[:, None] * d[None, :]
        X[k] = _bilinear(ux, uy, Z, pts)
        if len(truth_pts):
            # label = any ground-truth cell within `radius` of the ray point
            d2 = ((pts[:, None, :] - truth_pts[None, :, :]) ** 2).sum(-1)
            Y[k] = (d2.min(axis=1) < radius ** 2).astype(np.float32)
    return normalize_trace(X, detrend_degree), Y


def rays_from_run(run_dir: str, rays_per_sample: int = 200,
                  n_points: int = 128, seed: int = 0, **kw):
    """rays_from_sample() over every sample_* folder in a run directory."""
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for name in sorted(os.listdir(run_dir)):
        sdir = os.path.join(run_dir, name)
        need = os.path.join(sdir, "numpy", "simulation",
                            "charge_sensing_data.npy")
        if name.startswith("sample_") and os.path.isfile(need):
            x, y = rays_from_sample(sdir, rays_per_sample, n_points,
                                    rng=rng, **kw)
            Xs.append(x); Ys.append(y)
    if not Xs:
        raise FileNotFoundError(f"no usable sample_* folders in {run_dir}")
    return np.concatenate(Xs), np.concatenate(Ys)


# ----------------------------------------------------------------------
# Source 2: fully synthetic 1D traces (no sample folders needed)
# ----------------------------------------------------------------------

def synthetic_rays(n_rays: int = 2000, n_points: int = 128,
                   seed: int = 0,
                   detrend_degree: int = DETREND_DEGREE
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge-sensing-like traces: a few logistic steps of random sign, height,
    width and position, plus a slow background drift.  Labels mark points
    within 2 transition-widths of each step centre.  Mimics what a ray sees
    when it crosses dot-to-lead lines.

    The traces are clean, matching the noiseless simulator.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_points)
    X = np.empty((n_rays, n_points), dtype=np.float32)
    Y = np.zeros((n_rays, n_points), dtype=np.float32)
    for k in range(n_rays):
        z = rng.normal(0, 0.3) * t + rng.normal(0, 0.2) * np.sin(
            2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 2 * np.pi))
        for _ in range(rng.integers(0, 5)):                 # 0-4 transitions
            c = rng.uniform(0.05, 0.95)
            w = rng.uniform(0.004, 0.02)                    # transition width
            h = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])
            z = z + h / (1.0 + np.exp(-(t - c) / w))
            Y[k, np.abs(t - c) < 2 * w] = 1.0
        X[k] = z
    # Same normalization as the real rays — pretraining on traces preprocessed
    # differently from the fine-tuning data is a silent domain gap.
    return normalize_trace(X, detrend_degree), Y


# ----------------------------------------------------------------------
# CLI: build and save a dataset
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", help="training_data run with sample_* folders "
                    "(<project>/training_data/<run>)")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="number of synthetic rays instead of --run-dir")
    ap.add_argument("--rays-per-sample", type=int, default=200)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--out", default="ray_dataset.npz")
    a = ap.parse_args()
    if a.run_dir:
        X, Y = rays_from_run(a.run_dir, a.rays_per_sample, a.n_points)
    elif a.synthetic:
        X, Y = synthetic_rays(a.synthetic, a.n_points)
    else:
        ap.error("give --run-dir or --synthetic N")
    np.savez_compressed(a.out, X=X, Y=Y)
    print(f"saved {a.out}: X{X.shape}, positives "
          f"{100 * Y.mean():.1f}% of points")
