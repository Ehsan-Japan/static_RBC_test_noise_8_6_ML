"""
train_test_config.py — train and test devices from DISJOINT capacitance
intervals.

A held-out split by sample only shows the model interpolates: train and test
devices come from the same distribution, so the network can score well by
having learned the shape of one particular honeycomb family.  The claim worth
publishing is stronger — the model recovers transition lines for device
geometry it has never seen.  That needs the test devices to be drawn from
capacitance ranges the training devices could not have come from.

This module splits the geometry-controlling entries of DEFAULT_INTERVALS in
half and hands back one CapacitanceConfig per side:

    d1g1, d2g2   primary gate capacitances  -> set the line spacing
    d1g2, d2g1   cross gate capacitances    -> set the two line slopes
    d1d2         interdot capacitance       -> sets the interdot segment

Everything else (sensor couplings, d1d1, d2d2) is left shared: it does not
move the lines, and some entries cannot be split anyway — s1g2 is [0.05, 0.05],
a single point.

The honeycomb constraint survives the split for free.  Validity needs
max(cross) < min(primary); cross never exceeds 0.4 and primary never drops
below 1.0, in either half, so both halves still produce visible honeycombs.

    from dqd.ml.train_test_config import split_configs
    train_cfg, test_cfg = split_configs()          # low half / high half
"""
import copy
from typing import Dict, List, Tuple

from ..config.capacitance_config import DEFAULT_INTERVALS, CapacitanceConfig

# The entries that actually move the transition lines.
GEOMETRY_KEYS: Dict[str, List[str]] = {
    "Cdd": ["d1d2"],
    "Cgd": ["d1g1", "d1g2", "d2g1", "d2g2"],
}


def _halves(lo: float, hi: float, gap: float) -> Tuple[List[float], List[float]]:
    """
    Split [lo, hi] into a low and a high interval separated by a dead band.

    The gap (a fraction of the range) guarantees the two halves do not merely
    touch: without it a train draw at 2.4999 and a test draw at 2.5001 are the
    same device, and "disjoint intervals" would be true only on paper.
    """
    mid, span = 0.5 * (lo + hi), hi - lo
    half_gap = 0.5 * gap * span
    return [lo, mid - half_gap], [mid + half_gap, hi]


def split_intervals(intervals: Dict = None,
                    gap: float = 0.1,
                    swap: bool = False) -> Tuple[Dict, Dict]:
    """
    (train_intervals, test_intervals) with disjoint geometry ranges.

    gap  : dead band between the halves, as a fraction of each range
    swap : test on the LOW half instead of the high one.  Worth running both
           ways — if the two directions disagree a lot, the result is about
           one end of the range, not about generalisation.
    """
    base = intervals or DEFAULT_INTERVALS
    train = copy.deepcopy(base)
    test = copy.deepcopy(base)
    for matrix, keys in GEOMETRY_KEYS.items():
        for key in keys:
            lo, hi = base[matrix][key]
            if hi <= lo:
                continue                       # degenerate, nothing to split
            low, high = _halves(lo, hi, gap)
            train[matrix][key], test[matrix][key] = (
                (high, low) if swap else (low, high))
    return train, test


def split_configs(gap: float = 0.1, swap: bool = False
                  ) -> Tuple[CapacitanceConfig, CapacitanceConfig]:
    """Ready-to-use CapacitanceConfig for each side, both validated."""
    tr, te = split_intervals(gap=gap, swap=swap)
    train_cfg, test_cfg = CapacitanceConfig(tr), CapacitanceConfig(te)
    for name, cfg in (("train", train_cfg), ("test", test_cfg)):
        if not cfg.validate():
            raise ValueError(f"{name} split produced invalid intervals")
    return train_cfg, test_cfg


def describe(gap: float = 0.1, swap: bool = False) -> str:
    """Human-readable table of what was split — paste it into the paper."""
    tr, te = split_intervals(gap=gap, swap=swap)
    lines = [f"{'key':>6}  {'train':>16}  {'test':>16}"]
    for matrix, keys in GEOMETRY_KEYS.items():
        for key in keys:
            a, b = tr[matrix][key], te[matrix][key]
            lines.append(f"{key:>6}  [{a[0]:6.3f}, {a[1]:6.3f}]  "
                         f"[{b[0]:6.3f}, {b[1]:6.3f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
