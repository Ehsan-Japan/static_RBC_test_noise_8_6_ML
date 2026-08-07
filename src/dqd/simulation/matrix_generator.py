"""
CapacitanceMatrixGenerator — generates random capacitance matrices
from interval specifications.  Absorbs the old matrix_utils.py.
"""
import math
import random
from typing import Dict, Iterable, List, Optional


class CapacitanceMatrixGenerator:
    """
    Generates random capacitance matrices (symmetric and general) from
    interval specifications, with optional seed-based reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        # ONE stream for the whole lifetime of the generator.  Each generate_*
        # call used to build its own random.Random(self.seed): with a seed set
        # that made every matrix and every sample identical, so a seeded run
        # produced N copies of one device.  Reuse one stream and a seed means
        # what it should — reproducible, but still varied.
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Internal: one draw
    # ------------------------------------------------------------------

    def _draw(self, label: str, intervals: Dict[str, List[float]],
              log_keys: Iterable[str]) -> float:
        """
        One value for `label`, uniform over its interval — or log-uniform if
        the label is in `log_keys`.

        Log-uniform exists for entries whose effect on the stability diagram
        goes as 1/value (d1g1, d2g2 set the honeycomb period).  Drawing those
        uniformly concentrates most devices at the short-period end and leaves
        the long-period end thinly covered; drawing log-uniformly spreads the
        period itself evenly.  Falls back to uniform if either bound is <= 0,
        where the logarithm is undefined.
        """
        lo, hi = intervals.get(label, [0, 1])
        if label in log_keys and lo > 0 and hi > 0:
            return round(math.exp(self._rng.uniform(math.log(lo), math.log(hi))), 4)
        return round(self._rng.uniform(lo, hi), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_symmetric(
        self,
        intervals: Dict[str, List[float]],
        size: int,
        labels: List[List[str]],
        log_keys: Iterable[str] = (),
    ) -> List[List[float]]:
        """
        Generate a symmetric square matrix whose entries are drawn uniformly
        from the given intervals.

        Parameters
        ----------
        intervals : dict  {label: [min, max]}
        size      : int   side length of the square matrix
        labels    : list  size×size label grid (upper triangle mirrors lower)
        """
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                value = self._draw(labels[i][j], intervals, log_keys)
                matrix[i][j] = matrix[j][i] = value
        return matrix

    def generate_general(
        self,
        intervals: Dict[str, List[float]],
        shape: List[int],
        labels: List[List[str]],
        log_keys: Iterable[str] = (),
    ) -> List[List[float]]:
        """
        Generate a general (non-symmetric) matrix.

        Parameters
        ----------
        intervals : dict       {label: [min, max]}
        shape     : [rows, cols]
        labels    : list       rows×cols label grid
        """
        rows, cols = shape
        matrix = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = self._draw(labels[i][j], intervals, log_keys)
        return matrix

    # ------------------------------------------------------------------
    # Convenience: generate all four DQD matrices at once
    # ------------------------------------------------------------------

    def generate_all(self, config) -> Dict[str, List]:
        """
        Generate Cdd, Cgd, Cds, Cgs from a CapacitanceConfig instance.

        Returns a dict with keys "Cdd", "Cgd", "Cds", "Cgs".
        """
        intervals = config.intervals
        # Older configs predate log-uniform sampling and have no such attribute.
        log_keys = getattr(config, "log_uniform_keys", frozenset())
        return {
            "Cdd": self.generate_symmetric(
                intervals["Cdd"], size=2, labels=config.labels_Cdd,
                log_keys=log_keys,
            ),
            "Cgd": self.generate_general(
                intervals["Cgd"], shape=[2, 3], labels=config.labels_Cgd,
                log_keys=log_keys,
            ),
            "Cds": self.generate_general(
                intervals["Cds"], shape=[1, 2], labels=config.labels_Cds,
                log_keys=log_keys,
            ),
            "Cgs": self.generate_general(
                intervals["Cgs"], shape=[1, 3], labels=config.labels_Cgs,
                log_keys=log_keys,
            ),
        }

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    @staticmethod
    def display(matrix: List[List[float]], labels: List[List[str]], name: str) -> None:
        """Pretty-print a matrix with its labels."""
        print(f"\nGenerated {name} Matrix:")
        for i, row in enumerate(matrix):
            row_strs = [f"{labels[i][j]}={matrix[i][j]}" for j in range(len(row))]
            print("[ " + ", ".join(row_strs) + " ]")

# =====================================================================
# HOW IT WORKS:
# 1. RANDOM SAMPLING VIA SEED: The generator class can accept a 'seed' 
#    integer. This guarantees that the "random" physics values generated 
#    can be identically reproduced on any machine during testing.
#
# 2. TWO MATRIX STRATEGIES:
#      - Symmetric Generation: Used for things like dot-to-dot interactions 
#        (Cdd) where the effect of Dot 1 on Dot 2 is identical to Dot 2 on 
#        Dot 1. It only generates the upper half and mirrors it to the lower half.
#      - General Generation: Used for standard grids (like gates-to-dots) 
#        where dimensions are non-square and values are fully independent.
#
# 3. RANGE READING: It loops through the row/column layout, looks up the 
#    text label (e.g., 'd1g1') in the configuration file, finds its allowed 
#    [min, max] range, and picks a uniform random float rounded to 4 decimals.
#
# 4. MASTER ENGINE ('generate_all'): This maps directly to the companion config 
#    file. It passes the specific shapes (e.g., 2x2 for Cdd, 2x3 for Cgd) 
#    and processes all 4 required physics matrices in a single execution block.
# =====================================================================