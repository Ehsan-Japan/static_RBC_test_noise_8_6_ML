"""
CapacitanceConfig — holds capacitance matrix parameter ranges and labels,
and validates them.  Merges the old config.py + validation.py.
"""
from typing import Dict, List


# Default intervals (from the original config.py)
#
# WHICH ENTRIES ACTUALLY CHANGE THE PICTURE.  Measured by sweeping one entry
# at a time through the simulator and reading off the honeycomb period at
# res=100 over the ±1 V window:
#
#   d1g1 / d2g2   period 32 px at 1.5 → 12 px at 4.0 → 4 px at 12.0   STRONG
#   d1g2 / d2g1   changes the two line SLOPES; period moves only at the top end
#   d1d2          period 19.5 px at 0.1 → 18.5 px at 8.0              WEAK
#   d1d1 / d2d2   period 19.5 px at 0.1 → 19.5 px at 8.0              NONE
#   d1g3, d2g3, Cds, Cgs   sensor coupling only — never move a transition line
#
# So the geometry has essentially TWO free parameters, d1g1 and d2g2.  Widening
# anything else buys no diversity; widening those two is the whole game.
DEFAULT_INTERVALS: Dict[str, Dict[str, List[float]]] = {
    "Cdd": {
        # Provably inert (see table above): the dot self-capacitances cancel
        # out of the transition condition.  Left in place so the matrix keeps
        # its physical form, but do not expect variety from them.
        "d1d1": [0.1, 1],
        # Interdot (mutual) capacitance.  Kept at a scale comparable to the
        # primary gate capacitances so the interdot transition lines are always
        # visibly long relative to the honeycomb cell size.
        "d1d2": [0.5, 3.0],
        "d2d2": [0.1, 1],
    },
    "Cgd": {
        # The honeycomb structure requires that each dot is driven primarily by
        # its OWN gate (d1g1 for dot-1, d2g2 for dot-2) and only weakly by the
        # other gate (d1g2, d2g1).
        #
        # Dot-1 transition slope  = -d1g1 / d1g2  → steep  when d1g1 >> d1g2
        # Dot-2 transition slope  = -d2g1 / d2g2  → shallow when d2g2 >> d2g1
        #
        # Enforcing  max(cross) < min(primary)  guarantees the two slope
        # families are ALWAYS distinct → honeycomb is visible in every sample.
        #
        # d1g1/d2g2 WIDENED from [1.0, 4.0] and sampled LOG-uniformly (see
        # LOG_UNIFORM_KEYS).  Two reasons:
        #   * the old low end was dead.  Period ∝ 1/d1g1, so at d1g1 = 1.0 the
        #     period exceeds the voltage window and the device shows NO
        #     transition line at all — wasted samples at the bottom of the
        #     range, and the train half of the split ([1.0, 2.35]) was mostly
        #     that end.  1.2 is the smallest value that always shows lines.
        #   * uniform sampling of d1g1 puts most devices at one period, because
        #     the period is the reciprocal.  Log-uniform spreads the PERIOD
        #     evenly, which is the quantity the detector actually sees.
        # Effect on the train split: line spacing spread went from 1.6x
        # (20-33 px) to 2.8x (12-33 px) across devices.
        "d1g1": [1.2, 8.0],   # primary gate for dot 1 — always large
        "d1g2": [0.02, 0.45], # cross gate for dot 1  — max(0.45) < min(d1g1)=1.2
        "d1g3": [0.01, 0.1],
        "d2g1": [0.02, 0.45], # cross gate for dot 2  — max(0.45) < min(d2g2)=1.2
        "d2g2": [1.2, 8.0],   # primary gate for dot 2 — always large
        "d2g3": [0.01, 1],
    },
    "Cds": {
        "s1d1": [0.01, 0.1],
        "s1d2": [0.02, 0.1],
    },
    "Cgs": {
        "s1g1": [0.03, 0.05],
        "s1g2": [0.05, 0.05],
        "s1g3": [0.1, 1],
    },
}

# Entries drawn LOG-uniformly instead of uniformly over their interval.
# Only for entries whose effect on the image goes as 1/value: sampling those
# uniformly piles most devices onto one length scale.  Both bounds must be > 0.
LOG_UNIFORM_KEYS = frozenset({"d1g1", "d2g2"})

DEFAULT_LABELS_Cdd = [
    ["d1d1", "d1d2"],
    ["d1d2", "d2d2"],
]

DEFAULT_LABELS_Cgd = [
    ["d1g1", "d1g2", "d1g3"],
    ["d2g1", "d2g2", "d2g3"],
]

DEFAULT_LABELS_Cds = [["s1d1", "s1d2"]]

DEFAULT_LABELS_Cgs = [["s1g1", "s1g2", "s1g3"]]


class CapacitanceConfig:
    """
    Stores capacitance matrix parameter ranges and label layouts,
    and exposes a validate() method.
    """

    # Required keys for each matrix (from the original validation.py)
    _REQUIRED_KEYS: Dict[str, List[str]] = {
        "Cdd": ["d1d1", "d1d2", "d2d2"],
        "Cgd": ["d1g1", "d1g2", "d1g3", "d2g1", "d2g2", "d2g3"],
        "Cds": ["s1d1", "s1d2"],
        "Cgs": ["s1g1", "s1g2", "s1g3"],
    }

    def __init__(
        self,
        intervals: Dict[str, Dict[str, List[float]]] = None,
        labels_Cdd: List[List[str]] = None,
        labels_Cgd: List[List[str]] = None,
        labels_Cds: List[List[str]] = None,
        labels_Cgs: List[List[str]] = None,
        log_uniform_keys=None,
    ):
        self.intervals = intervals if intervals is not None else DEFAULT_INTERVALS
        # Which entries the generator draws log-uniformly.  Carried on the
        # config so a custom config (train/test split) keeps the same sampling
        # law as the defaults it was derived from.
        self.log_uniform_keys = frozenset(
            LOG_UNIFORM_KEYS if log_uniform_keys is None else log_uniform_keys)
        self.labels_Cdd = labels_Cdd if labels_Cdd is not None else DEFAULT_LABELS_Cdd
        self.labels_Cgd = labels_Cgd if labels_Cgd is not None else DEFAULT_LABELS_Cgd
        self.labels_Cds = labels_Cds if labels_Cds is not None else DEFAULT_LABELS_Cds
        self.labels_Cgs = labels_Cgs if labels_Cgs is not None else DEFAULT_LABELS_Cgs

    def validate(self) -> bool:
        """Return True if all intervals are valid, False (with error prints) otherwise."""
        for matrix, required_keys in self._REQUIRED_KEYS.items():
            if matrix not in self.intervals:
                print(f"Error: Missing interval dictionary for {matrix}.")
                return False
            for key in required_keys:
                if key not in self.intervals[matrix]:
                    print(f"Error: Missing key '{key}' in interval dictionary for {matrix}.")
                    return False
                interval = self.intervals[matrix][key]
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    print(
                        f"Error: Interval for {key} in {matrix} must be a list or tuple of two numbers."
                    )
                    return False
                min_val, max_val = interval
                if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
                    print(f"Error: Interval values for {key} in {matrix} must be numeric.")
                    return False
                if min_val > max_val:
                    print(
                        f"Error: For {key} in {matrix}, min value {min_val} is greater than "
                        f"max value {max_val}."
                    )
                    return False
        return True

    def print_summary(self) -> None:
        """Print a human-readable summary of all interval ranges."""
        for matrix, intervals in self.intervals.items():
            print(f"\n{matrix}:")
            for key, interval in intervals.items():
                print(f"  {key}: {interval}")
                
# =====================================================================
# HOW IT WORKS:
# 1. CENTRALIZED STORAGE: The class aggregates configuration metadata (ranges and
#    structural labels) required to model a double quantum dot (DQD) system.
# 2. FAIL-SAFE INITIALIZATION: If a user instantiates the class without passing 
#    custom ranges, it gracefully defaults to pre-vetted physics ranges that 
#    guarantee a clear experimental "honeycomb" structure.
# 3. ROBUST VALIDATION MECHANISM: Calling `.validate()` loops through a strict 
#    blueprint (`_REQUIRED_KEYS`). It flags structural gaps (missing metrics), 
#    type inconsistencies (non-numeric limits), and logical errors (inverted boundaries).
#
# FIVE CONCRETE USAGE EXAMPLES:
#
# EXAMPLE 1: Standard Initialization, Validation, and Summary
#   config = CapacitanceConfig()
#   if config.validate():
#       config.print_summary()  # Prints out all default ranges cleanly
#
# EXAMPLE 2: Modifying an Existing Parameter Within Safe Bounds
#   config = CapacitanceConfig()
#   config.intervals["Cdd"]["d1d1"] = [0.2, 0.8] # Tweak bounds dynamically
#   print(config.validate())                    # Returns True (valid change)
#
# EXAMPLE 3: Failing Validation due to an Inverted Range (Logical Error)
#   config = CapacitanceConfig()
#   config.intervals["Cdd"]["d1d2"] = [5.0, 1.0] # Fatal: min is greater than max
#   print(config.validate())                    # Returns False, prints specific Error
#
# EXAMPLE 4: Failing Validation due to Incomplete Data Structure (Missing Key)
#   config = CapacitanceConfig()
#   del config.intervals["Cgs"]["s1g1"]         # Missing expected key rule
#   print(config.validate())                    # Returns False, highlights missing 's1g1'
#
# EXAMPLE 5: Passing Fully Customized Matrices for Alternative Layouts
#   custom_intervals = {
#       "Cdd": {"d1d1": [0.5, 0.6], "d1d2": [1.0, 2.0], "d2d2": [0.5, 0.6]},
#       "Cgd": {"d1g1": [2., 3.], "d1g2": [0.1, 0.2], "d1g3": [0., 0.],
#               "d2g1": [0.1, 0.2], "d2g2": [2., 3.], "d2g3": [0., 0.]},
#       "Cds": {"s1d1": [0.0, 0.1], "s1d2": [0.0, 0.1]},
#       "Cgs": {"s1g1": [0.0, 0.1], "s1g2": [0.0, 0.1], "s1g3": [0.5, 0.6]}
#   }
#   custom_config = CapacitanceConfig(intervals=custom_intervals)
#   print(custom_config.validate())             # Returns True
# =====================================================================