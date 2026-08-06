"""
device_factory.py — make the device folders the ML study reads.  LIBRARY ONLY.

There is no command line here on purpose; the main programs live in scripts/.
Both scripts/run_experiment.py and scripts/generate_ml_data.py call generate(),
so there is exactly one copy of "how a device is made".

One device folder is:

    <out_dir>/sample_<i>/numpy/simulation/
        charge_sensing_data.npy    the measurement
        double_dot_data.npy        the occupation map it came from
        ground_truth_labels.npy    the answer

That is all the ML study needs.  run_simulation.py runs the FULL per-device
analysis — ray plots, peak crops, four sweeps per peak, GIFs, overlays — which
is what you want when inspecting ONE device and far too slow for thousands.
The rays are cut later, offline, at whatever budget you ask for, so you never
regenerate data to change the number of rays or points.

generate() is RESUMABLE and never overwrites: a device whose
ground_truth_labels.npy already exists is skipped.  Pointing it at a folder
that is already full is therefore free, and asking for more devices than are
there only makes the missing ones.  That is what lets a pipeline request its
dataset on every run and pay for it only once.
"""
import os
import time
from typing import Tuple

from ..config.axis_labels import set_axis_labels
from ..visualization.overlay import OverlayRenderer
from .dqd_simulator import DQDSimulator
from .matrix_generator import CapacitanceMatrixGenerator

# The arrays we keep.  Everything else the simulator writes is pruned.
KEEP_FILES = ("charge_sensing_data.npy", "double_dot_data.npy",
              "ground_truth_labels.npy")


def dataset_name(split: str, n_devices: int, resolution: int,
                 disjoint: bool = True) -> str:
    """
    The canonical folder name for a dataset, e.g.

        dataset_name("train", 2000, 100)  ->  "ml_train_split_n2000_res100"

    Every program builds its dataset paths through this, so "the folder the
    generator wrote" and "the folder the sweep looks for" cannot drift apart.
    The name carries the settings that change the DEVICES (which split, how
    many, what resolution); budgets are cut from them later and never appear.
    """
    tag = "split" if disjoint else "same"
    return f"ml_{split}_{tag}_n{n_devices}_res{resolution}"


def generate(out_dir: str,
             n_devices: int,
             config,
             seed: int,
             resolution: int = 100,
             voltage_window: Tuple[float, float, float, float] = (-1.0, 1.0,
                                                                  -1.0, 1.0),
             coulomb_peak_width: float = 0.01,
             temperature: float = 0.00001,
             keep_images: bool = False,
             progress_every: int = 25) -> str:
    """
    Simulate n_devices into out_dir and return out_dir.

    config : a CapacitanceConfig — the intervals the device geometry is drawn
             from.  Train and test get different ones (train_test_config.py).
    seed   : one reproducible random stream per split.  Same seed, same
             devices, in the same order.

    voltage_window is (vx_min, vx_max, vy_min, vy_max), the gate-voltage
    window swept on both axes.
    """
    vx_min, vx_max, vy_min, vy_max = voltage_window
    set_axis_labels(x_name="P1", y_name="P2", x_unit="mV", y_unit="mV")
    gen = CapacitanceMatrixGenerator(seed=seed)      # one stream, reproducible
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing devices to {os.path.abspath(out_dir)}")

    t0, made = time.time(), 0
    for i in range(1, n_devices + 1):
        device_dir = os.path.join(out_dir, f"sample_{i}")
        sim_dir = os.path.join(device_dir, "numpy", "simulation")
        gt_path = os.path.join(sim_dir, "ground_truth_labels.npy")
        if os.path.isfile(gt_path):
            continue                                  # resume, don't redo
        os.makedirs(sim_dir, exist_ok=True)

        sim_params = {
            "save_dir": device_dir,
            "capacitance": gen.generate_all(config),
            "model_params": {"coulomb_peak_width": coulomb_peak_width,
                             "T": temperature},
            "xlabel": "P1 (mV)", "ylabel": "P2 (mV)",
            "voltage_sweep": {"vx_min": vx_min, "vx_max": vx_max,
                              "vy_min": vy_min, "vy_max": vy_max,
                              "n_points_x": resolution,
                              "n_points_y": resolution},
            "optimal_Vg": [0.0, 0.0, 0.0],
            "plot_options": {
                "charge_sensing_save_path": os.path.join(
                    device_dir, "charge_sensing.jpg"),
                "charge_sensing_grad_save_path": os.path.join(
                    device_dir, "charge_sensing2.jpg"),
                "dpi": 60,          # the jpgs are a by-product; keep them cheap
            },
        }
        try:
            DQDSimulator(sim_params).run()
        except Exception as exc:
            print(f"[skip] device {i}: {exc}")
            continue

        # The simulator writes into device_dir; move the arrays we need into
        # the numpy/simulation/ layout the ML loaders expect.
        for name in ("charge_sensing_data.npy", "double_dot_data.npy"):
            src = os.path.join(device_dir, name)
            if os.path.isfile(src):
                os.replace(src, os.path.join(sim_dir, name))

        OverlayRenderer.generate_ground_truth_array(
            data_path=os.path.join(sim_dir, "double_dot_data.npy"),
            output_npy_path=gt_path,
        )
        if not keep_images:
            prune(device_dir, sim_dir)

        made += 1
        if progress_every and made % progress_every == 0:
            rate = (time.time() - t0) / made
            print(f"  {i}/{n_devices}  {rate:.2f}s/device  "
                  f"~{(n_devices - i) * rate / 60:.1f} min left")

    if made:
        print(f"{made} new devices in {os.path.abspath(out_dir)}  "
              f"({time.time() - t0:.0f}s)\n")
    else:
        print(f"nothing to make — {os.path.abspath(out_dir)} already has "
              f"its devices\n")
    return out_dir


def prune(device_dir: str, sim_dir: str) -> None:
    """Delete everything in a device folder except the arrays KEEP_FILES."""
    for root, dirs, files in os.walk(device_dir, topdown=False):
        for f in files:
            path = os.path.join(root, f)
            if os.path.dirname(path) == sim_dir and f in KEEP_FILES:
                continue
            try:
                os.remove(path)
            except OSError:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass
