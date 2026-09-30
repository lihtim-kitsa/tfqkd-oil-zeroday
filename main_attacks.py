"""
main_attacks.py
===============
Week 3 driver script.

1. Runs a key-rate sweep (vs distance) under all attack modes:
   Nominal, FIM, TWIRL, and all four Zero-Day sub-types (PNI, CDA, DSI, CC).
   Saves results/key_rate_under_attacks.png.

2. Generates the labelled dataset (train/val/test splits) via simulator.dataset.
   With default settings this takes several minutes (Numba JIT compile + sims).
   Pass --skip-dataset to only regenerate the plots.

Usage
-----
    python main_attacks.py
    python main_attacks.py --skip-dataset
    python main_attacks.py --seed 123
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

from simulator.tf_qkd import TFQKDProtocol, estimate_decoy_parameters, calculate_key_rate
from simulator.config import ATTENUATION_COEF
from simulator.attacks import (
    NOMINAL, FIM, TWIRL, PNI, CDA, DSI, CC,
    CFI, CAM, DPJ, SAM, RTN, MTC,
    ATTACK_NAMES, attack_params_for_mode,
)
from simulator.dataset import generate_dataset


# ── Plotting style ─────────────────────────────────────────────────────────────
ATTACK_STYLE = {
    NOMINAL: dict(color="#2196F3", ls="-",    lw=2.5,  marker="o", label="Nominal"),
    FIM:     dict(color="#F44336", ls="--",   lw=2.0,  marker="s", label="FIM"),
    TWIRL:   dict(color="#FF9800", ls="-.",   lw=2.0,  marker="^", label="TWIRL"),
    PNI:     dict(color="#9C27B0", ls=":",    lw=1.8,  marker="D", label="PNI (ZD-A)"),
    CDA:     dict(color="#4CAF50", ls=":",    lw=1.8,  marker="v", label="CDA (ZD-B)"),
    DSI:     dict(color="#00BCD4", ls=":",    lw=1.8,  marker="P", label="DSI (ZD-C)"),
    CC:      dict(color="#FF5722", ls=":",    lw=1.8,  marker="*", label="CC  (ZD-D)"),
    CFI:     dict(color="#E91E63", ls=":",    lw=1.8,  marker="x", label="CFI (ZD-E)"),
    CAM:     dict(color="#8BC34A", ls=":",    lw=1.8,  marker="+", label="CAM (ZD-F)"),
    DPJ:     dict(color="#3F51B5", ls=":",    lw=1.8,  marker="d", label="DPJ (ZD-G)"),
    SAM:     dict(color="#795548", ls=":",    lw=1.8,  marker="p", label="SAM (ZD-H)"),
    RTN:     dict(color="#607D8B", ls=":",    lw=1.8,  marker="1", label="RTN (ZD-I)"),
    MTC:     dict(color="#FFEB3B", ls=":",    lw=1.8,  marker="2", label="MTC (ZD-J)"),
}


def plob_bound(distance_km):
    loss_dB = ATTENUATION_COEF * distance_km
    eta     = 10 ** (-loss_dB / 10.0)
    return -np.log2(1 - eta)


def sweep_key_rate(attack_mode, attack_params, distances, n_pulses=30_000):
    """Run key-rate estimation at each distance for a given attack mode."""
    rates = []
    for d in distances:
        proto  = TFQKDProtocol(d, n_pulses=n_pulses, use_oil=True)
        result = proto.run(attack_mode=attack_mode, **attack_params)
        Y1, e1, Qmm, Emm = estimate_decoy_parameters(result)
        rates.append(calculate_key_rate(Y1, e1, Qmm, Emm))
    return rates


def plot_key_rates(distances, rates_by_mode, output_path="results/key_rate_under_attacks.png"):
    """Generate and save the key-rate comparison figure."""
    plt.figure(figsize=(12, 8))

    plob = [plob_bound(d) for d in distances]
    plt.semilogy(distances, plob, "k--", lw=2, label="PLOB bound")

    for mode, rates in rates_by_mode.items():
        style = ATTACK_STYLE[mode]
        valid = [(d, r) for d, r in zip(distances, rates) if r > 0]
        if not valid:
            print(f"  [warn] No positive rates for {ATTACK_NAMES[mode]}, skipping.")
            continue
        ds, rs = zip(*valid)
        plt.semilogy(ds, rs, **style)

    plt.xlabel("Distance (km)", fontsize=13)
    plt.ylabel("Secure Key Rate (bits/pulse)", fontsize=13)
    plt.title("TF-QKD Key Rate Under Attack — Week 3 Validation", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[main_attacks] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Week 3: attack injection validation + dataset generation.")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip dataset generation; only run key-rate sweep.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for dataset generation (default: 42).")
    args = parser.parse_args()

    distances = np.array([10, 50, 100, 150, 200])   # Reduced distances for faster sweep

    # ── Key-rate sweep ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Week 3 — Key-Rate Sweep Under All Attack Modes")
    print("=" * 60)

    ALL_MODES = [NOMINAL, FIM, TWIRL, PNI, CDA, DSI, CC, CFI, CAM, DPJ, SAM, RTN, MTC]
    rates_by_mode = {}

    for mode in ALL_MODES:
        params = attack_params_for_mode(mode)
        print(f"\n[sweep] {ATTACK_NAMES[mode]} ...")
        rates = sweep_key_rate(mode, params, distances)
        rates_by_mode[mode] = rates

        # Print table row
        print(f"  {'km':>6}  {'rate':>14}")
        for d, r in zip(distances, rates):
            marker = "" if r > 0 else "  (zero)"
            print(f"  {d:6.0f}  {r:14.3e}{marker}")

    plot_key_rates(distances, rates_by_mode)

    # ── Dataset generation ────────────────────────────────────────────────────
    if not args.skip_dataset:
        print("\n" + "=" * 60)
        print("Week 3 — Dataset Generation")
        print("=" * 60)
        print(f"  seed = {args.seed}")
        print("  (First run triggers Numba JIT compilation — may take ~30–60 s extra)")
        generate_dataset(seed=args.seed, verbose=True)
    else:
        print("\n[main_attacks] --skip-dataset: dataset generation skipped.")

    print("\n[main_attacks] Done.")


if __name__ == "__main__":
    main()
