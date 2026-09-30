"""
simulator/dataset.py
====================
Generates the labelled pulse-level dataset for Deep SVDD training.

Dataset structure
-----------------
Features (7 per pulse):
  0  phase_A      — Alice's post-OIL phase (rad)
  1  phase_B      — Bob's post-OIL phase (rad)
  2  intensity_A  — Alice's pulse intensity
  3  intensity_B  — Bob's pulse intensity
  4  phase_diff   — (phase_A − phase_B) mod 2π
  5  click_D0     — D0 detector outcome (0/1)
  6  click_D1     — D1 detector outcome (0/1)

Labels:
  0 — Nominal (clean OIL dynamics)
  1 — FIM
  2 — TWIRL
  3 — Zero-Day Family  (random mix of PNI, CDA, DSI, CC at generation time)

Split policy
------------
  train : 70 % — nominal ONLY  (Deep SVDD trains on clean data)
  val   : 15 % — all 4 labels  (for anomaly threshold selection)
  test  : 15 % — all 4 labels  (for final evaluation, frozen after this script)

Files written to results/
  dataset_train.npz          — keys: X (float32), y (int8)
  dataset_val.npz            — keys: X (float32), y (int8)
  dataset_test.npz           — keys: X (float32), y (int8)
  splits_seed.txt            — records the RNG seed for reproducibility
  zeroday_subtypes_val.npy   — per-trace ZD sub-type (int8) for val split
  zeroday_subtypes_test.npy  — per-trace ZD sub-type (int8) for test split
    (non-ZD traces are coded as -1)
"""

import os
import numpy as np

from simulator.tf_qkd import TFQKDProtocol
from simulator.attacks import (
    NOMINAL, FIM, TWIRL, PNI, CDA, DSI, CC,
    CFI, CAM, DPJ, SAM, RTN, MTC,
    ZERO_DAY_SUBTYPES, ATTACK_NAMES,
    sample_attack_params,
)
from simulator.config import DEFAULT_N_PULSES

def extract_strength(mode, params):
    if mode == FIM: return params.get("fim_mod_depth", 0.0)
    elif mode == TWIRL: return params.get("twirl_sweep_hz", 0.0)
    elif mode == PNI: return params.get("pni_sigma", 0.0)
    elif mode == CDA: return params.get("cda_amplitude", 0.0)
    elif mode == DSI: return params.get("dsi_ratio", 0.0)
    elif mode == CC: return params.get("cc_ratio_boost", 0.0)
    elif mode == CFI: return params.get("cfi_sweep_hz", 0.0)
    elif mode == CAM: return params.get("cam_sigma", 0.0)
    elif mode == DPJ: return params.get("dpj_phase_shift", 0.0)
    elif mode == SAM: return params.get("sam_amplitude", 0.0)
    elif mode == RTN: return params.get("rtn_amplitude", 0.0)
    elif mode == MTC: return params.get("mtc_ratio", 0.0)
    return 0.0


# ── Dataset generation constants ──────────────────────────────────────────────
DATASET_DISTANCE_KM = 100        # fixed mid-range distance for all traces
PULSES_PER_BATCH    = 100_000    # pulses per batch (equals one window!)
N_BATCHES_NOMINAL   = 70         # 70 batches × 100k = 7.0M nominal pulses
N_BATCHES_ATTACK    = 50         # 50 batches × 100k = 5.0M per attack class
TRAIN_FRAC          = 0.70
VAL_FRAC            = 0.15
# TEST_FRAC is implicit: 1 - TRAIN_FRAC - VAL_FRAC = 0.15


def _run_protocol_batch(distance_km, n_pulses, attack_mode, attack_params, alpha=None):
    """Run one batch and return the raw feature array (float32, shape [n, 7])."""
    protocol_kwargs = {} if alpha is None else {"alpha": alpha}
    proto  = TFQKDProtocol(
        distance_km, n_pulses=n_pulses, use_oil=True, **protocol_kwargs
    )
    result = proto.run(attack_mode=attack_mode, **attack_params)

    phi_A  = result["phi_A"].astype(np.float32)
    phi_B  = result["phi_B"].astype(np.float32)
    I_A    = result["I_A"].astype(np.float32)
    I_B    = result["I_B"].astype(np.float32)
    d_phi  = ((phi_A - phi_B) % (2 * np.pi)).astype(np.float32)
    c_D0   = result["clicks_D0"].astype(np.float32)
    c_D1   = result["clicks_D1"].astype(np.float32)

    return np.stack([phi_A, phi_B, I_A, I_B, d_phi, c_D0, c_D1], axis=1)


def _run_checkpointed_batch(cache_dir, key, run_batch):
    """Load a completed simulation batch or atomically compute and cache it."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            return cached["X"].copy()

    X = run_batch()
    temp_path = cache_path + ".tmp.npz"
    np.savez_compressed(temp_path, X=X)
    os.replace(temp_path, cache_path)
    return X


def generate_dataset(
    seed: int = 42,
    distance_km: float = DATASET_DISTANCE_KM,
    pulses_per_batch: int = PULSES_PER_BATCH,
    n_batches_nominal: int = N_BATCHES_NOMINAL,
    n_batches_attack: int  = N_BATCHES_ATTACK,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:
    """
    Generate and save the labelled dataset.

    Parameters
    ----------
    seed : int
        Master RNG seed. Recorded in results/splits_seed.txt.
    distance_km : float
        Channel distance used for all simulated traces.
    pulses_per_batch : int
        Number of pulses per simulation batch.
    n_batches_nominal : int
        Number of nominal batches (total nominal = n_batches_nominal × pulses_per_batch).
    n_batches_attack : int
        Number of batches per attack class (FIM, TWIRL, zero-day family).
    output_dir : str
        Directory where .npz and .npy files are saved.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                    'zd_subtypes_val', 'zd_subtypes_test'
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Keep expensive pulse batches across interruption. Bump v1 if the
    # simulator equations, attack parameter distributions, or batch semantics change.
    cache_dir = os.path.join(
        output_dir,
        f".dataset_cache_v1_seed{seed}_d{distance_km}_p{pulses_per_batch}_"
        f"nom{n_batches_nominal}_atk{n_batches_attack}",
    )

    # ── 1. Simulate nominal traces ────────────────────────────────────────────
    if verbose:
        print(f"[dataset] Simulating nominal traces ({n_batches_nominal} batches × {pulses_per_batch:,} pulses)...")

    X_nom_list = []
    for i in range(n_batches_nominal):
        if verbose and i % 10 == 0:
            print(f"  nominal batch {i+1}/{n_batches_nominal}")
        batch_seed = int(rng.integers(0, 2**31))
        np.random.seed(batch_seed)
        X_nom_list.append(_run_checkpointed_batch(
            cache_dir, f"{distance_km:g}km_nominal_{i:04d}",
            lambda: _run_protocol_batch(distance_km, pulses_per_batch, NOMINAL, {}),
        ))
    X_nom = np.concatenate(X_nom_list, axis=0)
    y_nom = np.zeros(len(X_nom), dtype=np.int8)

    # ── 2. Simulate FIM traces ────────────────────────────────────────────────
    if verbose:
        print(f"[dataset] Simulating FIM traces ({n_batches_attack} batches)...")

    X_fim_list = []
    strength_fim_list = []
    for i in range(n_batches_attack):
        if verbose and i % 10 == 0:
            print(f"  FIM batch {i+1}/{n_batches_attack}")
        batch_seed = int(rng.integers(0, 2**31))
        np.random.seed(batch_seed)
        params = sample_attack_params(FIM, rng)
        strength = extract_strength(FIM, params)
        X_fim_list.append(_run_checkpointed_batch(
            cache_dir, f"{distance_km:g}km_fim_{i:04d}",
            lambda: _run_protocol_batch(distance_km, pulses_per_batch, FIM, params),
        ))
        strength_fim_list.append(np.full(pulses_per_batch, strength, dtype=np.float32))
    X_fim = np.concatenate(X_fim_list, axis=0)
    y_fim = np.ones(len(X_fim), dtype=np.int8)  # label 1
    s_fim = np.concatenate(strength_fim_list, axis=0)

    # ── 3. Simulate TWIRL traces ──────────────────────────────────────────────
    if verbose:
        print(f"[dataset] Simulating TWIRL traces ({n_batches_attack} batches)...")

    X_twirl_list = []
    strength_twirl_list = []
    for i in range(n_batches_attack):
        if verbose and i % 10 == 0:
            print(f"  TWIRL batch {i+1}/{n_batches_attack}")
        batch_seed = int(rng.integers(0, 2**31))
        np.random.seed(batch_seed)
        params = sample_attack_params(TWIRL, rng)
        strength = extract_strength(TWIRL, params)
        X_twirl_list.append(_run_checkpointed_batch(
            cache_dir, f"{distance_km:g}km_twirl_{i:04d}",
            lambda: _run_protocol_batch(distance_km, pulses_per_batch, TWIRL, params),
        ))
        strength_twirl_list.append(np.full(pulses_per_batch, strength, dtype=np.float32))
    X_twirl = np.concatenate(X_twirl_list, axis=0)
    y_twirl = np.full(len(X_twirl), 2, dtype=np.int8)  # label 2
    s_twirl = np.concatenate(strength_twirl_list, axis=0)

    # ── 4. Simulate Zero-Day Family traces ────────────────────────────────────
    if verbose:
        print(f"[dataset] Simulating Zero-Day family traces ({n_batches_attack} batches, random sub-types)...")

    X_zd_list    = []
    zd_sub_list  = []   # per-trace sub-type int
    strength_zd_list = []

    for i in range(n_batches_attack):
        subtype = int(rng.choice(ZERO_DAY_SUBTYPES))
        if verbose and i % 10 == 0:
            print(f"  ZD batch {i+1}/{n_batches_attack} → sub-type: {ATTACK_NAMES[subtype]}")
        batch_seed = int(rng.integers(0, 2**31))
        np.random.seed(batch_seed)
        params = sample_attack_params(subtype, rng)
        strength = extract_strength(subtype, params)
        X_batch = _run_checkpointed_batch(
            cache_dir, f"{distance_km:g}km_zd_{i:04d}",
            lambda: _run_protocol_batch(distance_km, pulses_per_batch, subtype, params),
        )
        X_zd_list.append(X_batch)
        zd_sub_list.append(np.full(len(X_batch), subtype, dtype=np.int8))
        strength_zd_list.append(np.full(len(X_batch), strength, dtype=np.float32))

    X_zd     = np.concatenate(X_zd_list,   axis=0)
    y_zd     = np.full(len(X_zd), 3, dtype=np.int8)   # label 3
    zd_subs  = np.concatenate(zd_sub_list, axis=0)
    s_zd     = np.concatenate(strength_zd_list, axis=0)

    # ── 5. Build splits ───────────────────────────────────────────────────────
    # Train: nominal only, sequential to preserve windows
    nom_idx = np.arange(len(X_nom))
    n_train = int(len(X_nom) * TRAIN_FRAC / 1.0)   # use all nominal for train
    # We still carve val/test from nominal too so the threshold-setting
    # split has clean in-distribution negatives mixed with attack positives.
    n_nom_val  = int(len(X_nom) * VAL_FRAC)
    n_nom_test = len(X_nom) - n_train - n_nom_val

    train_nom_idx = nom_idx[:n_train]
    val_nom_idx   = nom_idx[n_train : n_train + n_nom_val]
    test_nom_idx  = nom_idx[n_train + n_nom_val :]

    X_train = X_nom[train_nom_idx]
    y_train = y_nom[train_nom_idx]

    # Val & test: balance nominal negatives with attack positives
    # Equal numbers of each attack class in val & test
    def split_attack(X_atk, y_atk, s_atk, zd_sub=None):
        idx = np.arange(len(X_atk))
        n_val  = int(len(X_atk) * (VAL_FRAC  / (VAL_FRAC + (1 - TRAIN_FRAC - VAL_FRAC))))
        X_val_ = X_atk[idx[:n_val]]
        y_val_ = y_atk[idx[:n_val]]
        s_val_ = s_atk[idx[:n_val]]
        X_test_= X_atk[idx[n_val:]]
        y_test_= y_atk[idx[n_val:]]
        s_test_= s_atk[idx[n_val:]]
        if zd_sub is not None:
            return X_val_, y_val_, s_val_, X_test_, y_test_, s_test_, zd_sub[idx[:n_val]], zd_sub[idx[n_val:]]
        return X_val_, y_val_, s_val_, X_test_, y_test_, s_test_

    X_fim_val,  y_fim_val,  s_fim_val, X_fim_test,  y_fim_test, s_fim_test  = split_attack(X_fim, y_fim, s_fim)
    X_twirl_val,y_twirl_val,s_twirl_val,X_twirl_test,y_twirl_test,s_twirl_test= split_attack(X_twirl, y_twirl, s_twirl)
    X_zd_val,  y_zd_val, s_zd_val, X_zd_test,  y_zd_test, s_zd_test, zd_subs_val, zd_subs_test = split_attack(X_zd, y_zd, s_zd, zd_subs)

    # Non-ZD traces get sub-type -1
    neg_val_subs  = np.full(n_nom_val + len(X_fim_val)  + len(X_twirl_val),  -1, dtype=np.int8)
    neg_test_subs = np.full(n_nom_test+ len(X_fim_test) + len(X_twirl_test), -1, dtype=np.int8)

    X_val  = np.concatenate([X_nom[val_nom_idx],  X_fim_val,  X_twirl_val,  X_zd_val],  axis=0)
    y_val  = np.concatenate([y_nom[val_nom_idx],  y_fim_val,  y_twirl_val,  y_zd_val],  axis=0)
    s_val  = np.concatenate([np.zeros(n_nom_val, dtype=np.float32), s_fim_val, s_twirl_val, s_zd_val], axis=0)
    zd_subtypes_val  = np.concatenate([neg_val_subs,  zd_subs_val],  axis=0)

    X_test = np.concatenate([X_nom[test_nom_idx], X_fim_test, X_twirl_test, X_zd_test], axis=0)
    y_test = np.concatenate([y_nom[test_nom_idx], y_fim_test, y_twirl_test, y_zd_test], axis=0)
    s_test = np.concatenate([np.zeros(n_nom_test, dtype=np.float32), s_fim_test, s_twirl_test, s_zd_test], axis=0)
    zd_subtypes_test = np.concatenate([neg_test_subs, zd_subs_test], axis=0)

    # Final shuffle within val and test removed to preserve temporal pulse windows
    # Window-level shuffling is handled during training by DataLoader

    # ── 6. Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(os.path.join(output_dir, "dataset_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, "dataset_val.npz"),   X=X_val,   y=y_val, strength=s_val)
    np.savez_compressed(os.path.join(output_dir, "dataset_test.npz"),  X=X_test,  y=y_test, strength=s_test)
    np.save(os.path.join(output_dir, "zeroday_subtypes_val.npy"),  zd_subtypes_val)
    np.save(os.path.join(output_dir, "zeroday_subtypes_test.npy"), zd_subtypes_test)

    # ── 7. Generate Multi-Distance OOD test sets ─────────────────────────────
    if verbose:
        print("\n[dataset] Generating multi-distance OOD test sets (50km, 150km)...")
    
    def _generate_test_at_dist(dist_km):
        n_test_nom = n_nom_test // pulses_per_batch
        n_test_atk = int(n_batches_attack * (1.0 - TRAIN_FRAC - VAL_FRAC) / (1.0 - TRAIN_FRAC))
        # Wait, simple approximation: just generate some batches
        n_test_atk = max(1, n_batches_attack // 4)
        n_test_nom = max(1, n_batches_nominal // 4)
        
        X_list, y_list, s_list, zd_list = [], [], [], []
        # Nominal
        for i in range(n_test_nom):
            batch_seed = int(rng.integers(0, 2**31))
            np.random.seed(batch_seed)
            X_list.append(_run_checkpointed_batch(
                cache_dir, f"{dist_km:g}km_nominal_{i:04d}",
                lambda: _run_protocol_batch(dist_km, pulses_per_batch, NOMINAL, {}),
            ))
            y_list.append(np.zeros(pulses_per_batch, dtype=np.int8))
            s_list.append(np.zeros(pulses_per_batch, dtype=np.float32))
            zd_list.append(np.full(pulses_per_batch, -1, dtype=np.int8))
        # FIM
        for i in range(n_test_atk):
            batch_seed = int(rng.integers(0, 2**31))
            np.random.seed(batch_seed)
            params = sample_attack_params(FIM, rng)
            X_list.append(_run_checkpointed_batch(
                cache_dir, f"{dist_km:g}km_fim_{i:04d}",
                lambda: _run_protocol_batch(dist_km, pulses_per_batch, FIM, params),
            ))
            y_list.append(np.ones(pulses_per_batch, dtype=np.int8))
            s_list.append(np.full(pulses_per_batch, extract_strength(FIM, params), dtype=np.float32))
            zd_list.append(np.full(pulses_per_batch, -1, dtype=np.int8))
        # TWIRL
        for i in range(n_test_atk):
            batch_seed = int(rng.integers(0, 2**31))
            np.random.seed(batch_seed)
            params = sample_attack_params(TWIRL, rng)
            X_list.append(_run_checkpointed_batch(
                cache_dir, f"{dist_km:g}km_twirl_{i:04d}",
                lambda: _run_protocol_batch(dist_km, pulses_per_batch, TWIRL, params),
            ))
            y_list.append(np.full(pulses_per_batch, 2, dtype=np.int8))
            s_list.append(np.full(pulses_per_batch, extract_strength(TWIRL, params), dtype=np.float32))
            zd_list.append(np.full(pulses_per_batch, -1, dtype=np.int8))
        # ZD
        for i in range(n_test_atk):
            subtype = int(rng.choice(ZERO_DAY_SUBTYPES))
            params = sample_attack_params(subtype, rng)
            batch_seed = int(rng.integers(0, 2**31))
            np.random.seed(batch_seed)
            X_list.append(_run_checkpointed_batch(
                cache_dir, f"{dist_km:g}km_zd_{i:04d}",
                lambda: _run_protocol_batch(dist_km, pulses_per_batch, subtype, params),
            ))
            y_list.append(np.full(pulses_per_batch, 3, dtype=np.int8))
            s_list.append(np.full(pulses_per_batch, extract_strength(subtype, params), dtype=np.float32))
            zd_list.append(np.full(pulses_per_batch, subtype, dtype=np.int8))
            
        return (np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), 
                np.concatenate(s_list, axis=0), np.concatenate(zd_list, axis=0))

    for d in [50, 150]:
        Xd, yd, sd, zdd = _generate_test_at_dist(d)
        np.savez_compressed(os.path.join(output_dir, f"dataset_test_{d}km.npz"), X=Xd, y=yd, strength=sd)
        np.save(os.path.join(output_dir, f"zeroday_subtypes_test_{d}km.npy"), zdd)

    with open(os.path.join(output_dir, "splits_seed.txt"), "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"distance_km={distance_km}\n")
        f.write(f"pulses_per_batch={pulses_per_batch}\n")
        f.write(f"n_batches_nominal={n_batches_nominal}\n")
        f.write(f"n_batches_attack={n_batches_attack}\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_val   shape: {X_val.shape}\n")
        f.write(f"X_test  shape: {X_test.shape}\n")
        f.write(f"Train label counts: {dict(zip(*np.unique(y_train, return_counts=True)))}\n")
        f.write(f"Val   label counts: {dict(zip(*np.unique(y_val,   return_counts=True)))}\n")
        f.write(f"Test  label counts: {dict(zip(*np.unique(y_test,  return_counts=True)))}\n")

    if verbose:
        print("\n[dataset] ── Saved files ──────────────────────────────────")
        print(f"  results/dataset_train.npz  {X_train.shape}")
        print(f"  results/dataset_val.npz    {X_val.shape}")
        print(f"  results/dataset_test.npz   {X_test.shape}")
        print(f"  results/splits_seed.txt")
        print(f"  results/zeroday_subtypes_val.npy")
        print(f"  results/zeroday_subtypes_test.npy")
        _print_label_balance("Train", y_train)
        _print_label_balance("Val",   y_val)
        _print_label_balance("Test",  y_test)

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        zd_subtypes_val=zd_subtypes_val,
        zd_subtypes_test=zd_subtypes_test,
    )


def _print_label_balance(split_name: str, y: np.ndarray):
    label_names = {0: "Nominal", 1: "FIM", 2: "TWIRL", 3: "Zero-Day"}
    vals, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n  {split_name} split ({total:,} pulses):")
    for v, c in zip(vals, counts):
        name = label_names.get(int(v), str(v))
        print(f"    label {v} ({name:8s}): {c:>8,}  ({100*c/total:5.1f}%)")
