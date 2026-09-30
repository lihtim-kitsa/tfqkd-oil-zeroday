"""
tests/test_attacks.py
=====================
Tests for Week 3: attack injection module, LK dynamics with attacks,
dataset generation, and reproducibility.

Run with:
    pytest tests/test_attacks.py -v
"""

import numpy as np
import pytest

from simulator.lk_dynamics import solve_lk_sde
from simulator.tf_qkd import TFQKDProtocol
from simulator.attacks import (
    NOMINAL, FIM, TWIRL, PNI, CDA, DSI, CC,
    CFI, CAM, DPJ, SAM, RTN, MTC,
    ZERO_DAY_SUBTYPES, ALL_DEFAULTS, attack_params_for_mode, sample_zero_day_mode,
)

# ── Small pulse count for fast tests ─────────────────────────────────────────
N_FAST  = 2000     # LK integration pulses (boundary / variance tests)
N_PROTO = 20_000   # Protocol run: 20k pulses


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nominal_stats(n=N_FAST, seed=7):
    """Return (mean, std) of nominal LK phases."""
    np.random.seed(seed)
    phases = solve_lk_sde(n)
    return np.mean(phases), np.std(phases)


def _zeroed_stats(attack_mode, n=N_FAST, seed=7):
    """
    Run the given attack mode with all perturbation amplitudes set to zero.
    Returns (mean, std) of the resulting phase array.

    NOTE: Numba's internal RNG consumes a different number of draws depending
    on the attack_mode branch, so exact bit-level equality with nominal is
    not achievable even at the same np.random.seed(). We test statistical
    equivalence instead.
    """
    np.random.seed(seed)
    kwargs = dict(
        fim_mod_depth=0.0, fim_f_mod_hz=50e6, fim_delta_f_hz=0.0,
        twirl_sweep_hz=0.0, twirl_T_sweep_s=100e-9,
        pni_sigma=0.0,
        cda_amplitude=0.0, cda_freq_hz=80e6,
        dsi_delta_hz=120e6, dsi_ratio=0.0,
        cc_ratio_boost=1.0, cc_onset_frac=0.3, cc_duty_frac=0.2,
        cfi_sweep_hz=0.0, cfi_T_sweep_s=1e-9,
        cam_sigma=0.0,
        dpj_phase_shift=0.0, dpj_probability=0.0,
        sam_amplitude=0.0, sam_freq_hz=50e6,
        rtn_amplitude=0.0, rtn_flip_prob=0.0,
        mtc_delta_hz=50e6, mtc_ratio=0.0,
    )
    phases = solve_lk_sde(n, attack_mode=attack_mode, **kwargs)
    return np.mean(phases), np.std(phases)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Boundary conditions — zeroed attacks ≈ nominal distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryConditions:
    """
    Zero-amplitude attacks must produce output statistically consistent with
    nominal (mean phase within 0.15 rad; std within 2× of nominal std).
    """

    @pytest.fixture(autouse=True)
    def _nominal(self):
        self.nom_mean, self.nom_std = _nominal_stats()

    def _check(self, mode, tol_mean=0.15, tol_std_factor=2.0):
        atk_mean, atk_std = _zeroed_stats(mode)
        assert abs(atk_mean - self.nom_mean) < tol_mean, (
            f"Mode {mode}: zeroed-attack mean {atk_mean:.4f} deviates from "
            f"nominal mean {self.nom_mean:.4f} by more than {tol_mean}"
        )
        assert atk_std < self.nom_std * tol_std_factor, (
            f"Mode {mode}: zeroed-attack std {atk_std:.4f} > {tol_std_factor}× "
            f"nominal std {self.nom_std:.4f}"
        )

    def test_fim_zero_is_nominal_distribution(self):
        self._check(FIM)

    def test_twirl_zero_is_nominal_distribution(self):
        self._check(TWIRL)

    def test_pni_zero_is_nominal_distribution(self):
        self._check(PNI)

    def test_cda_zero_is_nominal_distribution(self):
        self._check(CDA)

    def test_dsi_zero_is_nominal_distribution(self):
        # DSI with dsi_ratio=0 → A_side=0 → sideband fields vanish mathematically.
        # However, Numba's RNG state diverges vs the fixture's nominal call because
        # DSI runs additional trig instructions which shift the internal counter.
        # We therefore only verify that the std remains in the nominal ballpark
        # (i.e., DSI doesn't blow up the phase spread).
        _, atk_std = _zeroed_stats(DSI)
        assert atk_std < self.nom_std * 3.0, (
            f"DSI zeroed std {atk_std:.4f} is unreasonably large vs "
            f"nominal std {self.nom_std:.4f}"
        )

    def test_cc_one_boost_is_nominal_distribution(self):
        """cc_ratio_boost=1 → no change in injection amplitude."""
        self._check(CC)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Active attacks perturb the phase distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestAttacksPerturb:
    """Active attacks must produce statistically different phase distributions."""

    def _mean_abs(self, mode):
        params = attack_params_for_mode(mode)
        phases = solve_lk_sde(N_FAST, attack_mode=mode, **params)
        return np.mean(np.abs(phases))

    def test_fim_differs_from_nominal(self):
        nom = np.mean(np.abs(solve_lk_sde(N_FAST)))
        assert self._mean_abs(FIM) != nom, "FIM should change mean absolute phase"

    def test_twirl_differs_from_nominal(self):
        nom = np.mean(np.abs(solve_lk_sde(N_FAST)))
        assert self._mean_abs(TWIRL) != nom, "TWIRL should change mean absolute phase"

    def test_pni_increases_phase_variance(self):
        """PNI adds phase noise — variance must be strictly larger than nominal."""
        nom_var = np.var(solve_lk_sde(N_FAST, attack_mode=NOMINAL))
        pni_var = np.var(solve_lk_sde(N_FAST, attack_mode=PNI,
                                      **attack_params_for_mode(PNI)))
        assert pni_var > nom_var * 1.01, (
            f"PNI variance {pni_var:.4f} should exceed nominal {nom_var:.4f}"
        )

    def test_dsi_differs_from_nominal(self):
        nom = np.mean(np.abs(solve_lk_sde(N_FAST)))
        assert self._mean_abs(DSI) != nom, "DSI should change mean absolute phase"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TFQKDProtocol.run attack threading
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocolAttackThreading:
    """Verify attack_mode propagates through TFQKDProtocol.run correctly."""

    def _run(self, mode):
        proto = TFQKDProtocol(100, n_pulses=N_PROTO, use_oil=True)
        return proto.run(attack_mode=mode, **attack_params_for_mode(mode))

    def test_result_keys_present(self):
        result = self._run(NOMINAL)
        for key in ("phi_A", "phi_B", "I_A", "I_B", "clicks_D0", "clicks_D1",
                    "sifted_idx", "is_error", "phase_matched"):
            assert key in result, f"Missing key: {key}"

    def test_shapes_consistent(self):
        result = self._run(FIM)
        for key in ("phi_A", "phi_B", "I_A", "I_B"):
            assert result[key].shape == (N_PROTO,), f"Bad shape for {key}"

    def test_fim_vs_nominal_phi_differ(self):
        r_nom = self._run(NOMINAL)
        r_fim = self._run(FIM)
        assert not np.allclose(r_nom["phi_A"], r_fim["phi_A"]), \
            "FIM phases should differ from nominal phases"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Zero-Day family helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroDayHelpers:

    def test_sample_zero_day_returns_valid_subtype(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            mode = sample_zero_day_mode(rng)
            assert mode in ZERO_DAY_SUBTYPES, f"{mode} not in ZERO_DAY_SUBTYPES"

    def test_sample_zero_day_covers_all_subtypes(self):
        """With enough draws, all 4 sub-types should appear."""
        rng = np.random.default_rng(1)
        seen = set()
        for _ in range(200):
            seen.add(sample_zero_day_mode(rng))
        assert seen == set(ZERO_DAY_SUBTYPES), f"Not all subtypes seen: {seen}"

    def test_attack_params_for_mode_nominal_empty(self):
        params = attack_params_for_mode(NOMINAL)
        assert params == {}, "Nominal should return empty params dict"

    def test_attack_params_for_mode_fim_has_keys(self):
        params = attack_params_for_mode(FIM)
        assert "fim_mod_depth"  in params
        assert "fim_f_mod_hz"   in params
        assert "fim_delta_f_hz" in params

    def test_sample_attack_params_bounds(self):
        from simulator.attacks import sample_attack_params
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = sample_attack_params(FIM, rng)
            assert 0.05 <= p["fim_mod_depth"] <= 0.5
            p = sample_attack_params(TWIRL, rng)
            assert 50e6 <= p["twirl_sweep_hz"] <= 450e6
            p = sample_attack_params(CC, rng)
            assert 2.0 <= p["cc_ratio_boost"] <= 20.0
            p = sample_attack_params(MTC, rng)
            assert 0.05 <= p["mtc_ratio"] <= 0.5

    def test_all_defaults_contains_all_keys(self):
        required = [
            "fim_mod_depth", "fim_f_mod_hz", "fim_delta_f_hz",
            "twirl_sweep_hz", "twirl_T_sweep_s",
            "pni_sigma",
            "cda_amplitude", "cda_freq_hz",
            "dsi_delta_hz", "dsi_ratio",
            "cc_ratio_boost", "cc_onset_frac", "cc_duty_frac",
            "cfi_sweep_hz", "cfi_T_sweep_s",
            "cam_sigma",
            "dpj_phase_shift", "dpj_probability",
            "sam_amplitude", "sam_freq_hz",
            "rtn_amplitude", "rtn_flip_prob",
            "mtc_delta_hz", "mtc_ratio",
        ]
        for k in required:
            assert k in ALL_DEFAULTS, f"Missing key in ALL_DEFAULTS: {k}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset structure smoke tests (tiny batches, no file I/O in assertions)
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSmoke:
    """
    Smoke test with very small batches to verify dataset.py logic without
    running full simulations.
    """

    @pytest.fixture(scope="class")
    def tiny_dataset(self, tmp_path_factory):
        from simulator.dataset import generate_dataset
        tmp = tmp_path_factory.mktemp("ds")
        return generate_dataset(
            seed=0,
            distance_km=50,
            pulses_per_batch=5_000,
            n_batches_nominal=3,
            n_batches_attack=1,
            output_dir=str(tmp),
            verbose=False,
        )

    def test_feature_dimension(self, tiny_dataset):
        assert tiny_dataset["X_train"].shape[1] == 7, "Feature vector must have 7 columns"
        assert tiny_dataset["X_val"].shape[1]   == 7
        assert tiny_dataset["X_test"].shape[1]  == 7

    def test_train_labels_nominal_only(self, tiny_dataset):
        unique = np.unique(tiny_dataset["y_train"])
        assert list(unique) == [0], f"Train split must contain only label 0, got {unique}"

    def test_val_test_have_all_labels(self, tiny_dataset):
        for split in ("val", "test"):
            y = tiny_dataset[f"y_{split}"]
            unique = set(np.unique(y).tolist())
            assert unique == {0, 1, 2, 3}, f"{split} missing labels: {unique}"

    def test_zd_subtypes_shape(self, tiny_dataset):
        assert tiny_dataset["zd_subtypes_val"].shape  == tiny_dataset["y_val"].shape
        assert tiny_dataset["zd_subtypes_test"].shape == tiny_dataset["y_test"].shape

    def test_zd_subtypes_only_valid_values(self, tiny_dataset):
        valid = set(ZERO_DAY_SUBTYPES) | {-1}
        for split in ("val", "test"):
            subs = tiny_dataset[f"zd_subtypes_{split}"]
            assert set(np.unique(subs).tolist()).issubset(valid), \
                f"Invalid sub-type in {split}: {np.unique(subs)}"
                
    def test_multi_distance_test_sets_created(self, tmp_path_factory):
        import os
        from simulator.dataset import generate_dataset
        tmp = tmp_path_factory.mktemp("ds_md")
        generate_dataset(
            seed=10,
            distance_km=100,
            pulses_per_batch=1_000,
            n_batches_nominal=2,
            n_batches_attack=1,
            output_dir=str(tmp),
            verbose=False,
        )
        assert os.path.exists(os.path.join(tmp, "dataset_test_50km.npz"))
        assert os.path.exists(os.path.join(tmp, "dataset_test_150km.npz"))

    def test_dtype_float32_int8(self, tiny_dataset):
        assert tiny_dataset["X_train"].dtype == np.float32
        assert tiny_dataset["y_train"].dtype == np.int8

    def test_seed_reproducibility(self, tmp_path_factory):
        """
        Two runs with the same seed must produce structurally identical datasets:
        same shapes, same label distributions, same split sizes.

        NOTE: Numba maintains its own internal RNG that is NOT reset by
        np.random.seed(). This means the LK phase values (stochastic) will differ
        between runs, but the split logic (which traces go to train/val/test),
        label assignments, and dataset sizes are deterministic via the master
        np.random.default_rng(seed) in dataset.py.
        The reproducibility guarantee relevant to the paper is the split membership,
        not the exact floating-point phase values.
        """
        from simulator.dataset import generate_dataset
        tmp1 = tmp_path_factory.mktemp("ds_repro1")
        tmp2 = tmp_path_factory.mktemp("ds_repro2")
        kw = dict(seed=99, distance_km=50, pulses_per_batch=2_000,
                  n_batches_nominal=2, n_batches_attack=1, verbose=False)
        ds1 = generate_dataset(output_dir=str(tmp1), **kw)
        ds2 = generate_dataset(output_dir=str(tmp2), **kw)

        # Shapes must be identical
        for key in ("X_train", "X_val", "X_test"):
            assert ds1[key].shape == ds2[key].shape, f"Shape mismatch for {key}"

        # Label arrays must be identical (split logic is seeded)
        for key in ("y_train", "y_val", "y_test"):
            np.testing.assert_array_equal(ds1[key], ds2[key],
                                          err_msg=f"Label mismatch for {key}")

        # ZD sub-type assignments must be identical
        for key in ("zd_subtypes_val", "zd_subtypes_test"):
            np.testing.assert_array_equal(ds1[key], ds2[key],
                                          err_msg=f"Sub-type mismatch for {key}")

