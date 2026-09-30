"""Gradient-based FIM evasion search against Deep SAD and XGBoost.

The Lang-Kobayashi simulator is not differentiable, so this script fits
parameter-to-detector-score neural surrogates on randomized simulator samples,
optimizes attack parameters with Adam, then validates candidates in simulation.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
import yaml

from models.deep_svdd import DeepSVDD
from models.features import extract_features
from simulator.attacks import FIM
from simulator.tf_qkd import TFQKDProtocol, calculate_key_rate, estimate_decoy_parameters


N_SURROGATE = 48
N_RESTARTS = 8
OPT_STEPS = 300
PULSES = 100_000
LOW = np.array([0.05, np.log(10e6), 5e6], dtype=np.float64)
HIGH = np.array([0.5, np.log(150e6), 100e6], dtype=np.float64)
PARAM_NAMES = ["fim_mod_depth", "log_fim_f_mod_hz", "fim_delta_f_hz"]


class ScoreSurrogate(torch.nn.Module):
    def __init__(self, width=64, outputs=2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, outputs),
        )

    def forward(self, x):
        return self.network(x)


def params_from_unit(unit):
    physical = LOW + unit * (HIGH - LOW)
    return {
        "fim_mod_depth": float(physical[0]),
        "fim_f_mod_hz": float(np.exp(physical[1])),
        "fim_delta_f_hz": float(physical[2]),
    }


def load_deep_score_model():
    with open("hyperparams.yaml", "r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)
    model = DeepSVDD(**hyperparams["deep_svdd"])
    model.net.load_state_dict(torch.load(
        "models/saved/deep_svdd_seed1.pth",
        map_location=model.device,
        weights_only=True,
    ))
    with open("models/saved/deep_svdd_scaler_seed1.pkl", "rb") as f:
        model.scaler = pickle.load(f)

    train_archive = np.load("results/dataset_train.npz")
    X_train = extract_features(train_archive["X"], 100_000, 1_000)
    X_scaled = model.scaler.transform(X_train)
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor, torch.zeros(len(tensor))),
        batch_size=hyperparams["deep_svdd"]["batch_size"], shuffle=False,
    )
    model.init_center_c(loader)
    model.net.eval()
    with torch.no_grad():
        z = model.net(tensor.to(model.device)).cpu().numpy()
    inverse_covariance = np.linalg.inv(
        np.cov(z, rowvar=False) + np.eye(model.rep_dim) * 1e-4
    )
    return model, model.c.detach(), inverse_covariance


def score_candidates(unit_points, deep_model, center, inverse_covariance, xgb):
    feature_rows = []
    key_rates = []
    for i, unit in enumerate(unit_points):
        params = params_from_unit(unit)
        print(
            f"Simulating design point {i+1}/{len(unit_points)}: "
            f"depth={params['fim_mod_depth']:.3f}, "
            f"fmod={params['fim_f_mod_hz']/1e6:.2f} MHz, "
            f"offset={params['fim_delta_f_hz']/1e6:.2f} MHz",
            flush=True,
        )
        np.random.seed(71000 + i)
        protocol_result = TFQKDProtocol(100, n_pulses=PULSES, use_oil=True).run(
            attack_mode=FIM, **params
        )
        raw = np.stack([
            protocol_result["phi_A"], protocol_result["phi_B"],
            protocol_result["I_A"], protocol_result["I_B"],
            (protocol_result["phi_A"] - protocol_result["phi_B"]) % (2 * np.pi),
            protocol_result["clicks_D0"], protocol_result["clicks_D1"],
        ], axis=1).astype(np.float32)
        features = extract_features(raw, PULSES, PULSES)
        feature_rows.append(features[0])
        y1, e1, qmm, emm = estimate_decoy_parameters(protocol_result)
        key_rates.append(calculate_key_rate(y1, e1, qmm, emm))

    features = np.asarray(feature_rows)
    scaled = deep_model.scaler.transform(features)
    tensor = torch.tensor(scaled, dtype=torch.float32).to(deep_model.device)
    with torch.no_grad():
        z = deep_model.net(tensor)
        delta = z - center.to(deep_model.device)
        deep_scores = torch.einsum(
            "ni,ij,nj->n", delta,
            torch.tensor(inverse_covariance, dtype=delta.dtype, device=delta.device),
            delta,
        ).cpu().numpy()
    xgb_scores = xgb.predict_proba(features)[:, 1]
    return features, deep_scores, xgb_scores, np.asarray(key_rates)


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep, center, inverse_covariance = load_deep_score_model()
    with open("models/saved/xgboost_seed1.pkl", "rb") as f:
        xgb = pickle.load(f)

    unit = qmc.LatinHypercube(d=3, seed=425).random(N_SURROGATE)
    _, deep_scores, xgb_scores, key_rates = score_candidates(
        unit, deep, center, inverse_covariance, xgb
    )

    targets = np.column_stack([np.log1p(np.maximum(deep_scores, 0)), xgb_scores])
    target_scaler = StandardScaler().fit(targets)
    X_tensor = torch.tensor(unit, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(
        target_scaler.transform(targets), dtype=torch.float32, device=device
    )
    surrogate = ScoreSurrogate().to(device)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(1500):
        optimizer.zero_grad()
        loss = torch.mean((surrogate(X_tensor) - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
    surrogate.eval()

    candidates = []
    for detector_idx, detector in enumerate(("deep_svdd", "xgboost")):
        for restart in range(N_RESTARTS):
            point = torch.tensor(
                unit[(restart * 5 + detector_idx) % len(unit)].copy(),
                dtype=torch.float32, device=device, requires_grad=True,
            )
            optimizer = torch.optim.Adam([point], lr=0.025)
            for _ in range(OPT_STEPS):
                optimizer.zero_grad()
                prediction = surrogate(point.clamp(0, 1))
                objective = prediction[detector_idx]
                objective.backward()
                optimizer.step()
                with torch.no_grad():
                    point.clamp_(0, 1)
            candidates.append((detector, point.detach().cpu().numpy()))

    unique_candidates = []
    seen = set()
    for detector, point in candidates:
        key = tuple(np.round(point, 4))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((detector, point))
    candidate_unit = np.array([point for _, point in unique_candidates])
    _, candidate_deep, candidate_xgb, candidate_key_rates = score_candidates(
        candidate_unit, deep, center, inverse_covariance, xgb
    )

    rows = []
    for i, (target, point) in enumerate(unique_candidates):
        p = params_from_unit(point)
        rows.append({
            "optimized_against": target,
            **p,
            "deep_svdd_score": float(candidate_deep[i]),
            "xgboost_attack_probability": float(candidate_xgb[i]),
            "simulated_key_rate": float(candidate_key_rates[i]),
            "surrogate_validation": "real simulator rerun",
        })
    validation = pd.DataFrame(rows)
    validation.to_csv("results/adaptive_surrogate_candidates.csv", index=False)

    physical_parameters = [params_from_unit(point) for point in unit]
    pd.DataFrame({
        "fim_mod_depth": [p["fim_mod_depth"] for p in physical_parameters],
        "fim_f_mod_hz": [p["fim_f_mod_hz"] for p in physical_parameters],
        "fim_delta_f_hz": [p["fim_delta_f_hz"] for p in physical_parameters],
        "deep_svdd_score": deep_scores,
        "xgboost_attack_probability": xgb_scores,
        "simulated_key_rate": key_rates,
    }).to_csv("results/adaptive_surrogate_design.csv", index=False)

    with open("results/adaptive_surrogate.md", "w", encoding="utf-8") as f:
        f.write("# Gradient-Based Adaptive Attacker (Surrogate Search)\n\n")
        f.write(
            "The non-differentiable Lang-Kobayashi simulator was sampled at 48 "
            "Latin-hypercube FIM parameter settings. A small neural score surrogate "
            "was optimized with Adam to lower either Deep SVDD distance or the "
            "XGBoost attack probability, then each distinct candidate was rerun in "
            "the simulator. This is an exploratory adaptive attack: surrogate "
            "fit/generalization uncertainty is not yet a formal confidence bound, "
            "and the simulated key-rate estimator is asymptotic.\n\n"
        )
        f.write("## Surrogate-design samples\n\n")
        f.write(pd.DataFrame({
            "parameter": PARAM_NAMES,
            "lower_bound": LOW,
            "upper_bound": HIGH,
        }).to_markdown(index=False))
        f.write("\n\n## Simulator-validated candidate points\n\n")
        f.write(validation.to_markdown(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(deep_scores, xgb_scores, c=unit[:, 0], cmap="viridis", label="Design")
    axes[0].scatter(candidate_deep, candidate_xgb, marker="x", s=70, c="red", label="Optimized")
    axes[0].set_xlabel("Deep SVDD anomaly score")
    axes[0].set_ylabel("XGBoost attack probability")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].scatter(unit[:, 0], deep_scores, label="Deep SVDD design")
    axes[1].scatter(candidate_unit[:, 0], candidate_deep, marker="x", c="red", label="Optimized")
    axes[1].set_xlabel("FIM modulation depth")
    axes[1].set_ylabel("Deep SVDD anomaly score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/adaptive_surrogate.png", dpi=180)
    plt.close(fig)
    print("Saved surrogate attacker design, validated candidates, report, and figure.", flush=True)


if __name__ == "__main__":
    main()
