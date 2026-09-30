import optuna
import numpy as np
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.oil_rate_equations import OILSimulator
from sim.attack_injectors import FIMInjector, TWIRLInjector
from data.feature_extraction import extract_features

# Suppress optuna logging to keep console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

class EvasionSimulator:
    def __init__(self, model_path='checkpoints/xgboost.pkl'):
        """
        Loads the target surrogate model for evasion.
        """
        try:
            self.model = joblib.load(model_path)
            # Check if model has predict_proba
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model must have predict_proba for continuous optimization.")
        except Exception as e:
            print(f"Failed to load surrogate model: {e}")
            self.model = None

    def run_simulation(self, attack_class, **kwargs):
        """Run a single simulation with given attack parameters and return extracted features."""
        simulator = OILSimulator()
        t_span = (0, 1e-6)
        t_eval = np.linspace(0, 1e-6, 1000)
        y0 = [1.0, 0.0, 1e8]
        base_E_ref = 1.0
        base_phi_ref = 0.0
        def current_func(t): return 2.0e17
        
        if attack_class == 'fim':
            injector = FIMInjector(m=kwargs['m'], f_fim=kwargs['f_fim'])
            def E_ref_func(t): return injector.inject_E_ref(t, base_E_ref)
            def phi_ref_func(t): return base_phi_ref
        elif attack_class == 'twirl':
            injector = TWIRLInjector(delta_lambda=kwargs['delta_lambda'])
            def E_ref_func(t): return base_E_ref
            def phi_ref_func(t): return injector.inject_phi_ref(t, base_phi_ref)
            
        sol = simulator.simulate(t_span, y0, E_ref_func, phi_ref_func, current_func, t_eval)
        features = extract_features(sol.t, sol.y[0], sol.y[1], sol.y[2])
        return features

    def objective_fim(self, trial):
        # Constrained evasion search space for FIM
        m = trial.suggest_float('m', 0.05, 0.40)
        f_fim = trial.suggest_float('f_fim', 1e8, 1e10, log=True)
        
        # Must maintain minimum attack efficacy (e.g. m > 0.1 for useful photon inflation)
        # We can penalize m < 0.1 by adding a heavy cost
        
        feats = self.run_simulation('fim', m=m, f_fim=f_fim)
        
        # Evaluate against surrogate model
        probs = self.model.predict_proba(feats)
        if probs.shape[1] > 2:
            p_attack = 1 - probs[:, 0]
        else:
            p_attack = probs[:, 1]
            
        # Minimize max probability of attack across the windows
        avg_p_attack = np.mean(p_attack)
        
        # Penalize low m (weak attack)
        penalty = max(0, 0.15 - m) * 10
        return avg_p_attack + penalty

    def objective_twirl(self, trial):
        # Constrained evasion search space for TWIRL
        delta_lambda = trial.suggest_float('delta_lambda', 0.1, 5.0)
        
        feats = self.run_simulation('twirl', delta_lambda=delta_lambda)
        
        probs = self.model.predict_proba(feats)
        if probs.shape[1] > 2:
            p_attack = 1 - probs[:, 0]
        else:
            p_attack = probs[:, 1]
            
        avg_p_attack = np.mean(p_attack)
        return avg_p_attack

    def find_evasion_parameters(self, attack_class='fim', n_trials=50):
        if self.model is None:
            return None
            
        study = optuna.create_study(direction='minimize')
        if attack_class == 'fim':
            study.optimize(self.objective_fim, n_trials=n_trials)
        elif attack_class == 'twirl':
            study.optimize(self.objective_twirl, n_trials=n_trials)
            
        print(f"\n--- Best Evasion Parameters for {attack_class.upper()} ---")
        print(f"Minimum P_attack achieved: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        return study.best_params

if __name__ == "__main__":
    print("Initializing Evasion Simulator...")
    evader = EvasionSimulator()
    print("Running Bayesian Optimization for FIM evasion...")
    best_fim = evader.find_evasion_parameters('fim', n_trials=20)
    
    print("Running Bayesian Optimization for TWIRL evasion...")
    best_twirl = evader.find_evasion_parameters('twirl', n_trials=20)
    
    print("\nEvasion simulation complete. Best parameters can be used for Adversarial Training.")
