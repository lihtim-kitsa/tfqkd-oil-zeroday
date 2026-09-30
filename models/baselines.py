import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class StatisticalThreshold(BaseEstimator, ClassifierMixin):
    """
    Simple threshold detector that flags anomalies if features
    deviate significantly from the nominal distribution.
    """
    def __init__(self, threshold_std=3.0):
        self.threshold_std = threshold_std
        self.scaler = StandardScaler()
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        # Fit only on nominal data if y is provided and 0 is nominal
        if y is not None:
            X_nom = X[y == 0]
        else:
            X_nom = X
        self.scaler.fit(X_nom)
        self.mean = np.zeros(X.shape[1])
        self.std = np.ones(X.shape[1])
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        # Compute max deviation across features
        max_dev = np.max(np.abs(X_scaled), axis=1)
        # Convert to a pseudo-probability
        prob_anomaly = 1.0 - np.exp(-max_dev / self.threshold_std)
        probs = np.vstack([1 - prob_anomaly, prob_anomaly]).T
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

class QSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=5, C=1.0):
        self.n_qubits = n_qubits
        self.C = C
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.svm = SVC(kernel=self.kernel_matrix, probability=True, C=self.C)
        self.scaler = StandardScaler()
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            qml.IQPEmbedding(x1, wires=range(self.n_qubits))
            qml.adjoint(qml.IQPEmbedding)(x2, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        
        self.kernel_circuit = kernel_circuit

    def kernel_matrix(self, X1, X2):
        N1, N2 = len(X1), len(X2)
        matrix = np.zeros((N1, N2))
        for i in range(N1):
            for j in range(N2):
                # Fidelity is the probability of measuring all zeros
                matrix[i, j] = self.kernel_circuit(X1[i], X2[j])[0]
        return matrix

    def fit(self, X, y):
        # Subsample to avoid prohibitively long kernel matrix computation
        if len(X) > 200:
            idx = np.random.choice(len(X), 200, replace=False)
            X, y = X[idx], y[idx]
            
        X_scaled = self.scaler.fit_transform(X)
        self.X_fit_ = X_scaled
        self.svm.fit(X_scaled, (y > 0).astype(int))
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

class VQC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=5, n_layers=4, epochs=20, lr=0.1, batch_size=32):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def qnode(inputs, weights):
            qml.IQPEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))
        
        self.qnode = qnode
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        from pennylane import numpy as pnp
        self.weights = pnp.random.normal(0, np.pi, weight_shape, requires_grad=True)
        
    def fit(self, X, y):
        # Convert y to { -1, 1 } for PauliZ expectation
        y_binary = (y > 0).astype(int)
        y_scaled = np.where(y_binary == 1, 1, -1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        opt = qml.AdamOptimizer(stepsize=self.lr)
        
        from pennylane import numpy as pnp
        def cost(weights, X_batch, y_batch):
            preds = pnp.stack([self.qnode(x, weights) for x in X_batch])
            return pnp.mean((preds - y_batch) ** 2)

        for epoch in range(self.epochs):
            indices = np.arange(len(X_scaled))
            np.random.shuffle(indices)
            for i in range(0, len(X_scaled), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_scaled[batch_idx]
                y_batch = y_scaled[batch_idx]
                self.weights, _ = opt.step_and_cost(lambda w: cost(w, X_batch, y_batch), self.weights)
            
            if (epoch + 1) % 5 == 0:
                print(f"[VQC] Epoch {epoch+1}/{self.epochs}")
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        preds = np.array([self.qnode(x, self.weights) for x in X_scaled])
        # Convert expectation [-1, 1] to probability [0, 1]
        probs = (preds + 1) / 2
        return np.vstack([1 - probs, probs]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
