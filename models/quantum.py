import pennylane as qml
import numpy as np
import torch
from sklearn.svm import SVC

class VQC(torch.nn.Module):
    def __init__(self, num_qubits=5, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def _circuit(inputs, weights):
            # IQP feature map for data embedding
            qml.templates.IQPEmbedding(features=inputs, wires=range(self.num_qubits))
            # Strongly Entangling Layers for processing
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            # Expectation value of PauliZ on the first qubit for classification
            return qml.expval(qml.PauliZ(0))
            
        self.circuit = _circuit
        
        # Initialize weights
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        self.weights = torch.nn.Parameter(torch.rand(shape, requires_grad=True) * 2 * np.pi)

    def forward(self, x):
        # We process a batch by applying the circuit to each element
        out = torch.stack([self.circuit(xi, self.weights) for xi in x])
        return out.unsqueeze(1)

def get_qsvm_kernel():
    """Returns a precomputed fidelity kernel function for QSVM."""
    dev = qml.device("default.qubit", wires=5)
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.templates.IQPEmbedding(features=x1, wires=range(5))
        qml.adjoint(qml.templates.IQPEmbedding)(features=x2, wires=range(5))
        return qml.probs(wires=range(5))
        
    def q_kernel(A, B):
        # Computes fidelity |<x1|x2>|^2
        kernel_matrix = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                probs = kernel_circuit(A[i], B[j])
                kernel_matrix[i, j] = probs[0] # Probability of measuring |00000>
        return kernel_matrix

    return q_kernel
