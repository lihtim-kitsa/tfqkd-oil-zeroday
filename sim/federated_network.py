import torch
import copy
import numpy as np

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())
        
    def aggregate_weights(self, client_weights_list):
        """Perform Federated Averaging (FedAvg)."""
        num_clients = len(client_weights_list)
        if num_clients == 0:
            return
            
        # Initialize an empty dict for averaged weights
        avg_weights = copy.deepcopy(client_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
            
        # Sum weights
        for client_weights in client_weights_list:
            for key in avg_weights.keys():
                avg_weights[key] += client_weights[key]
                
        # Average
        for key in avg_weights.keys():
            avg_weights[key] = torch.div(avg_weights[key], num_clients)
            
        # Update global model
        self.global_model.load_state_dict(avg_weights)

class FederatedClient:
    def __init__(self, client_id, is_malicious=False):
        self.client_id = client_id
        self.is_malicious = is_malicious
        
    def train(self, model_dict, X_train, y_train, epochs=5, lr=0.01):
        """Train locally on client data and return updated weights."""
        import torch.nn as nn
        import torch.optim as optim
        from models.federated_mlp import FederatedMLP
        
        # Load local model with global weights
        model = FederatedMLP()
        model.load_state_dict(model_dict)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # We assume X_train and y_train are already torch tensors
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # If malicious, execute Byzantine data poisoning (scale weights or add noise)
        updated_weights = copy.deepcopy(model.state_dict())
        if self.is_malicious:
            for key in updated_weights.keys():
                # Evasion strategy: Add massive Gaussian noise
                noise = torch.randn_like(updated_weights[key]) * 2.0
                updated_weights[key] = -updated_weights[key] + noise
                
        return updated_weights
