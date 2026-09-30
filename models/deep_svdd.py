import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import RobustScaler

class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[32, 16], rep_dim=8):
        super(DeepSVDDNet, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, bias=False))
            layers.append(nn.BatchNorm1d(h_dim, affine=False))
            layers.append(nn.LeakyReLU(0.1))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, rep_dim, bias=False))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepSVDD:
    def __init__(self, input_dim=10, hidden_dims=[32, 16], rep_dim=8, lr=1e-3, epochs=100, batch_size=128, margin=10.0):
        self.rep_dim = rep_dim
        self.net = DeepSVDDNet(input_dim=input_dim, hidden_dims=hidden_dims, rep_dim=rep_dim)
        self.c = None # Center of the hypersphere
        self.scaler = RobustScaler()
        self.margin = margin
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

    def init_center_c(self, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the nominal data."""
        n_samples = 0
        c = torch.zeros(self.rep_dim, device=self.device)
        self.net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                
                # Only use nominal data for initializing the center
                nom_inputs = inputs[labels == 0]
                if len(nom_inputs) > 0:
                    outputs = self.net(nom_inputs)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)
                    
        if n_samples > 0:
            c /= n_samples
        # If c is too close to zero, set to eps
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps
        self.c = c

    def fit(self, X_train, y_train=None):
        """Train using Semi-Supervised Deep SAD if y_train is provided, else Deep SVDD."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        tensor_X = torch.tensor(X_train_scaled, dtype=torch.float32)
        
        if y_train is None:
            tensor_y = torch.zeros(len(tensor_X), dtype=torch.float32)
        else:
            tensor_y = torch.tensor(y_train, dtype=torch.float32)
            
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.init_center_c(train_loader)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        eta = 1.0  # Hyperparameter controlling the inverse distance penalty margin
        
        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                
                # Nominal data loss: minimize distance to c
                loss_nom = torch.where(labels == 0, dist, torch.zeros_like(dist))
                
                # Anomaly data loss: margin-based contrastive loss
                loss_anom = torch.where(labels > 0, torch.relu(self.margin - dist), torch.zeros_like(dist))
                
                loss = torch.mean(loss_nom + loss_anom)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
                
            if (epoch + 1) % 10 == 0:
                print(f"[DeepSAD] Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.6f}")

    def score_samples(self, X):
        """Compute anomaly scores (distance to center c). Larger is more anomalous."""
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.net(tensor_X)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
        return dist.cpu().numpy()
