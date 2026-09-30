import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.federated_mlp import FederatedMLP
from sim.federated_network import FederatedServer, FederatedClient

def evaluate_federated_model(model_state, test_data, test_labels):
    model = FederatedMLP(input_dim=5, output_dim=3)
    model.load_state_dict(model_state)
    model.eval()

    inputs = torch.tensor(test_data, dtype=torch.float32)
    labels = torch.tensor(test_labels, dtype=torch.long)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    preds_np = preds.numpy()
    labels_np = labels.numpy()
    
    acc = accuracy_score(labels_np, preds_np)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    rec = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

    print(f"Federated Model Evaluation -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 (Macro): {f1:.4f}")
    return acc, prec, rec, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of federated clients')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='Fraction of malicious clients')
    parser.add_argument('--rounds', type=int, default=10, help='Federated communication rounds')
    args = parser.parse_args()

    print(f"--- TF-QKD Federated Learning Simulation ---")
    print(f"Nodes: {args.num_nodes} | Poison Ratio: {args.poison_ratio} | Rounds: {args.rounds}")

    # 1. Load Data
    print("Loading physical dataset...")
    df = pd.read_parquet('data/dataset_v1.parquet')
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Global train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # 2. Partition Data among Clients
    num_samples = len(X_train_full) // args.num_nodes
    clients = []
    
    num_malicious = int(args.num_nodes * args.poison_ratio)
    malicious_ids = []
    if num_malicious > 0:
        malicious_ids = np.random.choice(range(args.num_nodes), num_malicious, replace=False)

    client_datasets = []
    for i in range(args.num_nodes):
        start = i * num_samples
        end = (i + 1) * num_samples
        
        X_client = torch.FloatTensor(X_train_full[start:end])
        y_client = torch.LongTensor(y_train_full[start:end])
        client_datasets.append((X_client, y_client))
        
        is_malicious = (i in malicious_ids)
        clients.append(FederatedClient(client_id=i, is_malicious=is_malicious))
        
    print(f"Initialized {args.num_nodes} clients ({num_malicious} malicious).")

    # 3. Initialize Global Model & Server
    global_model = FederatedMLP()
    server = FederatedServer(global_model)

    # 4. Federated Training Loop
    for rnd in range(1, args.rounds + 1):
        print(f"\n--- Round {rnd}/{args.rounds} ---")
        global_weights = server.get_global_weights()
        
        client_weight_updates = []
        for i, client in enumerate(clients):
            X_client, y_client = client_datasets[i]
            # Train locally for 3 epochs per round
            weights = client.train(global_weights, X_client, y_client, epochs=3)
            client_weight_updates.append(weights)
            
        # Aggregate on server
        server.aggregate_weights(client_weight_updates)
        
        # Evaluate global model
        acc, p, r, f1 = evaluate_federated_model(server.global_model.state_dict(), X_test, y_test)

    # Save final model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(server.global_model.state_dict(), 'checkpoints/federated_mlp.pth')
    print("\nFederated training complete. Global model saved.")

if __name__ == '__main__':
    main()
