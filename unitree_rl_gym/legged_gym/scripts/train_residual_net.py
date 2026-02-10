"""
Train residual network: Δτ = τ_isaac - τ_pd
Then in MuJoCo: τ = τ_pd + Δτ_learned
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class ResidualNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_residual_net(data_path, epochs=200, batch_size=512):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Input: pos_error, velocity
    X = df[['pos_error', 'velocity']].values
    y = df['tau_residual'].values.reshape(-1, 1)
    
    print(f"Dataset: {len(X)} samples")
    print(f"Residual stats: mean={y.mean():.3f}, std={y.std():.3f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_s)
    X_test_t = torch.FloatTensor(X_test_s)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)
    
    model = ResidualNet(input_dim=2, hidden_dims=[64, 32], output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_train_t))
        total_loss = 0
        n_batch = 0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = idx[i:i+batch_size]
            X_b = X_train_t[batch_idx]
            y_b = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batch += 1
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred_test = model(X_test_t)
                test_loss = criterion(pred_test, y_test_t).item()
                rmse = np.sqrt(test_loss)
            print(f"Epoch {epoch+1}: Train Loss={total_loss/n_batch:.4f}, Test RMSE={rmse:.4f} N·m")
            
            if test_loss < best_loss:
                best_loss = test_loss
    
    # Save
    save_dir = os.path.dirname(data_path)
    
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 2))
    torch.jit.save(traced, os.path.join(save_dir, "residual_net.pt"))
    
    with open(os.path.join(save_dir, "residual_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler_X, f)
    
    print(f"\nSaved model to: {save_dir}/residual_net.pt")
    print(f"Saved scaler to: {save_dir}/residual_scaler.pkl")
    
    # Final test
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).numpy()
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        print(f"\nFinal Test RMSE: {rmse:.4f} N·m")

if __name__ == "__main__":
    train_residual_net(
        os.path.expanduser("~/6619_ws/unitree_rl_gym/logs/residual_data.csv"),
        epochs=200
    )
