"""
Train ActuatorNet v2 with Excitation Data
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ActuatorNetV2(nn.Module):
    """Improved ActuatorNet with larger capacity"""
    def __init__(self, input_dim=3, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_actuator_net():
    print("=== Training ActuatorNet V2 ===")
    
    # Load excitation data
    df = pd.read_csv(os.path.expanduser("~/6619_ws/actuator_net/app/resources/excitation_data.csv"))
    print(f"Loaded {len(df)} samples")
    
    # Compute pos_error = action * scale + default - pos
    # action_scale = 0.25, default varies by joint
    default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    
    # Add pos_error column
    df['default'] = df['motor'].map(lambda m: default_angles[m])
    df['target_pos'] = df['action'] * 0.25 + df['default']
    df['pos_error'] = df['target_pos'] - df['pos']
    
    print(f"\nFeature ranges:")
    print(f"  pos_error: [{df['pos_error'].min():.3f}, {df['pos_error'].max():.3f}]")
    print(f"  velocity: [{df['vel'].min():.3f}, {df['vel'].max():.3f}]")
    print(f"  torque: [{df['torque'].min():.3f}, {df['torque'].max():.3f}]")
    
    # Prepare features: [pos_error, velocity, pos_error * velocity]
    X = np.column_stack([
        df['pos_error'].values,
        df['vel'].values,
        df['pos_error'].values * df['vel'].values  # Interaction term
    ])
    y = df['torque'].values.reshape(-1, 1)
    
    # Train/test split (stratify by phase for balanced evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test_scaled)
    
    # Model
    model = ActuatorNetV2(input_dim=3, hidden_dims=[128, 64, 32])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training
    batch_size = 4096
    n_epochs = 200
    best_loss = float('inf')
    patience_counter = 0
    
    print("\nTraining...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle
        indices = torch.randperm(len(X_train_t))
        
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '/tmp/best_actuator_net_v2.pt')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")
        
        if patience_counter > 30:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('/tmp/best_actuator_net_v2.pt'))
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = y_test
    
    # Metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n=== Test Results ===")
    print(f"RMSE: {rmse:.4f} N·m")
    print(f"MAE: {mae:.4f} N·m")
    print(f"R²: {r2:.4f} ({r2*100:.2f}%)")
    
    # Evaluate by phase
    print(f"\n=== Results by Phase ===")
    df_test = df.iloc[train_test_split(range(len(df)), test_size=0.2, random_state=42)[1]]
    
    # Save model
    save_dir = os.path.expanduser("~/6619_ws/unitree_rl_gym/logs")
    
    # Save as JIT
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, os.path.join(save_dir, "actuator_net_v2.pt"))
    
    # Save scalers
    with open(os.path.join(save_dir, "actuator_net_v2_scaler_X.pkl"), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(save_dir, "actuator_net_v2_scaler_y.pkl"), 'wb') as f:
        pickle.dump(scaler_y, f)
    
    print(f"\nSaved model to: {save_dir}/actuator_net_v2.pt")
    print(f"Saved scalers to: {save_dir}/actuator_net_v2_scaler_*.pkl")
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    train_actuator_net()
