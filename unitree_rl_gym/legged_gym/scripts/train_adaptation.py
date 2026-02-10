import isaacgym
"""
Phase 2 - Step 2: Train Adaptation Module

Supervised learning: obs_history → ê ≈ e (encoder output)
Loss: MSE(adaptation_module(obs_history), encoder(env_params))

Usage:
  cd ~/6619_ws/unitree_rl_gym
  python legged_gym/scripts/train_adaptation.py --data_path=logs/go2_full_rma/Feb09_15-09-59_/adaptation_data/rma_adaptation_data.pt
"""



import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from rsl_rl.modules.actor_critic_rma import AdaptationModule


def train_adaptation(args):
    device = torch.device(args.device)
    
    # =========================================================================
    # Load dataset
    # =========================================================================
    print(f"Loading data from: {args.data_path}")
    data = torch.load(args.data_path, map_location='cpu')
    
    obs_histories = data['obs_histories']  # [N, history_len * obs_dim]
    env_encodings = data['env_encodings']  # [N, encoding_dim]
    
    obs_history_len = data['obs_history_len']
    base_obs_dim = data['base_obs_dim']
    env_encoding_dim = data['env_encoding_dim']
    
    print(f"Dataset: {obs_histories.shape[0]} samples")
    print(f"  obs_history: {obs_history_len} x {base_obs_dim} = {obs_histories.shape[1]}")
    print(f"  env_encoding: {env_encoding_dim}")
    
    # =========================================================================
    # Train/Val split
    # =========================================================================
    dataset = TensorDataset(obs_histories, env_encodings)
    
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    # =========================================================================
    # Create Adaptation Module
    # =========================================================================
    adaptation_module = AdaptationModule(
        obs_dim=base_obs_dim,
        history_len=obs_history_len,
        encoding_dim=env_encoding_dim,
        hidden_dims=args.hidden_dims,
    ).to(device)
    
    total_params = sum(p.numel() for p in adaptation_module.parameters())
    print(f"\nAdaptation Module: {total_params:,} parameters")
    print(f"  Input: {base_obs_dim * obs_history_len}")
    print(f"  Hidden: {args.hidden_dims}")
    print(f"  Output: {env_encoding_dim}")
    
    # =========================================================================
    # Training
    # =========================================================================
    optimizer = optim.Adam(adaptation_module.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # --- Train ---
        adaptation_module.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for obs_hist, enc_true in train_loader:
            obs_hist = obs_hist.to(device)
            enc_true = enc_true.to(device)
            
            enc_pred = adaptation_module(obs_hist)
            loss = criterion(enc_pred, enc_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # --- Validate ---
        adaptation_module.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for obs_hist, enc_true in val_loader:
                obs_hist = obs_hist.to(device)
                enc_true = enc_true.to(device)
                
                enc_pred = adaptation_module(obs_hist)
                loss = criterion(enc_pred, enc_true)
                
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / n_val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = adaptation_module.state_dict()
        
        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.1e}")
    
    print("-" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Load best weights
    adaptation_module.load_state_dict(best_state)
    
    # =========================================================================
    # Evaluate quality
    # =========================================================================
    adaptation_module.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for obs_hist, enc_true in val_loader:
            obs_hist = obs_hist.to(device)
            enc_pred = adaptation_module(obs_hist)
            all_preds.append(enc_pred.cpu())
            all_trues.append(enc_true)
    
    preds = torch.cat(all_preds)
    trues = torch.cat(all_trues)
    
    # Per-dimension analysis
    print(f"\n=== Per-dimension MSE ===")
    for i in range(env_encoding_dim):
        dim_mse = ((preds[:, i] - trues[:, i]) ** 2).mean().item()
        dim_corr = torch.corrcoef(torch.stack([preds[:, i], trues[:, i]]))[0, 1].item()
        print(f"  Dim {i}: MSE={dim_mse:.6f}, Corr={dim_corr:.4f}")
    
    # Overall R²
    ss_res = ((preds - trues) ** 2).sum().item()
    ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot
    print(f"\n  Overall R² = {r2:.4f}")
    
    # =========================================================================
    # Save
    # =========================================================================
    save_dir = os.path.dirname(args.data_path)
    
    # Save adaptation module
    save_path = os.path.join(save_dir, 'adaptation_module.pt')
    torch.save({
        'model_state_dict': adaptation_module.state_dict(),
        'obs_dim': base_obs_dim,
        'history_len': obs_history_len,
        'encoding_dim': env_encoding_dim,
        'hidden_dims': args.hidden_dims,
        'best_val_loss': best_val_loss,
        'r2_score': r2,
        'epochs': args.epochs,
        'timestamp': datetime.now().isoformat(),
    }, save_path)
    print(f"\nSaved adaptation module: {save_path}")
    
    # Save training plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(train_losses, label='Train', alpha=0.8)
    ax.plot(val_losses, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Adaptation Module Training (R²={r2:.4f})')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, 'adaptation_training.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    # Save pred vs true scatter
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < env_encoding_dim:
            ax.scatter(trues[:500, i], preds[:500, i], alpha=0.3, s=5)
            ax.plot([-3, 3], [-3, 3], 'r--', alpha=0.5)
            dim_corr = torch.corrcoef(torch.stack([preds[:, i], trues[:, i]]))[0, 1].item()
            ax.set_title(f'Dim {i} (r={dim_corr:.3f})')
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Adaptation Module: Predicted vs True Encoding')
    fig.tight_layout()
    scatter_path = os.path.join(save_dir, 'adaptation_scatter.png')
    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter: {scatter_path}")
    
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to rma_adaptation_data.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=50, help='LR decay every N epochs')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    train_adaptation(args)
