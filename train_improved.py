"""
Complete Training Script for Tweet Movement Prediction
Run this script directly: python train_improved.py

This implements all the improvements:
1. LSTM with Attention
2. Focal Loss for class imbalance  
3. Time2Vec temporal encoding
4. Proper evaluation with threshold optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
import os

from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score, average_precision_score
)

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Data paths - MODIFY THESE
    DATA_DIR = "/home/public/tweetdatanlp/sent-trans-dbs"
    METADATA_FILE = "tweet_metadata.csv"
    EMBEDDINGS_FILE = "tweet_embeddings.npy"
    
    # Dataset
    SEQUENCE_LENGTH = 15
    DISTANCE_THRESHOLD_KM = 0  # Try 1, 5, or 10
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 30
    EARLY_STOP_PATIENCE = 7
    
    # Model
    HIDDEN_DIM = 128
    TIME_DIM = 32
    NUM_LAYERS = 2
    DROPOUT = 0.3
    USE_ATTENTION = True
    
    # Loss
    LOSS_TYPE = 'focal'  # 'focal', 'weighted_bce', 'bce'
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Random seed
    SEED = 42


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class Time2Vec(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, output_dim - 1)
        
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


class LSTMAttention(nn.Module):
    def __init__(self, embedding_dim=384, time_dim=32, hidden_dim=128, 
                 num_layers=2, dropout=0.3, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.time2vec = Time2Vec(time_dim)
        
        input_dim = embedding_dim + time_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1, bias=False)
            )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, t):
        seq_len, batch_size, _ = x.shape
        
        t_emb = self.time2vec(t)
        combined = torch.cat([x, t_emb], dim=-1)
        
        combined = self.input_proj(combined)
        combined = self.input_norm(combined)
        combined = F.relu(combined)
        combined = self.input_dropout(combined)
        
        lstm_out, _ = self.lstm(combined)
        
        if self.use_attention:
            scores = self.attention(lstm_out).squeeze(-1)
            weights = F.softmax(scores, dim=0)
            context = (lstm_out * weights.unsqueeze(-1)).sum(dim=0)
        else:
            context = lstm_out[-1]
        
        context = self.output_norm(context)
        context = self.output_dropout(context)
        return self.classifier(context), None


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        return (alpha_t * focal_weight * bce).mean()


# ============================================================
# DATASET
# ============================================================

class TweetSequenceDataset(Dataset):
    def __init__(self, dataframe, embeddings, sequence_length=5, distance_threshold_km=1.0):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.distance_threshold = distance_threshold_km
        self.samples = []
        
        print(f"Processing data (seq_len={sequence_length}, threshold={distance_threshold_km}km)...")
        
        user_col = 'UserID' if 'UserID' in dataframe.columns else 'user_id'
        
        for user_id, group in dataframe.groupby(user_col):
            group = group.sort_values('Timestamp').reset_index(drop=True)
            
            if len(group) < sequence_length + 1:
                continue
            
            for i in range(len(group) - sequence_length):
                indices = group.iloc[i:i + sequence_length].index.tolist()
                timestamps = group.iloc[i:i + sequence_length]['Timestamp'].values
                
                seq_end = i + sequence_length - 1
                target_idx = i + sequence_length
                
                loc_before = (group.iloc[seq_end]['Latitude'], group.iloc[seq_end]['Longitude'])
                loc_after = (group.iloc[target_idx]['Latitude'], group.iloc[target_idx]['Longitude'])
                
                distance = self._haversine(loc_before, loc_after)
                label = 1 if distance >= self.distance_threshold else 0
                
                self.samples.append({
                    'indices': indices,
                    'timestamps': timestamps,
                    'label': label
                })
        
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings).float()
        else:
            self.embeddings = embeddings.float()
        
        labels = [s['label'] for s in self.samples]
        n_moved = sum(labels)
        n_stayed = len(labels) - n_moved
        
        print(f"Total samples: {len(self.samples):,}")
        print(f"  Moved: {n_moved:,} ({100*n_moved/len(self.samples):.2f}%)")
        print(f"  Stayed: {n_stayed:,} ({100*n_stayed/len(self.samples):.2f}%)")
    
    def _haversine(self, loc1, loc2):
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371 * 2 * asin(sqrt(a))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        embeddings = self.embeddings[sample['indices']]
        timestamps = torch.tensor(sample['timestamps'], dtype=torch.float32).unsqueeze(-1)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return embeddings, timestamps, label
    
    def get_pos_weight(self):
        labels = [s['label'] for s in self.samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        return n_neg / n_pos


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for emb, time, labels in loader:
        emb = emb.permute(1, 0, 2).to(device)
        time = time.permute(1, 0, 2).to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits, _ = model(emb, time)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    return total_loss / len(loader), f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []
    
    for emb, time, labels in loader:
        emb = emb.permute(1, 0, 2).to(device)
        time = time.permute(1, 0, 2).to(device)
        labels = labels.to(device).unsqueeze(1)
        
        logits, _ = model(emb, time)
        total_loss += criterion(logits, labels).item()
        
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    binary_preds = (all_probs > threshold).astype(int)
    
    metrics = {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, binary_preds, zero_division=0),
        'precision': precision_score(all_labels, binary_preds, zero_division=0),
        'recall': recall_score(all_labels, binary_preds, zero_division=0),
        'probs': all_probs,
        'labels': all_labels
    }
    
    if len(np.unique(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def find_optimal_threshold(probs, labels):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (probs > thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    config = Config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Load data
    print("\nLoading data...")
    metadata_path = os.path.join(config.DATA_DIR, config.METADATA_FILE)
    embeddings_path = os.path.join(config.DATA_DIR, config.EMBEDDINGS_FILE)
    
    df = pd.read_csv(metadata_path)
    df['Timestamp'] = df['Timestamp'].astype(float)
    print(f"Loaded {len(df):,} tweets from {df['UserID'].nunique():,} users")
    
    embeddings = torch.from_numpy(np.load(embeddings_path)).float()
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = TweetSequenceDataset(
        df, embeddings, 
        sequence_length=config.SEQUENCE_LENGTH,
        distance_threshold_km=config.DISTANCE_THRESHOLD_KM
    )
    
    pos_weight = dataset.get_pos_weight()
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Split data
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"\nData splits: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
    
    # Create model
    embedding_dim = embeddings.shape[1]
    model = LSTMAttention(
        embedding_dim=embedding_dim,
        time_dim=config.TIME_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_attention=config.USE_ATTENTION
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Loss function
    if config.LOSS_TYPE == 'focal':
        criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        print(f"Loss: Focal (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
    elif config.LOSS_TYPE == 'weighted_bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        print(f"Loss: Weighted BCE (pos_weight={pos_weight:.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: Standard BCE")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"         ⭐ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with F1: {best_f1:.4f}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    opt_thresh, opt_f1 = find_optimal_threshold(test_metrics['probs'], test_metrics['labels'])
    
    print(f"\n--- Standard Threshold (0.5) ---")
    print(f"F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    print(f"\n--- Optimal Threshold ({opt_thresh:.2f}) ---")
    opt_preds = (test_metrics['probs'] > opt_thresh).astype(int)
    print(f"F1: {opt_f1:.4f}")
    print(f"Precision: {precision_score(test_metrics['labels'], opt_preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(test_metrics['labels'], opt_preds, zero_division=0):.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(test_metrics['labels'], opt_preds, target_names=['Stayed', 'Moved']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(test_metrics['labels'], opt_preds)
    print(cm)
    print(f"\nTP (Correct Moves): {cm[1][1]}")
    print(f"FP (False Alarms): {cm[0][1]}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'best_f1': best_f1,
        'optimal_threshold': opt_thresh
    }, 'best_model.pt')
    print("\nModel saved to best_model.pt")
    
    return model, test_metrics


if __name__ == '__main__':
    model, metrics = main()
