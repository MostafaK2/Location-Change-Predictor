# Standard library imports
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from haversine import haversine

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




## +++++++++++++++++++++++++++++++++++++++++++++++++++ Preprocessing Code +++++++++++++++++++++++++++++++++++ ##
# Returns a Dataframe
def perform_preprocessing(dataset_path = "/home/public/tweetdatanlp/GeoText.2010-10-12/full_text.txt"):
    print("Performing Preprocessing to return final dataframe ...")

    import pandas as pd
    from TweetNormalizer import normalizeTweet

    col_names = ["UserID", "Timestamp", "Place", "Latitude", "Longitude", "TweetText"]
    # load the dataset
    df = pd.read_csv(
        dataset_path,
        sep="\t",          
        encoding="latin1",
        dtype={"TweetText": str}, 
        names=col_names,

    )
    # -------------------- Preprocessing --------------------------#
    # convert to timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['UnixTimestamp'] = df['Timestamp'].astype('int64') // 10**9

    # drop nulls
    df_cleaned = df.dropna()

    # Normalize the tweets (takes 40 seconds)
    df_cleaned['NormalizedText'] = df_cleaned['TweetText'].apply(normalizeTweet)

    # Dropping outlier
    df_cleaned['Tokens'] = df_cleaned['NormalizedText'].apply(lambda x: x.split())
    df_cleaned = df_cleaned[df_cleaned['Tokens'].apply(len) <= 64].reset_index(drop=True)

    # Normalizing timestamps to [0,1]
    MIN_TIME = df_cleaned['UnixTimestamp'].min()
    MAX_TIME = df_cleaned['UnixTimestamp'].max()
    df_cleaned["Timestamp"] = (df_cleaned['UnixTimestamp'] - MIN_TIME) / (MAX_TIME - MIN_TIME)

    # Cols to keep
    col_to_keep= ["UserID", "Timestamp", "NormalizedText", "Latitude", "Longitude"]
    final_df  = df_cleaned[col_to_keep]
    print("Finshed Preprocessing Step \n")
    return final_df

class Time2Vec(nn.Module):
    def __init__(self, vector_size):
        super(Time2Vec, self).__init__()
        self.vector_size = vector_size

        # Linear[0] Periodic[1, .... n]
        self.l1 = nn.Linear(1, 1)
        self.periodic_term = nn.Linear(1, vector_size - 1)
        
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x0 = self.l1(t)
        x1n = torch.sin(self.periodic_term(t))
        return torch.cat([x0, x1n], dim=-1)

## =================================================== Heiriacal LSTM Model Class ======================================================== ##
class LSTM_HE(nn.Module):
    def __init__(self, nwords, device, word_embed_size=128, time_vec_size=32, sen_embedding_size=384, hidden_size=60, batch_first=True, dropout=0.3, num_lstm_layers=2):
        super(LSTM_HE, self).__init__()

        self.device = device
        self.sen_embedding_size = sen_embedding_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.time_vec_size = time_vec_size

        # Embeddings for the sentence words 
        self.embedding = nn.Embedding(nwords, word_embed_size)
        self.embedding_dropout = nn.Dropout(p=dropout)

        # Word-level LSTM with configurable number of layers
        self.sen_embedding = nn.LSTM(
            input_size=word_embed_size, 
            hidden_size=sen_embedding_size,
            num_layers=num_lstm_layers,
            bidirectional=False,
            batch_first=batch_first,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Projection layer for residual connection at word-level
        self.word_residual_proj = nn.Linear(word_embed_size, sen_embedding_size)
        self.ln1 = nn.LayerNorm(sen_embedding_size)
        self.dropout1 = nn.Dropout(p=dropout)

        self.time2vec = Time2Vec(time_vec_size)
        
        # Sentence-level LSTM with configurable number of layers
        self.lstm2 = nn.LSTM(
            input_size=sen_embedding_size + time_vec_size, 
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_lstm_layers,
            bidirectional=False,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Projection layer for residual connection at sentence-level
        self.sen_residual_proj = nn.Linear(sen_embedding_size + time_vec_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)

        # Binary classification with hidden layer for more capacity
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better gradient flow"""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier initialization
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1.0 for EACH LAYER
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'residual_proj' in name and 'weight' in name:
                # Normal Xavier initialization WITHOUT scaling down
                nn.init.xavier_uniform_(param)
            elif 'time2vec' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
                
    def forward(self, x_padded, y_padded, t_padded, seq_length, token_length):

        batch_size, max_seq, max_token = x_padded.shape
        
        # Step 1: Flatten to process all tweets together
        x_flat = x_padded.view(batch_size * max_seq, max_token).to(self.device)

        # Step 2: Get embeddings for all tokens
        embedded = self.embedding(x_flat)
        embedded = self.embedding_dropout(embedded)

        # Step 3: Flatten token_length 
        token_lengths_flat = []
        for user_lengths in token_length:
            token_lengths_flat.extend(user_lengths)
        token_lengths_flat = torch.tensor(token_lengths_flat, dtype=torch.long)

        # Step 4: Filter out padded tweets (length == 0)
        mask = token_lengths_flat > 0
        embedded_real = embedded[mask]
        token_lengths_real = token_lengths_flat[mask]

        # Step 5: Pack sequences
        packed_embedded = pack_padded_sequence(
            embedded_real,
            token_lengths_real.cpu(),
            batch_first=True, 
            enforce_sorted=False
        )

        # Step 6: Pass through word-level LSTM
        packed_output, (hidden, cell) = self.sen_embedding(packed_embedded)
        
        # Take the LAST layer output
        tweet_embeddings_real = hidden[-1, :, :]
        
        assert tweet_embeddings_real.dim() == 2, f"Expected 2D tensor, got {tweet_embeddings_real.shape}"

        # Step 7: Create full tensor with padding
        sen_embedding_size = tweet_embeddings_real.shape[-1]
        num_real_tweets = tweet_embeddings_real.shape[0]
        
        assert num_real_tweets == mask.sum().item(), f"Mismatch: {num_real_tweets} real tweets but {mask.sum().item()} mask entries"
        
        tweet_embeddings_full = torch.zeros(batch_size * max_seq, sen_embedding_size, 
                                            device=tweet_embeddings_real.device)
        tweet_embeddings_full[mask] = tweet_embeddings_real
        
        # Step 8: Reshape back to (batch_size, max_seq, sen_embedding_size)
        tweet_embeddings = tweet_embeddings_full.view(batch_size, max_seq, -1)

        # === RESIDUAL CONNECTION 1: Word-level ===
        embedded_full = torch.zeros(batch_size * max_seq, max_token, self.embedding.embedding_dim,
                                    device=embedded.device)
        embedded_full[mask] = embedded_real
        embedded_reshaped = embedded_full.view(batch_size, max_seq, max_token, -1)
        
        # Reshape token_lengths for proper broadcasting
        token_lengths_reshaped = token_lengths_flat.view(batch_size, max_seq).to(embedded.device)
        
        # Mean pool over tokens (ignoring padding)
        embedded_mean = embedded_reshaped.sum(dim=2) / token_lengths_reshaped.unsqueeze(-1).clamp(min=1).float()
        embedded_projected = self.word_residual_proj(embedded_mean)
        
        # Simple residual connection (no scaling)
        tweet_embeddings = tweet_embeddings + embedded_projected

        # Apply LayerNorm and dropout
        tweet_embeddings = self.ln1(tweet_embeddings)
        tweet_embeddings = self.dropout1(tweet_embeddings)

        time_embeddings = self.time2vec(t_padded)
        tweet_time_concat = torch.cat([tweet_embeddings, time_embeddings], dim=-1)

        # ========== Sentence-level LSTM ==========
        seq_length_tensor = torch.tensor(seq_length, dtype=torch.long)
        packed_tweets = pack_padded_sequence(
            tweet_time_concat,
            seq_length_tensor.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass through sentence-level LSTM
        packed_output, (hidden, cell) = self.lstm2(packed_tweets)

        # Unpack to get outputs for ALL timesteps
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # === RESIDUAL CONNECTION 2: Sentence-level ===
        tweet_embeddings_projected = self.sen_residual_proj(tweet_time_concat)
        
        # Simple residual connection (no scaling)
        lstm_output = lstm_output + tweet_embeddings_projected

        # Apply LayerNorm and dropout
        lstm_output = self.ln2(lstm_output)
        lstm_output = self.dropout2(lstm_output)

        # Classification with hidden layer
        hidden_out = self.fc1(lstm_output)
        hidden_out = self.relu(hidden_out)
        logits = self.fc2(hidden_out)

        return logits

## =================================================== Dataset Class For Heirical LSTM =================================================== ##
class HELocationChangeDataset(Dataset):
    def __init__(self, df, user_col='UserID', tweet_col='NormalizedText', 
                 lat_col='Latitude', lon_col='Longitude', time_col='Timestamp',
                 vocab=None, window_size=15, stride=5,
                 distance_threshold=0.01, min_freq=5):
        super().__init__()
        
        self.user_col = user_col
        self.tweet_col = tweet_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_col = time_col
        self.window_size = window_size
        self.stride = stride
        self.distance_threshold = distance_threshold

        # Filter users by total tweet count
        self.user_groups = [(user_id, group) for user_id, group in df.groupby(user_col) 
                            if 2 < len(group) <= 100]  # Keep users with reasonable tweet counts
        
        # Build vocabulary
        self.vocab = self._build_vocab(df, tweet_col, min_freq)
        
        # Create sliding windows from all users
        self.windows = self._create_sliding_windows()
        
        print(f"[✔️] Created {len(self.windows)} windows from {len(self.user_groups)} users")
    
    def _build_vocab(self, df, text_column, min_freq):
        """Build vocabulary from all tweets"""
        word_counts = Counter()
        
        # Count all words from sequences
        for normalized_sentence in df[text_column].tolist():
            words = str(normalized_sentence).lower().split()
            word_counts.update(words)
        
        # Create vocabulary with special tokens
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
                
        print(f"[✔️] Vocabulary size: {len(vocab)}")
        return vocab
    
    def _create_sliding_windows(self):
        """Create sliding windows from all user sequences"""
        windows = []
        
        for user_id, group in self.user_groups:
            # Sort by timestamp
            group = group.sort_values(self.time_col).reset_index(drop=True)
            
            total_tweets = len(group)
            
            # If sequence is shorter than window, use the whole sequence
            if total_tweets <= self.window_size:
                windows.append((user_id, group))
            else:
                # Create sliding windows
                start_idx = 0
                while start_idx + self.window_size <= total_tweets:
                    window_group = group.iloc[start_idx:start_idx + self.window_size]
                    windows.append((user_id, window_group))
                    start_idx += self.stride
                
                # # Optionally add the last window if there's remaining data
                # if start_idx < total_tweets:
                #     window_group = group.iloc[-self.window_size:]
                #     windows.append((user_id, window_group))
        
        return windows
    
    def _encode_tweet(self, text):
        """
        Encode a single tweet text into a list of token IDs using the given vocab.
        Unknown words → <UNK>
        """
        tokens = str(text).lower().split()
        encoded = []

        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                encoded.append(self.vocab["<UNK>"])
        return encoded
    
    def _get_vocab(self):
        return self.vocab
    
    def _encode_single_sequences(self, sequence):
        """Encode all tweets in a sequence"""
        return [self._encode_tweet(tweet) for tweet in sequence]

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get a single window"""
        user_id, group = self.windows[idx]

        tweets = group[self.tweet_col].tolist()
        timestamps = group[self.time_col].values
        locs = list(zip(group[self.lat_col], group[self.lon_col]))

        # Encode tweets (all except last for input)
        x = self._encode_single_sequences(tweets[:-1])
        
        # Calculate location changes
        y = []
        for i in range(1, len(locs)):
            dist = haversine(locs[i-1], locs[i])
            y.append(1 if dist > self.distance_threshold else 0)
        y = torch.tensor(y, dtype=torch.long)
        
        # Timestamps (all except last)
        t = torch.tensor(timestamps[:-1], dtype=torch.float).unsqueeze(-1)
        
        return x, y, t


# ==================================================== TRAINING FUNCTIONS ======================================================== ##

def train_epoch(device, epoch, model, optimizer, criterion, train_loader, diagnostic=True):
    
    model.train()
    print(f"\n Starting Epoch [{epoch+1}] ...")

    train_loss = 0
    
    for i, (batch_tweets, batch_y, batch_t, seq_lengths, token_lengths) in enumerate(tqdm(train_loader, desc="Training")):
        batch_tweets = batch_tweets.to(device)
        batch_y = batch_y.unsqueeze(-1).to(device)
        batch_t = batch_t.float().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model.forward(batch_tweets, batch_y, batch_t, seq_lengths, token_lengths)
        
        loss = criterion(logits, batch_y.float())
        
        # Backward pass
        loss.backward()
        
        # CHECK GRADIENTS - first batch and every 50 batches
        if diagnostic and i == 0 and epoch == 0:
            print(f"\nINITIAL GRADIENT CHECK (Epoch {epoch+1}, Batch {i})")
            grad_stats = check_gradients(model, threshold=1e-7)
            monitor_gradient_flow(model)
        elif diagnostic and i == 0:
            print(f"\nGRADIENT CHECK (Epoch {epoch+1}, Batch {i})")
            monitor_gradient_flow(model)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    train_loss = train_loss / len(train_loader)
    return train_loss

def valid_epoch(device, model, criterion, val_loader):
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_tweets, batch_y, batch_t, seq_lengths, token_lengths in tqdm(val_loader, desc="Validation"):
            batch_tweets = batch_tweets.to(device)
            batch_y = batch_y.unsqueeze(-1).to(device)
            batch_t = batch_t.float().to(device)
            
            logits = model.forward(batch_tweets, batch_y, batch_t, seq_lengths, token_lengths)
            loss = criterion(logits, batch_y.float())
            
            valid_loss += loss.item()
        
        valid_loss = valid_loss / len(val_loader)
        return valid_loss

# ==================================================== DIAGNOSTIC FUNCTIONS ==================== #

def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL PARAMETER COUNT")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)
    
    # Breakdown by layer
    print("\nPARAMETER BREAKDOWN BY LAYER:")
    print("-" * 60)
    for name, param in model.named_parameters():
        print(f"{name:50s} | {param.numel():>10,} | {str(param.shape):>20}")
    print("=" * 60)
    
    return total_params, trainable_params


def check_gradients(model, threshold=1e-7):
    """
    Check for vanishing/exploding gradients after backward pass.
    Call this AFTER loss.backward() but BEFORE optimizer.step()
    """
    grad_stats = {
        'vanishing': [],
        'exploding': [],
        'healthy': [],
        'none': []
    }
    
    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            
            # Categorize gradient health
            if grad_norm < threshold:
                grad_stats['vanishing'].append(name)
                status = "⚠️  VANISHING"
            elif grad_norm > 100:
                grad_stats['exploding'].append(name)
                status = "🔥 EXPLODING"
            else:
                grad_stats['healthy'].append(name)
                status = "✅ HEALTHY"
            
            print(f"{name:50s} | {status:15s} | "
                  f"Norm: {grad_norm:>10.6f} | "
                  f"Mean: {grad_mean:>10.6f} | "
                  f"Std: {grad_std:>10.6f} | "
                  f"Max: {grad_max:>10.6f}")
        else:
            grad_stats['none'].append(name)
    
    print("=" * 80)
    print("\nGRADIENT SUMMARY:")
    print("-" * 80)
    print(f"✅ Healthy gradients:   {len(grad_stats['healthy']):3d} layers")
    print(f"⚠️  Vanishing gradients: {len(grad_stats['vanishing']):3d} layers")
    print(f"🔥 Exploding gradients: {len(grad_stats['exploding']):3d} layers")
    print(f"❌ No gradients:        {len(grad_stats['none']):3d} layers")
    print("=" * 80)
    
    if grad_stats['vanishing']:
        print("\n⚠️  VANISHING GRADIENT LAYERS:")
        for layer in grad_stats['vanishing']:
            print(f"  - {layer}")
    
    if grad_stats['exploding']:
        print("\n🔥 EXPLODING GRADIENT LAYERS:")
        for layer in grad_stats['exploding']:
            print(f"  - {layer}")
    
    return grad_stats


def monitor_gradient_flow(model):
    """Monitor average gradient flow through each layer type"""
    layers = {
        'embedding': [],
        'lstm_ih': [],
        'lstm_hh': [],
        'layernorm': [],
        'projection': [],
        'fc': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if 'embedding' in name:
                layers['embedding'].append(grad_norm)
            elif 'weight_ih' in name:
                layers['lstm_ih'].append(grad_norm)
            elif 'weight_hh' in name:
                layers['lstm_hh'].append(grad_norm)
            elif 'ln' in name or 'layernorm' in name.lower():
                layers['layernorm'].append(grad_norm)
            elif 'proj' in name:
                layers['projection'].append(grad_norm)
            elif 'fc' in name:
                layers['fc'].append(grad_norm)
    
    print("\n" + "=" * 60)
    print("GRADIENT FLOW BY LAYER TYPE")
    print("=" * 60)
    
    for layer_type, norms in layers.items():
        if norms:
            avg_norm = np.mean(norms)
            min_norm = np.min(norms)
            max_norm = np.max(norms)
            print(f"{layer_type:20s} | "
                  f"Avg: {avg_norm:>10.6f} | "
                  f"Min: {min_norm:>10.6f} | "
                  f"Max: {max_norm:>10.6f}")
    print("=" * 60)
    
    return layers

# ============================= Custom Collete for variable length batching ============================= #
def collate_fn(batch):
    """
    batch = [(x_i, y_i, t_i)]
    x_i = list of tweets, each tweet = list of token IDs
    y_i = tensor of labels
    t_i = tensor
    """
    batch_tweets = []
    batch_y = []
    batch_t = []
    seq_lengths = []
    token_lengths = []

    max_token = 0
    max_seq = 0

    # First pass: get lengths
    for x, _, _ in batch:
        seq_lengths.append(len(x))
        max_seq = max(max_seq, len(x))
        token_len_user = [len(tweet) for tweet in x]
        max_token = max(max_token, max(token_len_user))

    # Second pass: pad
    for i, (x, y, t) in enumerate(batch):
        padded_tweets = []
        user_token_lengths = [len(tweet) for tweet in x]
        
        for tweet in x:
            pad_len = max_token - len(tweet)
            padded_tweets.append(tweet + [0] * pad_len)
        
        padded_y = y.tolist() if y.dim() > 0 else [y.item()] * len(x)
        
        if t.dim() > 0:
            padded_t = t.squeeze().tolist() if t.shape[0] > 1 else [t.item()] * len(x)
        else:
            padded_t = [t.item()] * len(x)
        
        while len(padded_tweets) < max_seq:
            padded_tweets.append([0] * max_token)
            padded_y.append(-100)
            padded_t.append(0)
            user_token_lengths.append(0)
        
        batch_tweets.append(torch.tensor(padded_tweets, dtype=torch.long))
        batch_y.append(torch.tensor(padded_y, dtype=torch.long))
        batch_t.append(torch.tensor(padded_t, dtype=torch.float))
        token_lengths.append(user_token_lengths)

    batch_tweets = torch.stack(batch_tweets, dim=0)
    batch_y = torch.stack(batch_y, dim=0)
    batch_t = torch.stack(batch_t, dim=0)
    batch_t = batch_t.unsqueeze(-1)
            
    return batch_tweets, batch_y, batch_t, seq_lengths, token_lengths


# ============================== Evaluate function =========================== #

def evaluate_model(model, best_model_state, val_loader, device, criterion):
    """
    Comprehensive evaluation of model on validation set
    Returns metrics and predictions
    """
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    print("\n" + "="*80)
    print("🔍 EVALUATING MODEL ON VALIDATION SET")
    print("="*80)
    
    with torch.no_grad():
        for batch_tweets, batch_y, batch_t, seq_lengths, token_lengths in val_loader:
            batch_tweets = batch_tweets.to(device)
            batch_y_input = batch_y.unsqueeze(-1).to(device)
            batch_t = batch_t.float().to(device)

            # Forward pass
            logits = model(batch_tweets, batch_y_input, batch_t, seq_lengths, token_lengths)
            
            # Calculate loss
            loss = criterion(logits, batch_y_input.float())
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Filter out padded positions
            batch_y_flat = batch_y.to(device)
            mask_flat = (batch_y_flat != -100)
            
            all_preds.extend(preds.squeeze()[mask_flat].cpu().numpy())
            all_labels.extend(batch_y_flat[mask_flat].cpu().numpy())
            all_probs.extend(probs.squeeze()[mask_flat].cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Average loss
    avg_loss = total_loss / len(val_loader)
    
    # Print results
    print("\nVALIDATION METRICS:")
    print("-"*80)
    print(f"Loss:           {avg_loss:.4f}")
    print(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    
    print("\nCLASS DISTRIBUTION:")
    print("-"*80)
    num_class_0 = (all_labels == 0).sum()
    num_class_1 = (all_labels == 1).sum()
    print(f"True Labels  - Class 0 (No Change): {num_class_0:5d} ({num_class_0/len(all_labels)*100:.1f}%)")
    print(f"True Labels  - Class 1 (Change):    {num_class_1:5d} ({num_class_1/len(all_labels)*100:.1f}%)")
    
    num_pred_0 = (all_preds == 0).sum()
    num_pred_1 = (all_preds == 1).sum()
    print(f"\nPredictions  - Class 0 (No Change): {num_pred_0:5d} ({num_pred_0/len(all_preds)*100:.1f}%)")
    print(f"Predictions  - Class 1 (Change):    {num_pred_1:5d} ({num_pred_1/len(all_preds)*100:.1f}%)")
    
    print("\nCONFUSION MATRIX:")
    print("-"*80)
    print(f"                    Predicted")
    print(f"                No Change  |  Change")
    print(f"Actual No Change    {tn:5d}   |  {fp:5d}")
    print(f"Actual Change       {fn:5d}   |  {tp:5d}")
    
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-"*80)
    print(classification_report(all_labels, all_preds, 
                                target_names=['No Change (0)', 'Change (1)'],
                                digits=4))
    
    # Check if model is just predicting one class
    print("\nMODEL BEHAVIOR ANALYSIS:")
    print("-"*80)
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print(f"🚨 WARNING: Model is predicting ONLY class {int(unique_preds[0])}!")
        print(f"   This indicates the model is NOT learning properly.")
    else:
        print(f"✅ Model predicts both classes")
        
    # Check prediction confidence
    avg_prob = all_probs.mean()
    print(f"\nAverage prediction probability: {avg_prob:.4f}")
    if 0.45 < avg_prob < 0.55:
        print(f"⚠️  WARNING: Predictions are close to 0.5 (uncertain)")
    
    print("="*80)
    
    # Return metrics dictionary
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


def plot_losses(train_loss, val_loss, filename="HE_Deep_LSTM_train_val_curve.png"):
    """
    Plots training and validation loss curves.

    Parameters:
        train_loss (list or array): Training loss values per epoch.
        val_loss (list or array): Validation loss values per epoch.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


class Config:
    # Data paths - MODIFY THESE
    DATA_DIR = "/home/public/tweetdatanlp/GeoText.2010-10-12/full_text.txt"

    # DATASET
    SEQUENCE_LENGTH = 11
    DISTANCE_THRESHOLD_KM = 1  # Try 1, 5, or 10
    WINDOW_SIZE=9
    STRIDE=3
    MIN_FREQ=9          # Try between 7-15 or higher 
        

    # Training
    TRAINING_SPLIT_RATIO = 0.8  # Other percent used for validation
    BATCH_SIZE = 64
    EPOCHS = 50
    EARLY_STOP_PATIENCE = 7

    ## MODEL PARAMS
    WORDEMBSIZE = 32   
    SENEMBSIZE = 64    
    HIDDENSIZE = 64     
    TIMEEMBSIZE = 32     # Not added yet
    NUM_LSTM_LAYERS = 2  # Number of LSTM layers (Dont do: 1 (LSTM breaks))
    DROPOUT = 0.5

    ## Optimizer Params
    LEARNING_RATE = 1.75e-5  # From Learning Rate Range Test
    WEIGHT_DECAY = 1e-5

    # Random seed
    SEED = 42

    DIAGNOSTIC  = True   # set to true for diagnostic

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def main():
    config = Config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*100}")
    print(f"Device: {device}")
    print(f"{'='*100}\n\n")

    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load data
    print("\n" + "="*100)
    data_frame = perform_preprocessing(config.DATA_DIR)

    print("Setting up dataset...")
    
    dataset = HELocationChangeDataset(
        data_frame,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        min_freq=config.MIN_FREQ,
        distance_threshold=config.DISTANCE_THRESHOLD_KM,
    )
    print("Finshed Setting up Dataset class\n")
    

    # TRAINING SPLIT RATIOS
    total_len = len(dataset)
    train_pct = config.TRAINING_SPLIT_RATIO
    train_len = int(train_pct * total_len)
    val_len = total_len - train_len


    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Train/Validation Split ready: {len(train_set)} train, {len(val_set)} val batches\n")
    print("="*100 + "\n\n")

    # ============================= MODEL INITIALIZATION ============================= #
    print("="*100)
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTM_HE(
        nwords=len(dataset._get_vocab()), 
        device=device, 
        word_embed_size=config.WORDEMBSIZE,  
        sen_embedding_size=config.SENEMBSIZE, 
        hidden_size=config.HIDDENSIZE,
        dropout=config.DROPOUT,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    ).to(device)

    print("\nAnalyzing model architecture...")
    count_parameters(model)

    # ============================= LOSS, OPTIMIZER, SCHEDULER ============================= #
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,         # Restart every 10 epochs
        T_mult=2,       # Double the restart interval after each restart
        eta_min=1e-7    # Minimum LR
    )

    print(f"\nOptimizer: Adam with lr={config.LEARNING_RATE}, weight decay={config.WEIGHT_DECAY}")
    print(f"Loss: BCEWithLogitsLoss")

    print("\n\n\n\n" + "="*100)
    print("STARTING TRAINING")
    print("="*100)
    
    t_curve = []
    v_curve = []
    best_model_state = None
    patience_counter = 0
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(device, epoch, model,optimizer, criterion, train_loader, diagnostic=config.DIAGNOSTIC)
        valid_loss = valid_epoch(device, model, criterion, val_loader)

        t_curve.append(train_loss)
        v_curve.append(valid_loss)

        print(f"Training Loss: {train_loss}  ||   Validation Loss: {valid_loss}")
        if valid_loss < train_loss:
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        scheduler.step()
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE!")
    print("="*100)

    evaluate_model(model, best_model_state, val_loader, device, criterion)

    plot_losses(t_curve,  v_curve)

    # Save model
    torch.save(model.state_dict(), 'best_LSTM_HE_model.pt')



if __name__ == "__main__":
    main()









