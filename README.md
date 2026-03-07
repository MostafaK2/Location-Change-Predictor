# Tweet Movement Prediction Model

A PyTorch implementation for predicting user movement based on tweet sequences using LSTM with attention, temporal encoding, and focal loss for handling class imbalance.

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn
```

### Running the Model

```bash
python train_improved.py
```

The script will automatically:
1. Load tweet metadata and embeddings
2. Create sequences of tweets for each user
3. Train the model with early stopping
4. Evaluate on test set with optimal threshold
5. Save the trained model to `best_model.pt`

## Configuration

Edit the `Config` class in `train_improved.py` to customize:

### Data Paths
```python
DATA_DIR = "/home/public/tweetdatanlp/sent-trans-dbs"
METADATA_FILE = "tweet_metadata.csv"          # CSV with columns: UserID, Timestamp, Latitude, Longitude
EMBEDDINGS_FILE = "tweet_embeddings.npy"     # NumPy array of shape (N_tweets, 384)
```

### Dataset Parameters
```python
SEQUENCE_LENGTH = 15                           # Number of tweets in each sequence
DISTANCE_THRESHOLD_KM = 0                      # Min distance (km) to classify as "moved"
                                               # Try: 1, 5, 10 for different sensitivity
```

### Training Parameters
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
EARLY_STOP_PATIENCE = 7                        # Stop if no improvement for 7 epochs
```

### Model Architecture
```python
HIDDEN_DIM = 128                               # LSTM hidden size
TIME_DIM = 32                                  # Temporal embedding dimension
NUM_LAYERS = 2                                 # LSTM layers
DROPOUT = 0.3                                  # Dropout rate
USE_ATTENTION = True                           # Use attention mechanism
```

### Loss Function
```python
LOSS_TYPE = 'focal'                           # Options: 'focal', 'weighted_bce', 'bce'
FOCAL_ALPHA = 0.25                            # Focal loss alpha (balance factor)
FOCAL_GAMMA = 2.0                             # Focal loss gamma (focusing parameter)
```

## Model Architecture

### Input
- **Tweet Embeddings**: Sequence of tweet embeddings (typically 384-dim from sentence transformers)
- **Timestamps**: Unix timestamps for each tweet in the sequence

### Processing Pipeline

```
Tweet Embeddings (seq_len, batch, 384)
        ↓
Time2Vec Encoding (seq_len, batch, time_dim)
        ↓
Concatenate → Project to hidden_dim
        ↓
LayerNorm + ReLU + Dropout
        ↓
LSTM (num_layers, hidden_dim)
        ↓
Attention Mechanism (learns which tweets matter)
        ↓
Output Normalization + Dropout
        ↓
Classifier (hidden_dim → 1)
        ↓
Sigmoid → Probability of movement
```

### Key Components

**Time2Vec**: Encodes temporal information as:
```
t_emb = [Linear(t), sin(Periodic(t))]
```
Captures both linear time trends and periodic patterns.

**Attention Layer**: For each position in the sequence:
```
score = tanh(Linear(lstm_output))
weight = softmax(score)
context = sum(lstm_output * weight)
```

**Focal Loss**: Addresses class imbalance by focusing on hard examples:
```
FL = -α(1-p_t)^γ * BCE
```
Where `p_t` is the model's confidence on the correct class.

## Data Format

### Metadata CSV (`tweet_metadata.csv`)
Required columns:
- `UserID`: User identifier
- `Timestamp`: Unix timestamp (float)
- `Latitude`: Tweet location latitude
- `Longitude`: Tweet location longitude

Example:
```
UserID,Timestamp,Latitude,Longitude
user_123,1609459200.0,40.7128,-74.0060
user_123,1609545600.0,40.7130,-74.0065
user_456,1609632000.0,34.0522,-118.2437
```

### Embeddings Array (`tweet_embeddings.npy`)
NumPy array of shape `(N_tweets, embedding_dim)`. Typically 384 dimensions from sentence transformers like `all-MiniLM-L6-v2` or similar models.

## Output


### Saved Model (`best_model.pt`)
```python
# Load trained model
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
config_dict = checkpoint['config']
optimal_threshold = checkpoint['optimal_threshold']  # 0.35 in example above
best_f1 = checkpoint['best_f1']
```

## Advanced Usage

### Using a Pretrained Model

```python
import torch

# Load checkpoint
checkpoint = torch.load('best_model.pt')
config_dict = checkpoint['config']

# Recreate model
model = LSTMAttention(
    embedding_dim=384,
    time_dim=config_dict['TIME_DIM'],
    hidden_dim=config_dict['HIDDEN_DIM'],
    num_layers=config_dict['NUM_LAYERS'],
    dropout=config_dict['DROPOUT'],
    use_attention=config_dict['USE_ATTENTION']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
optimal_threshold = checkpoint['optimal_threshold']

# Inference
model.eval()
with torch.no_grad():
    logits, _ = model(embeddings, timestamps)
    probs = torch.sigmoid(logits)
    predictions = (probs > optimal_threshold).int()
```

### Adjusting Loss Functions

To experiment with different loss functions:

```python
config.LOSS_TYPE = 'weighted_bce'  # Class-weighted BCE
# or
config.LOSS_TYPE = 'bce'           # Standard BCE (not recommended for imbalanced data)
```

## License

FREE TO USE

## References

- Vaswani et al. (2017): Attention Is All You Need
- Kazemi et al. (2019): Time2Vec: Learning a Vector Representation of Time
- Lin et al. (2017): Focal Loss for Dense Object Detection
- Hochreiter & Schmidhuber (1997): Long Short-Term Memory

## Questions & Support

For issues, questions, or suggestions, please open an issue on the repository.
