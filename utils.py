# Returns a Dataframe
def perform_preprocessing(dataset_path = "/home/public/tweetdatanlp/GeoText.2010-10-12/full_text.txt"):
    print("🔧 Performing Preprocessing to return final dataframe ...")

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
    print(f"The minimum unix time: {MIN_TIME} and Max unix time: {MAX_TIME}")


    # Remove Outliers
    df_cleaned = plot_clean_scatter(df_cleaned)

    # Cols to keep
    col_to_keep= ["UserID", "Timestamp", "NormalizedText", "Latitude", "Longitude"]
    final_df  = df_cleaned[col_to_keep]

    
    print(final_df.head())
    print("✔️ Finshed Preprocessing Step ")
    return final_df


def plot_clean_scatter(df, std_threshold=5, if_plot=False):
    """Remove outliers and plot lat/lon scatter"""
    
    # Remove outliers using z-score method
    lat_mean, lat_std = df['Latitude'].mean(), df['Latitude'].std()
    lon_mean, lon_std = df['Longitude'].mean(), df['Longitude'].std()
    
    df_clean = df[
        (np.abs(df['Latitude'] - lat_mean) < std_threshold * lat_std) &
        (np.abs(df['Longitude'] - lon_mean) < std_threshold * lon_std)
    ]

    
    print(f"Original: {len(df)} tweets")
    print(f"After removing outliers: {len(df_clean)} tweets")
    print(f"Removed: {len(df) - len(df_clean)} outliers")

    if (if_plot):
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.scatter(df_clean['Longitude'], df_clean['Latitude'], 
                s=10, c='red', alpha=0.6, edgecolors='black', linewidth=0.3)
        ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        ax.set_title('Tweet Locations (Outliers Removed)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('scatter_clean.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df_clean








import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

def lr_range_test(model, train_loader, device, start_lr=1e-7, end_lr=1, num_iter=None, 
                  smooth_factor=0.05, divergence_threshold=4):
    """
    Learning Rate Range Test (LR Finder)
    
    Systematically increases learning rate and tracks loss to find optimal LR.
    
    Args:
        model: Your neural network model
        train_loader: Training data loader
        device: cuda or cpu
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 1)
        num_iter: Number of iterations (default: one epoch)
        smooth_factor: Smoothing factor for loss curve (default: 0.05)
        divergence_threshold: Stop if loss exceeds best_loss * threshold
    
    Returns:
        lrs: List of learning rates tested
        losses: List of corresponding losses
        suggested_lr: Suggested starting learning rate
    """
    print("\n" + "="*100)
    print("LEARNING RATE RANGE TEST")
    print("="*100)
    print(f"Range: {start_lr:.2e} → {end_lr:.2e}")
    
    # Make a copy of the model to avoid messing up the original
    model_copy = copy.deepcopy(model)
    model_copy.train()
    
    # Setup
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model_copy.parameters(), lr=start_lr)
    
    # Calculate number of iterations
    if num_iter is None:
        num_iter = len(train_loader)
    
    # Calculate LR multiplication factor
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    # Storage
    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0
    batch_num = 0
    
    # Iterate through training data
    iterator = iter(train_loader)
    
    print(f"\nTesting {num_iter} iterations...")
    
    for iteration in tqdm(range(num_iter), desc="LR Range Test"):
        batch_num += 1
        
        # Get batch
        try:
            batch_tweets, batch_y, batch_t, seq_lengths, token_lengths = next(iterator)
        except StopIteration:
            # If we run out of data, restart iterator
            iterator = iter(train_loader)
            batch_tweets, batch_y, batch_t, seq_lengths, token_lengths = next(iterator)
        
        batch_tweets = batch_tweets.to(device)
        batch_y = batch_y.unsqueeze(-1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model_copy(batch_tweets, batch_y, batch_t, seq_lengths, token_lengths)
        loss = criterion(logits, batch_y.float())
        
        # Compute smoothed loss
        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Stop if loss is diverging
        if avg_loss > divergence_threshold * best_loss:
            print(f"\n⚠️  Stopping early - loss is diverging (iteration {iteration})")
            break
        
        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(avg_loss)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    print("\n✅ LR Range Test Complete!")
    
    # Find suggested learning rate
    suggested_lr = suggest_lr(lrs, losses)
    
    # Plot results
    plot_lr_finder(lrs, losses, suggested_lr)
    
    return lrs, losses, suggested_lr


def suggest_lr(lrs, losses):
    """
    Suggest optimal learning rate based on the steepest descent
    """
    # Find the steepest negative gradient
    gradients = np.gradient(losses)
    min_gradient_idx = np.argmin(gradients)
    
    # Suggested LR is typically where gradient is steepest (before minimum)
    # Use LR that's ~10x smaller than where loss is minimum
    min_loss_idx = np.argmin(losses)
    
    # Take the LR at steepest descent, or 1/10th of min loss LR
    if min_gradient_idx < len(lrs) * 0.8:  # Only if not too close to end
        suggested_idx = min_gradient_idx
    else:
        suggested_idx = max(0, min_loss_idx - len(lrs) // 10)
    
    suggested_lr = lrs[suggested_idx]
    
    return suggested_lr


def plot_lr_finder(lrs, losses, suggested_lr):
    """
    Plot the learning rate range test results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss vs Learning Rate (log scale)
    ax1.plot(lrs, losses, linewidth=2, color='#2E86AB')
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (smoothed)', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Range Test', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark suggested LR
    ax1.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2, 
                label=f'Suggested LR: {suggested_lr:.2e}')
    ax1.legend(fontsize=11)
    
    # Plot 2: Loss vs Iteration
    ax2.plot(losses, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (smoothed)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Over Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_range_test.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved as 'lr_range_test.png'")
    plt.show()
    
    # Print recommendations
    print("\n" + "="*100)
    print("LEARNING RATE RECOMMENDATIONS")
    print("="*100)
    print(f"\n📍 Suggested Starting LR: {suggested_lr:.2e}")
    print(f"\n💡 Guidelines:")
    print(f"   • Start training with LR around: {suggested_lr:.2e}")
    print(f"   • Consider trying: {suggested_lr/3:.2e} (more conservative)")
    print(f"   • Or try: {suggested_lr*3:.2e} (more aggressive)")
    print(f"\n⚠️  Look for:")
    print(f"   • LR where loss decreases fastest (steepest downward slope)")
    print(f"   • Pick a value BEFORE the loss starts increasing")
    print(f"   • Common choice: 1/10th of the LR at minimum loss")
    print("="*100 + "\n")
