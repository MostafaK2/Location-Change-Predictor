import torch
import numpy as np
import pandas as pd
from haversine import haversine


from torch.utils.data import Dataset

class TweetEmbeddingDataset(Dataset):
    def __init__(self, emb_path, meta_path, distance_threshold=0.001):
        super().__init__()

        
        self.embeddings = np.load(emb_path) # (376318, 384) embeddings
        self.meta = pd.read_csv(meta_path)  # (376318, 4)

        assert len(self.meta) == self.embeddings.shape[0], "Embeddings and metadata row count mismatch!"
        
        self.user_groups = list(self.meta.groupby("UserID"))
        self.distance_threshold = distance_threshold

    def __len__(self):
        return len(self.user_groups)
    
    def __getitem__(self):
        idx = 0
        _, group = self.user_groups[idx]  # userID, grp
        indices = group.index.tolist()

        x = torch.tensor(self.embeddings[indices], dtype=torch.float)
        t = torch.tensor(group["Timestamp"].values, dtype=torch.float).unsqueeze(-1)[:-1]
        
        locs = list(zip(group["Latitude"], group["Longitude"]))

        x = x[:-1]
        y = []

        for i in range(1, len(locs)):
            dist = haversine(locs[i-1], locs[i])
            y.append(1 if dist > 0 else 0)
        y = torch.tensor(y, dtype=torch.float)
        print(x.shape, y.shape, t.shape)


