import pandas as pd
import numpy as np
import os
import utils.data as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

data = './hyy_data/selected_features/' # directory where data is stored
train_path = os.path.join(data, 'training_dataset.pkl') # path

df = pd.read_pickle(train_path)

class_counts = df['type'].value_counts()
max_samples = class_counts.max()
num_classes = df['type'].nunique()

# Calculate the inverse of each class count
class_weights = {cls: 1/class_count for cls, class_count in class_counts.items()}

# Calculate the weight of each sample
sample_weights = df['type'].apply(lambda x: class_weights[x])

# Normalize sample weights so they sum up to 1
sample_weights = sample_weights / sum(sample_weights)

desired_samples = int(num_classes * max_samples / 2) # dividing by two gives approximately the same amount of data as the pytorch WeightedRandomSampler
# Oversample the dataframe
oversampled_df = df.sample(n=desired_samples, replace=True, weights=sample_weights)

unique_class_counts = oversampled_df['type'].value_counts().sort_index()
print(unique_class_counts)

upsample_path = os.path.join(data,'upsample_training_dataset.pkl')
oversampled_df.to_pickle(upsample_path)
