import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# custom packages for datasets, network architecture and training
import utils.data as ds
import utils.networks as nw
import utils.train as tr
import utils.plotting as pl
import utils.testing as te

start_time = time.time()
remove_class = 2

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Cuda device: {torch.cuda.get_device_name(device_num)} has been found and set to default device.")

data = './hyy_data/selected_features/' # directory where data is stored
test_path = os.path.join(data, 'test_dataset.pkl')
test_dataset = ds.BoostDataset_Adv(test_path,test_mode=True)

if remove_class != None:
    test_dataset.remove_class(remove_class)
    print(f'Class {remove_class} removed from datasets.')

mean_std_path = os.path.join(data,'mean_std.pkl')
mean_std = pd.read_pickle(mean_std_path)
means = [mean_std[feature][0] for feature in mean_std.columns.tolist()]
stds = [mean_std[feature][1] for feature in mean_std.columns.tolist()]
mean_std = [means, stds]

top_models_path = './GridSearch/AN/'
tester = te.Tester(device=device, folder_path=top_models_path, test_dataset=test_dataset, mean_std=mean_std, num_cores=int(os.cpu_count()/2))

for model_index in range(1, 6):
    epoch, loss, params, state_dict = tester.load_model_ANN(model_index)
    tester.test_AN(model_index, epoch, loss, params, state_dict)

end_time = time.time()
total_time = (end_time-start_time)/3600
print(f'\nTesting complete. Test plots saved. \nTime for tests: {total_time:.2f}hrs')
