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
import utils.feature_analysis as fa

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

# loading data
data = './hyy_data/selected_features/' # directory where data is stored
test_path = os.path.join(data, 'test_dataset.pkl')
test_dataset = ds.BoostDataset(test_path)
test_dataset_adv = ds.BoostDataset_Adv(test_path,test_mode=True)


if remove_class != None:
    test_dataset.remove_class(remove_class)
    test_dataset_adv.remove_class(remove_class)
    print(f'Class {remove_class} removed from datasets.')

test_loader = DataLoader(test_dataset, batch_size=2500, num_workers=int(os.cpu_count()/2), shuffle=False)
test_loader_adv = DataLoader(test_dataset_adv, batch_size=3000, num_workers=int(os.cpu_count()/2), shuffle=False)
num_classes = test_dataset.num_classes()

'''
print('DNN D_Hyy calcualtion...')
# loading DNN
DNN4Class_folder_path = './models/DNN/4Class'
DNN4Class = os.path.join(DNN4Class_folder_path, 'DNN4Class.pth')
model = torch.load(DNN4Class)
model.eval()


#print('Generating D_Hyy values...')
D_values = []
f_values = test_dataset.jet_fractions().to(device)
model.eval()

with torch.no_grad():
    for data, labels, _ in tqdm(test_loader, desc=f"Calculating D_Hyy"):
        data = data.to(device)
        labels = labels.cpu().numpy()
        #batch_size = len(labels)

        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        #epsilon = 1e-10 # A small constant to prevent division by zero

        numerator = f_values[0] * probs[:, 0]
        denominator_terms = [f_values[j] * probs[:, j] for j in range(1, num_classes)]  # Exclude class 0
        denominator = sum(denominator_terms)# + epsilon
        D0 = torch.log(numerator / denominator).cpu().numpy()

        #paired_values = list(zip(labels, D0))
        D_values.extend(D0)

D_values = np.array(D_values)
feature_analysis = fa.FeatureAnalysis(data_path=test_path, output_path=DNN4Class_folder_path, n_neighbors=4, rng=2023, D_values=D_values,remove_class=remove_class)
mi_df, nmi_df = feature_analysis.MI()

print('ANN D_Hyy calcualtion...')
ANN4Class_folder_path = './models/AN/4Class'
ANN4Class = os.path.join(ANN4Class_folder_path, 'ANN4Class.pth')
model = torch.load(ANN4Class)
model.eval()

#print('Generating D_Hyy values...')
D_values = []
f_values = test_dataset_adv.jet_fractions().to(device)
model.eval()

with torch.no_grad():
    for data, labels, _ in tqdm(test_loader_adv, desc=f"Calculating D_Hyy"):
        data = data.to(device)
        labels = labels.cpu().numpy()
        #batch_size = len(labels)

        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        #epsilon = 1e-10 # A small constant to prevent division by zero

        numerator = f_values[0] * probs[:, 0]
        denominator_terms = [f_values[j] * probs[:, j] for j in range(1, num_classes)]  # Exclude class 0
        denominator = sum(denominator_terms)# + epsilon
        D0 = torch.log(numerator / denominator).cpu().numpy()

        #paired_values = list(zip(labels, D0))
        D_values.extend(D0)

D_values = np.array(D_values)
feature_analysis = fa.FeatureAnalysis(data_path=test_path, output_path=ANN4Class_folder_path, n_neighbors=4, rng=2023, D_values=D_values,remove_class=remove_class)
mi_df, nmi_df = feature_analysis.MI()
