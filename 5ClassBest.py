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
torch.multiprocessing.set_sharing_strategy('file_system')
# custom packages for datasets, network architecture and training
import utils.data as ds
import utils.networks as nw
import utils.train as tr
import utils.plotting as pl
import utils.testing as te

start_time = time.time()
remove_class = None

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
test_path = os.path.join(data, 'validation_dataset.pkl')
test_dataset = ds.BoostDataset(test_path)

if remove_class != None:
    test_dataset.remove_class(remove_class)
    print(f'Class {remove_class} removed from datasets.')

#test_loader = DataLoader(test_dataset_adv, batch_size=3000, num_workers=int(os.cpu_count()/2), shuffle=False)
num_classes = test_dataset.num_classes()

# loading DNN
DNN4Class_folder_path = './models/AN/5ClassMK5'
DNN4Class = os.path.join(DNN4Class_folder_path, 'ANN4Class.pth')
model = torch.load(DNN4Class)
model.eval()

#log = None
log = pd.read_csv(os.path.join(DNN4Class_folder_path,'training_log.csv'))
# testing DNN
plotter = pl.plotting5Class(log=log, new_folder_path=DNN4Class_folder_path, num_classes=num_classes)
test_loader=DataLoader(test_dataset, batch_size=2500, num_workers=int(os.cpu_count()/2), shuffle=False)
bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
plotter.DNN_confusion_matrix(data_loader=test_loader,model=model)
plotter.lossandsignal()
plotter.ScalarDiscriminant(jet_fractions=test_dataset.jet_fractions().to(device),data_loader=test_loader,model=model)
plotter.ScalarDiscriminantZee(jet_fractions=test_dataset.jet_fractions().to(device),data_loader=test_loader,model=model)
plotter.JetRejectionRate(data_loader=test_loader,model=model)

plotter.MulticlassROC(data_loader=test_loader,model=model)
plotter.SignalEff_vs_pTandEta(data_loader=test_loader, dataset=test_dataset, model=model)
#plotter.SignalEff_vs_FatElectron_cM_bins(data_loader=test_loader,dataset=test_dataset, model=model, col_index=test_dataset.column_index('FatElectron_cM'), bins=bins,DNN=True)
