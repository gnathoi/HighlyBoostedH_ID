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

# loading data
data = './hyy_data/selected_features/' # directory where data is stored
test_path = os.path.join(data, 'test_dataset.pkl')
test_path2 = os.path.join(data, 'test_dataset.pkl')
test_dataset_adv = ds.BoostDataset_Adv(test_path,test_mode=True)
test_dataset = ds.BoostDataset(test_path2)

if remove_class != None:
    test_dataset_adv.remove_class(remove_class)
    test_dataset.remove_class(remove_class)
    print(f'Class {remove_class} removed from datasets.')

#test_loader = DataLoader(test_dataset_adv, batch_size=3000, num_workers=int(os.cpu_count()/2), shuffle=False)
num_classes = test_dataset.num_classes()

# loading DNN
DNN4Class_folder_path = './models/DNN/4Class'
DNN4Class = os.path.join(DNN4Class_folder_path, 'DNN4Class.pth')
model1 = torch.load(DNN4Class)
model1.eval()

log_DNN = pd.read_csv(os.path.join(DNN4Class_folder_path,'training_log.csv'))
plotter_DNN = pl.plotting(log=log_DNN, new_folder_path=DNN4Class_folder_path, num_classes=num_classes)
test_loader=DataLoader(test_dataset, batch_size=2500, num_workers=int(os.cpu_count()/2), shuffle=False)
plotter_DNN.DNN_confusion_matrix(data_loader=test_loader,model=model1)
#plotter_DNN.SignalEff_vs_pTandEta(data_loader=test_loader, dataset=test_dataset, model=model1)
#plotter_DNN.JetRejectionRate(data_loader=test_loader,model=model1)

#log = pd.read_csv(os.path.join(DNN4Class_folder_path,'training_log.csv'))

ANN4Class_folder_path = './models/AN/4Class'
ANN4Class = os.path.join(ANN4Class_folder_path, 'ANN4Class.pth')
model2 = torch.load(ANN4Class)
model2.eval()
#log = None
log = pd.read_csv(os.path.join(ANN4Class_folder_path,'training_log.csv'))
# testing DNN
plotter = pl.plotting(log=log, new_folder_path=ANN4Class_folder_path, num_classes=num_classes)
test_loader=DataLoader(test_dataset_adv, batch_size=3000, num_workers=int(os.cpu_count()/2), shuffle=False)
bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
plotter.lossandsignal()
plotter.DNN_confusion_matrix(data_loader=test_loader,model=model2)
plotter.ScalarDiscriminant(jet_fractions=test_dataset_adv.jet_fractions().to(device),data_loader=test_loader,model=model2)
plotter.ScalarDiscriminantZee(jet_fractions=test_dataset_adv.jet_fractions().to(device),data_loader=test_loader,model=model2)
plotter.JetRejectionRate(data_loader=test_loader,model=model2)
plotter.MulticlassROC(data_loader=test_loader,model=model2)
plotter.SignalEff_vs_pTandEta(data_loader=test_loader, dataset=test_dataset_adv, model=model2)
plotter.SignalEff_vs_FatElectron_cM_bins(data_loader=test_loader,dataset=test_dataset_adv, model=model2, col_index=test_dataset_adv.column_index('FatElectron_cM_bins'), bins=bins,DNN=False)
model_configs = [{'model':model1,'DNN':True,'label':'DNN','col_index':test_dataset.column_index('FatElectron_cM'),'test_dataset':test_dataset,'batch_size':2500},
                {'model':model2,'DNN':False,'label':'ANN','col_index':test_dataset_adv.column_index('FatElectron_cM_bins'),'test_dataset':test_dataset_adv,'batch_size':3000}]
plotter.SignalEff_vs_FatElectron_cM_binsANN_DNN(model_configs=model_configs, bins=bins)
plotter.JRRvsMass(model_configs=model_configs, bins=bins)
