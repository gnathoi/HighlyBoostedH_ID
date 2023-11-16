import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import gc

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

class Tester:
    def __init__(self,device,folder_path,test_dataset,mean_std,state_dicts_folder='state_dicts',num_cores=os.cpu_count()):
        self.device = device
        self.folder_path = folder_path
        self.test_dataset = test_dataset
        self.state_dicts_folder = state_dicts_folder
        self.mean_std = mean_std
        self.num_cores = num_cores

    def load_model(self, model_index):
        model_path = os.path.join(self.folder_path, f'top_model_{model_index}.pkl')
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        #print(f"Loaded model info for top_model_{model_index}: {model_info}")

        epoch = model_info['stop_epoch']
        loss = model_info['val_loss']
        params = model_info['hyperparameters']

        state_dict_path = os.path.join(self.folder_path, self.state_dicts_folder, f'top_model_{model_index}.pth')
        state_dict = torch.load(state_dict_path)
        return epoch, loss, params, state_dict

    def test_DNN(self, model_index, epoch, loss, params, state_dict):
        print(f'\nModel {model_index}')
        num_classes = self.test_dataset.num_col('type')
        input_size = self.test_dataset.input_size()
        print(f"Validation loss: {loss}")
        print(f"Testing with params: {params}")

        model = nw.HiggsNet(
            input_size=input_size,
            hidden_nodes=int(params['hidden_nodes']),
            num_layers=int(params['num_layers']),
            activation_fn=params['activation_fn'],
            drop_p=float(params['drop_p']),
            mean_std=self.mean_std,
            num_classes=num_classes
        ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        test_folder = f'./GridSearch/DNN/testing/top_model_{model_index}'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        test_loader = DataLoader(self.test_dataset, batch_size=int(params['batch_size']), num_workers=self.num_cores, shuffle=False)
        plotter = pl.plotting(log=None, new_folder_path=test_folder, num_classes=num_classes)
        plotter.SignalEff_vs_pTandEta(data_loader=test_loader, dataset=self.test_dataset, model=model)
        plotter.ScalarDiscriminantZee(jet_fractions=self.test_dataset.jet_fractions().to(self.device),data_loader=test_loader,model=model)
        bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
        plotter.DNN_confusion_matrix(data_loader=test_loader,model=model)
        plotter.JetRejectionRate(data_loader=test_loader,model=model)
        plotter.SignalEff_vs_FatElectron_cM_bins(data_loader=test_loader,dataset=self.test_dataset, model=model, col_index=self.test_dataset.column_index('FatElectron_cM'), bins=bins,DNN=True)
        plotter.ScalarDiscriminant(jet_fractions=self.test_dataset.jet_fractions().to(self.device),data_loader=test_loader,model=model)
        plotter.MulticlassROC(data_loader=test_loader,model=model)

    def test_AN(self, model_index, epoch, loss, params, state_dict):
        print(f'\nModel {model_index}')
        num_classes = self.test_dataset.num_col('type')
        input_size = self.test_dataset.input_size()
        cM_ind = self.test_dataset.column_index('FatElectron_cM_bins')
        print(f"Validation loss: {loss}")
        print(f"Testing with params: {params}")

        model = nw.HiggsJetTagger(input_size=input_size,
                                        hidden_nodes=params['hidden_nodes_D'],
                                        num_layers=params['num_layers_D'],
                                        activation_fn=params['activation_fn'],
                                        drop_p=params['drop_p_D'],
                                        mean_std=self.mean_std,
                                        num_classes=num_classes,
                                        cM_ind=cM_ind).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        test_folder = f'./GridSearch/AN/testing/top_model_{model_index}'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        test_loader = DataLoader(self.test_dataset, batch_size=int(params['batch_size']), num_workers=self.num_cores, shuffle=False)
        plotter = pl.plotting(log=None, new_folder_path=test_folder, num_classes=num_classes)
        #plotter.SignalEff_vs_pTandEta(data_loader=test_loader, dataset=self.test_dataset, model=model)
        #plotter.ScalarDiscriminantZee(jet_fractions=self.test_dataset.jet_fractions().to(self.device),data_loader=test_loader,model=model)
        bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
        #plotter.DNN_confusion_matrix(data_loader=test_loader,model=model)
        #plotter.JetRejectionRate(data_loader=test_loader,model=model)
        plotter.SignalEff_vs_FatElectron_cM_bins(data_loader=test_loader,dataset=self.test_dataset, model=model, col_index=self.test_dataset.column_index('FatElectron_cM_bins'), bins=bins,DNN=False)
        #plotter.ScalarDiscriminant(jet_fractions=self.test_dataset.jet_fractions().to(self.device),data_loader=test_loader,model=model)
        #plotter.MulticlassROC(data_loader=test_loader,model=model)

    def load_model_ANN(self, model_index):
        model_path = os.path.join(self.folder_path, f'top_model_{model_index}.pkl')
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        #print(f"Loaded model info for top_model_{model_index}: {model_info}")

        epoch = model_info['stop_epoch']
        loss = model_info['val_loss_D']
        params = model_info['hyperparameters']

        state_dict_path = os.path.join(self.folder_path, self.state_dicts_folder, f'top_model_{model_index}.pth')
        state_dict = torch.load(state_dict_path)
        return epoch, loss, params, state_dict


    '''
    def load_models(self):
        top_models = []
        for i in range(5):
            model_path = os.path.join(self.folder_path, f'top_model_{i+1}.pkl')
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            print(f"Loaded model info for top_model_{i+1}: {model_info}")
            epoch = model_info['stop_epoch']
            loss = model_info['val_loss']
            params = model_info['hyperparameters']
            top_models.append((epoch, loss, params))

        # Load state_dicts from individual files
        for i, (epoch, loss, params) in enumerate(top_models):
            state_dict_path = os.path.join(self.folder_path, self.state_dicts_folder, f'top_model_{i+1}.pth')
            state_dict = torch.load(state_dict_path)
            top_models[i] = (epoch, loss, params, state_dict)
            self.top_models = top_models
        return self.top_models

    def test_top_DNNs(self):
        for i, (_, _, params, state_dict) in enumerate(self.top_models):
            print(f'\nModel {i+1}/5')
            num_classes = self.test_dataset.num_col('type')
            input_size = self.test_dataset.input_size()
            print(f"Testing with params: {params}")

            model = nw.HiggsNet(input_size=input_size,
                             hidden_nodes=int(params['hidden_nodes']),
                             num_layers=int(params['num_layers']),
                             activation_fn=params['activation_fn'],
                             drop_p=float(params['drop_p']),
                             mean_std=self.mean_std,
                             num_classes=num_classes).to(self.device)

            model.load_state_dict(state_dict)
            model.eval()

            test_folder = f'./GridSearch/DNN/testing/top_model_{i+1}'
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            test_loader = DataLoader(self.test_dataset,batch_size=int(params['batch_size']),num_workers=self.num_cores,shuffle=False)

            plotter = pl.plotting(log=None, new_folder_path=test_folder, num_classes=num_classes)
            bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
            plotter.SignalEff_vs_FatElectron_cM_bins(data_loader=test_loader,dataset=self.test_dataset, model=model, col_index=self.test_dataset.column_index('FatElectron_cM'), bins=bins,DNN=True)
            plotter.ScalarDiscriminant(jet_fractions=self.test_dataset.jet_fractions().to(self.device),data_loader=test_loader,model=model)
            plotter.MulticlassROC(data_loader=test_loader,model=model)
            plotter.DNN_confusion_matrix(data_loader=test_loader,model=model)
            plotter.JetRejectionRate(data_loader=test_loader,model=model)
            del test_loader
            gc.collect()
        '''
