# standard packages
import pandas as pd
import numpy as np
import itertools
import json
import time
from tqdm import tqdm
import os
import pickle
import heapq
# machine learning packages, pytorch & sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import ParameterGrid
# custom packages for datasets, network architecture and training
import utils.data as ds
import utils.networks as nw
import utils.train as tr

start_time = time.time()
weight = False # leave weight tensor on
upsample = True
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
output = './outputs/AN/'
name = 'GridSearch' # output directory name here

new_folder_path = os.path.join(output, name)

# Create the new directory if it does not exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

#input_size = 15
# initialize your hyperparameters

hyperparameters = {
    "lambda_adv": [0.01,1,100],
    "activation_fn": [F.sigmoid,F.relu],
    "batch_size": [2000,2500,3000],
    "num_epochs": [30,50],
    "hidden_nodes_A": [100,200],
    "learning_rate_A": [1,0.01],
    "drop_p_A": [0.01,0.1],
    "weight_decay_A": [0],
    "num_layers_A": [4],
    "hidden_nodes_D": [200],
    "learning_rate_D": [0.001,0.01],
    "drop_p_D": [0.01],
    "weight_decay_D": [0],
    "num_layers_D": [4]
}
'''
hyperparameters = {
    "lambda_adv": [100],
    "activation_fn": [F.relu,F.sigmoid],
    "batch_size": [2400],
    "num_epochs": [1],
    "hidden_nodes_A": [64,128],
    "learning_rate_A": [1,0.1],
    "drop_p_A": [0.01],
    "weight_decay_A": [0],
    "num_layers_A": [3],
    "hidden_nodes_D": [128],
    "learning_rate_D": [0.002],
    "drop_p_D": [0.1],
    "weight_decay_D": [0],
    "num_layers_D": [3,4,5]
}
'''

for key, value in hyperparameters.items():
    print(f"{key}: {value}")

# importing the datasets
print('\nLoading data...')
if upsample == True:
    train_path = os.path.join(data, 'upsample_training_dataset.pkl')
else:
    train_path = os.path.join(data, 'training_dataset.pkl') # path
    
train_dataset = ds.BoostDataset_Adv(train_path) # calling the dataset class
val_path =  os.path.join(data, 'test_dataset.pkl') # path
train_bins, _ = train_dataset.massbins()
print("Train Bins: ", train_bins)  # Debug print
val_dataset = ds.BoostDataset_Adv(val_path,bins=train_bins) # calling the dataset class
val_bins, _ = val_dataset.massbins()
print("Validation Bins: ", val_bins)  # Debug print
print('Loaded!\n')
print(f'train. dataset: \n{train_dataset.count_type_per_bin()}')
print(f'val. dataset: \n{val_dataset.count_type_per_bin()}')

# remove any classes that have been specified.
if remove_class != None:
    train_dataset.remove_class(remove_class)
    val_dataset.remove_class(remove_class)
    print(f'Class {remove_class} removed from datasets.\n')

num_classes = train_dataset.num_col('type')
input_size = train_dataset.input_size()

if weight == True:
    #weights_tensor = train_dataset.class_weights_calculator()
    weights_path = os.path.join(data, 'weights_tensor.npy')
    weights = np.load(weights_path)
    print('Weights for CrossEntropyLoss:')
    if remove_class != None: # if we have removed a class need to make sure we also remove the weights for the loss function
        weights = np.delete(weights, remove_class)
        weights_tensor = torch.tensor(weights).float()

    class_labels = list(range(num_classes))  # print the weights used for each class
    for i, class_label in enumerate(class_labels):
        print(f"Weight for class {class_label}: {weights_tensor[i]:.2f}")
    criterion_D = nn.CrossEntropyLoss(weight=weights_tensor).to(device) # weight=weights_tensor
    print('Weights tensor implemented in loss function.')
else:
    criterion_D = nn.CrossEntropyLoss().to(device) # define loss function without class weights for evaluation
    print('No weights tensor in loss function.')

criterion_A = nn.CrossEntropyLoss()

# save the search space so that we don't forget it when we come to write
searchname = 'searchspace.pkl'
search_path = os.path.join(new_folder_path, searchname)
# Save the dictionary into a pickle file
with open(search_path, 'wb') as f:
    pickle.dump(hyperparameters, f)

print(f'\nSearching space defined by: \n{hyperparameters}')

# Generate a grid over the hyperparameters
param_grid = list(ParameterGrid(hyperparameters))
combinations = len(param_grid)

print(f'\nNo. of combinations in the hyperparameter space: {combinations}')

mean_std_path = os.path.join(data,'mean_std.pkl')
#print(f'mean_std_path: {mean_std_path}\n')

mean_std = pd.read_pickle(mean_std_path)
means = [mean_std[feature][0] for feature in mean_std.columns.tolist()]
stds = [mean_std[feature][1] for feature in mean_std.columns.tolist()]
mean_std = [means, stds]

num_classes = train_dataset.num_col('type')
cM_ind = train_dataset.column_index('FatElectron_cM_bins')

start_idx = 0
param_index_path = os.path.join(new_folder_path,'param_index.txt')
if os.path.exists(param_index_path):
    top_models = tr.load_top_models_adv(new_folder_path)
    with open(param_index_path, 'r') as f:
        start_idx = int(f.read()) + 1  # Start from the next index
else:
    top_models = []

param_interval = 5
# Iterate over each combination in the grid
for idx, params in enumerate(param_grid[start_idx:],start=start_idx):
    print(f'\nModel: {idx+1}/{combinations}')
    print(f'Hyperparameters: \n{params}')

    discriminator = nw.HiggsJetTagger(input_size=input_size,
                                    hidden_nodes=params['hidden_nodes_D'],
                                    num_layers=params['num_layers_D'],
                                    activation_fn=params['activation_fn'],
                                    drop_p=params['drop_p_D'],
                                    mean_std=mean_std,
                                    num_classes=num_classes,
                                    cM_ind=cM_ind).to(device)
    adversary = nw.Adversary(num_classes=1,
                            hidden_nodes=params['hidden_nodes_A'],
                            num_layers=params['num_layers_A'],
                            activation_fn=params['activation_fn'],
                            drop_p=params['drop_p_A'],
                            mean_std=None).to(device)

    # Initialize the optimizer
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params['learning_rate_D'], weight_decay=params.get('weight_decay_D', 0))
    optimizer_A = optim.Adam(adversary.parameters(), lr=params['learning_rate_A'], weight_decay=params.get('weight_decay_A', 0))
    # Check if both adversary and discriminator are on the same device, and that device is CUDA
    adversary_device = next(adversary.parameters()).device
    discriminator_device = next(discriminator.parameters()).device
    if adversary_device == discriminator_device and adversary_device == torch.device('cuda:0'):
        print('\nNetwork has successfully built and is on the cuda device.')
    else:
        print('\nWARNING: Network training is not using cuda device!\nTerminating training!')
        sys.exit(0)
    # DataLoaders
    print('\nSending data to loader...')
    num_cores = os.cpu_count() # use the number of cpu cores for the number of workers for the DataLoader
    train_loader = DataLoader(train_dataset, num_workers=num_cores, batch_size=int(params['batch_size']), shuffle=True)
    valid_loader = DataLoader(val_dataset, num_workers=num_cores, batch_size=int(params['batch_size']), shuffle=False)
    print('DataLoader ready!\n')

    bins, _ = val_dataset.massbins()
    # Trainer class
    trainer = tr.Trainer_Adv(cM_ind=cM_ind,
                            lambda_adv=params['lambda_adv'],
                            model_D=discriminator,
                            model_A=adversary,
                            device=device, train_loader=train_loader,
                            valid_loader=valid_loader,
                            val_dataset=val_dataset,
                            optimizer_D=optimizer_D,
                            optimizer_A=optimizer_A,
                            criterion_D=criterion_D,
                            criterion_A=criterion_A,
                            bins=bins)
    # Train the model
    stop_epoch, val_loss_D, tpr_std, tpr_mean = trainer.train(params['num_epochs'])

    # Check if the model should be in the top 5
    if val_loss_D <= 0.4 and tpr_std < 0.2: # make sure these values are right 0.4 and 0.2
        if len(top_models) < 5 or -tpr_mean < top_models[0][0]: # Assuming worst model is at index 0
            state_dict = discriminator.state_dict()
            if len(top_models) == 5: # If we already have 5 models, pop the worst one
                heapq.heappop(top_models)
            heapq.heappush(top_models, (-tpr_mean, stop_epoch, val_loss_D, params, state_dict)) # Add the current model's performance to the top

    if (idx + 1) % param_interval == 0:
        tr.save_top_models_adv(top_models,new_folder_path)
        with open(os.path.join(new_folder_path, f'param_index.txt'), 'w') as f:
            f.write(str(idx))
        print(f"Saved top models at index {idx}")

    print(f"Model {idx+1} - Validation Loss D: {val_loss_D:.4f}, Validation TPR mean: {tpr_mean:.4f}")

tr.save_top_models_adv(top_models,new_folder_path)
end_time = time.time()
total_time = (end_time-start_time)/3600
print(f'\nGridSearch complete. Top models saved. \nTime for run: {total_time:.2f}mins')
