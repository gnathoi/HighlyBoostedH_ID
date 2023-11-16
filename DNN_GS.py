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
weight = True # leave weight tensor on
upsample = False
remove_class = 2

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Cuda device: {torch.cuda.get_device_name(device_num)} has been found and set to default device.")

data = './hyy_data/selected_features1/' # directory where data is stored
output = './outputs/DNN/'
name = 'GridSearch' # output directory name here

new_folder_path = os.path.join(output, name)

# Create the new directory if it does not exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

#input_size = 15
# initialize your hyperparameters

hyperparameters = {
    "activation_fn": [F.sigmoid,nn.ReLU],
    "batch_size": np.arange(1500, 3001, 500),
    "num_epochs": np.arange(30, 60, 20),
    "hidden_nodes": np.arange(100, 260, 100),
    "learning_rate": [0.001,0.005,0.01],
    "drop_p": np.arange(0, 0.1, 0.025),
    "weight_decay": [0],
    "num_layers":[3,4]
}

for key, value in hyperparameters.items():
    print(f"{key}: {value}")

# importing the datasets
print('\nLoading data...')
if upsample == True:
    train_path = os.path.join(data, 'upsample_training_dataset.pkl')
else:
    train_path = os.path.join(data, 'training_dataset.pkl') # path

train_dataset = ds.BoostDataset(train_path) # calling the dataset class
val_path =  os.path.join(data, 'validation_dataset.pkl') # path
val_dataset = ds.BoostDataset(val_path) # calling the dataset class
print('Loaded!\n')

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
    criterion = nn.CrossEntropyLoss(weight=weights_tensor).to(device) # weight=weights_tensor
    print('Weights tensor implemented in loss function.')
else:
    criterion = nn.CrossEntropyLoss().to(device) # define loss function without class weights for evaluation
    print('No weights tensor in loss function.')

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

start_idx = 0
param_index_path = os.path.join(new_folder_path,'param_index.txt')
if os.path.exists(param_index_path):
    top_models = tr.load_top_models(new_folder_path)
    with open(param_index_path, 'r') as f:
        start_idx = int(f.read()) + 1  # Start from the next index
else:
    top_models = []


# List to keep track of top 5 models
 # [(val_loss1, params1, state_dict1), ...]
param_interval = 5
# Iterate over each combination in the grid
for idx, params in enumerate(param_grid[start_idx:],start=start_idx):
    print(f'\nModel: {idx+1}/{combinations}')
    print(f'Hyperparameters: \n{params}')

     # Create the model with the current hyperparameter combination
    model = nw.HiggsNet(input_size=input_size,
                     hidden_nodes=params['hidden_nodes'],
                     num_layers=params['num_layers'], # Not specified in hyperparameters
                     activation_fn=params['activation_fn'],
                     drop_p=params['drop_p'],
                     mean_std=mean_std,
                     num_classes=num_classes).to(device) # Number of classes not specified, assumed 4

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params.get('weight_decay', 0))
    # check that the training will happen on the GPU
    if torch.device('cuda:0') == next(model.parameters()).device:
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

    # Trainer class
    trainer = tr.Trainer(model=model,
                      device=device,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      val_dataset=val_dataset,
                      optimizer=optimizer,
                      criterion=criterion)

    # Train the model
    stop_epoch, val_loss = trainer.train(params['num_epochs'])

    # Check if the model should be in the top 5
    if len(top_models) < 5 or val_loss < top_models[0][0]: # Assuming worst model is at index 0
        state_dict = model.state_dict()
        if len(top_models) == 5: # If we already have 5 models, pop the worst one
            heapq.heappop(top_models)
        heapq.heappush(top_models, (val_loss, stop_epoch, params, state_dict)) # Add the current model's performance to the top

    if (idx + 1) % param_interval == 0:
        tr.save_top_models(top_models,new_folder_path)
        with open(os.path.join(new_folder_path, f'param_index.txt'), 'w') as f:
            f.write(str(idx))
        print(f"Saved top models at index {idx}")

    print(f"Model {idx+1} - Validation Loss: {val_loss:.4f}")

tr.save_top_models(top_models,new_folder_path)
end_time = time.time()
total_time = (end_time-start_time)/3600
print(f'\nGridSearch complete. Top models saved. \nTime for run: {total_time:.2f}hrs')
