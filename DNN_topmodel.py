import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # must be called before importing pyplot
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torcheval.metrics.functional import multiclass_f1_score

# custom packages to import
import utils.data as ds
import utils.networks as nw
import utils.plotting as pl
import utils.testing as te

import os
import sys

start_time = time.time()
remove_class = 2
upsample = False
weight = True

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Cuda device: {torch.cuda.get_device_name(device_num)} has been found and set to default device.")

data = './hyy_data/selected_features/' # directory where data is stored
root = './models/DNN/' # output
name = '4Class'

new_folder_path = os.path.join(root, name)

# Create the new directory if it does not exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

print('\nOutput folder: ', new_folder_path)

# importing the datasets
if upsample == False:
    train_path = os.path.join(data, 'training_dataset.pkl') # normal dataset path
else:
    train_path = os.path.join(data, 'upsample_training_dataset.pkl') # upsampled dataset path

print(f'\ntrain_path: {train_path}')
train_dataset = ds.BoostDataset(train_path) # calling the dataset class

val_path =  os.path.join(data, 'validation_dataset.pkl') # path
val_dataset = ds.BoostDataset(val_path) # calling the dataset class

# remove any classes that have been specified.
if remove_class != None:
    train_dataset.remove_class(remove_class)
    val_dataset.remove_class(remove_class)

mean_std_path = os.path.join(data,'mean_std.pkl')
print(f'mean_std_path: {mean_std_path}\n')

mean_std = pd.read_pickle(mean_std_path)
means = [mean_std[feature][0] for feature in mean_std.columns.tolist()]
stds = [mean_std[feature][1] for feature in mean_std.columns.tolist()]
mean_std = [means, stds]

num_classes = train_dataset.num_col('type')
input_size = train_dataset.input_size()

top_models_path = './GridSearch/DNN/'
tester = te.Tester(device=device, folder_path=top_models_path, test_dataset=val_dataset, mean_std=mean_std, num_cores=int(os.cpu_count()/2))
epoch, loss, params, state_dict = tester.load_model(1)

print(params)

model = nw.HiggsNet(
    input_size=input_size,
    hidden_nodes=int(params['hidden_nodes']),
    num_layers=int(params['num_layers']),
    activation_fn=params['activation_fn'],
    drop_p=0.03, # rounded to nearest 2 dp
    mean_std=mean_std,
    num_classes=num_classes
).to(device)

num_epochs = int(params['num_epochs'])

if weight == True:
    #weights_tensor = train_dataset.class_weights_calculator()
    weights_path = os.path.join(data, 'weights_tensor.npy')
    weights = np.load(weights_path)

    if remove_class != None: # if we have removed a class need to make sure we also remove the weights for the loss function
        weights = np.delete(weights, remove_class)
        weights_tensor = torch.tensor(weights).float()
    else:
        weights_tensor = torch.tensor(weights).float()

    class_labels = list(range(num_classes))  # print the weights used for each class
    for i, class_label in enumerate(class_labels):
        print(f"Weight for class {class_label}: {weights_tensor[i]}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor).to(device) # weight=weights_tensor
    print('Weights tensor implemented in loss function.')
else:
    criterion = nn.CrossEntropyLoss().to(device) # define loss function without class weights for evaluation
    print('No weights tensor in loss function.')

optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

num_cores = os.cpu_count() # use the number of cpu cores for the number of workers for the DataLoader
train_loader = DataLoader(train_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=True)#False,sampler=sampler) # using the loader
valid_loader = DataLoader(val_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=False) # using the loade
eval_train_loader = DataLoader(train_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=False)


# take a look at the model architecture
print('\nNetwork architecture:')
print(model)

# check that the training will happen on the GPU
if torch.device('cuda:0') == next(model.parameters()).device:
    print('\nNetwork has successfully built and is on the cuda device.')
else:
    print('\nWARNING: Network training is not using cuda device!\nTerminating training!')
    sys.exit(0) # the script terminates if we are not using a GPU. Due to the size of the dataset GPU use is necessary.

# Initialize a DataFrame to log training metrics
log = pd.DataFrame()#columns=['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy','train_acc_class0','valid_acc_class0','train_acc_class1','valid_acc_class1','train_acc_class2','valid_acc_class2','train_acc_class3','valid_acc_class3'])

# Calculate total number of samples per class in the train and valid datasets
total_in_class_train = train_dataset.countclasses()
#total_in_class_eval_train = train_dataset.countclasses()
total_in_class_valid = val_dataset.countclasses()


# Calculate the total number noise classes in the dataset
train_noise_total = sum(total_in_class_train[-(num_classes - 1):])
valid_noise_total = sum(total_in_class_valid[-(num_classes - 1):])

D_values = []
best_valid_loss = float('inf')  # Initialize with infinity
patience_counter = 0  # Counter for early stopping
delta = 0.0001  # Minimum change to qualify as an improvement
print('\nStarting training...')
#early_stopping = tr.EarlyStopping(patience=5, delta=0.001)  # Initialize early stopping
for epoch in range(num_epoch):
    start_time = time.time()
    train_losses = []
    train_correct = 0
    train_class_losses = [0] * num_classes
    train_correct_class = [0] * num_classes
    model.train()

    for data, targets, sample_weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        data = data.to(device)
        targets = targets.to(device)
        sample_weights = sample_weights.to(device)

        outputs = model(data)

        softmax_output = F.softmax(outputs, dim=1)
        _, predicted = torch.max(softmax_output.data, 1)
        train_correct += (predicted == targets).sum().item()

        for i in range(num_classes):
            class_outputs, class_targets, class_sample_weights = (arr[targets == i] for arr in [outputs, targets, sample_weights])
            class_loss = criterion(class_outputs, class_targets)
            class_loss = torch.mean(class_loss*class_sample_weights)
            train_class_losses[i] += class_loss.item()

        loss = criterion(outputs, targets)
        train_losses.append(loss.item())
        loss = torch.mean(loss*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
        #train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(num_classes):
            train_correct_class[i] += ((predicted == i) & (targets == i)).sum().item()

    train_loss = np.mean(train_losses)
    train_accuracy = train_correct / len(train_loader.dataset)
    train_acc_classes = [c / total for c, total in zip(train_correct_class, total_in_class_train)]

    # we compute all three types of F1 score
    train_f1_micro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='micro').cpu().item()
    train_f1_macro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='macro').cpu().item()
    train_f1_weighted = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='weighted').cpu().item() # need to move tensor to cpu to save to log later
    train_f1_None = multiclass_f1_score(outputs, targets, num_classes=num_classes, average=None).cpu().tolist()


    train_noise_correct = sum(train_correct_class[i] for i in range(1, num_classes))
    train_noise_accuracy = train_noise_correct / train_noise_total

    valid_losses = []
    valid_correct = 0
    valid_class_losses = [0] * num_classes
    valid_correct_class = [0] * num_classes

    model.eval()

    with torch.no_grad():
        for data, targets, sample_weights in tqdm(valid_loader, desc=f"Eval. valid data"):
            data = data.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)

            outputs = model(data)
            # Calculate valid loss per class

            for i in range(num_classes):
                class_outputs, class_targets, class_sample_weights = (arr[targets == i] for arr in [outputs, targets, sample_weights])
                class_loss = criterion(class_outputs, class_targets)
                class_loss = torch.mean(class_loss)# * class_sample_weights)
                valid_class_losses[i] += class_loss.item()

            loss = criterion(outputs, targets)
            loss = torch.mean(loss)# * sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
            valid_losses.append(loss.item())

            softmax_output = F.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_output.data, 1)
            valid_correct += (predicted == targets).sum().item()

            # Compute per-class correct predictions
            for i in range(num_classes):
                valid_correct_class[i] += ((predicted == i) & (targets == i)).sum().item()

    valid_loss = np.mean(valid_losses)
    valid_accuracy = valid_correct / len(valid_loader.dataset)
    valid_acc_classes = [c / total for c, total in zip(valid_correct_class, total_in_class_valid)]

    valid_f1_micro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='micro').cpu().item()
    valid_f1_macro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='macro').cpu().item()
    valid_f1_weighted = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='weighted').cpu().item()
    valid_f1_None = multiclass_f1_score(outputs, targets, num_classes=num_classes, average=None).cpu().tolist()

    valid_noise_correct = sum(valid_correct_class[i] for i in range(1, num_classes))
    valid_noise_accuracy = valid_noise_correct / valid_noise_total

    # saving everything to the log
    log.at[epoch, 'epoch'] = epoch + 1
    log.at[epoch, 'train_loss'] = train_loss
    log.at[epoch, 'valid_loss'] = valid_loss
    log.at[epoch, 'train_accuracy'] = train_accuracy
    log.at[epoch, 'valid_accuracy'] = valid_accuracy
    log.at[epoch, 'train_f1_micro'] = train_f1_micro
    log.at[epoch, 'valid_f1_micro'] = valid_f1_micro
    log.at[epoch, 'train_f1_macro'] = train_f1_macro
    log.at[epoch, 'valid_f1_macro'] = valid_f1_macro
    log.at[epoch, 'train_f1_weighted'] = train_f1_weighted
    log.at[epoch, 'valid_f1_weighted'] = valid_f1_weighted
    log.at[epoch, 'train_noise_acc'] = train_noise_accuracy
    log.at[epoch, 'valid_noise_acc'] = valid_noise_accuracy

    for i in range(num_classes):
        log.at[epoch, f'train_loss_class{i}'] = train_class_losses[i]
        log.at[epoch, f'valid_loss_class{i}'] = valid_class_losses[i]
        log.at[epoch, f'valid_acc_class{i}'] = valid_acc_classes[i]
        log.at[epoch, f'train_acc_class{i}'] = train_acc_classes[i]
        log.at[epoch, f'train_f1_None{i}'] = train_f1_None[i]
        log.at[epoch, f'valid_f1_None{i}'] = valid_f1_None[i]

    end_time = time.time()

    # printing updates
    print(f'\nTraining Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    print(f'Total Training Accuracy: {train_accuracy*100:.2f}%, Total Validation Accuracy: {valid_accuracy*100:.2f}%')

    print('\nF1 Scores')
    print(f'Training F1 score micro: {train_f1_micro:.4f}, Validation F1 score micro: {valid_f1_micro:.4f}')
    print(f'Training F1 score macro: {train_f1_macro:.4f}, Validation F1 score macro: {valid_f1_macro:.4f}')
    print(f'Training F1 score weighted: {train_f1_weighted:.4f}, Validation F1 score weighted: {valid_f1_weighted:.4f}')

    for i in range(num_classes):
        print(f'Class {i} Training F1 score: {train_f1_None[i]:.4f}, Validation F1 score: {valid_f1_None[i]:.4f}')


    print('\nAccuracies:')

    print(f'Signal training Accuracy: {train_acc_classes[0]*100:.2f}%, Signal validation Accuracy: {valid_acc_classes[0]*100:.2f}%')
    print(f'Total noise training Accuracy: {train_noise_accuracy*100:.2f}%, Total noise validation Accuracy: {valid_noise_accuracy*100:.2f}%\n')

    for i in range(1,num_classes):
        print(f'Class {i} Training accuracy: {train_acc_classes[i]*100:.2f}%, Validation accuracy: {valid_acc_classes[i]*100:.2f}%')

    print('\nLosses:')
    for i in range(num_classes):
        print(f'Class {i} Training loss: {train_class_losses[i]:.4f}, Validation loss: {valid_class_losses[i]:.4f}')

    print(f'\nTime for epoch: {end_time-start_time:.2f}s\n')

    loss_improved = best_valid_loss - valid_loss >= delta
    if loss_improved:
        best_valid_loss = valid_loss  # Update the best validation loss
        patience_counter = 0  # Reset patience counter
        print(f'Patience counter unchanged and is at {patience_counter}\n')
        # You could save the model here, if needed
        # torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1  # Increment patience counter
        print(f'Patience counter increased to {patience_counter}\n')
    if patience_counter >= 5:  # If we've waited for 5 epochs without improvement
        print("Early stopping triggered.")
        break  # Stop training

# saving the training log
filename = 'training_log.csv'
log_path = os.path.join(new_folder_path, filename)
log.to_csv(log_path, sep=',', index=False)

# we call the plotting script and generate our plots
#plotter = pl.plotting(log, new_folder_path, num_classes) # initialize the plotting class
#plotter.total_curves()
#plotter.SN_curves()
#plotter.per_class_loss_curves()
#plotter.per_class_acc_curves()
#plotter.training_all_f1()
#plotter.training_None_f1()
model_path = os.path.join(new_folder_path, 'DNN4Class.pth')
torch.save(model, model_path)
