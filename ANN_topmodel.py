import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # must be called before importing pyplot
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import multiclass_f1_score

# custom packages to import
import utils.data as ds
import utils.networks as nw
import utils.plotting as pl
import utils.testing as te

start_time1 = time.time()


data = './hyy_data/selected_features/' # directory where data is stored
root = './models/AN/' # output
name = '4Class'

upsample = True
weight = False
remove_class = None

new_folder_path = os.path.join(root, name)

# Create the new directory if it does not exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

print('\nOutput folder: ', new_folder_path)

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Cuda device: {torch.cuda.get_device_name(device_num)} has been found and set to default device.")

# importing the datasets
if upsample == False:
    train_path = os.path.join(data, 'training_dataset.pkl') # normal dataset path
else:
    train_path = os.path.join(data, 'upsample_training_dataset.pkl') # upsampled dataset path

print(f'\ntrain_path: {train_path}')
train_dataset = ds.BoostDataset_Adv(train_path) # calling the dataset class
train_cols = train_dataset.colcheck()
print(train_cols)
# we define a second instance of the training dataset that is never upsampled and is not shuffled, we can create analogous training curves fro the auxiliary network from this
eval_train_path = os.path.join(data, 'training_dataset.pkl')
eval_train_dataset = ds.BoostDataset_Adv(eval_train_path)
# we have a validation dataset that the discriminator has not seen
val_path =  os.path.join(data, 'validation_dataset.pkl') # path
val_dataset = ds.BoostDataset_Adv(val_path) # calling the dataset class

# remove any classes that have been specified.
if remove_class != None:
    train_dataset.remove_class(remove_class)
    eval_train_dataset.remove_class(remove_class)
    val_dataset.remove_class(remove_class)

mean_std_path = os.path.join(data,'mean_std.pkl')
print(f'mean_std_path: {mean_std_path}')

mean_std = pd.read_pickle(mean_std_path)
means = [mean_std[feature][0] for feature in mean_std.columns.tolist()]
stds = [mean_std[feature][1] for feature in mean_std.columns.tolist()]
mean_std = [means, stds]

top_models_path = './GridSearch/AN/'
tester = te.Tester(device=device, folder_path=top_models_path, test_dataset=train_dataset, mean_std=mean_std, num_cores=int(os.cpu_count()/2))
epoch, loss, params, state_dict = tester.load_model_ANN(1)
print(epoch)
print(params)

num_cores = os.cpu_count() # use the number of cpu cores for the number of workers for the DataLoader
train_loader = DataLoader(train_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=True)
eval_train_loader = DataLoader(eval_train_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=False)
valid_loader = DataLoader(val_dataset,batch_size=int(params['batch_size']),num_workers=num_cores,shuffle=False) # using the loader

num_classes = train_dataset.num_col('type')
cM_ind = train_dataset.column_index('FatElectron_cM_bins')
input_size = train_dataset.input_size()

# Calculate total number of samples per class in the train and valid datasets
#total_in_class_train = train_dataset.countclasses()
total_in_class_eval_train = eval_train_dataset.countclasses()
total_in_class_valid = val_dataset.countclasses()

# Calculate the total number noise classes in the dataset
train_noise_total = sum(total_in_class_eval_train[-(num_classes - 1):])
valid_noise_total = sum(total_in_class_valid[-(num_classes - 1):])

if weight == True:
    #weights_tensor = train_dataset.class_weights_calculator()
    weights_path = os.path.join(data, 'weights_tensor.npy')
    weights = np.load(weights_path)

    if remove_class != None: # if we have removed a class need to make sure we also remove the weights for the loss function
        weights = np.delete(weights, remove_class)
        weights_tensor = torch.tensor(weights).float()

    weights_tensor = torch.tensor(weights).float()
    class_labels = list(range(num_classes))  # print the weights used for each class
    for i, class_label in enumerate(class_labels):
        print(f'Weight for class {class_label}: {weights_tensor[i]}')
    criterion_aux = nn.CrossEntropyLoss(weight=weights_tensor).to(device) # weight=weights_tensor
    print('\nWeights tensor implemented in loss function.')
else:
    criterion_aux = nn.CrossEntropyLoss().to(device) # define loss function without class weights for evaluation
    print('No weights tensor in loss function.')


lambdas = [params['lambda_adv']] #np.array([100])
print(f'lambdas: {lambdas}')

for lambda_adv in tqdm(lambdas,desc='lambda search'):
    print(f'lambda for this run: {lambda_adv}')
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
    # loss functions
    adv_loss = nn.CrossEntropyLoss()  # Mean Squared Error for the adversary.
    # for the discriminator

    # Optimizers
    optimizer_A = optim.Adam(adversary.parameters(), lr=params['learning_rate_A'], weight_decay=params['weight_decay_A'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params['learning_rate_D'], weight_decay=params['weight_decay_D'])

    # Check if both adversary and discriminator are on the same device, and that device is CUDA
    adversary_device = next(adversary.parameters()).device
    discriminator_device = next(discriminator.parameters()).device

    if adversary_device == discriminator_device and adversary_device == torch.device('cuda:0'):
        print('\nNetwork has successfully built and is on the cuda device.\n')
    else:
        print('\nWARNING: Network training is not using cuda device!\nTerminating training!')
        sys.exit(0)  # the script terminates if we are not using a GPU. Due to the size of the dataset GPU use is necessary.
    #decreasing_count = 0

    log = pd.DataFrame() # initialize the log

    for w_epoch in range(5):
        print(f'Warm-up: {w_epoch+1}/{5}')
        loss_total_A = []
        for data, targets, sample_weights, weights_adv, _ in tqdm(train_loader, desc=f"Warming"):
            cM_ind_values = data[:, cM_ind].to(device)
            targets = targets.to(device).view(-1, 1).float()  # Ensure the targets are in the same shape as the predictions
            #targets = warmup_vals.to(self.device)
            sample_weights = sample_weights.to(device)
            weights_adv = weights_adv.to(device)

            # Training the adversary on true targets
            outputs_A = adversary(targets)
            loss_A = adv_loss(outputs_A.squeeze(), cM_ind_values)
            loss_A = torch.mean(loss_A * weights_adv)  # Apply sample weights if necessary

            # Backpropagate the loss for the adversary
            optimizer_A.zero_grad()
            loss_A.backward()
            optimizer_A.step()

    count = 0
    delta = 0.001
    best_tpr_mean = float('-inf')
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        #print(f'Decreasing count at: {decreasing_count}')
        start_time = time.time()

        loss_D = []
        loss_A = []
        loss_total = []

        discriminator.train()
        adversary.train()
        # Using tqdm for progress tracking
        for data, labels, sample_weights, weights_adv, _ in tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}"):
            data = data.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            cM_ind_values = data[:, cM_ind].to(device)
            weights_adv = weights_adv.to(device)

            # Forward pass through discriminator
            discriminator_output_raw = discriminator(data)
            softmax_output = F.softmax(discriminator_output_raw, dim=1)
            _, discriminator_predicted = torch.max(softmax_output.data, 1)
            discriminator_predicted = discriminator_predicted.view(-1,1).float()
            # Discriminator training
            optimizer_D.zero_grad()
            loss_discriminator = criterion_aux(discriminator_output_raw, labels)
            loss_discriminator = torch.mean(loss_discriminator*sample_weights) # weighting for pT bins
            loss_D.append(loss_discriminator.item())
            # Adversary's prediction and loss
            adversary_predictions = adversary(discriminator_predicted)
            loss_adversary = adv_loss(adversary_predictions.squeeze(), cM_ind_values)
            loss_adversary = torch.mean(loss_adversary*weights_adv)
            loss_A.append(loss_adversary.item())
            # Incorporate the adversary loss into the discriminator's loss
            total_loss = loss_discriminator - lambda_adv * loss_adversary
            loss_total.append(total_loss.item())
            total_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Adversary training
            optimizer_A.zero_grad()
            loss_adversary.backward()
            optimizer_A.step()

        loss_D = np.mean(loss_D)
        loss_A = np.mean(loss_A)
        loss_total = np.mean(loss_total)

        train_losses = []
        train_correct = 0
        train_class_losses = [0] * num_classes
        train_correct_class = [0] * num_classes

        valid_losses = []
        valid_correct = 0
        valid_class_losses = [0] * num_classes
        valid_correct_class = [0] * num_classes

        discriminator.eval()

        with torch.no_grad():

            for data, targets, sample_weights, _, _ in tqdm(eval_train_loader, desc=f"Eval. train data"):
                data = data.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)

                outputs = discriminator(data)

                for i in range(num_classes):
                    class_outputs, class_targets, class_sample_weights = (arr[targets == i] for arr in [outputs, targets, sample_weights])
                    class_loss = criterion_aux(class_outputs, class_targets)
                    class_loss = torch.mean(class_loss)# * class_sample_weights)
                    train_class_losses[i] += class_loss.item()

                loss = criterion_aux(outputs, targets)
                #loss = torch.mean(loss * sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
                train_losses.append(loss.item())

                softmax_output = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_output.data, 1)
                train_correct += (predicted == targets).sum().item()

                # Compute per-class correct predictions
                for i in range(num_classes):
                    train_correct_class[i] += ((predicted == i) & (targets == i)).sum().item()

            train_loss = np.mean(train_losses)
            train_accuracy = train_correct / len(eval_train_loader.dataset)
            train_acc_classes = [c / total for c, total in zip(train_correct_class, total_in_class_eval_train)]

            # we compute all three types of F1 score
            train_f1_micro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='micro').cpu().item()
            train_f1_macro = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='macro').cpu().item()
            train_f1_weighted = multiclass_f1_score(outputs, targets, num_classes=num_classes, average='weighted').cpu().item() # need to move tensor to cpu to save to log later
            train_f1_None = multiclass_f1_score(outputs, targets, num_classes=num_classes, average=None).cpu().tolist()


            train_noise_correct = sum(train_correct_class[i] for i in range(1, num_classes))
            train_noise_accuracy = train_noise_correct / train_noise_total

            all_targets = []
            all_predicted_D = []
            all_cM_ind_values = []
            bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
            for data, targets, sample_weights, _, _ in tqdm(valid_loader, desc=f"Eval. valid data"):
                data = data.to(device)
                targets = targets.to(device)
                cM_ind_values = data[:, cM_ind].to(device)
                sample_weights = sample_weights.to(device)
                all_targets.extend(targets.cpu().numpy())
                all_cM_ind_values.extend(cM_ind_values.cpu().numpy())

                outputs = discriminator(data)
                # Calculate valid loss per class

                for i in range(num_classes):
                    class_outputs, class_targets, class_sample_weights = (arr[targets == i] for arr in [outputs, targets, sample_weights])
                    class_loss = criterion_aux(class_outputs, class_targets)
                    class_loss = torch.mean(class_loss)# * class_sample_weights)
                    valid_class_losses[i] += class_loss.item()

                loss = criterion_aux(outputs, targets)
                loss = torch.mean(loss)# * sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
                valid_losses.append(loss.item())

                softmax_output = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_output.data, 1)
                valid_correct += (predicted == targets).sum().item()

                predicted_D = predicted.view(-1, 1).float()
                all_predicted_D.extend(predicted_D.cpu().numpy())

                # Compute per-class correct predictions
                for i in range(num_classes):
                    valid_correct_class[i] += ((predicted == i) & (targets == i)).sum().item()

        bins = torch.tensor(np.arange(len(bins) - 1)).to(device)
        tprs = []

        all_targets = torch.tensor(np.array(all_targets)).to(device)
        all_predicted_D = torch.tensor(np.array(all_predicted_D)).squeeze().to(device)
        all_cM_ind_values = torch.tensor(np.array(all_cM_ind_values)).to(device)

        for b in tqdm(range(len(bins)), desc="STD"):
            mask = all_cM_ind_values == b
            true_in_bin = all_targets[mask]
            preds_in_bin = all_predicted_D[mask]
            true_positives = torch.sum((preds_in_bin.squeeze() == 0) & (true_in_bin == 0))
            total_true_zeros = torch.sum(true_in_bin == 0)
            TPR = true_positives.float() / total_true_zeros.float() if total_true_zeros > 0 else torch.tensor(float('nan')).to(device)
            tprs.append(TPR.cpu().item())

        tprs = np.array(tprs)

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
        log.at[epoch, 'loss_D'] = loss_D
        log.at[epoch, 'loss_A'] = loss_A
        log.at[epoch, 'loss_total'] = loss_total
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
        print(f'\nAdversary loss: {loss_A:.4f}, Discriminator loss: {loss_D:.4f}, total loss: {loss_total:.4f}]')
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        print(f'Total Training Accuracy: {train_accuracy*100:.2f}%, Total Validation Accuracy: {valid_accuracy*100:.2f}%')

        print('\nF1 Scores:')
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

        # saving the training log
        filename = 'training_log.csv'
        log_path = os.path.join(new_folder_path, filename)
        log.to_csv(log_path, sep=',', index=False)
        model_path = os.path.join(new_folder_path, 'ANN4Class.pth')
        torch.save(discriminator, model_path)

        print(f'Signal efficiencies: {tprs}')
        print(f'STD: {np.sqrt(np.nanvar(tprs))}')
        print(f'\nTime for epoch: {end_time-start_time:.2f}s\n')

        tpr_mean = np.nanmean(tprs)
        if tpr_mean > best_tpr_mean + delta:
            best_tpr_mean = tpr_mean
            count = 0
        else:
            count += 1
            print(f'Count increased to {count}')
            if count >= 5:
                break

    filename = 'training_log.csv'
    log_path = os.path.join(new_folder_path, filename)
    log.to_csv(log_path, sep=',', index=False)

    model_path = os.path.join(new_folder_path, 'ANN4Class.pth')
    torch.save(discriminator, model_path)
    end_time1 = time.time()
    print(f'\nAll tasks completed! \nTime taken for all tasks: {(end_time1 - start_time1)/60:.2f}mins\n')
