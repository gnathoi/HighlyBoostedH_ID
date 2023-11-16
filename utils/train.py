import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# We define this very useful and lightweight class for the GridSearch.
# Once GridSearch is completed we will rerun the training with optimized hyperparameters
# using a heavier duty training script.
# We use this to evaluate the best model based on signal accuracy
class Trainer:
    def __init__(self, model, device, train_loader, valid_loader, val_dataset, optimizer, criterion):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.model.to(self.device)

        # Calculate total number of samples per class in the train and valid datasets
        #self.total_in_class_valid = self.val_dataset.countclasses()

    def train_epoch(self,epoch):
        self.model.train()
        train_losses = []
        #train_correct = 0
        for data, targets, sample_weights in tqdm(self.train_loader, desc=f"Epoch {epoch+1} training"):
            data = data.to(self.device)
            targets = targets.to(self.device)
            sample_weights = sample_weights.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss = torch.mean(loss*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
            train_losses.append(loss.item())

            #_, predicted = torch.max(outputs.data, 1)
            #train_correct += (predicted == targets).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss = np.mean(train_losses)
            #train_accuracy = train_correct / len(self.train_loader.dataset)

        return train_loss#, train_accuracy

    def validate(self):
        self.model.eval()
        valid_losses = []
        with torch.no_grad():
            for data, targets, sample_weights in tqdm(self.valid_loader,desc=f"Validating"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                sample_weights = sample_weights.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                #loss = torch.mean(loss * sample_weights)  # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
                valid_losses.append(loss.item())

        valid_loss = np.mean(valid_losses)
        return valid_loss

    def train(self,num_epochs):
        early_stopping = EarlyStopping(patience=5, delta=0.001)  # Initialize early stopping
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validate()

            print(f'\nEpoch: {epoch+1}, validation loss: {valid_loss:.4f}\n')

            if early_stopping(valid_loss):
                print(f"Early stopping activated at epoch {epoch+1}")
                return epoch + 1, valid_loss  # Return the epoch at which training was stopped and the last validation loss

        return None, valid_loss

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.count = 0
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop

def save_top_models(top_models,new_folder_path):
    for i, (stop_epoch, val_loss, params, state_dict) in enumerate(top_models):
        model_saves = os.path.join(new_folder_path, 'state_dicts')
        # Create the new directory if it does not exist
        if not os.path.exists(model_saves):
            os.makedirs(model_saves)
        # Save model
        model_path = os.path.join(model_saves, f'top_model_{i+1}.pth')
        torch.save(state_dict, model_path)
        # Save hyperparameters and validation loss
        params_path = os.path.join(new_folder_path, f'top_model_{i+1}.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump({"val_loss": val_loss,"stop_epoch": stop_epoch, "hyperparameters": params}, f)

def load_top_models(folder_path):
    top_models = []
    for i in range(1, 6):  # Assuming you save top 5 models
        state_dict_path = os.path.join(folder_path, 'state_dicts', f'top_model_{i}.pth')
        params_path = os.path.join(folder_path, f'top_model_{i}.pkl')

        if os.path.exists(state_dict_path) and os.path.exists(params_path):
            state_dict = torch.load(state_dict_path)
            with open(params_path, 'rb') as f:
                metadata = pickle.load(f)

            top_models.append((metadata['val_loss'],metadata['stop_epoch'], metadata['hyperparameters'], state_dict))
    return top_models


class Trainer_Adv:
    def __init__(self, cM_ind, lambda_adv, model_D, model_A, device, train_loader, valid_loader, val_dataset, optimizer_D, optimizer_A, criterion_D, criterion_A, bins):
        self.cM_ind = cM_ind
        self.lambda_adv = lambda_adv
        self.model_D = model_D
        self.model_A = model_A
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer_D = optimizer_D
        self.optimizer_A = optimizer_A
        self.criterion_D = criterion_D
        self.criterion_A = criterion_A

        self.bins = bins

        self.model_D.to(self.device)
        self.model_A.to(self.device)

        self.warmed_up = False

        # Calculate total number of samples per class in the train and valid datasets
        #self.total_in_class_valid = self.val_dataset.countclasses()

    def train_epoch(self,epoch):
        self.model_D.train()
        self.model_A.train()
        #losses_D = []
        #losses_A = []
        loss_total = []
        #train_correct = 0
        for data, targets, sample_weights, weights_adv, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1} training"):
            data = data.to(self.device)
            cM_ind_values = data[:, self.cM_ind].to(self.device)
            targets = targets.to(self.device)
            sample_weights = sample_weights.to(self.device)
            weights_adv = weights_adv.to(self.device)

            outputs_D = self.model_D(data)

            softmax_output = F.softmax(outputs_D, dim=1)
            _, predicted_D = torch.max(softmax_output.data, 1)
            predicted_D = predicted_D.view(-1,1).float()

            loss_D = self.criterion_D(outputs_D, targets)
            loss_D = torch.mean(loss_D*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.
            #train_losses.append(loss.item())
            #_, predicted_D = torch.max(outputs_D.data, 1)
            #predicted_D = predicted_D.view(-1,1).float()

            outputs_A = self.model_A(predicted_D)
            loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
            loss_A = torch.mean(loss_A*weights_adv)
            combined_loss = loss_D - self.lambda_adv * loss_A

            # discriminator training
            self.optimizer_D.zero_grad()
            combined_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # adversary training
            self.optimizer_A.zero_grad()
            loss_A.backward()
            self.optimizer_A.step()

        loss_total = np.mean(combined_loss.detach().cpu().numpy())
        return loss_total#, train_accuracy
    '''
    def validate(self):
        self.model_D.eval()
        self.model_A.eval()
        valid_losses = []
        with torch.no_grad():
            for data, targets, sample_weights, weights_adv in tqdm(self.valid_loader,desc=f"Validating"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                cM_ind_values = data[:, self.cM_ind].to(self.device)
                sample_weights = sample_weights.to(self.device)
                weights_adv = weights_adv.to(self.device)

                outputs_D = self.model_D(data)

                softmax_output = F.softmax(outputs_D, dim=1)
                _, predicted_D = torch.max(softmax_output.data, 1)
                predicted_D = predicted_D.view(-1,1).float()

                loss = self.criterion_D(outputs_D, targets)
                valid_losses.append(loss.item())
                loss_D = torch.mean(loss*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.

                outputs_A = self.model_A(predicted_D)
                loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
                loss_A = torch.mean(loss_A*weights_adv)
                combined_loss = loss_D - self.lambda_adv * loss_A
                #loss = torch.mean(loss * sample_weights)  # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.


        valid_loss = np.mean(valid_losses)
        return valid_loss, combined_loss
        '''
    def validate(self):
        self.model_D.eval()
        valid_losses = []
        all_targets = []
        all_predicted_D = []
        all_cM_ind_values = []

        with torch.no_grad():
            for data, targets, _, _, _ in tqdm(self.valid_loader, desc=f"Validating"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                cM_ind_values = data[:, self.cM_ind].to(self.device)

                all_targets.extend(targets.cpu().numpy())
                all_cM_ind_values.extend(cM_ind_values.cpu().numpy())

                outputs_D = self.model_D(data)
                softmax_output = F.softmax(outputs_D, dim=1)
                _, predicted_D = torch.max(softmax_output.data, 1)
                predicted_D = predicted_D.view(-1, 1).float()
                all_predicted_D.extend(predicted_D.cpu().numpy())

                loss = self.criterion_D(outputs_D, targets)
                valid_losses.append(loss.item())

        valid_loss = np.mean(valid_losses)

        bins = torch.tensor(np.arange(len(self.bins) - 1)).to(self.device)
        tprs = []

        all_targets = torch.tensor(np.array(all_targets)).to(self.device)
        all_predicted_D = torch.tensor(np.array(all_predicted_D)).squeeze().to(self.device)
        all_cM_ind_values = torch.tensor(np.array(all_cM_ind_values)).to(self.device)

        for b in tqdm(range(len(bins)), desc="STD"):
            mask = all_cM_ind_values == b
            true_in_bin = all_targets[mask]
            preds_in_bin = all_predicted_D[mask]
            true_positives = torch.sum((preds_in_bin.squeeze() == 0) & (true_in_bin == 0))
            total_true_zeros = torch.sum(true_in_bin == 0)
            TPR = true_positives.float() / total_true_zeros.float() if total_true_zeros > 0 else torch.tensor(float('nan')).to(self.device)
            tprs.append(TPR.cpu().item())

        tpr_var = np.nanvar(tprs)
        tpr_std = np.sqrt(tpr_var)
        tpr_mean = np.nanmean(tprs)
        print(f'TPRs arcross bins:\n{tprs}')
        #print(f'mean tpr: {tpr_mean}')
        return valid_loss, tpr_std, tpr_mean

    def train(self,num_epochs):
        early_stopping = EarlyStopping_adv(patience=5, delta=0.001)  # Initialize early stopping
        for epoch in range(num_epochs):
            if self.warmed_up == False:
                self.warmup()

            train_loss = self.train_epoch(epoch)
            valid_loss, tpr_std, tpr_mean = self.validate()

            print(f'\nEpoch: {epoch+1}, Validation Loss D: {valid_loss:.4f}, validation TPR mean: {tpr_mean:.4f}, Validation TPR std: {tpr_std:.4f}\n')

            if early_stopping(tpr_mean=tpr_mean, epoch=epoch):
                print(f"Early stopping activated at epoch {epoch+1}")
                return epoch + 1, valid_loss, tpr_std, tpr_mean

        return None, valid_loss, tpr_std, tpr_mean

    def warmup(self, num_warm_up=5):
        self.model_A.train()  # Set the model to training mode
        for w_epoch in range(num_warm_up):
            print(f'Warm-up: {w_epoch+1}/{num_warm_up}')
            loss_total_A = []
            for data, targets, sample_weights, weights_adv, _ in tqdm(self.train_loader, desc=f"Warming"):
                cM_ind_values = data[:, self.cM_ind].to(self.device)
                targets = targets.to(self.device).view(-1, 1).float()  # Ensure the targets are in the same shape as the predictions
                #targets = warmup_vals.to(self.device)
                sample_weights = sample_weights.to(self.device)
                weights_adv = weights_adv.to(self.device)

                # Training the adversary on true targets
                outputs_A = self.model_A(targets)
                loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
                loss_A = torch.mean(loss_A * weights_adv)  # Apply sample weights if necessary

                # Backpropagate the loss for the adversary
                self.optimizer_A.zero_grad()
                loss_A.backward()
                self.optimizer_A.step()

                loss_total_A.append(loss_A.detach().cpu().numpy())

            # Calculate the mean loss over all batches
            loss_mean_A = np.mean(loss_total_A)
            print(f'\nWarm-up Adv. Loss: {loss_mean_A:.4f}\n')
        self.warmed_up = True

class EarlyStopping_adv:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_tpr_mean = float('-inf')  # Initialize to negative infinity
        self.count = 0
        self.stop = False

    def __call__(self, tpr_mean, epoch):
        if tpr_mean > self.best_tpr_mean + self.delta:
            self.best_tpr_mean = tpr_mean
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

        return self.stop

def save_top_models_adv(top_models,new_folder_path):
    for i, (stop_epoch, val_loss_D, tpr_mean, params, state_dict) in enumerate(top_models):
        model_saves = os.path.join(new_folder_path, 'state_dicts')
        # Create the new directory if it does not exist
        if not os.path.exists(model_saves):
            os.makedirs(model_saves)
        # Save model
        model_path = os.path.join(model_saves, f'top_model_{i+1}.pth')
        torch.save(state_dict, model_path)
        # Save hyperparameters and validation loss
        params_path = os.path.join(new_folder_path, f'top_model_{i+1}.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump({"tpr_mean":-tpr_mean,"stop_epoch": stop_epoch, "val_loss_D": val_loss_D, "hyperparameters": params}, f)

def load_top_models_adv(folder_path):
    top_models = []
    for i in range(1, 6):  # Assuming you save top 5 models
        state_dict_path = os.path.join(folder_path, 'state_dicts', f'top_model_{i}.pth')
        params_path = os.path.join(folder_path, f'top_model_{i}.pkl')

        if os.path.exists(state_dict_path) and os.path.exists(params_path):
            state_dict = torch.load(state_dict_path)
            with open(params_path, 'rb') as f:
                metadata = pickle.load(f)

            top_models.append((-metadata['tpr_mean'],metadata['stop_epoch'],metadata['val_loss_D'], metadata['hyperparameters'], state_dict))
    return top_models



class Trainer_Adv_LOG:
    def __init__(self, cM_ind, lambda_adv, model_D, model_A, device, train_loader, valid_loader, val_dataset, optimizer_D, optimizer_A, criterion_D, criterion_A, bins):
        self.cM_ind = cM_ind
        self.lambda_adv = lambda_adv
        self.model_D = model_D
        self.model_A = model_A
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer_D = optimizer_D
        self.optimizer_A = optimizer_A
        self.criterion_D = criterion_D
        self.criterion_A = criterion_A

        self.bins = bins

        self.model_D.to(self.device)
        self.model_A.to(self.device)

        self.warmed_up = False

        # Calculate total number of samples per class in the train and valid datasets
        #self.total_in_class_valid = self.val_dataset.countclasses()

    def train_epoch(self,epoch):
        self.model_D.train()
        self.model_A.train()
        #losses_D = []
        #losses_A = []
        loss_total = []
        #losses_D = []
        #train_correct = 0
        for data, targets, sample_weights, weights_adv, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1} training"):
            data = data.to(self.device)
            cM_ind_values = data[:, self.cM_ind].to(self.device)
            targets = targets.to(self.device)
            sample_weights = sample_weights.to(self.device)
            weights_adv = weights_adv.to(self.device)

            outputs_D = self.model_D(data)

            softmax_output = F.softmax(outputs_D, dim=1)
            _, predicted_D = torch.max(softmax_output.data, 1)
            predicted_D = predicted_D.view(-1,1).float()

            loss_D = self.criterion_D(outputs_D, targets)
            #losses_D.append(loss_D.item())
            loss_D = torch.mean(loss_D*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.

            outputs_A = self.model_A(predicted_D)
            loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
            loss_A = torch.mean(loss_A*weights_adv)
            combined_loss = loss_D - self.lambda_adv * loss_A

            # discriminator training
            self.optimizer_D.zero_grad()
            combined_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # adversary training
            self.optimizer_A.zero_grad()
            loss_A.backward()
            self.optimizer_A.step()

        loss_total = np.mean(combined_loss.detach().cpu().numpy())
        #loss_D = np.mean(losses_D)
        return loss_total#, loss_D
    '''
    def validate(self):
        self.model_D.eval()
        self.model_A.eval()
        valid_losses = []
        with torch.no_grad():
            for data, targets, sample_weights, weights_adv in tqdm(self.valid_loader,desc=f"Validating"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                cM_ind_values = data[:, self.cM_ind].to(self.device)
                sample_weights = sample_weights.to(self.device)
                weights_adv = weights_adv.to(self.device)

                outputs_D = self.model_D(data)

                softmax_output = F.softmax(outputs_D, dim=1)
                _, predicted_D = torch.max(softmax_output.data, 1)
                predicted_D = predicted_D.view(-1,1).float()

                loss = self.criterion_D(outputs_D, targets)
                valid_losses.append(loss.item())
                loss_D = torch.mean(loss*sample_weights) # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.

                outputs_A = self.model_A(predicted_D)
                loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
                loss_A = torch.mean(loss_A*weights_adv)
                combined_loss = loss_D - self.lambda_adv * loss_A
                #loss = torch.mean(loss * sample_weights)  # the sample weights are created in the BoostedDataset class from weight_ey_2d, here we apply that weighting.


        valid_loss = np.mean(valid_losses)
        return valid_loss, combined_loss
        '''
    def validate(self):
        self.model_D.eval()
        valid_losses = []
        train_losses = []
        train_correct = 0
        valid_correct = 0


        # Initialize train_correct_class and valid_correct_class to 0 for each class before entering the loop.
        train_correct_class = [0] * num_classes
        valid_correct_class = [0] * num_classes

        # ... (rest of your code)

        for loader, desc in [(self.train_loader, 'Training'), (self.valid_loader, 'Validating')]:
            all_targets = []
            all_predicted_D = []

            for data, targets, _, _, _ in tqdm(loader, desc=f"{desc}"):
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs_D = self.model_D(data)
                softmax_output = F.softmax(outputs_D, dim=1)
                _, predicted_D = torch.max(softmax_output.data, 1)

                correct = (predicted_D == targets).sum().item()

                for i in range(num_classes):
                    correct_class_i = ((predicted_D == i) & (targets == i)).sum().item()

                    if desc == 'Training':
                        train_correct_class[i] += correct_class_i
                    else:
                        valid_correct_class[i] += correct_class_i

                if desc == 'Training':
                    train_correct += correct
                else:
                    valid_correct += correct

                predicted_D = predicted_D.view(-1, 1).float()
                all_predicted_D.extend(predicted_D.cpu().numpy())

                # ... (rest of your code)

        # After the loop, calculate the accuracy based on train_correct and valid_correct.
        train_accuracy = (train_correct / len(self.train_loader.dataset)) * 100
        valid_accuracy = (valid_correct / len(self.valid_loader.dataset)) * 100



        return train_loss, valid_loss, train_accuracy, valid_accuracy, train_correct_class, valid_correct_class

    def train(self,num_epochs):
        early_stopping = EarlyStopping_adv(patience=5, delta=0.001)  # Initialize early stopping
        log = pd.DataFrame()
        for epoch in range(num_epochs):
            if self.warmed_up == False:
                self.warmup()

            train_loss = self.train_epoch(epoch)
            valid_loss, tpr_std, tpr_mean = self.validate()

            print(f'\nEpoch: {epoch+1}, Validation Loss D: {valid_loss:.4f}, validation TPR mean: {tpr_mean:.4f}, Validation TPR std: {tpr_std:.4f}\n')

            if early_stopping(tpr_mean=tpr_mean, epoch=epoch):
                print(f"Early stopping activated at epoch {epoch+1}")
                return epoch + 1, valid_loss, tpr_std, tpr_mean

        return None, valid_loss, tpr_std, tpr_mean

    def warmup(self, num_warm_up=5):
        self.model_A.train()  # Set the model to training mode
        for w_epoch in range(num_warm_up):
            print(f'Warm-up: {w_epoch+1}/{num_warm_up}')
            loss_total_A = []
            for data, targets, sample_weights, weights_adv, _ in tqdm(self.train_loader, desc=f"Warming"):
                cM_ind_values = data[:, self.cM_ind].to(self.device)
                targets = targets.to(self.device).view(-1, 1).float()  # Ensure the targets are in the same shape as the predictions
                #targets = warmup_vals.to(self.device)
                sample_weights = sample_weights.to(self.device)
                weights_adv = weights_adv.to(self.device)

                # Training the adversary on true targets
                outputs_A = self.model_A(targets)
                loss_A = self.criterion_A(outputs_A.squeeze(), cM_ind_values)
                loss_A = torch.mean(loss_A * weights_adv)  # Apply sample weights if necessary

                # Backpropagate the loss for the adversary
                self.optimizer_A.zero_grad()
                loss_A.backward()
                self.optimizer_A.step()

                loss_total_A.append(loss_A.detach().cpu().numpy())

            # Calculate the mean loss over all batches
            loss_mean_A = np.mean(loss_total_A)
            print(f'\nWarm-up Adv. Loss: {loss_mean_A:.4f}\n')
        self.warmed_up = True
