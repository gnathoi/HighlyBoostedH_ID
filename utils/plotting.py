import os
import pandas as pd
import numpy as np
import seaborn as sns
import mplhep as hep
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
'''
# Change the overall theme, context and palette
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1)
sns.set_palette("pastel")
'''

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import matplotlib
matplotlib.use('Agg')  # must be called before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.lines import Line2D
plt.rcdefaults()

if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

class plotting():
    def __init__(self, log, new_folder_path, num_classes):
        self.log = log
        self.new_folder_path = new_folder_path
        self.num_classes = num_classes

        self.labels = [r'$\rm H \rightarrow \gamma\gamma $',r'$\rm Z \rightarrow e^+e^- $', r'$\rm e/\gamma $', r'$\rm (\tau)\tau$']
        self.colors = ['#4462a5', '#689aff', '#c50807', '#68cd67', 'tab:orange', 'tab:purple', 'tab:green']
        self.markers = ['o', 'v', 's', '^']
        self.linestyle = ['solid', (0, (1, 1)), 'dashed', (0, (5, 1)), 'solid', 'solid', 'solid', 'solid']
        self.atlas_label = 'Simulation'# Preliminary'

    def total_curves(self):
        print('Generating total training curves...')
        # total training curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.log['epoch'], self.log['train_loss'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_loss'], label='Validation')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.log['epoch'], self.log['train_accuracy'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_accuracy'], label='Validation')
        plt.title('Total Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        '''
                # Plot training and validation loss
        plt.subplot(1, 2, 1)
        sns.lineplot(x='epoch', y='value', hue='variable',
                     data=pd.melt(self.log, ['epoch'], value_vars=['train_loss', 'valid_loss']))
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(labels=['Training', 'Validation'])  # Update the legend labels

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        sns.lineplot(x='epoch', y='value', hue='variable',
                     data=pd.melt(self.log, ['epoch'], value_vars=['train_accuracy', 'valid_accuracy']))
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(labels=['Training', 'Validation'])  # Update the legend labels
        '''

        plt.tight_layout()

        filename = 'training_curves.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')


    def lossandsignal(self):
        print('Generating total training curves...')
        # total training curves
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.log['epoch'], self.log['train_loss'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_loss'], label='Validation')
        #plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch',fontsize=24)
        plt.ylabel('Loss',fontsize=24)
        plt.xlim([0,20])
        #plt.xticks(ticks=range(0, 31, 2), labels=[str(x) for x in range(0, 31, 2)])
        plt.gca().xaxis.set_major_locator(FixedLocator(range(0, 19, 5)))
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=20,loc='upper right')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.875), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.subplot(1, 2, 2)
        plt.plot(self.log['epoch'], self.log['train_acc_class0']*100, label='Train')
        plt.plot(self.log['epoch'], self.log['valid_acc_class0']*100, label='Validation')
        #plt.title('Total Accuracy vs. Epoch')
        plt.xlabel('Epoch',fontsize=24)
        plt.ylabel('Signal Accuracy [%]',fontsize=24)
        plt.xlim([0,20])
        plt.gca().xaxis.set_major_locator(FixedLocator(range(0, 19, 5)))
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=20,loc='lower right')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.875), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()

        filename = 'training_LandScurves.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def SN_curves(self):
        print('Generating signal and noise training curves...')
        # Plotting training and validation accuracies for class 0
        '''

        plt.figure(figsize=(10,6))

        plt.plot(self.log['epoch'], self.log['train_loss_class0'], label='Training')
        plt.plot(self.log['epoch'], self.log['valid_loss_class0'], label='Validation')

        plt.title('Training and Validation Losses for Class 0')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        filename = 'training_loss_curves_class0.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)

        # Plotting training and validation accuracies for class 0
        plt.figure(figsize=(10,6))

        plt.plot(self.log['epoch'], self.log['train_acc_class0'], label='Training')
        plt.plot(self.log['epoch'], self.log['valid_acc_class0'], label='Validation')

        plt.title('Validation Accuracy for Class 0')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        '''
        plt.figure(figsize=(12, 5))


        plt.subplot(1, 2, 1)
        plt.plot(self.log['epoch'], self.log['train_acc_class0'], label='Training')
        plt.plot(self.log['epoch'], self.log['valid_acc_class0'], label='Validation')
        plt.title('Signal Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.log['epoch'], self.log['train_noise_acc'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_noise_acc'], label='Validation')
        plt.title('Noise Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.tight_layout()

        filename = 'SN_training_curves.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def per_class_loss_curves(self):
        print('Generating per class loss training curves...')
        # tile the curves per class
        fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # adjust size as needed

        # Flatten axs for easy iteration
        axs = axs.ravel()

        for i in range(self.num_classes):  # adjust range if more classes
            axs[i].plot(self.log['epoch'], self.log[f'train_loss_class{i}'], label='Training')
            axs[i].plot(self.log['epoch'], self.log[f'valid_loss_class{i}'], label='Validation')

            axs[i].set_title(f'Loss for Class {i}')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Loss')
            axs[i].legend()

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()  # adjust subplot parameters for better spacing

        filename = 'training_loss_curves_all_classes_loss.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')


    def per_class_acc_curves(self):
        print('Generating per class accuracy training curves...')
        # tile the curves per class
        fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # adjust size as needed

        # Flatten axs for easy iteration
        axs = axs.ravel()

        for i in range(self.num_classes):  # adjust range if more classes
            axs[i].plot(self.log['epoch'], self.log[f'train_acc_class{i}'], label='Training')
            axs[i].plot(self.log['epoch'], self.log[f'valid_acc_class{i}'], label='Validation')

            axs[i].set_title(f'Accuracy for Class {i}')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Accuracy')
            axs[i].legend()

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()  # adjust subplot parameters for better spacing

        filename = 'training_curves_all_classes.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')


    def GAN_confusion_matrix(self,data_loader,model):
        #print('Generating confusion matrix...')
        # confusion matrix
        # switch model to evaluation mode
        model.eval()

        # Create lists to store predicted and actual labels
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        with torch.no_grad():
            for inputs, classes, _ in tqdm(data_loader, desc=f"Generating confusion matrix"):
            #for i, (inputs, classes,sample_weights) in enumerate(data_loader): # changed to accomodate the new dataset class
                inputs = inputs.to(device)
                classes = classes.to(device)
                _, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix

        # Plot confusion matrix
        plt.figure(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='.2%', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xticks(np.arange(len(self.labels)) + 0.5, self.labels)
        plt.yticks(np.arange(len(self.labels)) + 0.5, self.labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        #plt.title('Confusion Matrix')

        filename = 'GAN_confusion_matrix.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def DNN_confusion_matrix(self,data_loader,model):
        #print('Generating confusion matrix...')
        # confusion matrix
        # switch model to evaluation mode
        model.eval()

        # Create lists to store predicted and actual labels
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        with torch.no_grad():
            for inputs, classes, _ in tqdm(data_loader, desc=f"Generating confusion matrix"):
            #for i, (inputs, classes,sample_weights) in enumerate(valid_loader): # changed to accomodate the new dataset class
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                softmax_output = F.softmax(outputs, dim=1)
                _, preds = torch.max(softmax_output.data, 1)
                # Append batch prediction results
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix

        # Plot confusion matrix
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(conf_mat, annot=True, fmt='.2%', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels, annot_kws={"size": 16}, cbar=False)
        #ax.set_xticks([])
        #ax.set_yticks([])
        #plt.xlabel('Predicted labels')
        #plt.ylabel('True labels')
        #plt.title('Confusion Matrix')
        mappable = ax.collections[0]
        mappable.set_clim(0, 1)
        cbar = colorbar(mappable, ax=ax)
        # Define custom tick positions and labels
        tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
        tick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']

        # Set custom ticks and labels
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=14)
        ax.set_xlabel('Predicted labels', fontsize=18)
        ax.set_ylabel('True labels', fontsize=18)

        # Increase font size for tick labels on both axes
        ax.tick_params(axis='both', labelsize=18)

        filename = 'DNN_confusion_matrix.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def training_f1(self):
        print('Generating F1 score training curves...')
        # Plotting training and validation F1 scores

        plt.figure(figsize=(10,6))

        plt.plot(self.log['epoch'], self.log['train_f1_score'], label='Training')
        plt.plot(self.log['epoch'], self.log['valid_f1_score'], label='Validation')

        plt.title('F1 Scores vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        filename = 'training_F1_scores.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def training_all_f1(self):
        print('Generating F1 score training curves...')

        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.log['epoch'], self.log['train_f1_micro'], label='Training')
        plt.plot(self.log['epoch'], self.log['valid_f1_micro'], label='Validation')
        plt.title('F1 Score (Micro)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.log['epoch'], self.log['train_f1_macro'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_f1_macro'], label='Validation')
        plt.title('F1 Score (Macro)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(self.log['epoch'], self.log['train_f1_weighted'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_f1_weighted'], label='Validation')
        plt.title('F1 Score (Weighted)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.tight_layout()

        filename = 'all_F1_training_curves.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def training_None_f1(self):
        print('Generating F1 score per class training curves...')

        fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # adjust size as needed

        # Flatten axs for easy iteration
        axs = axs.ravel()

        for i in range(self.num_classes):  # adjust range if more classes
            axs[i].plot(self.log['epoch'], self.log[f'train_f1_None{i}'], label='Training')
            axs[i].plot(self.log['epoch'], self.log[f'valid_f1_None{i}'], label='Validation')

            axs[i].set_title(f'F1 Score for Class {i}')
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('F1 Score')
            axs[i].legend()

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()  # adjust subplot parameters for better spacing

        filename = 'F1_per_class.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def GAN_training_loss(self):
        print('Generating ACGAN training curves...')
        '''
        # total training curves
        plt.figure(figsize=(10, 6))

        plt.plot(self.log['epoch'], self.log['avg_g_loss'], label='Generator')
        plt.plot(self.log['epoch'], self.log['avg_d_loss'], label='Discriminator')
        plt.title('ACGAN Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()
        '''
        plt.figure(figsize=(12, 5))


        plt.subplot(1, 2, 1)
        plt.plot(self.log['epoch'], self.log['avg_g_loss'], label='Generator')
        plt.title('Generator Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.log['epoch'], self.log['avg_d_loss'], label='Discriminator')
        plt.title('Discriminator Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()

        plt.tight_layout()

        filename = 'ACGAN_training_curves.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def ScalarDiscriminant(self,jet_fractions,data_loader,model,ACGAN=False):
        print('Generating D_Hyy plot...')
        D_values = []
        f_values = jet_fractions
        model.eval()

        with torch.no_grad():
            for data, labels, _ in tqdm(data_loader, desc=f"Calculating D_Hyy"):
                data = data.to(device)
                labels = labels.cpu().numpy()

                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                probs = F.softmax(outputs, dim=1)
                #epsilon = 1e-10 # A small constant to prevent division by zero

                numerator = f_values[0] * probs[:, 0]
                denominator_terms = [f_values[j] * probs[:, j] for j in range(1, self.num_classes)]  # Exclude class 0
                denominator = sum(denominator_terms)# + epsilon
                D0 = torch.log(numerator / denominator).cpu().numpy()

                paired_values = list(zip(labels, D0))
                D_values.extend(paired_values)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        unique_labels = sorted(set(label for label, _ in D_values))
        for label in tqdm(unique_labels, desc='Plot D_i'):
            w = [D0 for lbl, D0 in D_values if lbl == label]
            plt.hist(w, histtype='step', weights=np.ones(len(w)) / len(w), bins=100, lw=2.5, label=self.labels[label],color=self.colors[label], linestyle=self.linestyle[label])

        custom_lines = [Line2D([0], [0], color=self.colors[0], lw=2.5, linestyle=self.linestyle[0]),
                        Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3])]
        '''
        hep.atlas.label(self.atlas_label,
                        data=True,
                        loc=4,
                        fontsize=20,
                        rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        '''
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.ylim(0.00001, 1)
        plt.yscale('log')
        plt.xlabel('$D_{\mathrm{H\gamma\gamma}}$', fontsize=24)
        plt.ylabel('Jet Fraction', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(custom_lines, [self.labels[0], self.labels[1], self.labels[2], self.labels[3]], loc='upper right', handlelength=1.5, fontsize=20)

        filename = 'ScalarDiscriminantDHyy.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def ScalarDiscriminantZee(self,jet_fractions,data_loader,model,ACGAN=False):
        print('Generating D_Zee plot...')
        D_values = []
        f_values = jet_fractions
        model.eval()

        with torch.no_grad():
            for data, labels, _ in tqdm(data_loader, desc=f"Calculating D_Zee"):
                data = data.to(device)
                labels = labels.cpu().numpy()

                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                probs = F.softmax(outputs, dim=1)
                #epsilon = 1e-10 # A small constant to prevent division by zero

                numerator = f_values[1] * probs[:, 1]
                denominator_terms = [f_values[j] * probs[:, j] for j in range(self.num_classes) if j != 1]  # Exclude class 1
                denominator = sum(denominator_terms)# + epsilon
                D0 = torch.log(numerator / denominator).cpu().numpy()

                paired_values = list(zip(labels, D0))
                D_values.extend(paired_values)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        unique_labels = sorted(set(label for label, _ in D_values))
        for label in tqdm(unique_labels, desc='Plot D_i'):
            w = [D0 for lbl, D0 in D_values if lbl == label]
            plt.hist(w, histtype='step', weights=np.ones(len(w)) / len(w), bins=100, lw=2.5, label=self.labels[label],color=self.colors[label], linestyle=self.linestyle[label])

        custom_lines = [Line2D([0], [0], color=self.colors[0], lw=2.5, linestyle=self.linestyle[0]),
                        Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3])]
        '''
        hep.atlas.label(self.atlas_label,
                        data=True,
                        loc=4,
                        fontsize=20,
                        rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        '''
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.ylim(0.00001, 1)
        plt.yscale('log')
        plt.xlabel('$D_{\mathrm{Zee}}$', fontsize=24)
        plt.ylabel('Jet Fraction', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(custom_lines, [self.labels[0], self.labels[1], self.labels[2], self.labels[3]], loc='upper right', handlelength=1.5, fontsize=20)

        filename = 'ScalarDiscriminantDzee.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def MulticlassROC(self, data_loader, model, ACGAN=False):
        print('Generating ROC curves...')
        model.eval()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Predictions for ROC"):
                data = data.to(device)
                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)
                # Convert the logits to probabilities using the softmax function
                probs = F.softmax(outputs, dim=1)
                y_scores.append(probs.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_scores = np.concatenate(y_scores, axis=0)

        # Convert true labels to one-hot encoding
        y_onehot = np.eye(self.num_classes)[y_true]

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        plt.annotate('Simulation', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.6, 0.855), xycoords='axes fraction', fontsize=24, verticalalignment='top')

        # Calculate ROC for each class
        for i in tqdm(range(self.num_classes), desc='Plotting ROC'):
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, label=f'{self.labels[i]} (area = {roc_auc:.2f})', color=self.colors[i],linestyle=self.linestyle[i])


        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        #plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right",fontsize=20)

        filename = 'MulticlassROC.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def JetRejectionRate(self, data_loader, model, ACGAN=False):
        print('Generating Jet rejection rate curves...')
        model.eval()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Predictions"):
                data = data.to(device)
                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)
                probs = F.softmax(outputs, dim=1)
                y_scores.append(probs.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_scores = np.concatenate(y_scores, axis=0)

        # Convert true labels to one-hot encoding
        #y_onehot = np.eye(self.num_classes)[y_true]
        # Jet Rejection Rate Plot
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        #fpr_original, tpr_original = {}, {}

        for i in tqdm([1, 2, 3],desc="Plotting"): # rejection classses
            targets_loc = (y_true == 0) | (y_true == i)
            fpr_original, tpr_original, _ = roc_curve(y_true[targets_loc], y_scores[targets_loc, 0],pos_label=0)
            tpr = tpr_original.copy()
            fpr = fpr_original.copy()
            '''
            thresholds = np.linspace(0.8,0.95,50)
            for j in thresholds:
                tpr = np.append(tpr, tpr_original[tpr_original > j])
                fpr = np.append(fpr, fpr_original[tpr_original > j])
            tpr = np.sort(tpr)
            fpr = np.sort(fpr)
            '''
            fpr_offset = fpr + 1e-10 # Small constant offset to avoid division by zero
            rejection_rate = 1./fpr_offset
            #tpr_offset = tpr +1e-10
            #rejection_rate = 1/tpr_offset
            rejection_rate[np.isinf(rejection_rate)] = np.nan  # replace infinite values with NaNs

            plt.plot(tpr*100, rejection_rate, label=self.labels[i], color=self.colors[i], ls=self.linestyle[i], lw=2.5)

            fpr_at_tpr_95 = np.interp(0.95, tpr, fpr)
            rejection_rate_at_tpr_95 = 1. / (fpr_at_tpr_95 + 1e-10)

            print(f"Rejection rate for curve {self.labels[i]} when tpr=0.95 is {rejection_rate_at_tpr_95}")

        plt.yscale('log')
        plt.xlabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$', fontsize=24)
        plt.ylabel('Jet Rejection Rate', fontsize=24)
        plt.xlim(48, 102)
        plt.ylim(1e+2, 1e+8)
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        custom_lines = [Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3])]

        plt.legend(custom_lines, [self.labels[1], self.labels[2], self.labels[3]], loc='upper right', handlelength=1.5, fontsize=20)

        rejection_filename = 'JetRejectionRate.pdf'
        full_rejection_path = os.path.join(self.new_folder_path, rejection_filename)
        plt.savefig(full_rejection_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def SignalEff_vs_Col(self, data_loader, model, col_index, column_name, ACGAN=False, n_bins=100):
        print('Generating signal eff. vs ' + column_name + '...')
        model.eval()
        y_true = []
        predictions = []
        column_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)

                column_values.extend(data[:, col_index].cpu().numpy())  # Extracting mass values

                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                pred_class = np.argmax(outputs.cpu().numpy(), axis=1)
                predictions.extend(pred_class)
                y_true.extend(targets.cpu().numpy())

        column_values = np.array(column_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        # Calculating the TPR and FPR for class 0
        quantiles = np.linspace(0, 1, n_bins+1)
        bins = np.quantile(column_values, quantiles)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2

        tprs = []
        fprs = []
        tprs_uncertainties = []
        fprs_uncertainties = []

        for b in tqdm(range(n_bins),desc="Signal efficiencies"):
            mask = (column_values >= bins[b]) & (column_values < bins[b + 1])
            n_samples = np.sum(mask)
            true_in_bin = y_true[mask]
            preds_in_bin = predictions[mask]

            true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
            total_true_zeros = np.sum(true_in_bin == 0)
            TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan

            false_positives = np.sum((preds_in_bin == 0) & (true_in_bin != 0))
            total_true_non_zeros = np.sum(true_in_bin != 0)
            FPR = false_positives / total_true_non_zeros if total_true_non_zeros > 0 else np.nan

            tprs.append(TPR)
            fprs.append(FPR)

            # Binomial uncertainties
            tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan
            fpr_uncertainty = np.sqrt(FPR * (1 - FPR) / n_samples) if not np.isnan(FPR) and n_samples > 0 else np.nan
            tprs_uncertainties.append(tpr_uncertainty)
            fprs_uncertainties.append(fpr_uncertainty)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')

        bin_errs = np.diff(bins)/2
        plt.errorbar(bin_midpoints, tprs, xerr=bin_errs, yerr=tprs_uncertainties, fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
        #plt.errorbar(bin_midpoints, fprs, xerr=bin_errs, yerr=fprs_uncertainties, fmt='o', color='red', label='False Positive Rate')

        #plt.xlim(0, 400)
        label_string = '$\\mathrm{' + column_name + '}\\ [GeV]$'
        plt.xlabel(label_string)
        #plt.xlabel(rf'$\mathrm{{column_name}}\ [GeV]$')
        plt.ylabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$')
        # plt.title('')
        plt.legend()

        filename = 'SignalEff_vs_' + column_name + '.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')


    def F1Score_vs_Col(self, data_loader, model, col_index, column_name, ACGAN=False, n_bins=100):
        print('Generating F1 score vs ' + column_name + '...')
        model.eval()
        y_true = []
        predictions = []
        column_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)
                column_values.extend(data[:, col_index].cpu().numpy())

                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                pred_class = np.argmax(outputs.cpu().numpy(), axis=1)
                predictions.extend(pred_class)
                y_true.extend(targets.cpu().numpy())

        column_values = np.array(column_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        quantiles = np.linspace(0, 1, n_bins+1)
        bins = np.quantile(column_values, quantiles)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2

        avg_f1_scores = []

        for b in tqdm(range(n_bins), desc="Calculating F1 scores"):
            mask = (column_values >= bins[b]) & (column_values < bins[b + 1])
            true_in_bin = y_true[mask]
            preds_in_bin = predictions[mask]

            F1 = f1_score(true_in_bin, preds_in_bin, average="weighted")
            avg_f1_scores.append(F1)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')

        bin_errs = np.diff(bins)/2
        plt.errorbar(bin_midpoints, avg_f1_scores, xerr=bin_errs, fmt='v', color=self.colors[1]) #label=self.labels[0])

        label_string = '$\\mathrm{' + column_name + '}\\ [GeV]$'
        plt.xlabel(label_string)
        plt.ylabel(r'$\mathrm{Weighted\ F1\ Score}$')
        #plt.legend()

        filename = 'F1Score_vs_' + column_name + '.png'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def SignalEff_vs_FatElectron_cM_bins(self, data_loader,dataset, model, col_index, bins, ACGAN=False, DNN=False):
        #print('Generating signal eff. vs FatElectron_cM_bins...')
        model.eval()
        y_true = []
        predictions = []
        column_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)

                column_values.extend(data[:, col_index].cpu().numpy())  # Extracting FatElectron_cM_bins values

                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                #pred_class = np.argmax(outputs.cpu().numpy(), axis=1)
                softmax_output = F.softmax(outputs, dim=1)
                _, pred_class = torch.max(softmax_output.data, 1)
                predictions.extend(pred_class.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        column_values = np.array(column_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        #quantiles = np.linspace(0, 1, n_bins + 1)
        #bins = np.quantile(column_values, quantiles)
        bins = np.array(bins)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        bin_errs = np.diff(bins) / 2

        tprs = []
        #fprs = []
        tprs_uncertainties = []
        #fprs_uncertainties = []

        for b in tqdm(range(len(bins)-1), desc="Signal efficiencies"):
            if DNN == True:
                mask = (column_values >= bins[b]) & (column_values < bins[b+1])
            else:
                mask = column_values == b
            n_samples = np.sum(mask)
            true_in_bin = y_true[mask]
            preds_in_bin = predictions[mask]

            true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
            total_true_zeros = np.sum(true_in_bin == 0)
            TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan

            #false_positives = np.sum((preds_in_bin == 0) & (true_in_bin != 0))
            #total_true_non_zeros = np.sum(true_in_bin != 0)
            #FPR = false_positives / total_true_non_zeros if total_true_non_zeros > 0 else np.nan

            tprs.append(TPR)
            #fprs.append(FPR)

            # Binomial uncertainties
            tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan # 0 if need be
            #fpr_uncertainty = np.sqrt(FPR * (1 - FPR) / n_samples) if not np.isnan(FPR) and n_samples > 0 else np.nan
            tprs_uncertainties.append(tpr_uncertainty)
            #fprs_uncertainties.append(fpr_uncertainty)

        # Plotting code remains the same, just update the x-label to reflect the binning on 'FatElectron_cM_bins'
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')


        #bin_midpoints = dataset.num_col('FatElectron_cM_bins')
        #print(len(bin_midpoints),len(tprs))
        #tprs = np.array(tprs)
        #tprs[np.isnan(tprs)] = 0 # or some value that makes sense in your context
        #tprs_uncertainties[np.isnan(tprs_uncertainties)] = 0

        plt.errorbar(bin_midpoints, np.array(tprs)*100,xerr=bin_errs, yerr=tprs_uncertainties, fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
        #plt.errorbar(bin_midpoints, fprs, xerr=bin_errs, yerr=fprs_uncertainties, fmt='o', color='red', label='False Positive Rate')
        mean_tpr = np.mean(tprs)
        std_tpr = np.std(tprs)

        plt.annotate(r'$\langle \epsilon_{{H\rightarrow\gamma\gamma}} \rangle$ = {:.2f}%'.format(mean_tpr*100),
             xy=(1, 1), xycoords='axes fraction',
             xytext=(-10, -10), textcoords='offset points',
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=20)

        plt.annotate(r'$\sigma_{{\mathrm{{\epsilon_{{H\rightarrow\gamma\gamma}}}}}}$ = {:.2f}%'.format(std_tpr*100),
                     xy=(1, 0.95), xycoords='axes fraction',
                     xytext=(-10, -10), textcoords='offset points',
                     horizontalalignment='right',
                     verticalalignment='top',
                     fontsize=20)


        #plt.xlim(0, 400)
        label_string = '$\\mathrm{Jet \ Mass}\ [GeV]$'
        plt.xlabel(label_string, fontsize=24)
        #plt.xlabel(rf'$\mathrm{{column_name}}\ [GeV]$')
        plt.ylabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$', fontsize=24)
        # plt.title('')
        #plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=20)

        filename = 'SignalEff_vs_mass_bins.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    # Rest of the plotting code

    def SignalEff_vs_pTandEta(self, data_loader, dataset, model):
        print('Generating signal eff. vs pT and eta bins...')
        model.eval()
        pT_col_index = dataset.column_index('pT_Bin')
        eta_col_index = dataset.column_index('Eta_Bin')
        pT_bins = dataset.bin_info('pT_Bin')
        eta_bins = dataset.bin_info('Eta_Bin')

        bins_pt = [400,520,540,560,580,605,630,660,700,740,780,840,900,1000,1100]
        bins_eta = [0,0.6,1.37,1.52,2.5]

        y_true = []
        predictions = []
        pT_values = []
        eta_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)

                pT_values.extend(data[:, pT_col_index].cpu().numpy())  # Extracting pT values
                eta_values.extend(data[:, eta_col_index].cpu().numpy())  # Extracting eta values

                outputs = model(data)
                softmax_output = F.softmax(outputs, dim=1)
                _, pred_class = torch.max(softmax_output.data, 1)
                predictions.extend(pred_class.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        pT_values = np.array(pT_values)
        eta_values = np.array(eta_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        pt_midpoints = (np.array(bins_pt[:-1]) + np.array(bins_pt[1:])) / 2
        pt_bin_widths = np.diff(bins_pt) / 2

        eta_midpoints = (np.array(bins_eta[:-1]) + np.array(bins_eta[1:])) / 2
        eta_bin_widths = np.diff(bins_eta) / 2

        plt.style.use([hep.style.ATLAS])
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ptxlim, etaxlim = [400,1100], [0,2.5]
        for idx, (col_values, bins, xlabel, xlim, midpoints, bin_widths) in enumerate(zip([pT_values, eta_values], [pT_bins, eta_bins], ['$p_T$ [GeV]', '$|\eta|$'], [ptxlim,etaxlim],[pt_midpoints, eta_midpoints],[pt_bin_widths, eta_bin_widths])):
            tprs = []
            tprs_uncertainties = []

            for b in tqdm(range(len(bins)), desc=f"Signal efficiencies for {xlabel}"):
                mask = col_values == b
                n_samples = np.sum(mask)
                true_in_bin = y_true[mask]
                preds_in_bin = predictions[mask]

                true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
                total_true_zeros = np.sum(true_in_bin == 0)
                TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan
                tprs.append(TPR)

                tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan
                tprs_uncertainties.append(tpr_uncertainty)
            '''
            bin_midpoints = np.array(bins) #+ 1#(np.array(bins[:-1]) + np.array(bins[1:])) / 2
            bin_errs = np.diff(bins) / 2
            '''

            #print(f"Length of bin_midpoints: {len(bin_midpoints)}")
            #print(f"Length of tprs: {len(tprs)}")
            axes[idx].errorbar(midpoints[:len(tprs)], np.array(tprs) * 100,
                       xerr=bin_widths[:len(tprs)], # adding x error bars
                       yerr=tprs_uncertainties,
                       fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
            #axes[idx].errorbar(bin_midpoints[:len(tprs)], np.array(tprs)*100,yerr=tprs_uncertainties, fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
            axes[idx].set_xlim(xlim)
            #axes[idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].set_xlabel(xlabel,fontsize=24)
            axes[idx].set_ylabel('Signal Efficiency [%]',fontsize=24)
            axes[idx].annotate('Simulation', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=24, verticalalignment='top')
            axes[idx].annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.825), xycoords='axes fraction', fontsize=24, verticalalignment='top')
            #std_of_tprs = np.std(tprs, ddof=1)
            #axes[idx].annotate(f'Stdev: {std_of_tprs:.4f}', xy=(0.7, 0.05), xycoords='axes fraction', fontsize=12)
            #axes[idx].legend([xlabel])
        #axes[0].set_xticks(range(14))
        axes[0].legend(loc='lower right',labels=[r'$\rm p_T$ bins'],fontsize=24)
        axes[1].legend(loc='lower right',labels=[r'$\rm |\eta|$ bins'],fontsize=24)
        axes[1].set_ylim(70,110)
        plt.tight_layout()
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        #plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20, verticalalignment='top')
        #plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.1, 0.925), xycoords='axes fraction', fontsize=20, verticalalignment='top')


        filename = 'SignalEff_vs_pTandEta_bins.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def SignalEff_vs_FatElectron_cM_binsANN_DNN(self, model_configs, bins, ACGAN=False):
        print('Generating signal eff. vs FatElectron_cM_bins...')
        '''
        model_configs = [
        {'model': model1, 'DNN': False, 'label': 'Model 1'},
        {'model': model2, 'DNN': True, 'label': 'Model 2'}]
        '''

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')

        for i, config in enumerate(model_configs):
            model = config['model']
            DNN = config['DNN']
            label = config['label']
            col_index = config['col_index']
            batch_size = config['batch_size']
            dataset = config['test_dataset']

            model.eval()
            y_true = []
            predictions = []
            column_values = []
            data_loader=DataLoader(dataset, batch_size=batch_size, num_workers=int(os.cpu_count()/2), shuffle=False)
            with torch.no_grad():
                for data, targets, _ in tqdm(data_loader, desc=f"Generating Predictions for {label}"):
                    data = data.to(device)

                    column_values.extend(data[:, col_index].cpu().numpy())

                    if ACGAN:
                        _, outputs = model(data)
                    else:
                        outputs = model(data)

                    softmax_output = F.softmax(outputs, dim=1)
                    _, pred_class = torch.max(softmax_output.data, 1)
                    predictions.extend(pred_class.cpu().numpy())
                    y_true.extend(targets.cpu().numpy())

            column_values = np.array(column_values)
            y_true = np.array(y_true)
            predictions = np.array(predictions)

            #quantiles = np.linspace(0, 1, n_bins + 1)
            #bins = np.quantile(column_values, quantiles)
            bins = np.array(bins)
            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            bin_errs = np.diff(bins) / 2

            tprs = []
            #fprs = []
            tprs_uncertainties = []
            #fprs_uncertainties = []

            for b in tqdm(range(len(bins)-1), desc="Signal efficiencies"):
                if DNN == True:
                    mask = (column_values >= bins[b]) & (column_values < bins[b+1])
                else:
                    mask = column_values == b
                n_samples = np.sum(mask)
                true_in_bin = y_true[mask]
                preds_in_bin = predictions[mask]

                true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
                total_true_zeros = np.sum(true_in_bin == 0)
                TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan

                #false_positives = np.sum((preds_in_bin == 0) & (true_in_bin != 0))
                #total_true_non_zeros = np.sum(true_in_bin != 0)
                #FPR = false_positives / total_true_non_zeros if total_true_non_zeros > 0 else np.nan

                tprs.append(TPR)
                #fprs.append(FPR)

                # Binomial uncertainties
                tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan # 0 if need be
                #fpr_uncertainty = np.sqrt(FPR * (1 - FPR) / n_samples) if not np.isnan(FPR) and n_samples > 0 else np.nan
                tprs_uncertainties.append(tpr_uncertainty)
                #fprs_uncertainties.append(fpr_uncertainty)

                #bin_midpoints = dataset.num_col('FatElectron_cM_bins')
                #print(len(bin_midpoints),len(tprs))
                #tprs = np.array(tprs)
                #tprs[np.isnan(tprs)] = 0 # or some value that makes sense in your context
                #tprs_uncertainties[np.isnan(tprs_uncertainties)] = 0
            print("Length of bin_midpoints:", len(bin_midpoints))
            print("Length of tprs:", len(tprs))
            std_tpr = np.nanstd(tprs)
            enhanced_label = f"{config['label']} ($\\sigma_{{\\epsilon_{{H \\rightarrow \\gamma\\gamma}}}}$= {std_tpr*100:.2f}%)"
            plt.errorbar(bin_midpoints, np.array(tprs)*100,xerr=bin_errs, yerr=tprs_uncertainties, fmt=self.markers[i+2], color=self.colors[i+2], label=enhanced_label)

        #plt.xlim(0, 400)
        label_string = '$\\mathrm{Jet \ Mass}\ [GeV]$'
        plt.xlabel(label_string, fontsize=24)
        #plt.xlabel(rf'$\mathrm{{column_name}}\ [GeV]$')
        plt.ylabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$', fontsize=24)
        # plt.title('')
        plt.legend(fontsize=24)

        # ... (Rest of the plotting code)

        filename = 'SignalEff_vs_mass_bins_DNNandANN.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')


    def JRRvsMass(self, model_configs, bins, ACGAN=False):
        print('Generating JRR vs mass...')
        plt.style.use([hep.style.ATLAS])
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, config in enumerate(model_configs):
            model = config['model']
            DNN = config['DNN']
            label = config['label']
            col_index = config['col_index']
            batch_size = config['batch_size']
            dataset = config['test_dataset']

            model.eval()
            y_true = []
            predictions = []
            column_values = []
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=int(os.cpu_count()/2), shuffle=False)

            with torch.no_grad():
                for data, targets, _ in tqdm(data_loader, desc=f"Generating Predictions for {label}"):
                    data = data.to(device)
                    column_values.extend(data[:, col_index].cpu().numpy())

                    if ACGAN:
                        _, outputs = model(data)
                    else:
                        outputs = model(data)

                    softmax_output = F.softmax(outputs, dim=1)
                    _, pred_class = torch.max(softmax_output.data, 1)
                    predictions.extend(pred_class.cpu().numpy())
                    y_true.extend(targets.cpu().numpy())

            column_values = np.array(column_values)
            y_true = np.array(y_true)
            predictions = np.array(predictions)
            bins = np.array(bins)

            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            bin_errs = np.diff(bins) / 2

            for bg_jet in [1,2,3]:
                jet_rejection_rates = []
                jet_rejection_uncertainties = []

                for b in tqdm(range(len(bins) - 1), desc=f"Jet {bg_jet}"):
                    if DNN == True:
                        mask = (column_values >= bins[b]) & (column_values < bins[b+1])
                    else:
                        mask = column_values == b

                    n_samples = np.sum(mask)
                    true_in_bin = y_true[mask]
                    preds_in_bin = predictions[mask]

                    true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
                    total_true_zeros = np.sum(true_in_bin == 0)
                    TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan

                    if TPR >= 0.95:
                        false_positives = np.sum((preds_in_bin == 0) & (true_in_bin == bg_jet))
                        total_true_bg_jet = np.sum(true_in_bin == bg_jet)
                        FPR = false_positives / total_true_bg_jet if total_true_bg_jet > 0 else np.nan
                        jet_rejection_rate = 1 / FPR if FPR > 0 else np.nan
                        if np.isnan(jet_rejection_rate):
                            print(f"NaN encountered! TPR: {TPR}, total_true_bg_jet: {total_true_bg_jet}, FPR: {FPR}")
                        jet_rejection_rates.append(jet_rejection_rate)
                        uncertainty = np.sqrt((1 - FPR) / (FPR * np.sum(true_in_bin == bg_jet)))
                        jet_rejection_uncertainties.append(jet_rejection_rate * uncertainty)
                    else:
                        jet_rejection_rates.append(np.nan)
                        jet_rejection_uncertainties.append(np.nan)

                print(f"\nJRR {bg_jet}: {jet_rejection_rates}")
                print(f"STD {bg_jet}: {np.sqrt(np.nanvar(jet_rejection_rates))}")
                ax = axes[1] if not DNN else axes[0]
                ax.errorbar(bin_midpoints, jet_rejection_rates,xerr=bin_errs,yerr=jet_rejection_uncertainties , fmt=self.markers[bg_jet], color=self.colors[bg_jet], label=self.labels[bg_jet])
                ax.set_xlim([95,270])
                ax.set_xlabel('$\\mathrm{Jet \ Mass}\ [GeV]$', fontsize=24)
                ax.set_ylabel('Jet Rejection Rate', fontsize=24)
                ax.legend(fontsize=20,loc='lower right')
                ax.annotate('Simulation', xy=(0.65, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
                ax.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.65, 0.875), xycoords='axes fraction', fontsize=24, verticalalignment='top')
                ax.annotate(r'$\rm \epsilon_{H\rightarrow\gamma\gamma}=95\%$', xy=(0.65, 0.775), xycoords='axes fraction', fontsize=20, verticalalignment='top', fontweight='normal')
                ax.tick_params(axis='both', which='major', labelsize=20)

        axes[0].annotate('DNN', xy=(0.65, 0.67), xycoords='axes fraction', fontsize=24, verticalalignment='top', fontweight='normal')
        axes[1].annotate('ANN', xy=(0.65, 0.67), xycoords='axes fraction', fontsize=24, verticalalignment='top', fontweight='normal')

        plt.tight_layout()
        filename = 'JRRvsMassComp.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)


class plotting5Class():
    def __init__(self, log, new_folder_path, num_classes):
        self.log = log
        self.new_folder_path = new_folder_path
        self.num_classes = num_classes

        self.labels = [r'$\rm H \rightarrow \gamma\gamma $',r'$\rm Z \rightarrow e^+e^- $',r'$\rm q/g$', r'$\rm e/\gamma $', r'$\rm (\tau)\tau$']
        self.colors = ['#4462a5', '#689aff', '#c50807', '#68cd67', 'tab:orange', 'tab:purple', 'tab:green']
        self.markers = ['o', 'v', 's', '^']
        self.linestyle = ['solid', (0, (1, 1)), 'dashed', (0, (5, 1)), 'solid', 'solid', 'solid', 'solid']
        self.atlas_label = 'Simulation'# Preliminary'

    def DNN_confusion_matrix(self,data_loader,model):
        #print('Generating confusion matrix...')
        # confusion matrix
        # switch model to evaluation mode
        model.eval()

        # Create lists to store predicted and actual labels
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        with torch.no_grad():
            for inputs, classes, _ in tqdm(data_loader, desc=f"Generating confusion matrix"):
            #for i, (inputs, classes,sample_weights) in enumerate(valid_loader): # changed to accomodate the new dataset class
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                softmax_output = F.softmax(outputs, dim=1)
                _, preds = torch.max(softmax_output.data, 1)
                # Append batch prediction results
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix

        # Plot confusion matrix
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(conf_mat, annot=True, fmt='.2%', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels, annot_kws={"size": 14},cbar=False)
        #ax.set_xticks([])
        #ax.set_yticks([])
        #plt.xlabel('Predicted labels')
        #plt.ylabel('True labels')
        #plt.title('Confusion Matrix')
        #cbar = ax.collections[0].colorbar
        mappable = ax.collections[0]
        mappable.set_clim(0, 1)
        cbar = colorbar(mappable, ax=ax)
        #cbar.set_clim(0, 1)
        # Define custom tick positions and labels
        tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
        tick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']

        # Set custom ticks and labels
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=14)
        ax.set_xlabel('Predicted labels', fontsize=16)
        ax.set_ylabel('True labels', fontsize=16)

        # Increase font size for tick labels on both axes
        ax.tick_params(axis='both', labelsize=14)

        filename = 'DNN_confusion_matrix.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def lossandsignal(self):
        print('Generating total training curves...')
        # total training curves
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.log['epoch'], self.log['train_loss'], label='Train')
        plt.plot(self.log['epoch'], self.log['valid_loss'], label='Validation')
        #plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch',fontsize=24)
        plt.ylabel('Loss',fontsize=24)
        plt.xlim([0,20])
        #plt.xticks(ticks=range(0, 31, 2), labels=[str(x) for x in range(0, 31, 2)])
        plt.gca().xaxis.set_major_locator(FixedLocator(range(0, 19, 5)))
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=20,loc='lower left')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.875), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.subplot(1, 2, 2)
        plt.plot(self.log['epoch'], self.log['train_acc_class0']*100, label='Train')
        plt.plot(self.log['epoch'], self.log['valid_acc_class0']*100, label='Validation')
        #plt.title('Total Accuracy vs. Epoch')
        plt.xlabel('Epoch',fontsize=24)
        plt.ylabel('Signal Accuracy [%]',fontsize=24)
        plt.xlim([0,20])
        plt.gca().xaxis.set_major_locator(FixedLocator(range(0, 19, 5)))
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=20,loc='lower right')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.875), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()

        filename = 'training_LandScurves.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def ScalarDiscriminant(self,jet_fractions,data_loader,model,ACGAN=False):
        print('Generating D_Hyy plot...')
        D_values = []
        f_values = jet_fractions
        model.eval()

        with torch.no_grad():
            for data, labels, _ in tqdm(data_loader, desc=f"Calculating D_Hyy"):
                data = data.to(device)
                labels = labels.cpu().numpy()

                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                probs = F.softmax(outputs, dim=1)
                #epsilon = 1e-10 # A small constant to prevent division by zero

                numerator = f_values[0] * probs[:, 0]
                denominator_terms = [f_values[j] * probs[:, j] for j in range(1, self.num_classes)]  # Exclude class 0
                denominator = sum(denominator_terms)# + epsilon
                D0 = torch.log(numerator / denominator).cpu().numpy()

                paired_values = list(zip(labels, D0))
                D_values.extend(paired_values)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        unique_labels = sorted(set(label for label, _ in D_values))
        for label in tqdm(unique_labels, desc='Plot D_i'):
            w = [D0 for lbl, D0 in D_values if lbl == label]
            plt.hist(w, histtype='step', weights=np.ones(len(w)) / len(w), bins=100, lw=2.5, label=self.labels[label],color=self.colors[label], linestyle=self.linestyle[label])

        custom_lines = [Line2D([0], [0], color=self.colors[0], lw=2.5, linestyle=self.linestyle[0]),
                        Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3]),
                        Line2D([0], [0], color=self.colors[4], lw=2.5, linestyle=self.linestyle[4])]
        '''
        hep.atlas.label(self.atlas_label,
                        data=True,
                        loc=4,
                        fontsize=20,
                        rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        '''
        plt.annotate('Simulation', xy=(0.75, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.75, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.ylim(0.00001, 1)
        plt.xlim(-60,10)
        plt.yscale('log')
        plt.xlabel('$D_{\mathrm{H\gamma\gamma}}$', fontsize=24)
        plt.ylabel('Jet Fraction', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(custom_lines, [self.labels[0], self.labels[1], self.labels[2], self.labels[3],self.labels[4]], loc='upper left', handlelength=1.5, fontsize=20)

        filename = 'ScalarDiscriminantDHyy.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def ScalarDiscriminantZee(self,jet_fractions,data_loader,model,ACGAN=False):
        print('Generating D_Zee plot...')
        D_values = []
        f_values = jet_fractions
        model.eval()

        with torch.no_grad():
            for data, labels, _ in tqdm(data_loader, desc=f"Calculating D_Zee"):
                data = data.to(device)
                labels = labels.cpu().numpy()

                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                probs = F.softmax(outputs, dim=1)
                #epsilon = 1e-10 # A small constant to prevent division by zero

                numerator = f_values[1] * probs[:, 1]
                denominator_terms = [f_values[j] * probs[:, j] for j in range(self.num_classes) if j != 1]  # Exclude class 1
                denominator = sum(denominator_terms)# + epsilon
                D0 = torch.log(numerator / denominator).cpu().numpy()

                paired_values = list(zip(labels, D0))
                D_values.extend(paired_values)

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        unique_labels = sorted(set(label for label, _ in D_values))
        for label in tqdm(unique_labels, desc='Plot D_i'):
            w = [D0 for lbl, D0 in D_values if lbl == label]
            plt.hist(w, histtype='step', weights=np.ones(len(w)) / len(w), bins=100, lw=2.5, label=self.labels[label],color=self.colors[label], linestyle=self.linestyle[label])

        custom_lines = [Line2D([0], [0], color=self.colors[0], lw=2.5, linestyle=self.linestyle[0]),
                        Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3]),
                        Line2D([0], [0], color=self.colors[4], lw=2.5, linestyle=self.linestyle[4])]
        '''
        hep.atlas.label(self.atlas_label,
                        data=True,
                        loc=4,
                        fontsize=20,
                        rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        '''
        plt.annotate('Simulation', xy=(0.75, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.75, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.ylim(0.00001, 1)
        plt.xlim(-60,10)
        plt.yscale('log')
        plt.xlabel('$D_{\mathrm{Zee}}$', fontsize=24)
        plt.ylabel('Jet Fraction', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(custom_lines, [self.labels[0], self.labels[1], self.labels[2], self.labels[3],self.labels[4]], loc='upper left', handlelength=1.5, fontsize=20)

        filename = 'ScalarDiscriminantDzee.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def MulticlassROC(self, data_loader, model, ACGAN=False):
        print('Generating ROC curves...')
        model.eval()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Predictions for ROC"):
                data = data.to(device)
                if ACGAN == True:
                    _, outputs = model(data)
                else:
                    outputs = model(data)
                # Convert the logits to probabilities using the softmax function
                probs = F.softmax(outputs, dim=1)
                y_scores.append(probs.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_scores = np.concatenate(y_scores, axis=0)

        # Convert true labels to one-hot encoding
        y_onehot = np.eye(self.num_classes)[y_true]

        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')

        # Calculate ROC for each class
        for i in tqdm(range(self.num_classes), desc='Plotting ROC'):
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, label=f'{self.labels[i]} (area = {roc_auc:.2f})', color=self.colors[i],linestyle=self.linestyle[i])


        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        #plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right",fontsize=20)

        filename = 'MulticlassROC.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()

    def JetRejectionRate(self, data_loader, model, ACGAN=False):
        print('Generating Jet rejection rate curves...')
        model.eval()
        y_true = []
        y_scores = []
        def moving_average(arr, window_size):
            return np.convolve(arr, np.ones(window_size) / window_size, mode='valid')

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Predictions"):
                data = data.to(device)
                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)
                probs = F.softmax(outputs, dim=1)
                y_scores.append(probs.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_scores = np.concatenate(y_scores, axis=0)

        # Convert true labels to one-hot encoding
        #y_onehot = np.eye(self.num_classes)[y_true]
        # Jet Rejection Rate Plot
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        #fpr_original, tpr_original = {}, {}

        for i in tqdm([1, 2, 3, 4],desc="Plotting"): # rejection classses
            targets_loc = (y_true == 0) | (y_true == i)
            fpr_original, tpr_original, _ = roc_curve(y_true[targets_loc], y_scores[targets_loc, 0],pos_label=0)
            window_size = 25
            tpr = moving_average(tpr_original.copy(),window_size)
            fpr = moving_average(fpr_original.copy(),window_size)
            '''
            thresholds = np.linspace(0.8,0.95,50)
            for j in thresholds:
                tpr = np.append(tpr, tpr_original[tpr_original > j])
                fpr = np.append(fpr, fpr_original[tpr_original > j])
            tpr = np.sort(tpr)
            fpr = np.sort(fpr)
            '''
            fpr_offset = fpr + 1e-8 # Small constant offset to avoid division by zero
            rejection_rate = 1./fpr_offset
            #tpr_offset = tpr +1e-10
            #rejection_rate = 1/tpr_offset
            rejection_rate[np.isinf(rejection_rate)] = np.nan  # replace infinite values with NaNs

            plt.plot(tpr*100, rejection_rate, label=self.labels[i], color=self.colors[i], ls=self.linestyle[i], lw=2.5)

            fpr_at_tpr_95 = np.interp(0.95, tpr, fpr)
            rejection_rate_at_tpr_95 = 1. / (fpr_at_tpr_95 + 1e-10)

            print(f"Rejection rate for curve {self.labels[i]} when tpr=0.95 is {rejection_rate_at_tpr_95}")

        plt.yscale('log')
        plt.xlabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$', fontsize=24)
        plt.ylabel('Jet Rejection Rate', fontsize=24)
        plt.xlim(48, 102)
        plt.ylim(1, 1e+6)
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.tick_params(axis='both', which='major', labelsize=20)
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        custom_lines = [Line2D([0], [0], color=self.colors[1], lw=2.5, linestyle=self.linestyle[1]),
                        Line2D([0], [0], color=self.colors[2], lw=2.5, linestyle=self.linestyle[2]),
                        Line2D([0], [0], color=self.colors[3], lw=2.5, linestyle=self.linestyle[3]),
                        Line2D([0], [0], color=self.colors[4], lw=2.5, linestyle=self.linestyle[4])]

        plt.legend(custom_lines, [self.labels[1], self.labels[2], self.labels[3],self.labels[4]], loc='upper right', handlelength=1.5, fontsize=20)

        rejection_filename = 'JetRejectionRate.pdf'
        full_rejection_path = os.path.join(self.new_folder_path, rejection_filename)
        plt.savefig(full_rejection_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()


    def SignalEff_vs_pTandEta(self, data_loader, dataset, model):
        print('Generating signal eff. vs pT and eta bins...')
        model.eval()
        pT_col_index = dataset.column_index('pT_Bin')
        eta_col_index = dataset.column_index('Eta_Bin')
        pT_bins = dataset.bin_info('pT_Bin')
        eta_bins = dataset.bin_info('Eta_Bin')

        bins_pt = [400,520,540,560,580,605,630,660,700,740,780,840,900,1000,1100]
        bins_eta = [0,0.6,1.37,1.52,2.5]

        y_true = []
        predictions = []
        pT_values = []
        eta_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)

                pT_values.extend(data[:, pT_col_index].cpu().numpy())  # Extracting pT values
                eta_values.extend(data[:, eta_col_index].cpu().numpy())  # Extracting eta values

                outputs = model(data)
                softmax_output = F.softmax(outputs, dim=1)
                _, pred_class = torch.max(softmax_output.data, 1)
                predictions.extend(pred_class.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        pT_values = np.array(pT_values)
        eta_values = np.array(eta_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        pt_midpoints = (np.array(bins_pt[:-1]) + np.array(bins_pt[1:])) / 2
        pt_bin_widths = np.diff(bins_pt) / 2

        eta_midpoints = (np.array(bins_eta[:-1]) + np.array(bins_eta[1:])) / 2
        eta_bin_widths = np.diff(bins_eta) / 2

        plt.style.use([hep.style.ATLAS])
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ptxlim, etaxlim = [400,1100], [0,2.5]
        for idx, (col_values, bins, xlabel, xlim, midpoints, bin_widths) in enumerate(zip([pT_values, eta_values], [pT_bins, eta_bins], ['$p_T$ [GeV]', '$|\eta|$'], [ptxlim,etaxlim],[pt_midpoints, eta_midpoints],[pt_bin_widths, eta_bin_widths])):
            tprs = []
            tprs_uncertainties = []

            for b in tqdm(range(len(bins)), desc=f"Signal efficiencies for {xlabel}"):
                mask = col_values == b
                n_samples = np.sum(mask)
                true_in_bin = y_true[mask]
                preds_in_bin = predictions[mask]

                true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
                total_true_zeros = np.sum(true_in_bin == 0)
                TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan
                tprs.append(TPR)

                tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan
                tprs_uncertainties.append(tpr_uncertainty)
            '''
            bin_midpoints = np.array(bins) #+ 1#(np.array(bins[:-1]) + np.array(bins[1:])) / 2
            bin_errs = np.diff(bins) / 2
            '''

            #print(f"Length of bin_midpoints: {len(bin_midpoints)}")
            #print(f"Length of tprs: {len(tprs)}")
            axes[idx].errorbar(midpoints[:len(tprs)], np.array(tprs) * 100,
                       xerr=bin_widths[:len(tprs)], # adding x error bars
                       yerr=tprs_uncertainties,
                       fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
            #axes[idx].errorbar(bin_midpoints[:len(tprs)], np.array(tprs)*100,yerr=tprs_uncertainties, fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
            axes[idx].set_xlim(xlim)
            #axes[idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].set_xlabel(xlabel,fontsize=24)
            axes[idx].set_ylabel('Signal Efficiency [%]',fontsize=24)
            axes[idx].annotate('Simulation', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=24, verticalalignment='top')
            axes[idx].annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.825), xycoords='axes fraction', fontsize=24, verticalalignment='top')
            #std_of_tprs = np.std(tprs, ddof=1)
            #axes[idx].annotate(f'Stdev: {std_of_tprs:.4f}', xy=(0.7, 0.05), xycoords='axes fraction', fontsize=12)
            #axes[idx].legend([xlabel])
        #axes[0].set_xticks(range(14))
        axes[0].legend(loc='lower right',labels=[r'$\rm p_T$ bins'],fontsize=24)
        axes[1].legend(loc='lower right',labels=[r'$\rm |\eta|$ bins'],fontsize=24)
        axes[1].set_ylim(70,110)
        plt.tight_layout()
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        #plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20, verticalalignment='top')
        #plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.1, 0.925), xycoords='axes fraction', fontsize=20, verticalalignment='top')


        filename = 'SignalEff_vs_pTandEta_bins.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')

    def SignalEff_vs_FatElectron_cM_bins(self, data_loader,dataset, model, col_index, bins, ACGAN=False, DNN=False):
        #print('Generating signal eff. vs FatElectron_cM_bins...')
        model.eval()
        y_true = []
        predictions = []
        column_values = []

        with torch.no_grad():
            for data, targets, _ in tqdm(data_loader, desc="Generating Predictions"):
                data = data.to(device)

                column_values.extend(data[:, col_index].cpu().numpy())  # Extracting FatElectron_cM_bins values

                if ACGAN:
                    _, outputs = model(data)
                else:
                    outputs = model(data)

                #pred_class = np.argmax(outputs.cpu().numpy(), axis=1)
                softmax_output = F.softmax(outputs, dim=1)
                _, pred_class = torch.max(softmax_output.data, 1)
                predictions.extend(pred_class.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        column_values = np.array(column_values)
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        #quantiles = np.linspace(0, 1, n_bins + 1)
        #bins = np.quantile(column_values, quantiles)
        bins = np.array(bins)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        bin_errs = np.diff(bins) / 2

        tprs = []
        #fprs = []
        tprs_uncertainties = []
        #fprs_uncertainties = []

        for b in tqdm(range(len(bins)-1), desc="Signal efficiencies"):
            if DNN == True:
                mask = (column_values >= bins[b]) & (column_values < bins[b+1])
            else:
                mask = column_values == b
            n_samples = np.sum(mask)
            true_in_bin = y_true[mask]
            preds_in_bin = predictions[mask]

            true_positives = np.sum((preds_in_bin == 0) & (true_in_bin == 0))
            total_true_zeros = np.sum(true_in_bin == 0)
            TPR = true_positives / total_true_zeros if total_true_zeros > 0 else np.nan

            #false_positives = np.sum((preds_in_bin == 0) & (true_in_bin != 0))
            #total_true_non_zeros = np.sum(true_in_bin != 0)
            #FPR = false_positives / total_true_non_zeros if total_true_non_zeros > 0 else np.nan

            tprs.append(TPR)
            #fprs.append(FPR)

            # Binomial uncertainties
            tpr_uncertainty = np.sqrt(TPR * (1 - TPR) / n_samples) if not np.isnan(TPR) and n_samples > 0 else np.nan # 0 if need be
            #fpr_uncertainty = np.sqrt(FPR * (1 - FPR) / n_samples) if not np.isnan(FPR) and n_samples > 0 else np.nan
            tprs_uncertainties.append(tpr_uncertainty)
            #fprs_uncertainties.append(fpr_uncertainty)

        # Plotting code remains the same, just update the x-label to reflect the binning on 'FatElectron_cM_bins'
        plt.style.use([hep.style.ATLAS])
        plt.figure(figsize=(15, 10))
        #hep.atlas.label(self.atlas_label, data=True, loc=4, fontsize=20, rlabel=r'$\rm \sqrt{s} = 13\ TeV$')
        plt.annotate('Simulation', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=24, verticalalignment='top')
        plt.annotate(r'$\rm \sqrt{s} = 13\ TeV$', xy=(0.05, 0.905), xycoords='axes fraction', fontsize=24, verticalalignment='top')


        #bin_midpoints = dataset.num_col('FatElectron_cM_bins')
        #print(len(bin_midpoints),len(tprs))
        #tprs = np.array(tprs)
        #tprs[np.isnan(tprs)] = 0 # or some value that makes sense in your context
        #tprs_uncertainties[np.isnan(tprs_uncertainties)] = 0

        plt.errorbar(bin_midpoints, np.array(tprs)*100,xerr=bin_errs, yerr=tprs_uncertainties, fmt=self.markers[0], color=self.colors[0], label=self.labels[0])
        #plt.errorbar(bin_midpoints, fprs, xerr=bin_errs, yerr=fprs_uncertainties, fmt='o', color='red', label='False Positive Rate')
        mean_tpr = np.mean(tprs)
        std_tpr = np.std(tprs)

        plt.annotate(r'$\langle \epsilon_{{H\rightarrow\gamma\gamma}} \rangle$ = {:.2f}%'.format(mean_tpr*100),
             xy=(1, 1), xycoords='axes fraction',
             xytext=(-10, -10), textcoords='offset points',
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=20)

        plt.annotate(r'$\sigma_{{\mathrm{{\epsilon_{{H\rightarrow\gamma\gamma}}}}}}$ = {:.2f}%'.format(std_tpr*100),
                     xy=(1, 0.95), xycoords='axes fraction',
                     xytext=(-10, -10), textcoords='offset points',
                     horizontalalignment='right',
                     verticalalignment='top',
                     fontsize=20)


        #plt.xlim(0, 400)
        label_string = '$\\mathrm{Jet \ Mass}\ [GeV]$'
        plt.xlabel(label_string, fontsize=24)
        #plt.xlabel(rf'$\mathrm{{column_name}}\ [GeV]$')
        plt.ylabel(r'$\mathrm{Signal\ Efficiency}\ [\%]$', fontsize=24)
        # plt.title('')
        #plt.legend()
        plt.tick_params(axis='both', which='major', labelsize=20)

        filename = 'SignalEff_vs_mass_bins.pdf'
        full_training_path = os.path.join(self.new_folder_path, filename)
        plt.savefig(full_training_path)
        plt.close('all')
        del data
        del outputs
        gc.collect()
