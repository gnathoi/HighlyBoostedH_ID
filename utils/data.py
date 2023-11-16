# DEFINE CUSTOM DATASET CLASS FOR DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

'''
This is our custom class for giving data to the loader.
As long as the input DataFrame has the same columns this class will enforce the correct ordering for HiggsNet and HiggsGAN.
The first three functions are necessary.
The other functions have been added for convenient calculation of the class weights, a list of the total number of samples per class,
how many classes there are in the dataset (useful for making the automating the output dimension of the NNs).
We can also modify the dataset easily and remove classes, this was useful for determining the use of additional noise classes.
'''

class BoostDataset(Dataset):
    def __init__(self,dataframe_path):
        super(BoostDataset, self).__init__()
        self.df = pd.read_pickle(dataframe_path)

        # enforcing input data ordering for data with the same features
        ind_list = ['FatElectron_Cluster1_dRToJet', 'FatElectron_Cluster2_dRToJet',
       'FatElectron_D2', 'FatElectron_EFraction', 'FatElectron_EMFrac',
       'FatElectron_FracSamplingMax', 'FatElectron_PF', 'FatElectron_Width',
       'FatElectron_balance', 'FatElectron_cM', 'FatElectron_dR_cc', 'Eta_Bin',
       'pT_Bin', 'FatElectron_nConstituents', 'FatElectron_ntrks', 'weight_ey_2d', 'type']
       # FatElectron_cM_bins
       # Check if all columns in ind_list are in self.df
        if set(ind_list).issubset(set(self.df.columns)):
            # Subset and rearrange the dataframe using reindex
            self.df = self.df.reindex(columns=ind_list)
        else:
            missing_cols = set(ind_list) - set(self.df.columns)
            print(f"Warning: The following columns are not present in the DataFrame: {missing_cols}")
        #index = pd.Index(ind_list, name='My Index')
        #self.df = self.df[index]
        self.df = self.df.reindex(columns=ind_list)

        # we define the weights associated with the transverse momentum (P_T-bins) and pseudorapidity bins (|eta|-bins)
        self.weights = torch.tensor(self.df.loc[:, self.df.columns.isin(['weight_ey_2d'])].values, dtype=torch.long).squeeze()
        # Separating features and target
        self.features = torch.tensor(self.df.loc[:, ~self.df.columns.isin(['weight_ey_2d', 'type'])].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df.loc[:, self.df.columns.isin(['type'])].values, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.features[index], self.targets[index], self.weights[index]
        return sample

    def get_labels(self):
        return self.targets

    def class_weights_calculator(self):
        class_weights = 1 / np.unique(self.df['type'], return_counts=True)[1]
        norm = np.linalg.norm(class_weights)
        class_weights /= norm
        class_weights = torch.tensor(class_weights).float()
        return class_weights

    def countclasses(self):
        totalclasslist = self.df['type'].value_counts().sort_index().tolist()
        return totalclasslist

    def input_size(self):
        t = self.features.size(1)
        return int(t)

    def num_classes(self):
        num = len(self.df['type'].unique())
        return num
    # merged, delete above function
    def num_col(self,col_name):
        num = len(self.df[col_name].unique())
        return num

    def remove_class(self, class_to_remove):
        # Remove the rows corresponding to the given class
        self.df = self.df[self.df['type'] != class_to_remove]
        # Shift the class labels of the remaining classes
        mask = self.df['type'] > class_to_remove
        self.df.loc[mask, 'type'] -= 1
        # Update the weights, features and targets
        self.weights = torch.tensor(self.df.loc[:, self.df.columns.isin(['weight_ey_2d'])].values, dtype=torch.long).squeeze()
        self.features = torch.tensor(self.df.loc[:, ~self.df.columns.isin(['weight_ey_2d', 'type'])].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df.loc[:, self.df.columns.isin(['type'])].values, dtype=torch.long).squeeze()

    def jet_fractions(self):
        _, counts = torch.unique(self.targets, return_counts=True)
        total_samples = self.targets.shape[0]
        return counts.float() / total_samples

    def column_index(self,column_name):
        col_ind = self.df.columns.get_loc(column_name)
        print(f'\nCol index: {col_ind}\n')
        return col_ind

    def bin_info(self,column_name):
        bins = sorted(self.df[column_name].unique())
        if bins[0] != 0:
            print("Lowest bin does not start at 0.")
        else:
            print("Lowest bin starts at 0.")
        return bins

    def RC_initial_weight(self,column_name):
        column_index = self.df.columns.get_loc(column_name)
        col_data = self.features[:, column_index].numpy()
        correlation_coefficient, _ = spearmanr(col_data, self.targets.numpy())
        print(f'Residual connection initialized weight = {correlation_coefficient:.2f}')
        return correlation_coefficient


class BoostDataset_Adv(Dataset):
    def __init__(self,dataframe_path,bins=None,test_mode=False):
        super(BoostDataset_Adv, self).__init__()
        self.test_mode = test_mode
        self.df = pd.read_pickle(dataframe_path)
        self.df['weight_adv'] = self.df['type'].apply(lambda x: 0 if x == 0 else 1)

        self.df, self.bins = self.bin_data(self.df, 'FatElectron_cM', 10, bins=bins)
        self.mass_df = pd.DataFrame(self.df['FatElectron_cM'], columns=['FatElectron_cM'])
        self.df = self.df.drop('FatElectron_cM', axis=1)

        # enforcing input data ordering for data with the same features
        ind_list = ['FatElectron_Cluster1_dRToJet', 'FatElectron_Cluster2_dRToJet',
       'FatElectron_D2', 'FatElectron_EFraction', 'FatElectron_EMFrac',
       'FatElectron_FracSamplingMax', 'FatElectron_PF', 'FatElectron_Width',
       'FatElectron_balance', 'FatElectron_dR_cc', 'Eta_Bin',
       'pT_Bin', 'FatElectron_nConstituents', 'FatElectron_ntrks','FatElectron_cM_bins', 'weight_ey_2d','weight_adv', 'type']
       # FatElectron_cM_bins
       # Check if all columns in ind_list are in self.df
        if set(ind_list).issubset(set(self.df.columns)):
            # Subset and rearrange the dataframe using reindex
            self.df = self.df.reindex(columns=ind_list)
        else:
            missing_cols = set(ind_list) - set(self.df.columns)
            print(f"Warning: The following columns are not present in the DataFrame: {missing_cols}")
        #index = pd.Index(ind_list, name='My Index')
        #self.df = self.df[index]
        self.df = self.df.reindex(columns=ind_list)
        # Create one-hot encoding for df['type']
        one_hot_type = pd.get_dummies(self.df['type'], prefix='type')
        sorted_columns = sorted(one_hot_type.columns, key=lambda x: int(x.split('_')[-1]))
        one_hot_type = one_hot_type[sorted_columns]
        # Convert to NumPy array
        one_hot_np = one_hot_type.values
        # Convert to PyTorch tensor
        self.warmup_vals = torch.tensor(one_hot_np, dtype=torch.float32)
        #print('number of df columns',len(self.df.columns))
        # we define the weights associated with the transverse momentum (P_T-bins) and pseudorapidity bins (|eta|-bins)
        self.weights = torch.tensor(self.df.loc[:, self.df.columns.isin(['weight_ey_2d'])].values, dtype=torch.long).squeeze()
        self.weights_adv = torch.tensor(self.df.loc[:, self.df.columns.isin(['weight_adv'])].values, dtype=torch.long).squeeze()
        # Separating features and target
        self.features = torch.tensor(self.df.loc[:, ~self.df.columns.isin(['weight_ey_2d','weight_adv', 'type'])].values, dtype=torch.float32)
        print(f"Number of columns in self.features: {self.features.size(1)}")
        self.targets = torch.tensor(self.df.loc[:, self.df.columns.isin(['type'])].values, dtype=torch.long).squeeze()

    def bin_data(self, df, col_name, n, bins):
        '''
        if bins is None:  # If bins are not provided, calculate new bins
            df[col_name + '_bins'], bins = pd.qcut(df[col_name], q=n, labels=False, retbins=True)
        else:
            df[col_name + '_bins'] = pd.cut(df[col_name], bins=bins, labels=False, include_lowest=True)
        '''
        bins = [50, 65, 75, 85, 90, 95, 112, 123, 127, 200, 776]
        df[col_name + '_bins'] = pd.cut(df[col_name], bins=bins, labels=False, include_lowest=True)
        return df, bins

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.test_mode == True:
            sample = self.features[index], self.targets[index], self.weights[index]
        else:
            sample = self.features[index], self.targets[index], self.weights[index], self.weights_adv[index], self.warmup_vals[index]
        return sample

    def get_labels(self):
        return self.targets

    def class_weights_calculator(self):
        class_weights = 1 / np.unique(self.df['type'], return_counts=True)[1]
        norm = np.linalg.norm(class_weights)
        class_weights /= norm
        class_weights = torch.tensor(class_weights).float()
        return class_weights

    def countclasses(self):
        totalclasslist = self.df['type'].value_counts().sort_index().tolist()
        return totalclasslist

    def input_size(self):
        t = self.features.size(1)
        return int(t)

    def num_classes(self):
        num = len(self.df['type'].unique())
        return num
    # merged, delete above function
    def num_col(self,col_name):
        num = len(self.df[col_name].unique())
        return num

    def remove_class(self, class_to_remove):
        # Remove the rows corresponding to the given class
        self.df = self.df[self.df['type'] != class_to_remove]
        # Shift the class labels of the remaining classes
        mask = self.df['type'] > class_to_remove
        self.df.loc[mask, 'type'] -= 1
        one_hot_type = pd.get_dummies(self.df['type'], prefix='type')
        sorted_columns = sorted(one_hot_type.columns, key=lambda x: int(x.split('_')[-1]))
        one_hot_type = one_hot_type[sorted_columns]
        one_hot_np = one_hot_type.values
        self.warmup_vals = torch.tensor(one_hot_np, dtype=torch.float32)
        # Update the weights, features and targets
        self.weights = torch.tensor(self.df.loc[:, self.df.columns.isin(['weight_ey_2d'])].values, dtype=torch.long).squeeze()
        self.features = torch.tensor(self.df.loc[:, ~self.df.columns.isin(['weight_ey_2d', 'type'])].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df.loc[:, self.df.columns.isin(['type'])].values, dtype=torch.long).squeeze()

    def jet_fractions(self):
        _, counts = torch.unique(self.targets, return_counts=True)
        total_samples = self.targets.shape[0]
        return counts.float() / total_samples

    def column_index(self,column_name):
        col_ind = self.df.columns.get_loc(column_name)
        print(f'\nCol index: {col_ind}\n')
        return col_ind

    def RC_initial_weight(self,column_name):
        column_index = self.df.columns.get_loc(column_name)
        col_data = self.features[:, column_index].numpy()
        correlation_coefficient, _ = spearmanr(col_data, self.targets.numpy())
        print(f'Residual connection initialized weight = {correlation_coefficient:.2f}')
        return correlation_coefficient

    def massbins(self):
        return self.bins, self.mass_df

    def colcheck(self):
        #print(self.df.columns)
        return self.df.columns

    def count_type_per_bin(self):
        grouped = self.df.groupby(['FatElectron_cM_bins', 'type']).size().reset_index(name='count')
        filtered = grouped[grouped['type'] == 0]
        return filtered[['FatElectron_cM_bins', 'count']]

    def bin_info(self,column_name):
        bins = sorted(self.df[column_name].unique())
        if bins[0] != 0:
            print("Lowest bin does not start at 0.")
        else:
            print("Lowest bin starts at 0.")
        return bins
