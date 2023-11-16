import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import entropy, gaussian_kde
from scipy.integrate import quad
from tqdm import tqdm
import time
import pymrmr

'''
It should be noted that class imbalance is an issue for divergence based metrics.
Therefore, we recommend that you use the upsampled dataset when calculating MI, NMI, mRMR.
'''

class FeatureAnalysis():
    def __init__(self,data_path,output_path,n_neighbors,rng,D_values=None,remove_class=None):
        self.data_path = data_path
        self.output_path = output_path
        self.n_neighbors = n_neighbors
        self.rng = rng
        self.D_values = D_values

        #print('Loading data...')
        self.dataset = pd.read_pickle(self.data_path)
        self.categ_var = ['Eta_Bin','pT_Bin','FatElectron_nConstituents','FatElectron_ntrks']
        self.cont_var = [col for col in self.dataset.columns if col not in self.categ_var and col != 'type' and col != 'weight_ey_2d' and col != 'D_Hyy']

        if remove_class != None:
            self.dataset = self.dataset[self.dataset['type']!=remove_class]

        if 'weight_ey_2d' in self.dataset.columns:
            self.dataset.drop('weight_ey_2d', axis=1, inplace=True)

        self.dataset = pd.concat([self.dataset['type'], self.dataset.drop('type', axis=1)], axis=1) # do this up here for mRMR to save memory
        if self.D_values is not None:
            self.dataset['D_Hyy'] = self.D_values
            self.X = self.dataset.drop(['type','D_Hyy'], axis=1)
            self.y = self.dataset['D_Hyy']
        else:
            self.X = self.dataset.drop('type', axis=1)
            self.y = self.dataset['type']

    # Function for mutual information and normalized mutual information
    def MI(self):

        start_time = time.time()

        output_MI = os.path.join(self.output_path,'MI_scores.txt')
        output_NMI = os.path.join(self.output_path,'NMI_scores.txt')
        '''
        def calc_entropy(column):
            # calculate entropy of a pandas Series, using base 2
            _, counts = np.unique(column, return_counts=True)
            return entropy(counts, base=2)
        '''
        def calc_entropy(column, var_type):
            if var_type == 'continuous':
                def integrand(x):
                    px = kde(x)
                    if px < 1e-10:  # Avoid log(0)
                        return 0.0
                    else:
                        return -px * np.log2(px)

                kernel_width = np.std(column) * np.power(len(column), -1.0 / 5)  # Scott's rule
                kde = gaussian_kde(column, bw_method=kernel_width)

                # These integrals are hard so we use Monte Carlo integration
                N = 10000  # number of samples
                samples = np.random.choice(column, N)
                estimates = [-p * np.log2(p) for p in kde(samples) if p > 1e-10]

                entropy_estimate = np.mean(estimates)
                return entropy_estimate
            elif var_type == 'categorical':
                _, counts = np.unique(column, return_counts=True)
                return entropy(counts, base=2)

        print('Calculating mutual information...')
        if self.D_values is not None:
            mi = mutual_info_regression(self.X,self.y,random_state=self.rng,n_neighbors=self.n_neighbors)
        else:
            mi = mutual_info_classif(self.X,self.y,random_state=self.rng,n_neighbors=self.n_neighbors)

        # Create a DataFrame with feature names from X_train and their corresponding mi scores

        print('Saving MI...')
        mi_df = pd.DataFrame({'Feature': self.X.columns, 'MI_Score': mi})

        # Sort the DataFrame by the MI_Score in descending order
        mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

        # Write the DataFrame to a text file
        mi_df.to_csv(output_MI, index=False, sep='\t')
        print(f'Saved to {output_MI}')

        print(f'MI result:\n{mi_df}')

        print('\nCalculating feature and target entropies...')
        '''
        # Calculate entropy for each feature
        feature_entropies = np.array([calc_entropy(self.X[col]) for col in self.X.columns])
        # Calculate entropy for target variable
        target_entropy = calc_entropy(self.y)
        '''
        feature_entropies = []
        for col in self.X.columns:
            if col in self.categ_var:
                feature_entropies.append(calc_entropy(self.X[col], 'categorical'))
            elif col in self.cont_var:
                feature_entropies.append(calc_entropy(self.X[col], 'continuous'))

        feature_entropies = np.array(feature_entropies)

        if self.D_values is not None:
            target_var_type = 'continuous'  # Change this to 'categorical' if needed
        else:
            target_var_type = 'categorical'

        target_entropy = calc_entropy(self.y, target_var_type)

        # Calculate NMI
        print('Calculaing normalized mutual infomation...')
        nmi = mi / np.sqrt(feature_entropies * target_entropy)

        nmi_df = pd.DataFrame({'Feature': self.X.columns, 'NMI_Score': nmi})
        # Sort the DataFrame by the NMI_Score in descending order
        nmi_df = nmi_df.sort_values(by='NMI_Score', ascending=False)

        # Write the DataFrame to a text file
        print('Saving NMI...')
        nmi_df.to_csv(output_NMI, index=False, sep='\t')
        print(f'Saved to {output_NMI}')
        print(f'NMI result:\n{nmi_df}')
        end_time = time.time()
        print(f'\nTime taken for MI and NMI: {(end_time - start_time)/60:.2f}mins\n')
        return mi_df, nmi_df

    # Function to calculate the per class mutual infomation with the classes
    def perclassMI(self):
        start_time = time.time()
        #output = os.path.join(output_path,'mi_per_class.txt')
        # Get unique classes in target variable
        classes = np.unique(self.y)

        # Initialize a dictionary to store MI for each class
        mi_per_class = {}

        print('Calculating mutual information per class...')
        for cls in tqdm(classes, desc='Processing classes'):
            # Subset the data for the current class
            X_cls = self.X[self.y == cls]

            # Calculate MI for the current class
            mi_cls = mutual_info_classif(X_cls, self.y[self.y == cls], random_state=self.rng, n_neighbors=self.n_neighbors)

            # Store the MI scores in the dictionary
            mi_per_class[cls] = pd.DataFrame({'Feature': X_cls.columns, 'MI_Score': mi_cls}).sort_values(by='MI_Score', ascending=False)

        print('Saving mutual information per class...')
        for cls, df in mi_per_class.items():
            output = os.path.join(self.output_path, f'mi_class_{cls}.txt')  # Create separate output file for each class
            df.to_csv(output, index=False, sep='\t')
            print(f'Saved to {output}')

        end_time = time.time()
        print(f'\nTime taken for MI per class: {(end_time - start_time)/60:.2f}mins\n')
        return mi_per_class

    def perclassNMI(self, select_class=None):
        start_time = time.time()

        # Get unique classes in target variable
        classes = np.unique(self.y)

        # If select_class is specified, only calculate for that class
        if select_class is not None:
            if select_class in classes:
                classes = [select_class]
            else:
                print(f"Class {select_class} not found in target variable. Please select a valid class or set select_class to None.")
                return

        # Initialize a dictionary to store NMI for each class
        nmi_per_class = {}

        # Function to calculate entropy
        def calc_entropy(column):
            _, counts = np.unique(column, return_counts=True)
            return entropy(counts, base=2)

        print('Calculating normalized mutual information per class...')
        for cls in tqdm(classes, desc='Processing classes'):
            # Subset the data for the current class
            X_cls = self.X[self.y == cls]

            # Calculate MI for the current class
            mi_cls = mutual_info_classif(X_cls, self.y[self.y == cls], random_state=self.rng, n_neighbors=self.n_neighbors)

            # Calculate entropy for each feature for the current class
            feature_entropies_cls = np.array([calc_entropy(X_cls[col]) for col in X_cls.columns])

            # Calculate entropy for target variable for the current class
            target_entropy_cls = calc_entropy(self.y[self.y == cls])

            # Calculate NMI for the current class
            epsilon = 1e-10
            nmi_cls = mi_cls / (np.sqrt(feature_entropies_cls * target_entropy_cls)+epsilon)

            # Store the NMI scores in the dictionary
            nmi_per_class[cls] = pd.DataFrame({'Feature': X_cls.columns, 'NMI_Score': nmi_cls}).sort_values(by='NMI_Score', ascending=False)

        print('Saving normalized mutual information per class...')
        for cls, df in nmi_per_class.items():
            output = os.path.join(self.output_path, f'nmi_class_{cls}.txt')  # Create separate output file for each class
            df.to_csv(output, index=False, sep='\t')
            print(f'Saved to {output}')

        end_time = time.time()
        print(f'\nTime taken for NMI per class: {(end_time - start_time)/60:.2f}mins\n')
        return nmi_per_class


    def mRMR(self,features):
        start_time = time.time()
        output =  os.path.join(self.output_path, f'mRMR_features{features}.txt')
        print(f'\nTotal number of features = {len(self.dataset.columns)-1}')
        print(f'Calculating {features} mRMR selected features...')
        selected_features = pymrmr.mRMR(self.dataset, 'MIQ', features) # this function treats the last column as the target

        print('Saving...')
        with open(output, 'w') as f:
            for item in selected_features:
                f.write("%s\n" % item)

        print(f'Saved at {output}')
        end_time = time.time()
        print(f'\nTime taken for mRMR: {(end_time - start_time)/60:.2f}mins\n')
        return selected_features
