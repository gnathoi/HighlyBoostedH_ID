import pandas as pd
import numpy as np
import scipy as sp
import os
from sklearn.model_selection import train_test_split

rng = 2023

output = './hyy_data/selected_features/'
if not os.path.exists(output):
    os.makedirs(output)

print('Loading data...')
df = pd.read_hdf('./hyy_data/new_ZeeAndhyyTagging_Inputfile_9thAugust2023.h5',key='FatElectron')

#print(df.columns)

cols_to_keep = ['Eta_Bin',
 'FatElectron_Cluster1_dRToJet',
 'FatElectron_Cluster2_dRToJet',
 'FatElectron_D2',
 'FatElectron_EFraction',
 'FatElectron_EMFrac',
 'FatElectron_FracSamplingMax',
 'FatElectron_PF',
 'FatElectron_Width',
 'FatElectron_balance',
 'FatElectron_cM',
 'FatElectron_dR_cc',
 'FatElectron_nConstituents',
 'FatElectron_ntrks',
 'pT_Bin',
 'type',
 'weight_ey_2d']

# Assuming you have a DataFrame df and a list of columns cols_to_keep
dropcolumns = [col for col in df.columns if col not in cols_to_keep]
#df = df.drop(columns=cols_to_drop)

#print(f'Dropping columns: \n{dropcolumns}')
df = df.drop(dropcolumns, axis=1)

categ_var = ['Eta_Bin','pT_Bin','FatElectron_nConstituents','FatElectron_ntrks']#'FatElectron_cM_bins']
# Select the categorical columns
categ_df = df[categ_var]

# Select the specific columns 'weight_ey_2d' and 'type'
special_cols_df = df[['weight_ey_2d', 'type']]

# Drop the selected columns from the original DataFrame to keep the remaining columns
df = df.drop(columns=categ_var + ['weight_ey_2d', 'type'])

# Concatenate all the parts in the desired order
df = pd.concat([df, categ_df, special_cols_df], axis=1)


# Separate features (X) and target column (y)
X = df.drop('type', axis=1)
y = df['type']

print('Converting types to correct ordinal encoding for pytorch...')
# convert types to the categories
y.replace({42: 0, 1: 1, 5: 2, 6: 2, 3: 3, 7: 3, 8: 4, 9: 4}, inplace=True)

print('Creating training dataset...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=rng, stratify=y)
print('Creating validation and test datasets...')
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=rng, stratify=y_test)


print('\nChecking stratification:')
print('y_train classes')
print(y_train.value_counts().sort_index())

print('y_val classes')
print(y_val.value_counts().sort_index())

print('y_test classes')
print(y_test.value_counts().sort_index())

print('Saving...')
train_path = os.path.join(output, 'training_dataset.pkl')
df_train = pd.concat([X_train,y_train],axis=1)
df_train.to_pickle(train_path)

valid_path = os.path.join(output, 'validation_dataset.pkl')
df_val = pd.concat([X_val,y_val],axis=1)
df_val.to_pickle(valid_path)

test_path = os.path.join(output, 'test_dataset.pkl')
df_test = pd.concat([X_test,y_test],axis=1)
df_test.to_pickle(test_path)

print(f'Saved to {output}')
