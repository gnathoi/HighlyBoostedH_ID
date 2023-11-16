import pandas as pd
import numpy as np
import os

data = './hyy_data/selected_features/' # directory where data is stored
train_path = os.path.join(data, 'training_dataset.pkl') # path

train_df = pd.read_pickle(train_path)


categ_var = ['Eta_Bin','pT_Bin','FatElectron_nConstituents','FatElectron_ntrks']
exclude_columns = categ_var + ['weight_ey_2d', 'type']
train_df = train_df.loc[:, ~train_df.columns.isin(exclude_columns)]

# calculate mean and standard deviation for each column
mean = train_df.mean()
std = train_df.std()

# create a new DataFrame that has the mean and standard deviation
mean_std = pd.DataFrame({'mean': mean, 'std': std})

# transpose the DataFrame so that it matches the structure expected by your code
mean_std = mean_std.transpose()

filepath = os.path.join(data, 'mean_std.pkl')
mean_std.to_pickle(filepath)
