import pandas as pd
import numpy as np
import os
import utils.data as ds

data = './hyy_data/selected_features/' # directory where data is stored

print('Loading data...')
train_path = os.path.join(data, 'training_dataset.pkl') # path
train_dataset = ds.BoostDataset(train_path) # calling the dataset class

print('Calculating the weights...')
weights_tensor = train_dataset.class_weights_calculator()
print(f'weights calculated as: \n{weights_tensor}')

print('Saving...')
weight_path = os.path.join(data,'weights_tensor.npy')
np.save(weight_path, weights_tensor)
print('Saved!')
