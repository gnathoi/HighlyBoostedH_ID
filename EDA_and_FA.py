import numpy as np
import pandas as pd
import time

# custom packages to import
import utils.data as ds
import utils.networks as nw
import utils.plotting as pl
import utils.exploratory_data_analysis as eda
import utils.feature_analysis as fa

rng = 2023

start_time = time.time()

data_eda = './hyy_data/selected_features/training_dataset.pkl'
output = './features/' # output


print('Beginning EDA...')
plotter = eda.EDA(data_path=data_eda,output=output)
plotter.comptomass()
plotter.corrplot()
plotter.violin()
plotter.bar()
plotter.enlarged_bar_plot()
plotter.box()
plotter.hist()
print(f'EDA completed and all plots saved to {output}')

print('Beginning feature analysis...')
FA = fa.FeatureAnalysis(data_path=data_eda,output_path=output,n_neighbors=4,rng=rng)

mi_df, nmi_df = FA.MI()
#mi_per_class = FA.perclassMI()
#mrmr_features = FA.mRMR(features=15)
#NMI_per_class = FA.perclassNMI()
print(f'Feature analysis completed and saved to {output}')

end_time = time.time()

print(f'\nAll tasks completed! \nTime taken for all tasks: {(end_time - start_time)/60:.2f}mins\n')
