HiggsNet

DNN and ANN classifiers for detection and tagging of boosted $H\rightarrow\gamma\gamma$ decays in the ATLAS detector.

In this repo there is:
-- ./hyy_data/
    Directory that contains the data
-- ./utils/
    Directory that contains the utility scripts imported by the main scripts
-- train_val_test.py
    Split the data into train., val., and test. datasets
-- weights_tensor_maker.py
    Find the weights for the loss function to address class imbalance
-- mean_std_maker.py
    Needed for the normalization layer in the architecture
-- upsample.py
    Create an upsampled training dataset
-- EDA_and_FA.py
    Perform exploratory data analysis and feature analysis
-- DNN_GS.py & ANN_GS.py
    The GridSearch scripts for each network architecture
-- DNN_toptests.py & ANN_toptests.py
    Run the tests on the top five models found during GridSearch
-- DNN_topmodel.py & ANN_topmodel.py
    Scripts to re-train the best models and save them
-- best_DNN_ANN.py
    Make the plots for the best DNN and ANN models
-- 5ClassBest.py
    For the appendix, make the plots for the 5 class versions of the DNN and ANN
-- D_Hyy_analysis.py
    Calculate the mutual information between features and scalar discriminants of the DNN and ANN
