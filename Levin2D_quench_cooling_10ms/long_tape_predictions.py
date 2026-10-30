import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime
import time
import sys
import os
import glob
import pickle
# Machine learning libraries
import torch


plt.close('all')

#######################################################################################################################
# Parameters:
#######################################################################################################################
# Settings:
save_model    = 1
old_model_name = '2024-11-26_16-42-43'
old_model_name = '2025-01-08_21-13-29'
old_model_name = 'NN-Moyen-3'

now = datetime.now()
now_str = '%i-%02i-%02i_%02i-%02i-%02i' % ( now.year, now.month, now.day, now.hour, now.minute, now.second )

# Ordinateur:
laptop = False

if laptop == True:
    user = "Bonjour"
else:
    user = "nbrea"

# Dataset:
directory     = "C:\\Users\\" + user + "\\OneDrive - polymtl.ca\\MAITRISE\\Codes\\Levin2D_cooling_10ms\\"
utils_path    = directory + "utils\\" 
path_material = "utils\\Material_properties\\data_spline\\" 

dataset_path = "data\\2024-11-19_16-01---20simu\\"
dataset_path = "data\\2024-12-31_13-20---20simu\\"

predictors = ['Tn-1(K)', 'Tn(K)','Q_ag(W/m2)','Q_supra(W/m2)','Q_inter(W/m2)','Q_conv(W/m2)']
target     = ['Tn+1(K)']
tag        = {'x':'xi(m)', 't':'tn(s)' }

sys.path.append(utils_path)
from utils.library_JH import *
from utils.library_nic import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "results\\2024-11-19_17-00-00-bon modèle pour résultats\\"
path = "results\\2024-11-26_16-42-43\\"
path = "results\\2025-01-12_22-43-37\\"
path= "results\\NN-Moyen-3\\"

model = torch.load( path + 'fML.pkl',  map_location=torch.device(device))
model.device = device
P_PDE = model.P_PDE
P = model.P


P_PDE['I'], P_PDE['ha'], P_PDE['R_inter'], P_PDE['Ic_montee'], P_PDE['xc'], P_PDE['prefact'] , P_PDE['sigma'] = load_geometric_parameters(directory + dataset_path + "truth.csv",P_PDE)


#######################################
# IMPORT DATA
#######################################



print("Loading DATA and preprocessing it...")
start_time = time.time()
DATA_test  = load_data( directory + dataset_path + "test/*.csv", P_PDE)
elapsed_time = time.time() - start_time
print(f"Preprocess done. Elapsed time: {elapsed_time:.2f} seconds")
####################################

multiple_20 = 10
DATA_test['Tn-1(K)'] = np.tile(DATA_test['Tn-1(K)'], (1, 1, multiple_20))
DATA_test['Tn(K)'] = np.tile(DATA_test['Tn(K)'], (1, 1, multiple_20))
DATA_test['Tn+1(K)'] = np.tile(DATA_test['Tn+1(K)'], (1, 1, multiple_20))
DATA_test['Q_ag(W/m2)'] = np.tile(DATA_test['Q_ag(W/m2)'], (1, 1, multiple_20))
DATA_test['Q_supra(W/m2)'] = np.tile(DATA_test['Q_supra(W/m2)'], (1, 1, multiple_20))
DATA_test['Q_inter(W/m2)'] = np.tile(DATA_test['Q_inter(W/m2)'], (1, 1, multiple_20))
DATA_test['Q_conv(W/m2)'] = np.tile(DATA_test['Q_conv(W/m2)'], (1, 1, multiple_20))
DATA_test['simu_id'] = np.linspace(1,20*multiple_20,20*multiple_20, dtype=int)
P_PDE['xc'] = np.tile(P_PDE['xc'],(multiple_20,1))
P_PDE['prefact'] = np.tile(P_PDE['prefact'],(multiple_20,1))
P_PDE['sigma'] = np.tile(P_PDE['sigma'],(multiple_20,1))
##############################################################
# Long Tape data manipulation:
geo_index = np.array(DATA_test['simu_id'])-1

DATA_test,P_PDE['x'] = reshape_dict_long_tape(DATA_test,P_PDE['x'],P_PDE['t'],geo_index)

P_PDE['xc'] = xc_translation_long_tape(P_PDE['xc'],geo_index,P_PDE)
P_PDE['prefact'] = np.expand_dims(P_PDE['prefact'].flatten(),axis=0)
P_PDE['sigma'] = np.expand_dims(P_PDE['sigma'].flatten(),axis=0)


########################################################################################################################
# Predictions
########################################################################################################################
print("Make predictions:")
#t0 = time.time()


# Predictions AI model
#model.eval()
#time_test, T_true, T_abc, Q_abc = model_predictions_hPGRNN(model,DATA_test, P_PDE)

# Predictions Dufort-Frankel
#t0_DF = time.time()
#T_DF, erreur_rel,erreur_abs = DufortFrankel_Levin1D(DATA_test,P_PDE)
#t1_DF = time.time()

# Calculate elapsed time for Dufort-Frankel
#time_DF = t1_DF - t0_DF
#print(f"Finite Difference Dufort-Frank resolution time: {time_DF:.6f} seconds")
#t1 = time.time()
#time_pred = t1- t0
#print(f"hPGRNN resolution time: {time_pred-time_DF:.6f} seconds")
#print(f"Total time for all predictions: {time_pred:.6f} seconds")
#t0_DF = time.time()
#model_DF = Levin1D(P_PDE)
#T_DF, Js_DF, rhos_DF = predictions_all_simulations(model_DF, DATA_test, P_PDE)
#t1_DF = time.time()
#print(f"Simulation time for 1-D Levin: {t1_DF- t0_DF:.6f} seconds")

# Prédictions par le modèle de RAPDi
t0 = time.time()
model.eval()
model = transfert_to_device(model,"cpu")
T_model, Js_model, rhos_model = predictions_all_simulations(model, DATA_test, P_PDE)
t1 = time.time()
print(f"Simulation time for RAPDi: {t1- t0:.6f} seconds")

# Données FEM
T_true, Js_true, rhos_true = solutions_FEM(DATA_test,P_PDE)

# Calcul des erreurs RMSE
RMSE_model      = RMSE_metric(T_true,T_model)
RMSE_DF         = RMSE_metric(T_true,T_DF)
RMSE_model  = RMSE_model[2:]
RMSE_DF     = RMSE_DF[2:]
# Calcul des norme euclidienne de l'erreur NEE en température
NEE_model      = eucli_metric(T_true,T_model)
NEE_DF         = eucli_metric(T_true,T_DF)

# Calcul des norme euclidienne de l'erreur NEE en Js
NEE_Js_model      = eucli_metric(Js_true,Js_model)
NEE_Js_DF         = eucli_metric(Js_true,Js_DF)

# Saving results in Lists
models_pred = []
STEP_models = []
RMSE_models = []


models_pred.append(T_model)
RMSE_models.append(RMSE_DF)
RMSE_models.append(RMSE_model)
models_name =["RAPDi"]

########################################################################################################################
# 4. Display Results
########################################################################################################################

print("Display results:")
os.mkdir('results\\Long_tape\\'+ now_str )
with open('results/Long_tape/' + now_str +'/T_models.pkl', 'wb') as f:
    pickle.dump(models_pred, f) 
with open('results/Long_tape/' + now_str +'/T_DF.pkl', 'wb') as f:
    pickle.dump(T_DF, f)
with open('results/Long_tape/' + now_str +'/T_true.pkl', 'wb') as f:
    pickle.dump(T_true, f)
torch.save(model,    'results\\Long_tape\\'+now_str+'\\fML.pkl' )

now_str = 'Long_tape\\'+ now_str
# 2.1) Save x vs T figures at different time steps for all models
multisteps_plot(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model)

# 2.2) Plot the final form
summary_multisteps_plot(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model,0)

# 2.3) Plot the final current form
summary_multisteps_plot_current(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model)

# 3) RMSE lot recursive-error over time 
plot_recursive_rmse(P_PDE,RMSE_models,models_name,now_str,save_model)

# 6) GIF - Visulalisation
for index_dataset in range(T_true.shape[2]):
    plot_gif_predictions(T_true[:,:,index_dataset],T_model[:,:,index_dataset],T_DF[:,:,index_dataset],P_PDE,now_str,index_dataset)


P['meanloss'] =  np.mean(RMSE_models[1])
P['finalloss'] = RMSE_models[1][-1]
print('######################')
print('Mean Loss: ' + str(np.mean(RMSE_models[1])))
print('Final Loss: ' + str(RMSE_models[1][-1]))


print("test")