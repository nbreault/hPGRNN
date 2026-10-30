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
run_training  = 'DisplayOld' # 'Run' or 'DisplayOld'
old_model_name = 'NN-Gros-4'


# Dataset:
directory     = ""
utils_path    = directory + "utils\\" 
path_material = "utils\\Material_properties\\data_spline\\" 


dataset_path = "data\\2025-05-01_14-36---100simu\\"

predictors = ['Tn-1(K)', 'Tn(K)','Q_ag(W/m2)','Q_supra(W/m2)','Q_inter(W/m2)','Q_conv(W/m2)']
target     = ['Tn+1(K)']
tag        = {'x':'xi(m)', 't':'tn(s)' }

sys.path.append(utils_path)
from utils.library_JH import *
from utils.library_nic import *

# Paramètres EDP:
P_PDE = {
    't_end'            : 0.01, # Temp total de la simulation      
    'L'                : 0.14,  # (m) Longueur tape 0.04
    'w'                : 0.002, # Largeur tape
    'Nt'               : 301,   # Nombre de points temporel
    'Nx'               : 301,   # Nombre de points spatial 
    'Tc'               : 90,    # Température critique
    'T0'               : 77,    # (K) Temperature Ambiante
    'E0'               : 1e-4,  # (V/m)
    'h_conv'           : 2000,  # (W/(m^2 K)
    'n'                : 20,    # Exposant PowerLaw
    'hs'               : 2e-6,  # (m) Épaisseur supra (ybco)
    'ha'               : 4e-6,  # (m) Épaisseur argent 
    'hhast'            : 50e-6, # (m) Épaisseur hastelloy
    'rhom_a'           : 10490, # (kg/m^3)
    'rhom_s'           : 6390,  # (kg/m^3)
    'rhom_hast'        : 8890,  # (kg/m^3)
    'Ic'               : 200,   # (A) Courant critique
    'rho_s_normal'     : 1e-6,  # Résistivité Supra T > Tc
    'toll_solverNR'    : 1e-15, # Tolérance relative L2, solver Newton-Raphson
        }

# Paramètres ML pour modèle CoefNet
P_MCE = {
    'kernel_size'          : 1,
    'dtype'                : torch.float, # torhc.float/float32 or float64
    'steps_init'           : 2,           # (2)       Initial number of recurrent steps during training 
    'tolerance'            : 1e-1,        # (5e-4)3e-1    Tolerance
    'steps_increment'      : 1,           # (1)       Steps increase by a amount of 'steps_increment' every 'epochs_before_new_step' epochs
    'steps_max'            : 2,           # (2)       Maximum number of epoch before increase steps if tolerance not reached
    'lambda_L2'            : 0,
    'dropout_prob'         : 0.1,         
    'batch_size'           : 32,         # (100)      Mini-batch size
    'l_r'                  : 1e-5,        # (1e-3)    Learning rate
    'epochs'               : 100,
    'scheduler'            : False,        # Do we want a constant ADAMS or a scheduler
    'total_attenuation'    : 1/100,       # atténuation du learning rate à la fin          
    'CONV1_frozen'         : True,        # 
    'CONV1_init'           : [1,0,1],     # [1.02,0,0.95]
    
    'CONV2_frozen'         : True,        # 
    'CONV2_init'           : [-1,0,1],    # [-1,0,1]c
    'CONV2_scale'          : 1,         # (0.5)
    
    'k_eff_scale'          : 2.3e-3,         # (0.3)   
    'nn_keff'             : [1,20,30,1], # [1,10,10,1] #----# Architecture du réseau de neurones
    
    'Ceff_scale'     : 130,         # (1-7)      
    'nn_C'        : [1,21,20,1], # [1,10,10,1] #----# Architecture du réseau de neurones
    'seq_len'              : 1,
    }

P_hPGRNN = {
    'kernel_size'          : 31, # Doit être un nombre impair
    'dtype'                : torch.float, # torhc.float/float32 or float64
    'steps_init'           : 2,           # (2)       Initial number of recurrent steps during training 
    'tolerance'            : 6e-1,        # (5e-4)3e-1    Tolerance
    'steps_increment'      : 1,           # (1)       Steps increase by a amount of 'steps_increment' every 'epochs_before_new_step' epochs
    'steps_max'            : 2,           # (2)       Maximum number of epoch before increase steps if tolerance not reached
    'lambda_L2'            : 0,
    'dropout_prob'         : 0.2,         
    'batch_size'           : 32,         # (100)      Mini-batch size
    'l_r'                  : 0.75e-4,        # (1e-3)    Learning rate
    'epochs'               : 200,
    'nn_structure'         : [186,400,399,401,400,401,6], # [1,10,10,1] #----# Architecture du réseau de neurones , 
    }

P = P_hPGRNN

P_PDE['dx'] = P_PDE['L']/(P_PDE['Nx']-1) # delta x
P_PDE['dt'] = P_PDE['t_end']/(P_PDE['Nt']-1) # delta t
P_PDE['As'] = P_PDE['w']*P_PDE['hs'] # Superconducter/ybco area
P_PDE['Aa'] = P_PDE['w']*P_PDE['ha'] # Ag area
P_PDE['Jc0'] = P_PDE['Ic']/(2*P_PDE['As']) # Critical current density
P_PDE['x'] = np.linspace(0,P_PDE['L'],P_PDE['Nx'])
P_PDE['t'] = np.linspace(0,P_PDE['t_end'],P_PDE['Nt'])
P_PDE['t_ML'] = P_PDE['t']
P_PDE['dt_ML'] = P_PDE['dt']

P_PDE['I'], P_PDE['ha'], P_PDE['R_inter'], P_PDE['Ic_montee'], P_PDE['xc'], P_PDE['prefact'] , P_PDE['sigma'] = load_geometric_parameters(directory + dataset_path + "truth.csv",P_PDE)

P_PDE['ha'][:] = 4e-6
#_PDE['R_inter'][:] = P_PDE['R_inter'].mean()

#######################################
# IMPORT DATA
#######################################

# Use GPU if avalaible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P['device'] = device

print("Loading DATA and preprocessing it...")
start_time = time.time()
DATA_train = load_data( directory + dataset_path + "train/*.csv", P_PDE)
DATA_val = load_data( directory + dataset_path + "val/*.csv", P_PDE)
DATA_test  = load_data( directory + dataset_path + "test/*.csv", P_PDE)
elapsed_time = time.time() - start_time
print(f"Preprocess done. Elapsed time: {elapsed_time:.2f} seconds")

# Supprime les simulations telles que I >= Ic = 200 dans le training set. 
if (2*P_PDE['I'] >= P_PDE['Ic']).any():
    dead_indices = np.where(2*P_PDE['I'] > 200)[0]
    Ns_test = len(DATA_test['simu_id'])
    Ns_val = len(DATA_val['simu_id'])

    # Retire les simulations qui sont dans le test et val set.
    dead_indices = dead_indices[dead_indices > Ns_test + Ns_val]
    dead_indices = dead_indices - (Ns_test + Ns_val)
    all_indices = np.arange(DATA_train['Tn+1(K)'].shape[2])
    keep_indices = np.setdiff1d(all_indices, dead_indices)
    DATA_train['Tn+1(K)'] = DATA_train['Tn+1(K)'][:,:,keep_indices]
    DATA_train['Tn(K)'] = DATA_train['Tn(K)'][:,:,keep_indices]
    DATA_train['Tn-1(K)'] = DATA_train['Tn-1(K)'][:,:,keep_indices]
    #DATA_train['Js(A/m2)'] = DATA_train['Js(A/m2)'][:,:,keep_indices]
    DATA_train['Q_ag(W/m2)'] = DATA_train['Q_ag(W/m2)'][:,:,keep_indices]
    DATA_train['Q_supra(W/m2)'] = DATA_train['Q_supra(W/m2)'][:,:,keep_indices]
    DATA_train['Q_inter(W/m2)'] = DATA_train['Q_inter(W/m2)'][:,:,keep_indices]
    DATA_train['Q_conv(W/m2)'] = DATA_train['Q_conv(W/m2)'][:,:,keep_indices]
    DATA_train['simu_id'] = [DATA_train['simu_id'][i] for i in keep_indices.tolist()]

# Initialisation des structures d'interpolation
load_materials_splines(P_PDE,P,utils_path)

P_PDE['rho_ag_spline'] = pickle.load( open( directory + path_material + "spline_rho_ag.pkl",   "rb" ) )
P_PDE['Cp_ag_spline'] = pickle.load( open( directory + path_material + "spline_Cp_ag.pkl",   "rb" ) )
P_PDE['k_ag_spline'] = pickle.load( open( directory + path_material + "spline_k_ag.pkl",   "rb" ) )
P_PDE['dkdT_ag_spline'] = pickle.load( open( directory + path_material + "spline_dkdT_ag.pkl",   "rb" ) )
P_PDE['Cp_ybco_spline'] = pickle.load( open( directory + path_material + "spline_Cp_ybco.pkl",   "rb" ) )
P_PDE['k_ybco_spline'] = pickle.load( open( directory + path_material + "spline_k_ybco.pkl",   "rb" ) )
P_PDE['dkdT_ybco_spline'] = pickle.load( open( directory + path_material + "spline_dkdT_ybco.pkl",   "rb" ) )
P_PDE['Cp_hast_spline'] = pickle.load( open( directory + path_material + "spline_Cp_hast.pkl",   "rb" ) )
P_PDE['k_hast_spline'] = pickle.load( open( directory + path_material + "spline_k_hast.pkl",   "rb" ) )
P_PDE['dkdT_hast_spline'] = pickle.load( open( directory + path_material + "spline_dkdT_hast.pkl",   "rb" ) )
P_PDE['k_eff_150'] = func_keff(np.array(150),P_PDE,0)
P_PDE['rhomCp_eff_150'] = func_rhomCpeff(np.array(150),P_PDE,0)

# Variables for normalization (Neural Network inputs) 
P_PDE['T_min'] = DATA_train[target[-1]].min()
P_PDE['T_max'] = DATA_train[target[-1]].max()
P_PDE['Q_min_cool'] = DATA_train['Q_conv(W/m2)'].min() 
P_PDE['Q_max_cool'] = DATA_train['Q_conv(W/m2)'].max()
P_PDE['Q_min_inter'] = DATA_train['Q_inter(W/m2)'].min() 
P_PDE['Q_max_inter'] = DATA_train['Q_inter(W/m2)'].max()
P_PDE['Q_min_ag'] = DATA_train['Q_ag(W/m2)'].min() 
P_PDE['Q_max_ag'] = DATA_train['Q_ag(W/m2)'].max()
P_PDE['Q_min_supra'] = DATA_train['Q_supra(W/m2)'].min() 
P_PDE['Q_max_supra'] = DATA_train['Q_supra(W/m2)'].max()
P_PDE['Q_min_ag_supra'] = torch.min(DATA_train['Q_ag(W/m2)'] + DATA_train['Q_supra(W/m2)'] )
P_PDE['Q_max_ag_supra'] = torch.max(DATA_train['Q_ag(W/m2)'] + DATA_train['Q_supra(W/m2)'] )


# Max Heat and Temperature outside the test set:
Q_max_train = torch.max(DATA_train['Q_ag(W/m2)']+DATA_train['Q_supra(W/m2)']+ DATA_train['Q_inter(W/m2)'] + DATA_train['Q_conv(W/m2)'])
Tmax_test =  DATA_test[target[-1]].max()
Qmax_test = torch.max(DATA_test['Q_ag(W/m2)']+ DATA_test['Q_inter(W/m2)']+ DATA_test['Q_supra(W/m2)']+DATA_test['Q_conv(W/m2)'])
Qmin_test = torch.min(DATA_test['Q_ag(W/m2)']+ DATA_test['Q_inter(W/m2)']+ DATA_test['Q_supra(W/m2)']+DATA_test['Q_conv(W/m2)'])
if Qmax_test > Q_max_train or Qmin_test < torch.min(P_PDE['Q_min_ag_supra']+P_PDE['Q_min_cool']) or Tmax_test > P_PDE['T_max']:
    Warning("Maximum/Minimum temperature or heat is higher in test set than training! Fix this!")
    print("Maximum/Minimum temperature or heat is higher in test set than training! Fix this!")

########################################################################################################################
# 2. Train model
########################################################################################################################

# Data-time
now = datetime.now()
now_str = '%i-%02i-%02i_%02i-%02i-%02i' % ( now.year, now.month, now.day, now.hour, now.minute, now.second )

if run_training.lower()=='run':

    # Timer
    print('Start training...')
    t0 = time.time()
    

    # Training model
    model = hPGRNN(P,P_PDE)
    #model = MCE(P,P_PDE,device)
    #model.to(device)

    #model = torch.load( 'results\\'+old_model_name+'\\fML.pkl',  map_location=torch.device(device))
    #model = transfert_to_device(model,"cuda")
    #model.device = device 
    #model.P = P
    model.train()
    Loss_train_epoch, Loss_val_epoch, STEPS = model.train_thermal( DATA_train, DATA_val, P['epochs'])
    
    print('Done')

    t1 = time.time()
    time_laps = t1 - t0
    print('Training Time: %.2f min (%.2f h)\n'%( (t1-t0)/60, (t1-t0)/3600 ))

    if save_model:
    # Create folder
        os.mkdir('results\\'+ now_str ) 
        torch.save(model,    'results\\'+now_str+'\\fML.pkl' )

elif run_training.lower()=='displayold':
        # Load model
        model = torch.load( 'results\\'+old_model_name+'\\fML.pkl',  map_location=torch.device(device))
        model.device = device
        os.mkdir('results\\'+ now_str )

########################################################################################################################
# 3. Predictions
########################################################################################################################
print("Make predictions:")

# Prédictions par le modèle de Levin1D
t0_DF = time.time()
model_DF = Levin1D(P_PDE)
T_DF, Js_DF, rhos_DF = predictions_all_simulations(model_DF, DATA_test, P_PDE)
t1_DF = time.time()
print(f"Simulation time for 1-D Levin: {t1_DF- t0_DF:.6f} seconds")

# Prédictions par le modèle de hPGRNN
t0 = time.time()
model.eval()
model = transfert_to_device(model,"cpu")
T_model, Js_model, rhos_model = predictions_all_simulations(model, DATA_test, P_PDE)
t1 = time.time()
print(f"Simulation time for hPGRNN: {t1- t0:.6f} seconds")

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

# Calcul des norme euclidienne de l'erreur NEE en Vs
E_true  = rhos_true*Js_true 
E_model = rhos_model*Js_model 
E_DF    = rhos_DF*Js_DF
NEE_Vs_model = erreur_moyenne_voltage(P_PDE['x'],E_true[:,::10,:],E_model[:,::10,:])[:-1]
NEE_Vs_DF = erreur_moyenne_voltage(P_PDE['x'],E_true[:,::10,:],E_DF[:,::10,:])[:-1]
# Saving results in Lists

LOSS_train = []
LOSS_val = []
models_pred = []
EPOCH = []
STEP_models = []
RMSE_models = []

if run_training.lower()=='run':
    LOSS_train.append(Loss_train_epoch)
    LOSS_val.append(Loss_val_epoch)
    EPOCH.append(P['epochs'])
    STEP_models.append(STEPS) 
models_pred.append(T_model)
RMSE_models.append(RMSE_DF)
RMSE_models.append(RMSE_model)
models_name =["hPGRNN"]

########################################################################################################################
# 4. Display Results
########################################################################################################################

print("Display results:")
with open('results/' + now_str +'/T_models.pkl', 'wb') as f:
    pickle.dump(models_pred, f) 
with open('results/' + now_str + '/Js_model.pkl', 'wb') as f:
    pickle.dump(Js_model, f)
with open('results/' + now_str + '/Js_DF.pkl', 'wb') as f:
    pickle.dump(Js_DF, f)
with open('results/' + now_str + '/Js_true.pkl', 'wb') as f:
    pickle.dump(Js_true, f)

P['meanloss_RMSE'] =  np.mean(RMSE_models[1])
P['finalloss_RMSE'] = RMSE_models[1][-1]

P['meanloss_eucli_T'] =  np.mean(NEE_model)
P['finalloss_eucli_T'] = NEE_model[-1]

P['meanloss_eucli_Js'] =  np.mean(NEE_Js_model)
P['finalloss_eucli_Js'] = NEE_Js_model[-1]
P['meanloss_eucli_Vs'] =  np.mean(NEE_Vs_model)
P['finalloss_eucli_Vs'] = NEE_Js_model[-1]

print('######################')
print('Final Loss T (K): ' + str(NEE_model[-1]))
print('Mean Loss T (K): ' + str(np.mean(NEE_model)))
print('Mean Loss Js (MA/cm^2): ' + str(np.mean(NEE_Js_model)/10**10))
print('Mean Loss Vs (V): ' + str(np.mean(NEE_Vs_model)))

# 1) Training Plot
if run_training.lower()=='run':
    plot_training_results(LOSS_train, LOSS_val, models_pred, models_name, EPOCH, STEP_models, now_str)

# 2.1) Save x vs T figures at different time steps for all models
multisteps_plot(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model)

# 2.2) Plot the final form
summary_multisteps_plot(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model,0)

# 2.3) Plot the final current form
#summary_multisteps_plot_current(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model)

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

# 4) Coefficients plots
P_PDE = dict_transfer_tensors_to_cpu(P_PDE)

# 5) save data
if save_model and (run_training.lower() == 'run'):
    save_data_txt(now_str,DATA_train,time_laps,device,predictors,target,directory,dataset_path,P,P_PDE)

print("test")