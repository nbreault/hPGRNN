"""
Auteur: Nicolas Breault
Some functions were taken from library_JH.
"""
# Standard modules                    
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import time
from scipy.integrate import simpson
from scipy.sparse import spdiags, diags, linalg
import seaborn as sns
import pickle
from typing import Dict
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.animation import FuncAnimation, PillowWriter
import math
from numba import njit

def dict_transfer_tensors_to_cpu(Dict):
    """ Parcours le dictionnaire pour trouver les tensors et les convertis sur le cpi
    """
    for key, value in Dict.items():
        if isinstance(value, torch.Tensor):
            Dict[key] = value.cpu()
    return Dict

def transfert_to_device(model,label):
    "model + label in string"

    if label.lower() == "cpu":
        model = model.cpu()
        model.device = torch.device('cpu')
        for attr in model.__dict__:
            val = getattr(model, attr)
            if torch.is_tensor(val):
                setattr(model, attr, val.cpu())

    elif label.lower() == "cuda":
        model = model.cuda()
        model.device = torch.device('cuda')
        for attr in model.__dict__:
            val = getattr(model, attr)
            if torch.is_tensor(val):
                setattr(model, attr, val.cuda())
    else:
        print('Only device available are "cpu" or "cuda" ')

    return model

def RMSE_metric(T_true, T_model):
    mse = np.mean((T_true - T_model) ** 2, axis=(0, 2))  # moyenne sur Nx et Ns
    rmse = np.sqrt(mse)
    return rmse  

def eucli_metric(T_true, T_model):
    # T_true et T_model ont la forme (Nx, Nt, Ns)
    diff = T_true - T_model  # (Nx, Nt, Ns)
    norm_spatial = np.linalg.norm(diff, axis=0)  # norme sur Nx → (Nt, Ns)
    e_t = np.mean(norm_spatial, axis=1)  # moyenne sur Ns → (Nt,)

    return e_t

def erreur_moyenne_voltage(x,E1,E2):
    Nx,Nt,Ns = E1.shape

    V1 = np.zeros((Nx,Nt,Ns))
    V2 = np.zeros((Nx,Nt,Ns))

    for s in range(Ns):
        for t in range(Nt-1):
            for xi in range(Nx-1):
                V1[xi+1,t,s] = -trapezoid(E1[0:xi+1,t,s], x[0:xi+1])
                V2[xi+1,t,s] = -trapezoid(E2[0:xi+1,t,s], x[0:xi+1])
        V1[:,:,s] = V1[:,:,s] - V1[-1,:,s]
        V2[:,:,s] = V2[:,:,s] - V2[-1,:,s]

    err = eucli_metric(V1,V2)
    return err

def load_geometric_parameters(path,P_PDE):
    """ Prends le fichier .csv et va chercher les paramètres géométriques xc, sigma et prefact"""

    parametres_geo = pd.read_csv(path)
    parametres_geo = parametres_geo.dropna(axis=1, how='all')

    # Les colonnes qui commencent par le préfixe qu'on désire, format (# Simu, # parametres pour un attribut)
    xc_columns = [col for col in parametres_geo.columns if col.startswith('xc')]
    prefact_columns = [col for col in parametres_geo.columns if col.startswith('prefact')]
    sigma_columns = [col for col in parametres_geo.columns if col.startswith('sigma')]

    # Conversion en matricef
    I = parametres_geo['I(A)'].values
    R_inter = parametres_geo['R_inter'].values
    h_ag = parametres_geo['h_ag'].values

    xc_matrix = parametres_geo[xc_columns].values
    prefact_matrix = parametres_geo[prefact_columns].values
    sigma_matrix = parametres_geo[sigma_columns].values
    Ic_montee_matrix = parametres_geo['Ic_montee'].values

    return I, h_ag, R_inter, Ic_montee_matrix, xc_matrix, prefact_matrix, sigma_matrix

def load_data(path,P_PDE=[]):
    # ****************************************************
    # Load dataset  from many csv files (add all simulations in a tensor)
    #       path: path of the folder containing the results of the COMSOL simulations (.csv files)
    #       labels: P_PDE if changes needed
    # Intput: dataset  from csv (Nt,Ns)
    # Output: Dictionnary containing all the required tensors (Nx,Nt,Ns)
    # ****************************************************
    # List of path of .csv files
    
    time_skip = P_PDE['time_skip_ML']
    spatial_skip = P_PDE['spatial_skip_ML']

    files = [f for f in glob.glob(path)]
    files.sort()


    T = []
    Q_ag = []
    Q_supra = []
    Q_int = []
    Q_cool = []
    ids = []
    i=0
    # Loop over COMSOL simulations (.csv files) 
    for file in files:
        i+=1
        
        df = pd.read_csv(file)

        # Remove empty spaces in header
        #df.columns = [c.replace(' ', '') for c in df.columns]

        # Add unique_index column
        str_id = file.split('Simu')[1].split('_')[0]
        simu_id = int(str_id)
        if not simu_id in ids:
            ids.append(simu_id)

        if 'Simu' + str_id + '_T' in file:
            df = pd.read_csv(file)
            tensor = np.array(df.values)
            T.append(tensor)
        
        elif 'Simu' + str_id + '_Qag' in file:
            df = pd.read_csv(file)
            tensor = torch.tensor(df.values, dtype=torch.float)
            Q_ag.append(tensor)

        elif 'Simu' + str_id + '_Qsupra' in file:
            df = pd.read_csv(file)
            tensor = torch.tensor(df.values, dtype=torch.float)
            Q_supra.append(tensor)

        elif 'Simu' + str_id + '_Qint' in file:
            df = pd.read_csv(file)
            tensor = torch.tensor(df.values, dtype=torch.float)
            Q_int.append(tensor)

        elif 'Simu' + str_id + '_Qcool' in file:
            df = pd.read_csv(file)
            tensor = torch.tensor(df.values, dtype=torch.float)
            Q_cool.append(tensor)

        #if i % 20 == 0:
            #print(f"Number of files processed: {i}/{len(files)}")
        # Drop columns with all NaN values
    T = np.stack(T, -1).transpose(1, 0, 2)
    Q_ag = torch.stack(Q_ag,-1).permute(1, 0, 2)[:,1:-1,:]
    Q_supra = torch.stack(Q_supra,-1).permute(1, 0, 2)[:,1:-1,:]
    Q_int = torch.stack(Q_int,-1).permute(1, 0, 2)[:,1:-1,:]
    Q_cool = torch.stack(Q_cool,-1).permute(1, 0, 2)[:,1:-1,:]
    
    # Preprocessed:
    DATA_preprocessed = {
        "Tn-1(K)": T[:, 0:-2, :],
        "Tn(K)": T[:, 1:-1, :],
        "Tn+1(K)": T[:, 2:, :],
        "Q_ag(W/m2)": Q_ag,
        "Q_supra(W/m2)": Q_supra,
        "Q_inter(W/m2)": Q_int,
        "Q_conv(W/m2)": Q_cool,
        "simu_id": ids
    }
    return DATA_preprocessed # Output: Dataframe

def load_materials_splines(P_PDE:Dict,P_model:Dict, utils_path:str):
    """ Input: P_PDE dictionnary with PDE parameters, string (path) to get data location
    Output: [], modify the P_PDE dictionnary without returning it
    """
    # Cp and k
    files_name = ["cp_hast","cp_silver","cp_ybco","k_hast","k_silver","k_YBCO","rho_silver"]
    new_variables_name = ["Cp_hast_data", "Cp_ag_data","Cp_ybco_data","k_hast_data","k_ag_data","k_ybco_data","rho_ag_data"]
    i=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for file_name in files_name:
        file_path = utils_path + "Material_properties\\data\\" + file_name + ".txt"
        data = np.loadtxt(file_path)
        data = torch.tensor(data,dtype=P_model['dtype'],device=device)
        P_PDE[new_variables_name[i]] = data 
        i+=1



def derivative_spline_torch(T, data):
    """
    Calculate the derivative of variable y with respect to temperature x using PyTorch. data[:,0]=x, data[:,1] = y

    Parameters:
    T : torch.Tensor
        Tensor of temperatures.
    
    data : torch.Tensor
        Tensor of shape (n, 2) where data[:,0] = x and data[:,1] = y

    Returns:
    torch.Tensor
        Tensor of the same shape as T with the derivatives of y.
    """
    
    x = data[:, 0].contiguous()  # Ensure x is contiguous
    y = data[:, 1]

    # Calculate the derivatives
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    derivative_data = dy / dx

    # Find which interval each T falls into
    intervals = torch.bucketize(T, x, right=True) - 1
    intervals = torch.clamp(intervals, 0, len(derivative_data) - 1)

    # Gather the derivatives for the corresponding intervals
    derivative_output = derivative_data[intervals]
    
    return derivative_output

def reshape_dataframe(old_dataframe,skipt,skipx):
    """ Reshape old_dataframe in a newdataframe which do higher time skip predictions
    pred: dt ---> pred: 2*dt, T(Nt,Nx) --> T(Nt/2,Nx)
    """
    tn = sorted(set(old_dataframe['tn(s)']))
    xi = sorted(set(old_dataframe['xi(m)']))
    Nx = len(xi)
    Nt = len(tn)+2
    T = old_dataframe['Tn(K)'].values.reshape(Nt-2,Nx)
    Q_ag = old_dataframe['Q_ag(W/m2)'].values.reshape(Nt-2,Nx)
    Q_supra = old_dataframe['Q_supra(W/m2)'].values.reshape(Nt-2,Nx)
    Q_inter = old_dataframe['Q_inter(W/m2)'].values.reshape(Nt-2,Nx)
    Q_conv = old_dataframe['Q_conv(W/m2)'].values.reshape(Nt-2,Nx)
    Q = np.stack((Q_ag, Q_supra, Q_inter,Q_conv), axis=2)
    #Q = old_dataframe['Q(W/m3)'].values.reshape(Nt-2,Nx)
    T_first = old_dataframe[old_dataframe['tn(s)'] == tn[0]]['Tn-1(K)']
    T_last = old_dataframe[old_dataframe['tn(s)'] == tn[-1]]['Tn+1(K)']
    T_reshape = np.concatenate((np.expand_dims(T_first, axis=0), T, np.expand_dims(T_last, axis=0)), axis=0)
    #Q_reshape = np.concatenate((np.expand_dims(T_first, axis=0), Q, np.expand_dims(T_last, axis=0)), axis=0) # Just to get the same shape, wont be use
    Q_reshape = np.concatenate((np.expand_dims(0*Q[0], axis=0), Q, np.expand_dims(0*Q[0], axis=0)), axis=0)
    entete = old_dataframe.columns.tolist()[0:-1]
    T = T_reshape[::skipt,::skipx]
    Q = Q_reshape[::skipt,::skipx,:]
    tn.insert(0, 0)
    tn.append(1e-3)
    tn = tn[::skipt]
    xi = xi[::skipx]
    data_total = None
    x_data = np.array(xi) 
    for i in range(len(tn)-2):
        t_data = tn[i+1]*np.ones(len(xi))
        Tn_m1   = T[i,:]       # Tn-1
        Tn      = T[i+1,:]     # Tn
        Tn_p1   = T[i+2,:]     # Tn+1
        #Q_data  = Q[i+1,:]
        Q_data_ag = Q[i+1,:,0]
        Q_data_supra = Q[i+1,:,1]
        Q_data_inter = Q[i+1,:,2]
        Q_data_conv = Q[i+1,:,3]
        #sous_data =  np.column_stack([t_data, x_data, Tn_m1, Tn, Tn_p1, Q_data])
        sous_data =  np.column_stack([t_data, x_data, Tn_m1, Tn, Tn_p1, Q_data_ag, Q_data_supra, Q_data_inter,Q_data_conv])
        if data_total is None:
            data_total = sous_data
        else:
            data_total = np.vstack([data_total, sous_data])
    new_dataframe = pd.DataFrame(data = data_total, columns=entete)

    return new_dataframe

def reshape_dict_long_tape(dict,x,t,simu_id):
    """ Dict contient tous les tenseurs pour les prédictions (Tn,Tn-1,Tn+1,Qi) avec dimensions (Nx,Nt-2, Ns). On prend concatonne spatialement toutes les simulations.
    Important de noter: Les premières Ns-1 simulations ont utilise Nx-1, alors que la dernière est Nx. Il faut rester cohérent sur le maillage pour une longueur L*Ns, même
    chose lors qu'on re-converti xc,prefact,sigma --> sur le domaine L*Ns.
    """
    L = x[-1]
    Nx = len(x)
    Nt = len(t)
    Ns = len(simu_id)

    # Concatonne le tensor (1:Nx-1,1:Nt-2,1:Ns-1) à (1:Nx, 1:Nt,Ns)
    T0 = np.concatenate((np.transpose(dict['Tn-1(K)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Tn-1(K)'][:,:,-1] , axis=2)), axis=0)
    T1 = np.concatenate((np.transpose(dict['Tn(K)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Tn(K)'][:,:,-1] , axis=2)), axis=0)
    T2 = np.concatenate((np.transpose(dict['Tn+1(K)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Tn+1(K)'][:,:,-1] , axis=2)), axis=0)
    Q_ag = np.concatenate((np.transpose(dict['Q_ag(W/m2)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Q_ag(W/m2)'][:,:,-1] , axis=2)), axis=0)
    Q_supra = np.concatenate((np.transpose(dict['Q_supra(W/m2)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Q_supra(W/m2)'][:,:,-1] , axis=2)), axis=0)
    Q_inter = np.concatenate((np.transpose(dict['Q_inter(W/m2)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Q_inter(W/m2)'][:,:,-1] , axis=2)), axis=0)
    Q_conv = np.concatenate((np.transpose(dict['Q_conv(W/m2)'][:-1,:,:-1], (2, 0, 1)).reshape((Nx-1) * (Ns-1), Nt-2, 1), np.expand_dims(dict['Q_conv(W/m2)'][:,:,-1] , axis=2)), axis=0)

    simu_id = [1]

    # Update des variables
    dict['Tn-1(K)']         = T0
    dict['Tn(K)']           = T1
    dict['Tn+1(K)']         = T2
    dict['Q_ag(W/m2)']      = Q_ag
    dict['Q_supra(W/m2)']   = Q_supra
    dict['Q_inter(W/m2)']   = Q_inter
    dict['Q_conv(W/m2)']    = Q_conv
    dict['simu_id']         = simu_id
    x_new                   = np.linspace(0,Ns*L,T0.shape[0])
    return dict,x_new

def reshape_dataframe_long_tape(old_dataframe,xi,tn,simu_id):
    Ns = len(simu_id)
    L = xi[-1]
    Nx = len(xi)
    Nt = len(tn)

    T_list = []
    T_first_list = []
    T_last_list = []
    Q_ag_list = []
    Q_supra_list = []
    Q_inter_list = []
    Q_conv_list = []
    for i in simu_id:
        T = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Tn(K)'].values.reshape(Nt-2,Nx)
        T_first = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Tn-1(K)'].values.reshape(Nt-2,Nx)[0,:]
        T_last = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Tn+1(K)'].values.reshape(Nt-2,Nx)[-1,:]
        Q_ag = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Q_ag(W/m2)'].values.reshape(Nt-2,Nx)
        Q_supra = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Q_supra(W/m2)'].values.reshape(Nt-2,Nx)
        Q_inter = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Q_inter(W/m2)'].values.reshape(Nt-2,Nx)
        Q_conv = old_dataframe.loc[old_dataframe['Simu_index'] == i, 'Q_conv(W/m2)'].values.reshape(Nt-2,Nx)

        if i != simu_id[-1]:
            T_list.append(T[:,:-1])
            T_first_list.append(T_first[:-1])
            T_last_list.append(T_last[:-1])
            Q_ag_list.append(Q_ag[:,:-1])
            Q_supra_list.append(Q_supra[:,:-1])
            Q_inter_list.append(Q_inter[:,:-1])
            Q_conv_list.append(Q_conv[:,:-1])
        else:
            T_list.append(T)
            T_first_list.append(T_first)
            T_last_list.append(T_last)
            Q_ag_list.append(Q_ag)
            Q_supra_list.append(Q_supra)
            Q_inter_list.append(Q_inter)
            Q_conv_list.append(Q_conv)
    T = np.concatenate(T_list, axis=1)
    T_first = np.concatenate(T_first_list,axis=0)
    T_last = np.concatenate(T_last_list,axis=0)
    Q_ag = np.concatenate(Q_ag_list, axis=1)
    Q_supra = np.concatenate(Q_supra_list, axis=1)
    Q_inter = np.concatenate(Q_inter_list, axis=1)
    Q_conv = np.concatenate(Q_conv_list, axis = 1)
    Q = np.stack((Q_ag, Q_supra, Q_inter,Q_conv), axis=2)
    
    T = np.concatenate((np.expand_dims(T_first, axis=0), T, np.expand_dims(T_last, axis=0)), axis=0)
    Q = np.concatenate((np.expand_dims(0*Q[0], axis=0), Q, np.expand_dims(0*Q[0], axis=0)), axis=0)
    entete = old_dataframe.columns.tolist()
    data_total = None
    x_data = np.array(xi) 
    Simu_index_long_tape = np.ones(len(xi)).astype(int)
    for i in range(len(tn)-2):
        t_data = tn[i+1]*np.ones(len(xi))
        Tn_m1   = T[i,:]       # Tn-1
        Tn      = T[i+1,:]     # Tn
        Tn_p1   = T[i+2,:]     # Tn+1
        Q_data_ag = Q[i+1,:,0]
        Q_data_supra = Q[i+1,:,1]
        Q_data_inter = Q[i+1,:,2]
        Q_data_conv = Q[i+1,:,3]
        sous_data =  np.column_stack([t_data, x_data, Tn_m1, Tn, Tn_p1, Q_data_ag, Q_data_supra, Q_data_inter,Q_data_conv,Simu_index_long_tape])
        if data_total is None:
            data_total = sous_data
        else:
            data_total = np.vstack([data_total, sous_data])
    new_dataframe = pd.DataFrame(data = data_total, columns=entete)

    return new_dataframe

def xc_translation_long_tape(xc,geo_index,P_PDE):
    """ xc matrix [Ns,# quenches possibles]. Prends xc et reshape en translation Ns_test simulations. Xc = (n-1)*L + xc. n: lignes de la matrice xc
    """
    L = P_PDE["L"] 
    Xc = np.copy(xc)
    for n in range(len(geo_index)):
        Xc[n,:] = n*L+xc[n,:]

    return np.expand_dims(Xc[geo_index].flatten(),axis=0)


def JcX_solo(x,xc_scalar,prefact_scalar,sigma_scalar):
    """ x: vector, xc, prefact,sigma sont des scalaires
    """
    return (1-prefact_scalar*np.exp(-(x-xc_scalar)**2/sigma_scalar**2))

def JcX_global(x,xc,prefact,sigma):
    """ x: vector, xc, prefact,sigma sont des vecteurs
    """
    JcX_final = 1
    for i in range(len(xc)):
        JcX_final = JcX_final*JcX_solo(x,xc[i],prefact[i],sigma[i])
    
    return JcX_final

def init_weights(m):
    if type(m) == torch.nn.Linear:
         torch.nn.init.xavier_uniform_(m.weight)
         m.bias.data.fill_ (0.01)

@njit
def JcX(x,xc_scalar,prefact_scalar,sigma_scalar):
        """ x: vector, xc, prefact,sigma sont des scalaires
        """
        return (1-prefact_scalar*np.exp(-(x-xc_scalar)**2/sigma_scalar**2))
    
@njit 
def Jc_XT(x,xc,amplitude,sigma,T,Jc0,Tc,T0):
        Jc = np.ones_like(x)
        for i in range(len(xc)):
            Jc = Jc*JcX(x,xc[i],amplitude[i],sigma[i])
        
        Jc = (Jc0*(Tc-T)/(Tc-T0)*(T<Tc) + (T>=Tc))*Jc
        return Jc

def solutions_FEM(DATA,P_PDE):
    Nx  = P_PDE['Nx']
    Nt  = P_PDE['Nt']
    Ns  = DATA['Tn+1(K)'].shape[2]
    x   = P_PDE['x']
    t   = P_PDE['t']
    As  = P_PDE['As']

    # Solutions thermiques
    T   = np.zeros((Nx,Nt,Ns))
    T[:,2:,:] = DATA['Tn+1(K)']
    T[:,0:2,:]  = DATA['Tn-1(K)'][:,0:2,:]

    # Solutions électriques
    Js  = np.zeros((Nx,Nt,Ns))
    rhos = np.zeros((Nx,Nt,Ns))

    for simu in range(Ns):
        I = Iapp(simu,P_PDE)
        Js[:,1,simu] += I(t[1])/As

        P_PDE['ha_current'] = P_PDE['ha'][simu]
        P_PDE['Aa'] = P_PDE['w']*P_PDE['ha_current']
        P_PDE['R_inter_current'] = P_PDE['R_inter'][simu]
        P_PDE['xc_current'] = P_PDE['xc'][simu,:]
        P_PDE['sigma_current'] = P_PDE['sigma'][simu,:]
        P_PDE['prefact_current'] = P_PDE['prefact'][simu,:]

        for j in range(Nt-1):
            Js[:,j+1,simu],_,rhos[:,j+1,simu] = solverNR_fast(x,T[:,j+1,simu],I(t[j+1]),Js[:,j,simu],P_PDE)

    return T,Js,rhos

def rhos_FEM(x,T,J,simu_index,P_PDE):
        """ T,J :[Nx,Nt], x: (Nx,), simu_index : int, P_PDE: dict
        """
        Nx, Nt = T.shape

        # Update paramètres géométriques pour calculer de rhos
        P_PDE['xc_current']          = P_PDE['xc'][simu_index,:]
        P_PDE['sigma_current']       = P_PDE['sigma'][simu_index,:]
        P_PDE['amplitude_current']   = P_PDE['prefact'][simu_index,:]
        P_PDE['R_inter_current']     = P_PDE['R_inter'][simu_index]
        P_PDE['ha_current']          = P_PDE['ha'][simu_index]
        P_PDE['Aa']                  = P_PDE['w']*P_PDE['ha_current']

        rhos = np.ones((Nx,Nt))*1e-16
        T_bool = T < P_PDE['Tc']
        n = P_PDE['n'] * T_bool + (~T_bool)  
        Jc = Jc_XT(x[:, None], P_PDE['xc_current'], P_PDE['amplitude_current'], P_PDE['sigma_current'], T,P_PDE['Jc0'],P_PDE['Tc'],P_PDE['T0'])

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(J / Jc)
            base = P_PDE['E0'] / Jc * ratio**(n - 1)
            rhos = np.clip(base, 1e-16, P_PDE['rho_s_normal'])

        return rhos


def Voltage_ChampElec(x,rho,J):
        
    Nx,Nt = rho.shape
    V = np.zeros((Nx-2,Nt))
    E = rho*J
    x_voltage = x[1:-1]

    for t in range(Nt):
        for xi in range(len(x_voltage)-1):
            #Simpson O(4), plus exacte mais plus couteuse
            #V[xi+1,t] = -simpson(E[0:xi+1,t],x[0:xi+1])

            # Simpson O(2), plus exacte mais plus couteuse
            V[xi+1,t] = -trapezoid(E[0:xi+1,t], x_voltage[0:xi+1]) 

    V = V - V[-1,:]
    return x_voltage,V,E

class Levin1D:
    def __init__(self,P_PDE):
        self.dx = P_PDE['dx']
        self.dt = P_PDE['dt']
        self.Nt = P_PDE['Nt']
        self.Nx = P_PDE['Nx']
        self.E0 = P_PDE['E0']
        self.n  = P_PDE['n']
        self.Ic = P_PDE['Ic']
        self.rho_s_normal = P_PDE['rho_s_normal']
        self.toll_solverNR = P_PDE['toll_solverNR']
        self.L = P_PDE['L']
        self.t_end = P_PDE['t_end']
        self.T0 = P_PDE['T0']
        self.Tc = P_PDE['Tc']
        self.h_conv = P_PDE['h_conv']
        self.Jc0 = P_PDE['Jc0']

        self.x = np.linspace(0,self.L,self.Nx)
        self.t = np.linspace(0,self.t_end,self.Nx)
        
        self.rhoa_spline   = P_PDE['rho_ag_spline']

        self.Ca_spline     = P_PDE['Cp_ag_spline']
        self.Cybco_spline  = P_PDE['Cp_ybco_spline']
        self.Chast_spline  = P_PDE['Cp_hast_spline']

        self.ka_spline     = P_PDE['k_ag_spline']
        self.kybco_spline  = P_PDE['k_ybco_spline']
        self.khast_spline  = P_PDE['k_hast_spline']

        self.dka_spline    = P_PDE['dkdT_ag_spline']
        self.dkybco_spline  = P_PDE['dkdT_ybco_spline']
        self.dkhast_spline = P_PDE['dkdT_hast_spline']

        self.rhom_a = P_PDE['rhom_a']
        self.rhom_ybco = P_PDE['rhom_s']
        self.rhom_hast = P_PDE['rhom_hast']

        self.ha = P_PDE['ha']
        self.hs = P_PDE['hs']
        self.hhast = P_PDE['hhast']

        self.simu_index = 0
        self.xc = P_PDE['xc']
        self.sigma = P_PDE['sigma']
        self.amplitude = P_PDE['prefact']
        self.R_inter = P_PDE['R_inter']
        self.Aa = P_PDE['Aa']
        self.As = P_PDE['As']
        self.w  = P_PDE['w']

    def k_effectif(self,T,ids):
        """ Selon Levin keff = sum_i h_i k_i
        """
        keff = self.ha[ids]*self.ka_spline(T) + self.hs*self.kybco_spline(T) + self.hhast*self.khast_spline(T)
        return keff

    def dkdT_effectif(self,T,ids):
        """ dk_eff/dT selon Levin
        """
        dkdT = self.ha[ids]*self.dka_spline(T) + self.hs*self.dkybco_spline(T) + self.hhast*self.dkhast_spline(T)
        return dkdT
    
    def rhomC_effectif(self,T,ids):
        """  rhom et Cp, rhom*Cp selon Levin
        """
        rhomCp_eff = self.ha[ids]*self.rhom_a*self.Ca_spline(T) +self.hs*self.rhom_ybco*self.Cybco_spline(T)+ self.hhast*self.rhom_hast*self.Chast_spline(T)
        return rhomCp_eff
    
    @staticmethod
    def JcX_solo(x,xc_scalar,prefact_scalar,sigma_scalar):
        """ x: vector, xc, prefact,sigma sont des scalaires
        """
        return (1-prefact_scalar*np.exp(-(x-xc_scalar)**2/sigma_scalar**2))
    
    @staticmethod
    def differentiation(x, Js, type):
        dJ = np.copy(Js)

        if type == "O2":
            dJ[0] = 0
            dJ[-1] = 0
            dJ[1:-1] = (Js[2:] - Js[:-2]) / (2 * (x[1] - x[0]))

        elif type == "O4":
            dJ[0] = 0
            dJ[1] = 0
            dJ[-1] = 0
            dJ[-2] = 0
            dJ[2:-2] = (Js[:-4] - 8 * Js[1:-3] + 8 * Js[3:-1] - Js[4:]) / (12 * (x[1] - x[0]))

        return dJ   
    
    def Jc(self,x,xc,amplitude,sigma,T):
        Jc = 1
        for i in range(len(xc)):
            Jc = Jc*self.JcX_solo(x,xc[i],amplitude[i],sigma[i])
        
        Jc = (self.Jc0*(self.Tc-T)/(self.Tc-self.T0)*(T<self.Tc) + (T>=self.Tc))*Jc
        return Jc
    
    def solverNR(self,x, T, Itot, Js_pre):
        Nx, dx = len(x), self.dx
        Js_frontiere = Itot/self.As
        α = -self.R_inter_current*self.As/self.Aa*self.ha_current
        Js = Js_frontiere/(Js_pre[0]+1)*Js_pre
        tol, err, k = self.toll_solverNR, 1e10, 0

        rhoa = self.rhoa_spline(T)
        n = self.n*(T<self.Tc) + (T>=self.Tc)
        Jc = self.Jc(x,self.xc_current,self.amplitude_current,self.sigma_current,T)

        # buffers hors boucle (dimension m = Nx-2, diag a,c de longueur m-1 = Nx-3)
        m = Nx - 2
        a = np.full(m-1, -α / dx**2)     # sous-diagonale (len = m-1)
        c = a.copy()                     # sur-diagonale (len = m-1)
        b = np.empty(m)                  # diagonale principale (len = m)
        d = np.empty(m)  

        while err >= tol:
            k += 1
            Ja = (Itot - self.As*Js) / self.Aa
            rhos = np.minimum(np.maximum(self.E0/Jc[1:-1]*np.abs(Js[1:-1]/Jc[1:-1])**(n[1:-1]-1),1e-16), self.rho_s_normal)
            g = rhoa[1:-1]*np.maximum(Ja[1:-1],0) - rhos*np.maximum(Js[1:-1],0)
            dg = -self.As/self.Aa*rhoa[1:-1] - n[1:-1]*rhos
            # diagonales

            b[:] = 2*α/dx**2 + dg
            # second dérivée
            d2 = (Js[0:-2] - 2*Js[1:-1] + Js[2:]) / dx**2
            # RHS
            d[:] = α*d2 - g
            # solve
            Δ = thomas(a, b, c, d)
            old = Js[1:-1].copy()
            Js[1:-1] = old + Δ
            Js[ 0 ], Js[-1] = Js_frontiere, Js_frontiere
            err = np.linalg.norm(Js[1:-1] - old) / np.linalg.norm(Js[1:-1])

            rhos = np.concatenate(([1e-16], rhos, [1e-16]))
        return Js, Ja, rhos, rhoa

    def electrique(self,x,T,I, Js_pre):

        Js, Ja, rhos, rhoa = self.solverNR(x, T, I, Js_pre)
        dJs = self.differentiation(x,Js,"O4")

        return Js,Ja,dJs,rhos,rhoa

    def thermique(self,x,T,Js,Ja,dJs,rhos,rhoa, simu_index ):

        T_pre = T[0,:]
        T_curr = T[-1,:]
        T_next = self.T0*np.ones((len(x)))

        # Chaleurs Levin
        Q_Ja    = self.ha_current*rhoa*Ja**2
        Q_Js    = self.hs*rhos*Js**2
        Q_conv  = -2*self.h_conv*(T_curr -self.T0)
        Q_inter = self.R_inter_current*self.ha_current**2*(self.As/self.Aa)**2*(dJs)**2

        Q = Q_Ja + Q_Js + Q_conv + Q_inter

        # Coefficients thermiques
        k_eff = self.k_effectif(T_curr,simu_index)
        dkdT_eff = self.dkdT_effectif(T_curr,simu_index)
        C_eff = self.rhomC_effectif(T_curr,simu_index)

        A_T  = 2 * k_eff[1:-1] * self.dt / ( C_eff[1:-1] * self.dx**2 )
        B_T = dkdT_eff[1:-1] * self.dt/ (2 * C_eff[1:-1] * self.dx**2 )
        gamma_T = 2*self.dt/( C_eff[1:-1] )

        Tn_ip = T_curr[2:]
        Tn_im = T_curr[:-2]

        # Doit prendre Nx-2 coordonnées car on évite les conditions frontières à cause de Ti-1, Ti+1
        T_next[1:-1] = 1/(1+A_T)* ( (1-A_T)*T_pre[1:-1] + A_T*(Tn_ip+Tn_im) + B_T*(Tn_ip-Tn_im)**2 + gamma_T*Q[1:-1])

        return T_next


    def predictions(self,simu_index,P_PDE):

        self.Nx = P_PDE['Nx']
        x = P_PDE['x']
        t = self.t
        self.xc_current         = self.xc[simu_index,:]
        self.sigma_current      = self.sigma[simu_index,:]
        self.amplitude_current  = self.amplitude[simu_index,:]
        self.R_inter_current    = self.R_inter[simu_index]
        self.ha_current         = self.ha[simu_index]
        self.Aa                 = self.w*self.ha_current

        T   = self.T0*np.ones((self.Nt,self.Nx))
        Js  = np.zeros((self.Nt,self.Nx))
        rhos = np.zeros((self.Nt,self.Nx))

        I = Iapp(simu_index,P_PDE)
        Js[1,:] += I(t[1])/self.As

        for i in range(self.Nt-2):

            Js[i+1,:],Ja,dJs,rhos[i+1,:],rhoa  = self.electrique(x,T[i+1,:],I(t[i+1]), Js[i,:])
            T[i+2,:]                    = self.thermique(x,T[i:i+2,:],Js[i+1,:],Ja,dJs,rhos[i+1,:],rhoa,simu_index )

        Js[-1,:],_,_,rhos[-1,:],_  = self.electrique(x,T[-1,:],I(t[-1]), Js[-2,:])

        return T.T, Js.T,rhos.T

    def rhos(self,x,T,J,simu_index):
        """ T,J :[Nx,Nt], x: (Nx,)
        """
        Nx, Nt = T.shape

        # Update paramètres géométriques pour calculer de rhos
        self.xc_current         = self.xc[simu_index,:]
        self.sigma_current      = self.sigma[simu_index,:]
        self.amplitude_current  = self.amplitude[simu_index,:]
        self.R_inter_current    = self.R_inter[simu_index]
        self.ha_current         = self.ha[simu_index]
        self.Aa                 = self.w*self.ha_current  

        rhos = np.ones((Nx,Nt))*1e-16
        T_bool = T < self.Tc
        n = self.n * T_bool + (~T_bool)  
        Jc = self.Jc(x[:, None], self.xc_current, self.amplitude_current, self.sigma_current, T)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(J / Jc)
            base = self.E0 / Jc * ratio**(n - 1)
            rhos = np.clip(base, 1e-16, self.rho_s_normal)

        return rhos
    
    def voltage(self,x,rho,J):
        """ rhos,J :[Nx,Nt], x: (Nx,)
        """
        x_voltage,V,E = Voltage_ChampElec(x,rho,J)
        return x_voltage,V,E
    
@njit
def thomas(a, b, c, d):
    """Solve Ax = d for tridiagonal A with subdiag a, diag b, supdiag c."""
    n = len(d)
    # forward sweep
    cp = np.empty(n-1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        tmp = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i] / tmp
        dp[i] = (d[i] - a[i-1]*dp[i-1]) / tmp
    dp[n-1] = (d[n-1] - a[n-2]*dp[n-2]) / (b[n-1] - a[n-2]*cp[n-2])
    # backward substitution
    x = np.empty(n)
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

class hPGRNN(Levin1D,nn.Module):
    def __init__(self, P,P_PDE):
        Levin1D.__init__(self, P_PDE)
        nn.Module.__init__(self)
        
        # CPU or GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float
        
        # Conserve le dictionnaire de paramètres P
        self.P = P
        self.P_PDE = P_PDE

        # Propriétés thermiques sur Pytorch
        self.k_ag_data      = P_PDE['k_ag_data']
        self.k_ybco_data    = P_PDE['k_ybco_data']
        self.k_hast_data    = P_PDE['k_hast_data']
        self.Cp_ag_data     = P_PDE['Cp_ag_data']
        self.Cp_ybco_data   = P_PDE['Cp_ybco_data']
        self.Cp_hast_data   = P_PDE['Cp_hast_data']

        self.ha = torch.tensor(P_PDE['ha'],device = self.device, dtype= torch.float)

        # Constante den ormalisation
        self.Tmin           = P_PDE['T_min']
        self.Tmax           = P_PDE['T_max']
        self.Q_min_cool     = P_PDE['Q_min_cool']
        self.Q_max_cool     = P_PDE['Q_max_cool']
        self.Q_min_inter    = P_PDE['Q_min_inter']
        self.Q_max_inter    = P_PDE['Q_max_inter']
        self.Q_min_ag_supra = P_PDE['Q_min_ag_supra']
        self.Q_max_ag_supra = P_PDE['Q_max_ag_supra']
        self.Q_min_ag       = P_PDE['Q_min_ag']
        self.Q_max_ag       = P_PDE['Q_max_ag']
        self.Q_min_supra    = P_PDE['Q_min_supra']
        self.Q_max_supra    = P_PDE['Q_max_supra']

        self.kernel_size    = P['kernel_size']
        self.batch_size     = P['batch_size']
        # Architecture pour les neural nets
        self.network = self.neural_net(P['nn_structure'],P['dropout_prob'])

        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)

    def neural_net(self, layers, dropout_prob):
        seq = []
        for i in range(len(layers) - 2):
            in_features = layers[i]
            out_features = layers[i + 1]

            # Using a residual block
            if in_features == out_features:
                seq.append((f'ResidualBlock{i}', ResidualBlock(in_features, out_features, dropout_prob, self.dtype)))
            else:
                # Fallback to standard linear + dropout + ReLU if dimensions don't match for residual block
                seq.append((f'Linear{i}', nn.Linear(in_features, out_features, dtype=self.dtype)))
                seq.append((f'Dropout{i}', nn.Dropout(dropout_prob)))
                seq.append((f'ReLU{i}', nn.ReLU()))
            if i == 0:
                seq.append((f'SelfAttention{i}', SelfAttention(out_features, num_heads=1)))
        # Adding the final linear layer
        seq.append((f'Linear{len(layers) - 2}', nn.Linear(layers[-2], layers[-1], dtype=self.dtype)))

        # Create the neural network
        model_seq = nn.Sequential(OrderedDict(seq)).to(self.device)

        # Initialize weights with a custom function if needed
        model_seq.apply(init_weights)
        return model_seq

    def linear_scaler(self,Q,min,max): # log et apres min-max
        return ( Q - min) / (max - min)
    
    def forward_network(self,B,Nx,inputs): # 1D tensor (flat)
        return self.network(inputs).view(B, Nx-2, 6).unbind(dim=2)

    def k_effectif(self, T: torch.Tensor, ids: int) -> torch.Tensor:
        k_ag = torch_interp1d(self.k_ag_data[:,0],self.k_ag_data[:,1],T)
        k_ybco = torch_interp1d(self.k_ybco_data[:,0],self.k_ybco_data[:,1],T)
        k_hast = torch_interp1d(self.k_hast_data[:,0],self.k_hast_data[:,1],T)

        return self.ha[ids].unsqueeze(1)*k_ag + self.hs*k_ybco + self.hhast*k_hast
    
    def rhomC_effectif(self, T: torch.Tensor, ids: int) -> torch.Tensor:
        Cp_ag = torch_interp1d(self.Cp_ag_data[:,0],self.Cp_ag_data[:,1],T)
        Cp_ybco = torch_interp1d(self.Cp_ybco_data[:,0],self.Cp_ybco_data[:,1],T)
        Cp_hast = torch_interp1d(self.Cp_hast_data[:,0],self.Cp_hast_data[:,1],T)

        Cpeff = self.ha[ids].unsqueeze(1)*self.rhom_a*Cp_ag + self.hs*self.rhom_ybco*Cp_ybco + self.hhast*self.rhom_hast*Cp_hast

        return Cpeff
    
    def dkdT_effectif(self, T: torch.Tensor, ids: int) -> torch.Tensor:
        dkdT_ag = derivative_spline_torch(T,self.k_ag_data)
        dkdT_ybco = derivative_spline_torch(T,self.k_ybco_data)
        dkdT_hast = derivative_spline_torch(T,self.k_hast_data)

        dkdT_eff = self.ha[ids].unsqueeze(1)*dkdT_ag + self.hs*dkdT_ybco + self.hhast*dkdT_hast
        return dkdT_eff
    
    def chaleurs(self,Js,Ja,dJs,rhos,rhoa):
        # Chaleurs Levin
        Q_a    = self.ha_current*rhoa*Ja**2
        Q_s    = self.hs*rhos*Js**2
        Q_inter = self.R_inter_current*self.ha_current**2*(self.As/self.Aa)**2*(dJs)**2

        Q = np.stack((Q_a, Q_s, Q_inter), axis=1)[None, :, :] 
        return torch.tensor(Q,dtype=torch.float) # Float32 vs float64 avec numpy

    def thermique(self,Tn,Tn1,Q, simu_index ):
        """Problème thermique, opère en Pytorch
        T^n:[batch_size, Nx],T^n-1"""
        K = self.kernel_size
        B = Tn.shape[0]
        Nx = self.Nx

        T_next  = torch.full_like(Tn, self.T0)

        Q_a     = Q[:,1:-1,0]
        Q_s     = Q[:,1:-1,1]
        Q_inter = Q[:,1:-1,2]
        Q_conv  = -2*self.h_conv*(Tn[:,1:-1]-self.T0)

        # Thermal and heat padding
        bound_kernelT       = self.T0* torch.ones((B, int((K-1)/2)),device=self.device)
        bound_kernelQ_ag    = Q_a[:,0].unsqueeze(1)*torch.ones((B, int((K-1)/2)),device=self.device)
        bound_kernelQ_supra = Q_s[:,0].unsqueeze(1)*torch.ones((B, int((K-1)/2)),device=self.device)
        bound_kernelQ_inter = Q_inter[:,0].unsqueeze(1)*torch.ones((B, int((K-1)/2)),device=self.device)
        bound_kernelQ_conv  = Q_conv[:,0].unsqueeze(1)*torch.ones((B, int((K-1)/2)),device=self.device)
        T_current_inputs    = torch.cat([bound_kernelT, Tn[:,1:-1], bound_kernelT], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)
        T_previous_inputs   = torch.cat([bound_kernelT, Tn1[:,1:-1], bound_kernelT], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)
        Q_ag_inputs         = torch.cat([bound_kernelQ_ag, Q_a, bound_kernelQ_ag], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)
        Q_supra_inputs      = torch.cat([bound_kernelQ_supra, Q_s, bound_kernelQ_supra], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)
        Q_inter_inputs      = torch.cat([bound_kernelQ_inter, Q_inter, bound_kernelQ_inter], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)
        Q_conv_inputs       = torch.cat([bound_kernelQ_conv, Q_conv, bound_kernelQ_conv], dim=1).unfold(dimension=1, size=K, step=1).reshape(-1,K)

        # Data normalisation
        T_current_inputs    = self.linear_scaler(T_current_inputs, self.Tmin, self.Tmax)
        T_previous_inputs   = self.linear_scaler(T_previous_inputs, self.Tmin, self.Tmax)
        Q_conv_inputs       = self.linear_scaler(Q_conv_inputs,self.Q_min_cool,self.Q_max_cool)
        Q_ag_inputs         = self.linear_scaler(Q_ag_inputs, self.Q_min_ag, self.Q_max_ag)
        Q_supra_inputs      = self.linear_scaler(Q_supra_inputs, self.Q_min_supra, self.Q_max_supra)
        Q_inter_inputs      = self.linear_scaler(Q_inter_inputs, self.Q_min_inter, self.Q_max_inter)

        # Modèle d'apprentissage profond
        inputs_network = torch.cat((T_current_inputs,T_previous_inputs,Q_ag_inputs, Q_supra_inputs, Q_inter_inputs, Q_conv_inputs), dim=1)
        a_T, b_T, c_T, d_T, e_T,f_T = self.forward_network(B,Nx,inputs_network)

        #a_T = a_T -a_T  + 1     
        #b_T = b_T -b_T  + 1   
        #c_T = c_T -c_T  + 1   
        #d_T = d_T -d_T  + 1   
        #e_T = e_T -e_T  + 1
        #f_T = f_T - f_T + 1

        # Coefficients thermiques
        Tn_contiguous = Tn.contiguous()
        k_eff = self.k_effectif(Tn_contiguous, simu_index)
        dkdT_eff = self.dkdT_effectif(Tn_contiguous, simu_index)
        C_eff = self.rhomC_effectif(Tn_contiguous, simu_index)

        A_T = 2 * k_eff[:,1:-1] * self.dt / (C_eff[:,1:-1] * self.dx**2)*a_T
        B_T = dkdT_eff[:,1:-1] * self.dt / (2 * C_eff[:,1:-1] * self.dx**2)*b_T
        gamma_T = 2 * self.dt / C_eff[:,1:-1]

        T_ip = Tn[:,2:]; T_im = Tn[:,:-2]

        # Doit prendre Nx-2 coordonnées car on évite les conditions frontières à cause de Ti-1, Ti+1
        T_next[:,1:-1] = (1/(1+A_T) * ((1-A_T)*Tn1[:,1:-1] + A_T*(T_ip+T_im)+ B_T*(T_ip-T_im)**2
                          + gamma_T *(c_T*(Q_a + Q_s) + d_T*Q_inter + e_T*Q_conv/2 + f_T*Q_conv/2) ))

        return T_next

    def predictions(self,simu_index,P_PDE):

        with torch.no_grad():
            self.Nx = P_PDE['Nx']
            x = P_PDE['x']
            self.x = x
            t = self.t
            self.xc_current         = P_PDE['xc'][simu_index,:]
            self.sigma_current      = P_PDE['sigma'][simu_index,:]
            self.amplitude_current  = P_PDE['prefact'][simu_index,:]
            self.R_inter_current    = P_PDE['R_inter'][simu_index]
            self.ha_current         = P_PDE['ha'][simu_index]
            self.Aa                 = np.array(self.P_PDE['w']*self.ha_current,dtype=np.float64)

            T   = self.T0*np.ones((self.Nt,self.Nx))
            Js  = np.zeros((self.Nt,self.Nx))
            rhos  = np.zeros((self.Nt,self.Nx))

            I = Iapp(simu_index,P_PDE)
            Js[1,:] += I(t[1])/self.As

            for i in range(self.Nt-2):

                Js[i+1,:],Ja,dJs,rhos[i+1,:],rhoa  = self.electrique(x,T[i+1,:],I(t[i+1]), Js[i,:])

                Q                           = self.chaleurs(Js[i+1,:], Ja, dJs, rhos[i+1,:], rhoa)
                Tn                          = torch.tensor(T[i+1:i+2,:],dtype=torch.float)
                Tn1                         = torch.tensor(T[i:i+1,:],dtype=torch.float)
                T[i+2,:]                    = self.thermique(Tn,Tn1,Q, torch.tensor([simu_index]))

            Js[-1,:],_,_,rhos[-1,:],_  = self.electrique(x,T[-1,:],I(t[-1]), Js[-2,:])

        return T.T, Js.T, rhos.T

    def train_thermal(self, DATA_train, DATA_val, epochs):
        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        
        if self.P['steps_max'] == 2: # Steps then won't change, only need to initiate tensor a single time
            Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
            Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)

        for epo in range(int(epochs)):
            if self.P['steps_max'] > 2:
                Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
                Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)
        
            # Number of samples
            N = Seq_train['Tn+1(K)'].size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(N)

            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define batch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train['Q(W/m2)'][indices]
                batch_Seq_train_Tn   = Seq_train['Tn+1(K)'][indices]
                batch_Seq_train_id   = Seq_train['simu_ids'][indices]-1
                
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list = [],[]#, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]

                # Pour sur l_s prédictions
                for n in range( 0, int(np.floor(steps-1)) ):  
                    Q_train       = batch_Seq_train_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)
                    Tn_predict_train = self.thermique( Tn_recur_current_train, Tn_recur_previous_train, Q_train,batch_Seq_train_id) # size: (K, 2*Nx)

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                difference_Tn_train = (torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                
                # Mean square error for one mini-batch
                loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #####################################################################################################
                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters     
                loss_train.backward()
                
                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()
                #####################################################################################################

                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch

            current_lr = self.optimizer.param_groups[0]['lr']

            # For loop for validation
            M = Seq_val['Tn+1(K)'].size()[0]
            permutation = torch.randperm(M)
            for j in range(0,M,K):
                indices = permutation[j:j+K]
                batch_Seq_val_Tn_2 = Seq_val['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_val_Tn_1 = Seq_val['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_val_Q_1  = Seq_val['Q(W/m2)'][indices]
                batch_Seq_val_Tn   = Seq_val['Tn+1(K)'][indices]
                batch_Seq_val_id   = Seq_val['simu_ids'][indices]

                # Initialize state at a starting point of a sequence
                Tn_recur_previous_val = batch_Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = batch_Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                Tn_pred_val_list, Tn_true_val_list = [],[]
                for n in range( 0, int(np.floor(steps-1)) ):
                    Q_val       = batch_Seq_val_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_val = batch_Seq_val_Tn[:,n,:]
                    Tn_predict_val = self.thermique( Tn_recur_current_val, Tn_recur_previous_val, Q_val,batch_Seq_val_id)
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                difference_Tn_val   = (torch.stack(Tn_pred_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx']) - torch.stack(Tn_true_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                #loss_val   = torch.mean(torch.square( difference_Tn_val ))
                loss_val   = torch.mean(torch.norm(difference_Tn_val,dim=1))
                loss_val_batch.append( loss_val.item() )    

            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) 
            ##############################################################################

            # Predict material properties:
            dim_inputs_T = int(self.kernel_size*2)
            T_evaluation =  self.Tmax/3*torch.ones(1,dim_inputs_T ).to(self.device)
            T_evaluation[:,5:] = 0.9*T_evaluation[:,5:]
            #Qas          =  self.Q_max_ag_supra/3*torch.ones(1,self.kernel_size ).to(self.device)
            Qa          =  self.P_PDE['Q_max_ag']/3*torch.ones(1,self.P['kernel_size'] ).to(self.device)
            Qs          =  self.P_PDE['Q_max_supra']/3*torch.ones(1,self.P['kernel_size'] ).to(self.device)
            Qinter       =  self.Q_max_inter/3*torch.ones(1,self.kernel_size  ).to(self.device)
            Qcool        =  self.Q_min_cool/3*torch.ones(1,self.kernel_size ).to(self.device)
            T_evaluation = self.linear_scaler(T_evaluation, self.Tmin, self.Tmax)
            #Qas          = self.linear_scaler(Qas, self.Q_min_ag_supra, self.Q_max_ag_supra)
            Qa          = self.linear_scaler(Qa,self.P_PDE['Q_min_ag'],self.P_PDE['Q_max_ag'])
            Qs          = self.linear_scaler(Qs,self.P_PDE['Q_min_supra'],self.P_PDE['Q_max_supra'])
            Qinter       = self.linear_scaler(Qinter, self.Q_min_inter, self.Q_max_inter)
            Qcool        = self.linear_scaler(Qcool,self.Q_min_cool,self.Q_max_cool)
            inputs_val   = torch.cat((T_evaluation,Qa,Qs,Qinter,Qcool), dim=1)

            self.a_150, self.b_150,self.c_150, self.d_150, self.e_150,self.f_150       = self.forward_network(1,1, inputs_val )
            print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, a(150): %.3f , b(150): %.3f , c(150): %.3f , d(150): %.3f , e(150): %.3f. f(150): %.3f,  lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.a_150, self.b_150,self.c_150,self.d_150,self.e_150,self.f_150,current_lr))
            
            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']
        return Loss_train_epoch, Loss_val_epoch, STEPS
    

def transfert_to_device(model,label):
    "model + label in string"

    if label.lower() == "cpu":
        model = model.cpu()
        model.device = torch.device('cpu')
        for attr in model.__dict__:
            val = getattr(model, attr)
            if torch.is_tensor(val):
                setattr(model, attr, val.cpu())

    elif label.lower() == "cuda":
        model = model.cuda()
        model.device = torch.device('cuda')
        for attr in model.__dict__:
            val = getattr(model, attr)
            if torch.is_tensor(val):
                setattr(model, attr, val.cuda())
    else:
        print('Only device available are "cpu" or "cuda" ')

    return model

def solverNR_fast(x, T, Itot, Js_pre, P_PDE):
    Nx, dx = len(x), x[1]
    As = P_PDE['As']
    Aa = P_PDE['Aa']
    ha_current = P_PDE['ha_current']
    R_inter_current = P_PDE['R_inter_current']
    xc_current = P_PDE['xc_current']
    prefact_current = P_PDE['prefact_current']
    sigma_current = P_PDE['sigma_current']
    toll_solver = P_PDE['toll_solverNR']
    n0 = P_PDE['n']
    Tc = P_PDE['Tc']
    T0 = P_PDE['T0']
    E0 = P_PDE['E0']
    Jc0 = P_PDE['Jc0']
    rho_s_normal = P_PDE['rho_s_normal']
    rhoa = P_PDE["rho_ag_spline"](T)

    Js_frontiere = (Itot / As)
    α = -R_inter_current*As/Aa*ha_current
    Js = Js_frontiere / (Js_pre[0] + 1e-8) * Js_pre.copy()
    tol, err, k = toll_solver, 1e10, 0

    n = n0*(T<Tc) + (T>=Tc)
    Jc = Jc_XT(x,xc_current,prefact_current,sigma_current,T,Jc0,Tc,T0)

    # buffers hors boucle
    a = np.empty(Nx-2)
    b = np.empty(Nx-2)
    c = np.empty(Nx-2)
    d = np.empty(Nx-2)
    a[:] = -α/dx**2
    c[:] = a[:]

    while err >= tol:
        k += 1
        Ja = (Itot - As*Js) / Aa
        rhos = np.minimum(np.maximum(E0/Jc[1:-1]*np.abs(Js[1:-1]/Jc[1:-1])**(n[1:-1]-1),1e-16), rho_s_normal)
        g = rhoa[1:-1]*np.maximum(Ja[1:-1],0) - rhos*np.maximum(Js[1:-1],0)
        dg = -As/Aa*rhoa[1:-1] - n[1:-1]*rhos
        # diagonales

        b[:] = 2*α/dx**2 + dg
        # second dérivée
        d2 = (Js[0:-2] - 2*Js[1:-1] + Js[2:]) / dx**2
        # RHS
        d[:] = α*d2 - g
        # solve
        Δ = thomas(a, b, c, d)
        old = Js[1:-1].copy()
        Js[1:-1] = old + Δ
        
        Js[0] = Js_frontiere
        Js[-1] = Js_frontiere
        err = np.linalg.norm(Js[1:-1] - old) / np.linalg.norm(Js[1:-1])

        rhos = np.append(np.append(1e-16, rhos), 1e-16)

    return Js, Ja, rhos

def predictions_all_simulations(model, DATA, P_PDE):
    Nx = DATA['Tn+1(K)'].shape[0]
    P_PDE['Nx'] = Nx
    Nt = P_PDE['Nt']
    Ns = len(DATA['simu_id'])
    T = np.zeros((Nx,Nt,Ns))
    Js = np.zeros((Nx,Nt,Ns))
    rhos = np.zeros((Nx,Nt,Ns))
    for i in range(Ns):
        model.simu_index = i
        T[:,:,i], Js[:,:,i],rhos[:,:,i] = model.predictions(i,P_PDE)

    return T, Js,rhos

def solverNR(x,T,Js_frontiere,Js_avant,P_PDE):
    """ x, T, Js & Js_avant (Js_n-1) vectors. Js_frontiere scalaire, conditions Dirichlet. P_PDE dict. stockant les paramètres 
    All vectors shape are (Nx,)
    """
    Nx = len(x)
    # Paramètres à calculer
    Itot = Js_frontiere*P_PDE['As']
    #alpha = -P_PDE['R_inter_current']*P_PDE['As']/P_PDE['Aa']*P_PDE['ha']
    alpha = -P_PDE['R_inter_current']*P_PDE['As']/P_PDE['Aa']*P_PDE['ha_current']
    rhoa = P_PDE['rho_ag_spline'](T)

    # Calcul des courants
    n = P_PDE['n'] * (T < P_PDE['Tc']) + (T >= P_PDE['Tc'])
    JcX = JcX_global(x,P_PDE['xc_current'],P_PDE['prefact_current'],P_PDE['sigma_current'])
    JcT = P_PDE['Jc0']*(P_PDE['Tc']-T)/(P_PDE['Tc']-P_PDE['T0'])*(T<P_PDE['Tc']) + (T>=P_PDE['Tc'])
    #JcT = P_PDE['Jc0'] * (0.00004016 * (T**4) - 0.0089 * (T**3) + 0.7256 * (T**2) - 35.7 * T + 1189) * (T < P_PDE['Tc']) / 202.7 + (T >= P_PDE['Tc'])
    Jc = JcT*JcX+1

    # Guess initial
    Js = Js_frontiere/(Js_avant[0]+1)*Js_avant
    
    erreur = 10
    k=0
    while erreur >= P_PDE['toll_solverNR']:
        k = k+1
        if k >= 100000:
            print("100 000 iterations reached.")
        # Constante pour le calcul
        Ja = (Itot-P_PDE['As']*Js)/P_PDE['Aa']
        rhos = np.minimum(np.maximum( P_PDE['E0']/Jc[1:-1]*np.abs(Js[1:-1]/Jc[1:-1])**(n[1:-1]-1),10**(-16)),P_PDE['rho_s_normal'])
        g = rhoa[1:-1]*np.maximum(Ja[1:-1],0) - rhos*np.maximum(Js[1:-1],0)
        dgdJ = -P_PDE['As']/P_PDE['Aa']*rhoa[1:-1]-n[1:-1]*rhos

        # Différences Finies
        d2J_dx2 = (Js[0:-2] - 2*Js[1:-1] + Js[2:])/P_PDE['dx']**2

        Mjj_k = (2 * alpha / P_PDE['dx']**2 + dgdJ)
        M_off = -alpha / P_PDE['dx']**2 * np.ones(Nx-2)

        M_k_data = [M_off, Mjj_k, M_off]
        M_k = spdiags(M_k_data, [-1, 0, 1], Nx-2, Nx-2,format='csr')
        R_k = (alpha * d2J_dx2 - g)

        # Solve system
        delta_Js = linalg.spsolve(M_k, R_k)
        J_k1 = Js[1:-1]
        Js = Js[1:-1] + delta_Js
        Js = np.concatenate(([Js_frontiere], Js, [Js_frontiere]))

        # Calculate error
        erreur = np.linalg.norm(Js[1:-1] - J_k1) / np.linalg.norm(Js[1:-1])

    return Js, rhos

def differentiation(x, Js, type):
    dJ = np.copy(Js)

    if type == "O2":
        dJ[0] = 0
        dJ[-1] = 0
        dJ[1:-1] = (Js[2:] - Js[:-2]) / (2 * (x[1] - x[0]))

    elif type == "O4":
        dJ[0] = 0
        dJ[1] = 0
        dJ[-1] = 0
        dJ[-2] = 0
        dJ[2:-2] = (Js[:-4] - 8 * Js[1:-3] + 8 * Js[3:-1] - Js[4:]) / (12 * (x[1] - x[0]))

    return dJ

def func_keff(T,P_PDE,ids):
    """ k_eff selon Levin
    """
    keff = P_PDE['ha'][ids]*P_PDE['k_ag_spline'](T) + P_PDE['hs']*P_PDE['k_ybco_spline'](T) + P_PDE['hhast']*P_PDE['k_hast_spline'](T)
    return keff

def func_dkeffdT(T,P_PDE,ids):
    """ dk_eff/dT selon Levin
    """
    dkdT = P_PDE['ha'][ids]*P_PDE['dkdT_ag_spline'](T) + P_PDE['hs']*P_PDE['dkdT_ybco_spline'](T)+ P_PDE['hhast']*P_PDE['dkdT_hast_spline'](T)
    return dkdT

def func_rhomCpeff(T, P_PDE,ids):
    """  rhom et Cp, rhom*Cp selon Levin
    """
    rhomCp_eff = P_PDE['ha'][ids]*P_PDE['rhom_a']*P_PDE['Cp_ag_spline'](T) + P_PDE['hs']*P_PDE['rhom_s']*P_PDE['Cp_ybco_spline'](T)+ P_PDE['hhast']*P_PDE['rhom_hast']*P_PDE['Cp_hast_spline'](T)
    return rhomCp_eff

def rho_superconductor_tensor(x,T,Js,JcX,P_PDE):
    n = P_PDE['n'] * (T < P_PDE['Tc']) + (T >= P_PDE['Tc'])
    #JcT = P_PDE['Jc0'] * (0.00004016 * (T**4) - 0.0089 * (T**3) + 0.7256 * (T**2) - 35.7 * T + 1189) * (T < P_PDE['Tc']) / 202.7 + (T >= P_PDE['Tc'])
    JcT = P_PDE['Jc0']*(P_PDE['Tc']-T)/(P_PDE['Tc']-P_PDE['T0'])*(T<P_PDE['Tc']) + (T>=P_PDE['Tc'])
    Jc = JcT*JcX+1
    rhos = np.minimum(np.maximum( P_PDE['E0']/Jc*np.abs(Js/Jc)**(n-1),10**(-16)),P_PDE['rho_s_normal'])
    return rhos

def Iapp(index:int,P_PDE:Dict):
    Ic_montee = P_PDE['Ic_montee'][index]
    t_end = P_PDE['t_end']
    Ic = P_PDE['Ic']
    I = P_PDE['I'][index]

    x = [0,0.1*t_end,t_end]
    y = [0,Ic_montee*Ic,I]
    Itot = InterpolatedUnivariateSpline( x, y, k=1, ext=3)
    return Itot 



def DufortFrankel_Levin1D(DATA_test,P_PDE):

    x = P_PDE['x']
    t = P_PDE['t'][1:-1] # Commence à dt, fini à t_end - dt
    dx = P_PDE['dx']
    dt = P_PDE['dt']
    
    Nt = P_PDE['Nt']
    Nx = len(x)
    Ns = len(DATA_test['simu_id'])
    

    # Structure des données (Nx,Nt-2,Ns)
    T_previous = DATA_test['Tn-1(K)']
    T_current = DATA_test['Tn(K)']
    T_true = DATA_test['Tn+1(K)']

    Js_avant = 10**(10)*np.ones((Nx,Ns))

    simu_index = np.array(DATA_test['simu_id'])-1 # Simulation indexes in Python
    

    # Structure tensor prédictions
    Js = np.zeros((Nx,Ns))
    dJs = np.zeros((Nx,Ns))
    T_pred = np.zeros((Nx,Nt-2,Ns))

    erreur_rel = []
    erreur_abs = []

    ## Boucle
    T_pre = T_previous[:,0,:]
    T_curr = T_current[:,0,:]

    for i in range(T_previous.shape[1]):

        # Initilisation constantes matériaux(T)
        keff = func_keff(T_curr,P_PDE,simu_index)
        dkdT_eff = func_dkeffdT(T_curr,P_PDE,simu_index)
        rhomCpeff = func_rhomCpeff(T_curr,P_PDE,simu_index)

        # Boucle sur le nombre de simulation pour trouver Js de chaque simu
        JcX = np.zeros((Nx,Ns))
        for simu in simu_index:
            # Conditions frontières et paramètres géométriques
            Itot = Iapp(simu,P_PDE)
            Js_frontiere = Itot(t[i])/P_PDE['As']
            P_PDE['xc_current'] = P_PDE['xc'][simu,:]
            P_PDE['sigma_current'] = P_PDE['sigma'][simu,:]
            P_PDE['prefact_current'] = P_PDE['prefact'][simu,:]
            P_PDE['R_inter_current'] = P_PDE['R_inter'][simu]
            P_PDE['ha_current'] = P_PDE['ha'][simu]

            # Superconductor resistivity
            P_PDE['Aa'] = P_PDE['w']*P_PDE['ha_current']
            JcX[:,simu] = JcX_global(x,P_PDE['xc_current'],P_PDE['prefact_current'],P_PDE['sigma_current'])
            # Solver
            Js[:,simu], rhos = solverNR(x,T_curr[:,simu],Js_frontiere,Js_avant[:,simu],P_PDE)
            dJs[:,simu] = differentiation(x,Js[:,simu],"O4")

        # For the next iteration
        P_PDE['Aa'] = P_PDE['w']*P_PDE['ha'][simu_index]
        Js_avant = Js
        Itotal = Js[0,:]*P_PDE['As']
        Ja = (Itotal-P_PDE['As']*Js)/P_PDE['Aa'] 

        # Résistivité
        rhoa = P_PDE['rho_ag_spline'](T_curr)
        rhos =  rho_superconductor_tensor(x,T_curr,Js,JcX,P_PDE) # PowerLaw
        #Q_Ja = P_PDE['ha']*rhoa*np.maximum(Ja,0)**2
        #Q_Js = P_PDE['hs']*rhos*np.maximum(Js,0)**2
        Q_Ja = P_PDE['ha'][simu_index]*rhoa*Ja**2
        Q_Js = P_PDE['hs']*rhos*Js**2
        Q_conv = -2*P_PDE['h_conv']*(T_curr - P_PDE['T0'])

        Q_inter = P_PDE['R_inter'][simu_index]*P_PDE['ha'][simu_index]**2*(P_PDE['As']/P_PDE['Aa'])**2*(dJs)**2

        if P_PDE['pred_recursif'] == False:
            Q_Ja = DATA_test['Q_ag(W/m2)'][:,i,:]
            Q_Js = DATA_test['Q_supra(W/m2)'][:,i,:]
            Q_inter = DATA_test['Q_inter(W/m2)'][:,i,:]
            Q_conv = DATA_test['Q_conv(W/m2)'][:,i,:]
        Q = Q_Ja + Q_Js + Q_inter + Q_conv

        # Thermal Equation
        A_T  = 2 * keff[1:-1] * P_PDE['dt'] / ( rhomCpeff[1:-1] * dx**2 )
        B_T = dkdT_eff[1:-1] * P_PDE['dt'] / (2 * rhomCpeff[1:-1] * dx**2 )
        gamma_T = 2*P_PDE['dt']/( rhomCpeff[1:-1] )

        Tn_ip = T_curr[2:]
        Tn_im = T_curr[:-2]

        # Doit prendre Nx-2 coordonnées car on évite les conditions frontières à cause de Ti-1, Ti+1
        T_pred[1:-1,i,:] = 1/(1+A_T)* ( (1-A_T)*T_pre[1:-1] + A_T*(Tn_ip+Tn_im) + B_T*(Tn_ip-Tn_im)**2 + gamma_T*Q[1:-1])

        # Bondary conditions
        T_pred[0,i,:] = P_PDE['T0']
        T_pred[-1,i,:] = P_PDE['T0']

        # Erreur relative du profil + Erreu absolue
        erreur_rel.append(np.linalg.norm(T_true[:,i,:] - T_pred[:,i,:],axis=0) / np.linalg.norm(T_true[:,i,:],axis=0))
        erreur_abs.append(np.max(T_true[:,i,:] - T_pred[:,i,:],axis=0))

        # Update T_curr & T_pre
        T_pre = T_curr
        T_curr = T_pred[:,i,:]

    return T_pred, erreur_rel,erreur_abs

def Levin1D_predictions(P_PDE):
    """ À partir des conditions intiales et frontières, prédit la physique selon les informations dans le dictionnaire P_PDE. 
    Il n'est pas nécessaire de donner un target avec lequel il va comparer. Conditions initiales assumé que c'est T0.
    """

    x = P_PDE['x']
    t = P_PDE['t'][1:-1] # Commence à dt, fini à t_end - dt
    dx = P_PDE['dx']
    dt = P_PDE['dt']
    
    Nt = P_PDE['Nt']
    Nx = len(x)
    Ns = 1

    # Structure des données (Nx,Ns) ---> général (Nx,Nt,Ns)
    T_pre = P_PDE['T0']*np.ones((Nx,Ns))
    T_curr = P_PDE['T0']*np.ones((Nx,Ns))

    Js_avant = 10**(10)*np.ones((Nx,Ns))

    # Structure tensor prédictions
    Js = np.zeros((Nx,Ns))
    dJs = np.zeros((Nx,Ns))
    T_pred = np.zeros((Nx,Nt-2,Ns))

    for i in range(T_pred.shape[1]):

        # Initilisation constantes matériaux(T)
        keff = func_keff(T_curr,P_PDE)
        dkdT_eff = func_dkeffdT(T_curr,P_PDE)
        rhomCpeff = func_rhomCpeff(T_curr,P_PDE)

        # Boucle sur le nombre de simulation pour trouver Js de chaque simu
        JcX = np.zeros((Nx,Ns))
        for simu in range(Ns):
            # Conditions frontières et paramètres géométriques
            Itot = Iapp(simu,P_PDE)
            Js_frontiere = Itot(t[i])/P_PDE['As']
            P_PDE['xc_current'] = P_PDE['xc']
            P_PDE['sigma_current'] = P_PDE['sigma']
            P_PDE['prefact_current'] = P_PDE['prefact']
            P_PDE['R_inter_current'] = P_PDE['R_inter']

            # Superconductor resistivity
            JcX[:,simu] = JcX_global(x,P_PDE['xc_current'],P_PDE['prefact_current'],P_PDE['sigma_current'])
            # Solver
            Js[:,simu], rhos = solverNR(x,T_curr[:,simu],Js_frontiere,Js_avant[:,simu],P_PDE)
            dJs[:,simu] = differentiation(x,Js[:,simu],"O4")

        # For the next iteration
        Js_avant = Js
        Itotal = Js[0,:]*P_PDE['As']
        Ja = (Itotal-P_PDE['As']*Js)/P_PDE['Aa'] 

        # Résistivité
        rhoa = P_PDE['rho_ag_spline'](T_curr)
        rhos =  rho_superconductor_tensor(x,T_curr,Js,JcX,P_PDE)

        Q_Ja = P_PDE['ha']*rhoa*Ja**2
        Q_Js = P_PDE['hs']*rhos*Js**2
        Q_conv = -2*P_PDE['h_conv']*(T_curr - P_PDE['T0'])

        Q_inter = P_PDE['R_inter_current']*P_PDE['ha']**2*(P_PDE['As']/P_PDE['Aa'])**2*(dJs)**2

        Q = Q_Ja + Q_Js + Q_inter + Q_conv

        # Thermal Equation
        A_T  = 2 * keff[1:-1] * P_PDE['dt'] / ( rhomCpeff[1:-1] * dx**2 )
        B_T = dkdT_eff[1:-1] * P_PDE['dt'] / (2 * rhomCpeff[1:-1] * dx**2 )
        gamma_T = 2*P_PDE['dt']/( rhomCpeff[1:-1] )

        Tn_ip = T_curr[2:]
        Tn_im = T_curr[:-2]

        # Doit prendre Nx-2 coordonnées car on évite les conditions frontières à cause de Ti-1, Ti+1
        T_pred[1:-1,i,:] = 1/(1+A_T)* ( (1-A_T)*T_pre[1:-1] + A_T*(Tn_ip+Tn_im) + B_T*(Tn_ip-Tn_im)**2 + gamma_T*Q[1:-1])

        # Bondary conditions
        T_pred[0,i,:] = P_PDE['T0']
        T_pred[-1,i,:] = P_PDE['T0']

        # Update T_curr & T_pre
        T_pre = T_curr
        T_curr = T_pred[:,i,:]
    
    return T_pred


def hPGRNN_predictions(model):
    """ Prédictions pour le modèle hPGRNN. À partir des conditions intiales et frontières, prédit la physique selon les informations dans le dictionnaire P_PDE. 
    Il n'est pas nécessaire de donner un target avec lequel il va comparer. Conditions initiales assumé que c'est T0.
    """
    P_PDE = model.P_PDE
    x = P_PDE['x']
    t = P_PDE['t'][1:-1] # Commence à dt, fini à t_end - dt
    dx = P_PDE['dx']
    dt = P_PDE['dt']
    
    Nt = P_PDE['Nt']
    Nx = len(x)
    Ns = 1

    # Structure des données (Nx,Ns) ---> général (Nx,Nt,Ns)
    T_pre = P_PDE['T0']*np.ones((Nx,Ns))
    T_curr = P_PDE['T0']*np.ones((Nx,Ns))

    Js_avant = 10**(10)*np.ones((Nx,Ns))

    # Structure tensor prédictions
    Js = np.zeros((Nx,Ns))
    dJs = np.zeros((Nx,Ns))
    T_pred = np.zeros((Nx,Nt-2,Ns))

    for i in range(T_pred.shape[1]):

        # Boucle sur le nombre de simulation pour trouver Js de chaque simu
        JcX = np.zeros((Nx,Ns))
        for simu in range(Ns):
            # Conditions frontières et paramètres géométriques
            Itot = Iapp(simu,P_PDE)
            Js_frontiere = Itot(t[i])/P_PDE['As']
            P_PDE['xc_current'] = P_PDE['xc']
            P_PDE['sigma_current'] = P_PDE['sigma']
            P_PDE['prefact_current'] = P_PDE['prefact']
            P_PDE['R_inter_current'] = P_PDE['R_inter']

            # Superconductor resistivity
            JcX[:,simu] = JcX_global(x,P_PDE['xc_current'],P_PDE['prefact_current'],P_PDE['sigma_current'])
            # Solver
            Js[:,simu], rhos = solverNR(x,T_curr[:,simu],Js_frontiere,Js_avant[:,simu],P_PDE)
            dJs[:,simu] = differentiation(x,Js[:,simu],"O4")

        # For the next iteration
        Js_avant = Js
        Itotal = Js[0,:]*P_PDE['As']
        Ja = (Itotal-P_PDE['As']*Js)/P_PDE['Aa'] 

        # Résistivité
        rhoa = P_PDE['rho_ag_spline'](T_curr)
        rhos =  rho_superconductor_tensor(x,T_curr,Js,JcX,P_PDE)

        Q_Ja = P_PDE['ha']*rhoa*Ja**2
        Q_Js = P_PDE['hs']*rhos*Js**2
        Q_conv = -2*P_PDE['h_conv']*(T_curr - P_PDE['T0'])

        Q_inter = P_PDE['R_inter_current']*P_PDE['ha']**2*(P_PDE['As']/P_PDE['Aa'])**2*(dJs)**2

        Q = np.stack((Q_Ja, Q_Js, Q_inter, Q_conv), axis=1) #(Nx,4)
        Q = Q.reshape(1,Q.shape[0],Q.shape[1]) #(1,Nx,3) for fitting forward() model shapes
      
        # Prediction

        Y_pred_recur = model.predict( T_curr.T, T_pre.T, Q)

        T_pred[:,i,:] = Y_pred_recur.cpu().detach().numpy().T # Convert to cpu
        T_pred[:, i, :][T_pred[:, i, :] < P_PDE['T0']] = P_PDE['T0']
        
        # Recurrence
        T_pre = np.copy(T_curr)
        T_curr = np.copy(T_pred[:,i,:])
    
    return T_pred

def init_weights_JH(m):
    if type(m) == torch.nn.Linear:
         torch.nn.init.xavier_uniform_(m.weight)
         m.bias.data.fill_ (0.01)

def cubes2sequences_JH(cube_DATA, steps):
        # Input: cube_DATA: dictionnary of cube of DATA 
        # The axis of the cubes are [i,n,s] i: index along "x", n: index along time, s: index along simulations
        # Output: dictionnary of sequences
        # Each sequence has the dimension [batch, sequence along time, Nx] or [N, L, Nx]
        tn = np.stack( sorted( set( cube_DATA['tn(s)'].flatten()) ) )
        
        # Loop over element in dictionnary
        sequences = {}
        for element in cube_DATA.keys():
            # Loop over starting points for multi-step predictions (tn)
            temp = []
            for i_t, t in enumerate( tn[:len(tn)-int(np.floor(steps))] ): 
                temp.append( cube_DATA[element][ :, i_t:(i_t+steps+1), : ] )
            
            sequences[element] = np.transpose( np.concatenate(temp,axis=2), (2,1,0) )
        return sequences        

def DATA_to_sequences(DATA,steps,P_PDE,P):
    """ Input DATA: Dict. with all tensors, steps: number of predictions steps Tn+1,Tn+2,..., P_PDE: Dictionnary with all PDE parameters
    Output: Dictionnary of sequences. Each sequence has the dimension [batch, sequence along time, Nx]
    """

    # Time taken for Tn(K)
    tn = P_PDE['t'][1:-1]
    dtype = P['dtype']
    device = P['device']

    sequences = {}
    simu_id_tensors = []
    simu_id_done = False

    for element in DATA.keys():
        if not element == 'simu_id':
            temp = []
            if '(K)' in element: # Temperature are in float64 numpy arrays
                for i_t,_ in enumerate( tn[:len(tn)-int(np.floor(steps))] ):
                    temp.append( DATA[element][ :, i_t:(i_t+steps+1), : ] )
                    if not simu_id_done:
                        simu_id_tensors.append( DATA['simu_id'] )
                
                if not simu_id_done:
                    simu_id_tensors = torch.tensor(simu_id_tensors,device=device).flatten()
                    simu_id_done = True

                sequences[element] = np.transpose( np.concatenate(temp,axis=2), (2,1,0) )
                sequences[element] = torch.tensor(sequences[element],dtype=dtype,device=device)
            else:
                for i_t,_ in enumerate( tn[:len(tn)-int(np.floor(steps))] ):
                    temp.append( DATA[element][ :, i_t:(i_t+steps+1), : ] )
                
                sequences[element] = torch.cat(temp, dim=2).permute(2, 1, 0)

    sequences['simu_ids'] = simu_id_tensors  
    sequences['Q(W/m2)'] = torch.stack((sequences['Q_ag(W/m2)'].to(dtype=dtype, device=device),
                                         sequences['Q_supra(W/m2)'].to(dtype=dtype, device=device), 
                                         sequences['Q_inter(W/m2)'].to(dtype=dtype, device=device),
                                         sequences['Q_conv(W/m2)'].to(dtype=dtype, device=device)), dim=3)
    del sequences['Q_ag(W/m2)']
    del sequences['Q_supra(W/m2)']
    del sequences['Q_inter(W/m2)']
    del sequences['Q_conv(W/m2)']
    return sequences

def DATA_to_sequences_remaster(DATA,L,P_PDE,P):
    """ DATA est un dictionnaire qui contient tous les tensors de température et de chaleur sous forme (Nx,Nt-2,Ns). T en numpy et Q en pytorch. L longueur de la séquence qu'on désire.
    """
    device = P['device']
    dtype = P['dtype']

    Nx, Nt, Ns = DATA['Tn+1(K)'].shape

    Tn2 = DATA['Tn+1(K)'].transpose().reshape(-1, Nx)
    Tn1 = DATA['Tn(K)'].transpose().reshape(-1, Nx)
    Tn0 = DATA['Tn-1(K)'].transpose().reshape(-1, Nx)

    T = torch.tensor(np.stack((Tn0, Tn1, Tn2), axis=1),device= P['device'])

    sequences = {}
    for element in DATA.keys():
        if '(W/m2)' in element:
            variable = DATA[element].permute(2,1,0) # Maintenant x.shape = (Ns, Nt-2, Nx)
            temp = []
            for s in range(Ns):
                for start_idx in range(Nt - L + 1):
                    seq = variable[s, start_idx:start_idx+L, :]
                    temp.append(seq)

            sequences[element] = torch.stack(temp)

    Q = torch.stack((sequences['Q_ag(W/m2)'].to(dtype=dtype, device=device),
                                         sequences['Q_supra(W/m2)'].to(dtype=dtype, device=device), 
                                         sequences['Q_inter(W/m2)'].to(dtype=dtype, device=device),
                                         sequences['Q_conv(W/m2)'].to(dtype=dtype, device=device)), dim=3)
    del sequences['Q_ag(W/m2)']
    del sequences['Q_supra(W/m2)']
    del sequences['Q_inter(W/m2)']
    del sequences['Q_conv(W/m2)']

    return sequences  


def DATA_to_sequences_v2(DATA, L, P_PDE, P):
    """
    Convertit les données en séquences temporelles de longueur L.
    
    Paramètres :
    - DATA : dict contenant des clés 'Tn+1(K)', 'Tn(K)', 'Tn-1(K)', et les flux Q_* (W/m2).
             Formes attendues :
               Tn+1(K), Tn(K), Tn-1(K) : (Nx, Nt, Ns) en numpy
               Q_ag(W/m2), Q_supra(W/m2), Q_inter(W/m2), Q_conv(W/m2) : (Nx, Nt, Ns) en torch
    - L : longueur de séquence temporelle désirée
    - P : dictionnaire contenant 'device' et 'dtype'
    
    Sorties :
    - T  tenseur regroupant les températures (Samples,L,Nx)
    - Q : tenseur regroupant les flux Q de forme (Samples, L, Nx, 4)
    
    Note : La forme finale des séquences est (Samples, L, Nx).
    """
    device = P['device']
    dtype = P['dtype']

    # Récupération des dimensions depuis Tn+1(K)
    Nx, _, Ns = DATA['Tn+1(K)'].shape

    # Convertir Tn-1(K), Tn(K), Tn+1(K) en tenseurs PyTorch et les réorganiser sous la forme (Nt*Ns, Nx) après transpose

    # Ici, transpose((1,2,0)) va donner (Nt, Ns, Nx), puis reshape(-1,Nx) => (Nt*Ns, Nx)
    Tn2 = torch.from_numpy(DATA['Tn+1(K)'].transpose(1,2,0).reshape(-1, Nx)).to(device=device, dtype=dtype)
    Tn1 = torch.from_numpy(DATA['Tn(K)'].transpose(1,2,0).reshape(-1, Nx)).to(device=device, dtype=dtype)
    Tn0 = torch.from_numpy(DATA['Tn-1(K)'].transpose(1,2,0).reshape(-1, Nx)).to(device=device, dtype=dtype)

    # Empilement des 3 temps : (Nt*Ns, 3, Nx)
    T = torch.stack((Tn0, Tn1, Tn2), dim=1)  

    Q_sequences = []  # On stockera ici les séquences Q, puis on concaténera

    selected_features = [
        'Q_ag(W/m2)', 
        'Q_supra(W/m2)', 
        'Q_inter(W/m2)', 
        'Q_conv(W/m2)', 
        'Tn(K)'
    ]
    # Traiter chaque variable du DATA
    for element in DATA.keys():
        if any(feature in element for feature in selected_features):

            # variable: (Nx, Nt, Ns) -> passer en (Ns, Nt, Nx) pour unfold sur Nt
            variable = DATA[element]  # Supposé torch.Tensor

            if 'Tn(K)' in element:
                variable = torch.tensor(variable, dtype=dtype, device=device)
                padding = P_PDE['T0']*torch.ones((Nx,L-1,Ns),device=device)
                variable = torch.cat([padding, variable],dim=1)
            else:
                # variable: (Nx, Nt, Ns) -> passer en (Ns, Nt, Nx) pour unfold sur Nt
                variable = variable.to(device) 
                padding = torch.zeros((Nx,L-1,Ns),device=device)
                variable = torch.cat([padding, variable],dim=1)

            variable = variable.permute(2,1,0)  # (Ns, Nt, Nx) nt-L+1 "fenêtres" le long de l'axe temporel
            # unfold : dimension=1 (l'axe temporel), size=L, step=1
            # Résultat: (Ns, Nt-L+1, L, Nx)
            variable_unfold = variable.unfold(dimension=1, size=L, step=1).permute(0, 1, 3, 2)

            # On reshape pour avoir (Ns*(Nt-L+1), L, Nx)
            var_seq = variable_unfold.reshape(-1, L, Nx)

            Q_sequences.append(var_seq.to(device=device, dtype=dtype))


    # Empilage des Q le long de la dimension feature

    # var_seq.shape = (Samples, L, Nx) pour chaque
    # On empile le long d'une nouvelle dimension => (Samples, L, Nx, 4)
    Q = torch.stack(Q_sequences, dim=3)

    return T, Q


def Multistep_predictions_Levin(model, DATA_test, simu_index, P_PDE):
    ## Test set Multi-step Prediction at each time(test set)
    Q_list, Y_pred_recur_list, Y_true_list, tn_list = [],[],[],[]
    
    # Data shape [Nx,Nt,Ns]
    x      = P_PDE['x']
    X1     = DATA_test['Tn-1(K)'] 
    X2     = DATA_test['Tn(K)']
    Y_true = DATA_test['Tn+1(K)']
    tn     = P_PDE['t'][1:-1]

    if not P_PDE['pred_recursif']:
        X3 = np.stack(( DATA_test['Q_ag(W/m2)'],  DATA_test['Q_supra(W/m2)'],  DATA_test['Q_inter(W/m2)'], DATA_test['Q_conv(W/m2)']), axis=1) #(Nx,3)

    
    Nx = X1.shape[0]
    Nt = X1.shape[1]
    
    # Initialize values at t=0
    previous_recur = np.expand_dims( X1[:,0,simu_index], 1).T # Voir documentation en ligne, c'est quand même clair.
    current_recur  = np.expand_dims( X2[:,0,simu_index], 1).T
    Js_avant = 10**10*np.ones(Nx)

    Itot = Iapp(simu_index,P_PDE)
    P_PDE['ha_current'] = P_PDE['ha'][simu_index]
    P_PDE['Aa'] = P_PDE['w']*P_PDE['ha_current']

    for n in range( 0, Nt ):

        if P_PDE['pred_recursif']:
            
            Itot_n = Itot(tn[n])
            Js_frontiere = Itot_n/P_PDE['As']
            P_PDE['xc_current'] = P_PDE['xc'][simu_index,:]
            P_PDE['sigma_current'] = P_PDE['sigma'][simu_index,:]
            P_PDE['prefact_current'] = P_PDE['prefact'][simu_index,:]

            P_PDE['R_inter_current'] = P_PDE['R_inter'][simu_index]
            
            T = current_recur.flatten()

            Js, rhos = solverNR(x,T,Js_frontiere,Js_avant,P_PDE)
            Js_avant = Js
            rhos = np.insert(rhos, [0, len(rhos)], [1E-16, 1E-16])
            dJs = differentiation(x,Js,"O4")
            Ja = (Itot_n-P_PDE['As']*Js)/P_PDE['Aa']

            Q_Ja = P_PDE['ha_current']*P_PDE['rho_ag_spline'](T)*np.maximum(Ja,0)**2
            Q_Js = P_PDE['hs']*rhos*np.maximum(Js,0)**2
            Q_inter = P_PDE['R_inter'][simu_index]*P_PDE['ha_current']**2*(P_PDE['As']/P_PDE['Aa'])**2*(dJs)**2
            Q_conv = -2*P_PDE['h_conv']*(T-P_PDE['T0'])
            Q = np.stack((Q_Ja, Q_Js, Q_inter, Q_conv), axis=1) #(Nx,4)
            Q = Q.reshape(1,Q.shape[0],Q.shape[1]) #(1,Nx,3) for fitting forward() model shapes

        else:
            print("WHATCH OUT, Q FORMAT CHANGED (Qag + Qsupra + Qinter)")
            Q = np.expand_dims( X3[:,n,simu_index],1 ).T

        
        # Prediction
        Y_pred_recur = model.predict( current_recur, previous_recur, Q, simu_index)
        Q_list.append(Q)

        Y_pred_recur = Y_pred_recur.cpu().detach().numpy() # Convert to cpu
        # Save data
        Y_pred_recur_list.append( Y_pred_recur )
        Y_true_list.append( np.expand_dims(Y_true[:,n,simu_index], 1).T )
        tn_list.append( tn[n]  )
        # Recurrence
        previous_recur = np.copy(current_recur)
        current_recur  = np.copy(Y_pred_recur)
        
    # Gather results
    out = {}
    out['time'] = np.array( tn_list )
    out['pred'] = np.squeeze( np.stack(Y_pred_recur_list),axis=1)
    out['Q'] = np.squeeze( np.stack(Q_list),axis=1).transpose(1,0,2)
    out['true'] = np.squeeze( np.array( Y_true_list), axis=1)
    return out

def relative_error(true,predict):
    error_norm = np.linalg.norm(true - predict, 2)
    true_norm = np.linalg.norm(true, 2)
    
    relative_error = error_norm / true_norm
    return relative_error

def max_absolute_error_mean(true,predict):

    # Max errors per column
    max_errors = np.max(np.abs(true - predict), axis=0) # vector

    # Mean error max
    mean_max_error = np.mean(max_errors) # scalar
    return mean_max_error

def RMSE_JH(true, predict, scale='lin'):
    if scale.lower() == 'lin':
        return np.sqrt(mean_squared_error( true, predict ))
    if scale.lower() == 'log':
        temp =  ( np.log10(predict.to_numpy()) - np.log10(true.to_numpy()))**2
        return np.sqrt( temp.sum()/len(temp) )

def model_predictions_hPGRNN(model,DATA_test, P_PDE):
    """ Do predictions on all the test set. Called Multistep_predictions_Levin which assume PINN with finite difference structure. Q might be calculated by solverNR at each iteration.
    model: pytorch model, DATA_test : Dictionnary with data preprocessed, P_PDE: Dictionnary with the problem parameters.
    """
    Ns_test = len(DATA_test['simu_id'])
    PRED_test, TRUE_test = [],[]
    Q_list = []
    for s in range(Ns_test):
        out = Multistep_predictions_Levin(model,DATA_test, s,P_PDE)
        PRED_test.append( out['pred'] )
        TRUE_test.append( out['true'] )
        Q_list.append(out['Q'])

    time_test = out['time']
    PRED_test = np.stack(PRED_test).T
    TRUE_test = np.stack(TRUE_test).T
    Q_list = np.stack(Q_list,axis=3)

    return time_test,TRUE_test, PRED_test, Q_list

def MetricsEvaluation(true,pred):
    """ Calculate RMSE, Maximum absolute error (mean between each test), and relative error L2 
    true and pred variables are tensors of shape (Nx,Nt,Ns)
    """
    RMSE = []
    E_rel = []
    MAEM = []
    for n in range(pred.shape[1]):
        RMSE.append( RMSE_JH( true[:,n,:],pred[:,n,:] )  ) 
        E_rel.append( relative_error( true[:,n,:],pred[:,n,:] )  ) 
        MAEM.append(max_absolute_error_mean(true[:,n,:],pred[:,n,:]))
    
    RMSE = np.stack(RMSE)
    E_rel = np.stack(E_rel)
    MAEM = np.stack(MAEM)

    return RMSE,E_rel,MAEM 

def plot_training_results(LOSS_train, LOSS_val, models_pred, models_name, EPOCH, STEP_models, now_str, save_model=True):
    # Add this line to initialize T_models
    T_models = models_name

    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    color_palette = sns.color_palette("YlGnBu", len(T_models) + 2)[::-1 ]

    plt.figure(figsize=(4, 6))

    # Display MSE
    plt.subplot(2, 1, 1)
    train_color = color_palette[0]
    val_color = color_palette[0]  # Use the same color for both train and val curves
    plt.semilogy(1 + np.arange(len(LOSS_train[-1])) * EPOCH[-1] / len(LOSS_train[-1]),
                 LOSS_train[-1], label='Train Coeff(T)', color=train_color)
    plt.semilogy(1 + np.arange(len(LOSS_val[-1])) * EPOCH[-1] / len(LOSS_val[-1]),
                 LOSS_val[-1], label='Val Coeff(T)', color=val_color, linestyle='--')
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add log grid
    plt.title(now_str)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('MSE')

    # Display steps
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(STEP_models[-1])) * EPOCH[-1] / (len(STEP_models[-1]) - 1),
             STEP_models[-1], label='', color=train_color)
    plt.grid(True)  # Add linear grid
    plt.legend()
    plt.ylabel('Multi-Steps training')
    plt.xlabel('Epoch')
    plt.tight_layout()

    if save_model:
        plt.savefig('results/' + now_str + '/Fig1_Training.png', dpi=300)
        with open('results/' + now_str + '/Loss_train.pkl', 'wb') as f:
            pickle.dump(LOSS_train, f)
        with open('results/' + now_str +'/Loss_val.pkl', 'wb') as f:
            pickle.dump(LOSS_val, f)


def Plot_prediction_error_general(choices_time,choice_simu,T_true,T_DF,T_models,models_name,P_PDE):
    sns.set(style="whitegrid")  # "whitegrid" or "darkgrid"
    color_palette = sns.color_palette("YlGnBu", len(T_models) + 2)  # "YlGnBu" similar to heatmap

    index_DF = np.where(np.isin(P_PDE['t'][2:], choices_time))[0]
    index_ML = np.where(np.isin(P_PDE['t_ML'][2:], choices_time))[0]
    xi_unique = P_PDE['x']

    title = P_PDE['title']

    plt.figure(figsize=(12,10))
    for i in range(len(choices_time)):
        plt.subplot(len(choices_time),2,2*i+1)
        plt.scatter(np.array(xi_unique),np.squeeze(T_true[  :,index_ML[i], choice_simu]),s=10, facecolors='none',edgecolors='k',label='t(n)=%.1f ms (Dataset)'  %(1e3*P_PDE['t_ML'][index_ML[i]]))
        plt.plot( np.array(xi_unique[:]), np.squeeze(T_DF[:,index_DF[i], choice_simu]),'r--', label='t(n)=%.1f ms (Dufort-F)' %(1e3*P_PDE['t'][index_DF[i]]) )
        for j in range(len(T_models)):
            T_pred = T_models[j]
            plt.plot( np.array(xi_unique),    np.squeeze(T_pred[  :,index_ML[i], choice_simu]),'--',  label='t(n)=%.1f ms ('%(1e3*P_PDE['t_ML'][index_ML[i]]) + models_name[j] + 'ML model)') #, color=color_palette[j + 2])
        
        if i==0: plt.title(title)
        sns.despine()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel('T(K)')
    plt.xlabel('x(m)')
    plt.tight_layout()

def multisteps_plot(T_true,T_DF,models_pred,models_name,P_PDE,now_str,save_model):
    #choi = np.arange(0,len(P_PDE['t_ML'][2:]), int(len(P_PDE['t_ML'][2:])/8) ) 
    choi = np.floor(np.linspace(0,len(P_PDE['t_ML'][2:]), 9))
    choi[-1] -= 1
    choice_simu_index = 0
    title = now_str+'\nTest set (multi-step prediction)'
    P_PDE['title'] = title 

    for i in range(3):
        choice_time_index = choi[i*3:(i+1)*3]
        choice_time = P_PDE['t_ML'][2:][choice_time_index.astype(int)]
        Plot_prediction_error_general(choice_time,choice_simu_index,T_true,T_DF,models_pred,models_name,P_PDE)
        if save_model:
            plt.savefig( 'results/' + now_str + '/Fig2-' + str(i+1) +'_Plot_test_Recursive.png', dpi=300)
    
def summary_multisteps_plot(T_true,T_DF,T_models,models_name,P_PDE,now_str,save_model,choice_simu):
    plt.style.use('default')
    #sns.set(style="whitegrid")  # "whitegrid" or "darkgrid"
    #color_palette = sns.color_palette("YlGnBu", len(T_models) + 2)  # "YlGnBu" similar to heatmap

    plt.figure(figsize=(10, 5))

    xi_unique = P_PDE['x']
    t_DF = P_PDE['t'][2:]
    t_ML = P_PDE['t_ML'][2:]
    dt_ML = P_PDE['dt_ML']
    #choice_simu = 0
    t1 = 0.06
    t2 = 0.08
    choices_time = [t1,t2]

    # Trouve les index les plus près
    index_DF = [np.argmin(np.abs(t_DF - t)) for t in choices_time]
    index_ML = [np.argmin(np.abs(t_ML - t)) for t in choices_time]

    # Plot for t_end
    plt.scatter(np.array(xi_unique), np.squeeze(T_true[:, -1, choice_simu]), s=10, facecolors='none', edgecolors='k', label='t(n)=%.2f ms (Dataset)' % (1e3 * t_ML[-1]))
    plt.plot(np.array(xi_unique), np.squeeze(T_DF[:,-1, choice_simu]), 'r--', label='t(n)=%.2f ms (Dufort-F)' % (1e3 * t_DF[-1]))
    for j in range(len(T_models)):
        T_pred = T_models[j]
        plt.plot( np.array(xi_unique),    np.squeeze(T_pred[:,-1, choice_simu]),'b--',  label='t(n)=%.2f ms ('%(1e3*t_ML[-1]) + models_name[j] + ' )')

    # Plot for tn_unique[t2]
    plt.scatter(np.array(xi_unique), np.squeeze(T_true[:, index_ML[-1], choice_simu]), s=10, facecolors='none', edgecolors='k', label='t(n)=%.2f ms (Dataset)' % (1e3 * t_ML[index_ML[-1]]))
    plt.plot(np.array(xi_unique), np.squeeze(T_DF[:, index_DF[-1], choice_simu]),'r--', label='t(n)=%.2f ms (Dufort-F)' % (1e3 * t_DF[index_DF[-1]]))
    for j in range(len(T_models)):
        T_pred = T_models[j]
        plt.plot( np.array(xi_unique),    np.squeeze(T_pred[:,index_ML[-1], choice_simu]),'b--',  label='t(n)=%.2f ms ('%(1e3*t_ML[index_ML[-1]]) + models_name[j]+ ' )' )

    # Plot for tn_unique[t1]
    plt.scatter(np.array(xi_unique), np.squeeze(T_true[:, index_ML[0], choice_simu]), s=10, facecolors='none', edgecolors='k', label='t(n)=%.2f ms (Dataset)' % (1e3 * t_ML[index_ML[0]]))
    plt.plot(np.array(xi_unique), np.squeeze(T_DF[:,  index_DF[0], choice_simu]), 'r--', label='t(n)=%.2f ms (Dufort-F)' % (1e3 *  t_DF[index_DF[0]]))
    for j in range(len(T_models)):
        T_pred = T_models[j]
        plt.plot( np.array(xi_unique),    np.squeeze(T_pred[:,index_ML[0], choice_simu]),'b--',  label='t(n)=%.2f ms ('%(1e3*t_ML[index_ML[0]]) + models_name[j]+ ' )')

    #sns.despine()
        #, color=color_palette[j + 2]
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('T(K)')
    plt.xlabel('x(m)')
    plt.title(now_str)
    plt.tight_layout()

    if save_model:
            plt.savefig('results/' + now_str + '/Fig2-4_Plot_test_Recursive.png', dpi=300)
            with open('results/' + now_str +'/T_true.pkl', 'wb') as f:
                pickle.dump(T_true, f)
            with open('results/' + now_str +'/T_DF.pkl', 'wb') as f:
                pickle.dump(T_DF, f)   
            #with open('results/' + now_str +'/T_models.pkl', 'wb') as f:
            #    pickle.dump(T_models, f) 
            with open('results/' + now_str +'/P_PDE.pkl', 'wb') as f:
                pickle.dump(P_PDE, f) 

def summary_last_multisteps_plot(T_true,T_DF,T_models,models_name,P_PDE,now_str,save_model,choice_simu):
    plt.style.use('default')
    #sns.set(style="whitegrid")  # "whitegrid" or "darkgrid"
    #color_palette = sns.color_palette("YlGnBu", len(T_models) + 2)  # "YlGnBu" similar to heatmap

    plt.figure(figsize=(10, 5))

    xi_unique = P_PDE['xi_unique']
    t_DF = P_PDE['tn_unique'] + P_PDE['tn_unique'][0]
    t_ML = P_PDE['tn_unique_ML'] + P_PDE['tn_unique_ML'][0]

    # Plot for t_end
    plt.scatter(np.array(xi_unique), np.squeeze(T_true[:, -1, choice_simu]), s=10, facecolors='none', edgecolors='k', label='t(n)=%.2f ms (Dataset)' % (1e3 * t_ML[-1]))
    plt.plot(np.array(xi_unique), np.squeeze(T_DF[:,-1, choice_simu]), 'r--', label='t(n)=%.2f ms (Dufort-F)' % (1e3 * t_DF[-1]))
    for j in range(len(T_models)):
        T_pred = T_models[j]
        plt.plot( np.array(xi_unique),    np.squeeze(T_pred[:,-1, choice_simu]),'b--',  label='t(n)=%.2f ms ('%(1e3*t_ML[-1]) + models_name[j] + ' )')

    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('T(K)')
    plt.xlabel('x(m)')
    plt.title(now_str)
    plt.tight_layout()

    if save_model:
            plt.savefig('results/' + now_str + '/Fig2-5_Plot_test_Recursive.png', dpi=300)

def summary_multisteps_plot_current(T_true,T_DF,T_models,models_name,P_PDE,now_str,save_model):
    plt.style.use('default')

    plt.figure(figsize=(12, 5))

    x = P_PDE['x']
    t_DF = P_PDE['t'][2:]
    t_ML = P_PDE['t_ML'][2:]
    Nx = len(x)

    choice_simu = 0
    t1 = 0.006
    index_DF = np.where(np.isin(t_DF, [t1, t_DF[-1]]))[0]
    index_ML = np.where(np.isin(t_ML,[t1, t_ML[-1]]))[0]

    I = P_PDE['I'][choice_simu]

    T_true = T_true[:,index_ML,:]
    T_DF = T_DF[:,index_DF,:]
    for i in range(len(T_models)):
        T_models[i] = T_models[i][:,index_ML,:]

    #Itot = I*t_DF[index_DF]/P_PDE['t_end']
    Iapplied = Iapp(choice_simu,P_PDE)
    Itot = Iapplied(t_DF[index_DF])

    for i in range(len(index_DF)):
        Js_frontiere = Itot[i]/(P_PDE['w']*P_PDE['hs'])
        Js_avant = Js_frontiere*np.ones(Nx)
        P_PDE['xc_current'] = P_PDE['xc'][choice_simu,:]
        P_PDE['sigma_current'] = P_PDE['sigma'][choice_simu,:]
        P_PDE['prefact_current'] = P_PDE['prefact'][choice_simu,:]
        P_PDE['R_inter_current'] = P_PDE['R_inter'][choice_simu]

        Js_true, _  = solverNR(x,T_true[:,i,choice_simu],Js_frontiere,Js_avant,P_PDE)
        Js_DF,_     = solverNR(x,T_DF[:,i,choice_simu],Js_frontiere,Js_avant,P_PDE)

        #plt.plot(x, Js_true*P_PDE['w']*P_PDE['hs'],'k', label='t(n)=%.2f ms (COMSOL)' % (1e3 * (t[time_index[i]]+ dt)))
        plt.scatter(x, Js_true*P_PDE['w']*P_PDE['hs'], s=30, facecolors='none', edgecolors='k',  label='t(n)=%.2f ms (COMSOL)' % (1e3 * (t_ML[index_ML[i]]) ))
        plt.plot(x, Js_DF*P_PDE['w']*P_PDE['hs'],'r--', linewidth=2, label='t(n)=%.2f ms (Diff. Finies)' % (1e3 * (t_DF[index_DF[i]])))

        for j in range(len(T_models)):
            Js_model,_ =  solverNR(x,T_models[j][:,i,choice_simu],Js_frontiere,Js_avant,P_PDE)
            plt.plot(x, Js_model*P_PDE['w']*P_PDE['hs'],'b--', linewidth=2, label='t(n)=%.2f ms ('%(1e3*(t_ML[index_ML[i]])) + models_name[j] + ')')  
    

    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Courant dans Supraconducteur (A)')
    plt.xlabel('x(m)')
    plt.title(now_str)
    plt.tight_layout()

    if save_model:
            plt.savefig('results/' + now_str + '/Fig2-4.5_Plot_J_test_Recursive.png', dpi=300)

def plot_recursive_rmse(P_PDE,RMSE_models,models_name,now_str,save_model):
    """ Assume que le premier éléments dans la liste RMSE_models est DF_DufortFrankel
    """
    #tn_models = P_PDE['tn_models']
    plt.style.use('default')
    plt.figure()
    plt.semilogy(P_PDE['t'][2:]*1000, RMSE_models[0],  label='FD (Dufort-F)' )
    for i in range(1,len(RMSE_models)):
        plt.semilogy( P_PDE['t_ML'][2:]*1000, RMSE_models[i],':', label= models_name[i-1])
        #plt.semilogy( tn_models[i-1], RMSE_models[i],':', label= models_name[i-1])
        
    
    #sns.despine()
    plt.ylabel('RMSE (K) (test set)')
    plt.xlabel('Time (ms)')
    plt.title(now_str)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_model: 
        plt.savefig( 'results/' + now_str + '/Fig3-1_RMSE_over_time.png', dpi=300)
        #with open('results/' + now_str + '/T_models.pkl', 'wb') as f:
        #    pickle.dump(RMSE_models, f)

def plot_ABC(model, P_PDE, now_str, save_model,type,c):
    device = torch.device('cpu')
    model.device = device
    x = np.linspace(P_PDE['T_min'], P_PDE['T_max'], 200)
    Qcool= np.linspace(P_PDE['Q_min_cool'], P_PDE['Q_max_cool'], 200)
    Q_ag_supra = np.linspace(P_PDE['Q_min_ag_supra_inter'], P_PDE['Q_max_ag_supra_inter'], 200)

    x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    Qcool_tensor = torch.tensor(Qcool, dtype=torch.float).unsqueeze(1)
    Q_ag_supra_tensor = torch.tensor(Q_ag_supra, dtype=torch.float).unsqueeze(1)

    if type.lower() == '2v':
        a_pred = model.forward_a(x_tensor,x_tensor,x_tensor).detach()
        b_pred = model.forward_b(x_tensor,x_tensor,x_tensor).detach()
        c_pred = model.forward_c(x_tensor,x_tensor,x_tensor).detach()
    elif type.lower() == 'pgrml2v':
        if c.lower() == 'c(q))':
            a_pred = model.forward_a(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
            b_pred = model.forward_b(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
            c_pred = model.forward_c2(Q_ag_supra_tensor,Q_ag_supra_tensor,Q_ag_supra_tensor,Q_ag_supra_tensor,Q_ag_supra_tensor).detach()
            e_pred = model.forward_e(Qcool_tensor,Qcool_tensor,Qcool_tensor,Qcool_tensor,Qcool_tensor).detach()
        else:
            a_pred = model.forward_a(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
            b_pred = model.forward_b(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
            c_pred = model.forward_c2(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
            e_pred = model.forward_e(x_tensor,x_tensor,x_tensor,x_tensor,x_tensor).detach()
    else:
        a_pred = model.forward_a(x_tensor).detach()
        b_pred = model.forward_b(x_tensor).detach()
        c_pred = model.forward_c(x_tensor).detach()
    if type.lower() == 'enhance':
        d_pred = model.forward_d(x_tensor).detach()

    fig = plt.figure(figsize=(12, 5))  # Adjust the figsize to your preference

    # Plot for a and c
    plt.subplot(1, 2, 1)
    plt.plot(x, a_pred, label='a(T)', color='blue', linewidth=2)
    if type.lower() == 'pgrml2v':
        plt.plot(x, b_pred, label='b(T)', color='orange', linewidth=2)
    else:
        plt.plot(x, c_pred, label='c(T)', color='green', linewidth=2)
    #plt.axvline(P_PDE['T_min'], color='k', dashes=[6, 2])
    #plt.axvline(P_PDE['T_max'], color='k', dashes=[6, 2])
    plt.grid()
    plt.legend()
    plt.xlabel('Temperature (K)')
    plt.ylabel('Coefficients [S.U.]')

    # Plot for b
    if type.lower() == 'pgrml2v':
        ax1 = fig.add_subplot(1, 2, 2)
        if c.lower() == 'c(q))':
            line1, = ax1.plot(Q_ag_supra, c_pred, label='c(Q)', color='green', linewidth=2)
            ax2 = ax1.twiny()
            line2, = ax2.plot(Qcool, e_pred, label='$\epsilon(Q)$', color='red', linewidth=2)
        else:
            line1, = ax1.plot(x, c_pred, label='c(T)', color='green', linewidth=2)
            ax2 = ax1.twiny()
            line2, = ax2.plot(x, e_pred, label='$\epsilon(T)$', color='red', linewidth=2)

        ax2.set_xlabel('Q$_{inter}$(W/m$^2$)', color="red")
        ax1.tick_params(axis='x', labelcolor="green")
        ax2.tick_params(axis='x', labelcolor="red")
        ax1.grid()
        ax2.grid()
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        combined_handles = handles + handles2
        combined_labels = labels + labels2
        ax1.legend(combined_handles, combined_labels, loc='upper left')  # Adjust the location as needed
        plt.grid()
        ax1.set_xlabel('Q$_{Ag-Supra}$(W/m$^2$)', color="green")
        ax1.set_ylabel('Coefficients [S.U.]')
        plt.subplots_adjust(wspace=0.2)  # Adjust the width space between subplots

    else:
        plt.subplot(1, 2, 2)
        plt.plot(x, b_pred, label='b (pred)', color='orange', linewidth=2)
        #if type == 'enhance':
        #    plt.plot(x, d_pred, label='d (pred)', color='red', linewidth=2)
            
        plt.axvline(P_PDE['T_min'], color='k', dashes=[6, 2])
        plt.axvline(P_PDE['T_max'], color='k', dashes=[6, 2])
        sns.despine()
        plt.legend()
        plt.xlabel('T(K)')
        plt.ylabel('Coefficients [S.U.]')
        plt.title(now_str)

        plt.subplots_adjust(wspace=0.2)  # Adjust the width space between subplots

    if save_model:
        plt.savefig('results/' + now_str + '/Fig4-coefficients_fictifs.png', dpi=300)
def plot_gif_predictions(true,model,DF,P_PDE,now_str,index_dataset):
    _,Nt = true.shape
    x = P_PDE['x']
    time_values = P_PDE['t']

    # Initialize the figure and axis
    fig, ax = plt.subplots()

    # Initialize three lines with the first time step data
    line1, = ax.plot(x, true[:, 0], 'ko', lw=0, label="2-D COMSOL", markersize=4, markerfacecolor='none')  # Black rings only
    line2, = ax.plot(x, model[:, 0], lw=2, linestyle='--', label="PGRML", color='#4d73b1')  # Darker blue dashed line for test2
    line3, = ax.plot(x, DF[:, 0], lw=2, linestyle='--', label="1-D Finite Differences", color='#c24c54')  # Red dashed line for test3

    # Set fixed x-axis limits based on the x-values
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Position along length [m]")
    ax.set_ylabel("Temperature inside HTS [K]")

    # Add a legend to distinguish between the three lines
    ax.legend()

    # Update function for each frame, with adaptive y-axis and time in title
    def update(t):
        # Update line data for test, test2, and test3
        line1.set_data(x, true[:, t])  # Only rings, no line
        line2.set_data(x, model[:, t])  # Dashed line for test2
        line3.set_data(x, DF[:, t])  # Dashed line for test3
        
        # Update the y-axis limits dynamically based on all three datasets
        all_data = np.concatenate((true[:, t], model[:, t], DF[:, t]))  # Combine all datasets
        ax.set_ylim(np.min(all_data), np.max(all_data) + 0.01 * np.max(all_data))
        
        # Update the title to show the current time step and simulation time
        current_time = time_values[t]  # Get the corresponding simulation time
        ax.set_title(f"Rinter = {P_PDE['R_inter'][index_dataset]/1e-10:.2f} uΩcm2,  Time step: {t}, Simulation time: {current_time*1000:.2f} ms")
        
        return line1, line2, line3

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, Nt, 5), interval=100, blit=False)

    anim.save('results/' + now_str + '/Predictions_evolution_ds' + str(index_dataset) + '.gif', writer=PillowWriter(fps=10))
    # Show the animation
    #plt.show()

def plot_gif_prefactors(T_true, T_abc, T_DF,Q_abc,model_ABCnet,index_simu,P_PDE,P_ABCnet,now_str,save_gif):
    true = T_true[:, :, index_simu]
    model = T_abc[:, :, index_simu]
    DF = T_DF[:, :, index_simu]
    Q = Q_abc[:, :, :, index_simu]
    Nx, Nt = true.shape
    x = P_PDE['x']
    time_values = P_PDE['t']

    # Initialize the figure with 3 subplots (one on the left, two on the right, stacked vertically)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])

    # Left subplot (original data)
    ax1 = fig.add_subplot(gs[:, 0])  # spans both rows on the left
    line1, = ax1.plot(x, true[:, 0], 'ko', lw=0, label="2-D COMSOL", markersize=4, markerfacecolor='none')
    line2, = ax1.plot(x, model[:, 0], lw=2, linestyle='--', label="PGRML", color='#4d73b1')
    line3, = ax1.plot(x, DF[:, 0], lw=2, linestyle='--', label="1-D Finite Differences", color='#c24c54')

    ax1.set_xlim(x[0], x[-1])
    ax1.legend()

    # Add xlabel and ylabel for the left subplot
    ax1.set_xlabel("Position along length [m]")
    ax1.set_ylabel("Temperature inside HTS [K]")

    # Right subplot (top: tensor Q)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bound = P_PDE['T0'] * torch.ones((Nt, 1), device=device)
    batch_front = P_PDE['T0'] * torch.ones((Nx, 1))
    batch_zeros = torch.zeros((Nx, 1))

    Q_current = torch.tensor(Q.transpose(1, 0, 2), device=device, dtype=torch.float)
    Q_ag = Q_current[:, :, 0]
    Q_supra = Q_current[:, :, 1]
    Q_inter = Q_current[:, :, 2]
    Q_conv = Q_current[:, :, 3]
    Q_ag_supra = Q_ag + Q_supra + Q_conv

    T_current = torch.tensor(model.T, device=device)
    T_in = T_current[:, 1:-1].flatten().unsqueeze(1)  # size( a*b, 1 )
    T_in_p = T_current[:, 0:-2].flatten().unsqueeze(1)  # T_in_(i-1) previous, spatial
    T_in_n = T_current[:, 2:].flatten().unsqueeze(1)  # T_in_(i+1) next, spatial
    T_in_pp = torch.cat((bound, T_current[:, :-4], bound), dim=1).flatten().unsqueeze(1)  # T_in_(i-2) previous, spatial
    T_in_nn = torch.cat((bound, T_current[:, 4:], bound), dim=1).flatten().unsqueeze(1)

    a_T = model_ABCnet.forward_a(T_in_pp, T_in_p, T_in, T_in_n, T_in_nn).reshape(Nt, Nx - 2)
    b_T = model_ABCnet.forward_b(T_in_pp, T_in_p, T_in, T_in_n, T_in_nn).reshape(Nt, Nx - 2)

    if P_ABCnet['coef_type'] == 'c(Q)':
        Q_inter_middle = Q_inter[:, 1:-1].flatten().unsqueeze(1)
        Q_inter_previous_i = Q_inter[:, 0:-2].flatten().unsqueeze(1)
        Q_inter_next_i = Q_inter[:, 2:].flatten().unsqueeze(1)
        Q_inter_previous_ii = torch.cat((bound, Q_inter[:, :-4], bound), dim=1).flatten().unsqueeze(1)
        Q_inter_next_ii = torch.cat((bound, Q_inter[:, 4:], bound), dim=1).flatten().unsqueeze(1)

        Q_ag_supra_middle = Q_ag_supra[:, 1:-1].flatten().unsqueeze(1)
        Q_ag_supra_previous_i = Q_ag_supra[:, 0:-2].flatten().unsqueeze(1)
        Q_ag_supra_next_i = Q_ag_supra[:, 2:].flatten().unsqueeze(1)
        Q_ag_supra_previous_ii = torch.cat((bound, Q_ag_supra[:, :-4], bound), dim=1).flatten().unsqueeze(1)
        Q_ag_supra_next_ii = torch.cat((bound, Q_ag_supra[:, 4:], bound), dim=1).flatten().unsqueeze(1)

        c_T = model_ABCnet.forward_c2(Q_ag_supra_previous_ii, Q_ag_supra_previous_i, Q_ag_supra_middle, Q_ag_supra_next_i,
                                    Q_ag_supra_next_ii).reshape(Nt, Nx - 2)
        e_T = model_ABCnet.forward_e(Q_inter_previous_ii, Q_inter_previous_i, Q_inter_middle, Q_inter_next_i,
                                    Q_inter_next_ii).reshape(Nt, Nx - 2)
    else:
        c_T = model_ABCnet.forward_c(T_in_pp, T_in_p, T_in, T_in_n, T_in_nn).reshape(Nt, Nx - 2)
        e_T = model_ABCnet.forward_e2(T_in_pp, T_in_p, T_in, T_in_n, T_in_nn).reshape(Nt, Nx - 2)

    # Convert tensors to numpy arrays for plotting
    a_T = a_T.cpu().detach().numpy().T # (Nx-2,Nt)
    b_T = b_T.cpu().detach().numpy().T
    c_T = c_T.cpu().detach().numpy().T

    # Top right subplot
    ax2 = fig.add_subplot(gs[0, 1])  # top right subplot
    line_a, = ax2.plot(x[1:-1], a_T[:, 0], lw=2, linestyle='-', label="a_T", color='orange')
    line_b, = ax2.plot(x[1:-1], b_T[:, 0], lw=2, linestyle='-', label="b_T", color='blue')
    line_c, = ax2.plot(x[1:-1], c_T[:, 0], lw=2, linestyle='-', label="c_T", color='green')

    ax2.set_xlim(x[0], x[-1])
    ax2.legend()

    # Bottom right subplot (normalized data)
    ax3 = fig.add_subplot(gs[1, 1])  # bottom right subplot
    norm_a_T = a_T - np.mean(a_T, axis=1)[:, np.newaxis]
    norm_b_T = b_T - np.mean(b_T, axis=1)[:, np.newaxis]
    norm_c_T = c_T - np.mean(c_T, axis=1)[:, np.newaxis]

    norm_a_T = norm_a_T / (np.max(norm_a_T, axis=1) - np.min(norm_a_T, axis=1))[:, np.newaxis]
    norm_b_T = norm_b_T / (np.max(norm_b_T, axis=1) - np.min(norm_b_T, axis=1))[:, np.newaxis]
    norm_c_T = norm_c_T / (np.max(norm_c_T, axis=1) - np.min(norm_c_T, axis=1))[:, np.newaxis]

    line_a_norm, = ax3.plot(x[1:-1], norm_a_T[:, 0], lw=2, linestyle='-', label="Normalized a_T", color='orange')
    line_b_norm, = ax3.plot(x[1:-1], norm_b_T[:, 0], lw=2, linestyle='-', label="Normalized b_T", color='blue')
    line_c_norm, = ax3.plot(x[1:-1], norm_c_T[:, 0], lw=2, linestyle='-', label="Normalized c_T", color='green')

    ax3.set_xlim(x[0], x[-1])
    ax3.legend()

    # Set initial y-limits for ax2 and ax3
    ax2.set_ylim(np.min((a_T, b_T, c_T)), np.max((a_T, b_T, c_T)))
    ax3.set_ylim(-1, 1)  # Normalized plot limits from 0 to 1

    # Add common xlabel and ylabel for the right subplots
    fig.supxlabel("Position along length [m]", x=0.75)
    #fig.supylabel("U.A.", y=0.5)

    # Animation update function
    def update(t):
        # Update left subplot (ax1)
        line1.set_data(x, true[:, t])
        line2.set_data(x, model[:, t])
        line3.set_data(x, DF[:, t])

        # Update y-limits dynamically for left subplot
        all_data = np.concatenate((true[:, t], model[:, t], DF[:, t]))
        ax1.set_ylim(np.min(all_data), np.max(all_data) + 0.01 * np.max(all_data))

        # Update the title for the left subplot
        current_time = time_values[t]
        ax1.set_title(f"Time step: {t}, Simulation time: {current_time * 1000:.2f} ms")

        # Update top right subplot (ax2)
        line_a.set_data(x[1:-1], a_T[:, t])
        line_b.set_data(x[1:-1], b_T[:, t])
        line_c.set_data(x[1:-1], c_T[:, t])

        # Update bottom right subplot (ax3 - normalized)
        line_a_norm.set_data(x[1:-1], norm_a_T[:, t])
        line_b_norm.set_data(x[1:-1], norm_b_T[:, t])
        line_c_norm.set_data(x[1:-1], norm_c_T[:, t])

        return line1, line2, line3, line_a, line_b, line_c, line_a_norm, line_b_norm, line_c_norm

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, Nt, 40), interval=100, blit=False)

    if save_gif:
        anim.save('results/' + now_str + '/Predictions_evolution_prefactors' + str(index_simu) + '.gif', writer=PillowWriter(fps=10))
    # Show the plot
    #plt.show()

def plot_coefnet(model, P_PDE, now_str, save_model):
    x = np.linspace(P_PDE['T_min'], P_PDE['T_max'], 200)
    x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=True).unsqueeze(1)
    keff_pred    = model.forward_keff(x_tensor).detach()
    rhomCpeff    = model.forward_rhomCpeff(x_tensor).detach()

    plt.figure(figsize=(12, 5))  # Adjust the figsize to your preference

    # Plot for a and c
    plt.subplot(1, 2, 1)
    plt.plot( x, func_keff( x, P_PDE ),'o', label='keff (true)' )
    plt.plot( x, keff_pred.detach(), label='k_eff (pred)' )
    plt.axvline(P_PDE['T_min'], color = 'k', dashes = [6,2])
    plt.axvline(P_PDE['T_max'], color = 'k', dashes = [6,2])
    sns.despine()
    plt.grid()
    plt.ylabel('k_eff')
    plt.xlabel('T(K)')

    # Plot for b
    plt.subplot(1, 2, 2)
    plt.plot( x, func_rhomCpeff( x, P_PDE ),'o', label='rhomCp_eff (true)' )
    plt.plot( x, rhomCpeff.detach(), label='rhomCp_eff (pred)' )
    plt.axvline(P_PDE['T_min'], color='k', dashes=[6, 2])
    plt.axvline(P_PDE['T_max'], color='k', dashes=[6, 2])
    sns.despine()
    plt.legend()
    plt.xlabel('T(K)')
    plt.ylabel('rhom*Cp_eff')
    plt.title(now_str)

    plt.subplots_adjust(wspace=0.2)  # Adjust the width space between subplots
    if save_model:
        plt.savefig('results/' + now_str + '/Fig5-Coefficients.png', dpi=300)

def plot_prefacteur(model, P_PDE, now_str, save_model):
    x = np.linspace(P_PDE['T_min'], P_PDE['T_max'], 200)
    x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=True).unsqueeze(1)
    #p_pred = model.forward_p(x_tensor).detach()
    g_pred = model.forward_g(x_tensor).detach()
    q_pred = model.forward_q(x_tensor).detach()

    plt.figure(figsize=(12, 5))  # Adjust the figsize to your preference

    # Plot for a and c
    plt.subplot(1,2,1)
    plt.plot(x, g_pred, label='g : grad T^(n)', color='orange', linewidth=2)
    plt.axvline(P_PDE['T_min'], color='k', dashes=[6, 2])
    plt.axvline(P_PDE['T_max'], color='k', dashes=[6, 2])
    sns.despine()
    plt.legend()
    plt.xlabel('T(K)')
    plt.ylabel('Coefficients [S.U.]')

    plt.subplot(1,2,2)
    plt.plot(x, q_pred, label='q : Q^(n)', color='green', linewidth=2)
    plt.axvline(P_PDE['T_min'], color='k', dashes=[6, 2])
    plt.axvline(P_PDE['T_max'], color='k', dashes=[6, 2])
    sns.despine()
    plt.legend()
    plt.xlabel('T(K)')
    plt.ylabel('Coefficients [S.U.]')

    plt.subplots_adjust(wspace=0.2)
    if save_model:
        plt.savefig('results/' + now_str + '/Fig4-coefficients_fictifs.png', dpi=300)

def save_data_txt(now_str,DATA_train,time_laps,device,predictors,target,directory,dataset_path,P,P_PDE):
    file = open( "results/" + now_str + "/Parameters" + ".txt", "w")
    file.write( "Train datapoints: %i \n" % (len(DATA_train)) )
    file.write( "Time: %.1f s (%.2f min)\n" % (time_laps, (time_laps)/60 ))
    file.write( "Device: %s\n\n" % (device) )
    file.write( "Predictors: " )
    file.write( str(predictors)+'\n' )
    file.write( "target: " )
    file.write( str(target)+'\n' )
    
    # Dataset:
    file.write( '\n# Training set:\n' )
    path = directory + dataset_path + "train/*.csv"
    for k in [f for f in glob.glob(path)]:
        file.write("%s \n" % ( k.split("/")[-1] ))
    file.write( '# Test set:\n' )
    path = directory + dataset_path + "test/*.csv"
    for k in [f for f in glob.glob(path)]:
        file.write("%s \n" % ( k.split("/")[-1] ))
    file.write( '# Validation set:\n' )
    path = directory + dataset_path + "val/*.csv"
    for k in [f for f in glob.glob(path)]:
        file.write("%s \n" % ( k.split("/")[-1] ))
    
    # Hyperparameters:
    file.write( '\n# Hyperparameters:\n' )
    for k in sorted (P.keys()):
        file.write("%s : %s \n" % (k, P[k]))
    # Parameters on PDE
    file.write( "\n# PDE parameters:\n" )
    for k in sorted (P_PDE.keys()):
        file.write("%s : %s \n" % (k, P_PDE[k]))
    file.close()

""""
def reshape_voisins(matrix, nb_voisins):
    matrice_resultante = []
    for ligne in matrix:
        # Convertir le tensor en une liste Python
        ligne = ligne.tolist()

        # Extraire les sous-matrices à partir de la ligne
        sous_matrice = [ligne[i:i+nb_voisins] for i in range(len(ligne) - nb_voisins + 1)]

        # Étendre la matrice résultante avec la sous-matrice obtenue
        matrice_resultante.extend(sous_matrice)

    # Convertir la liste en un tensor PyTorch
    matrice_resultante = torch.tensor(matrice_resultante)

    return matrice_resultante"""

def reshape_voisins(tensor, nb_voisins):
    # Appliquer une opération matricielle pour extraire les sous-matrices
    matrice_resultante = torch.cat([tensor[:, i:i+nb_voisins] for i in range(tensor.size(1) - nb_voisins + 1)], dim=0)

    return matrice_resultante



###########
# Taken from https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py
import contextlib 
class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True,allow_unused=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)
torch_interp1d = Interp1d.apply
##############

class PINN_CoefNet(nn.Module):
    def __init__(self, P, P_PDE, device, labels):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE  = P_PDE

        # CPU or GPU
        self.device = device
        # Architecture pour les neural nets
        self.net_keff = self.neural_net( P['k_eff_dh'] )
        self.net_rhomCpeff = self.neural_net( P['rhomCp_eff_dh'] )

        # Convolution layer pour faire convolution, + général et peut-être modifier
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 = np.array( P['CONV1_scale'] * np.array(P['CONV1_init']) )
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV1 = self.CONV1.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV1.named_parameters():
            param.requires_grad = not(P['CONV1_frozen']) # Freeze weights
        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array( P['CONV2_scale'] * np.array(P['CONV2_init']) ) #
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV2 = self.CONV2.to(self.device)
        
        # Set requires_grad=False
        for name, param in self.CONV2.named_parameters():
            param.requires_grad = not(P['CONV2_frozen']) # Freeze weights  

        # Display name of tensors who require grad
        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)

    def neural_net(self, layers):
            # General Neural network architecture
        seq = []
        for i in range(len(layers)-2):
            seq.append( ('Linear'+str(i), nn.Linear(layers[i], layers[i+1])) ) # (Nom,structure)
            seq.append( ('Relu'+str(i), nn.ReLU()) )
        seq.append( ('Linear'+str(i+1), nn.Linear(layers[i+1], layers[i+2])) )   
        # Create neural network
        model_seq = torch.nn.Sequential( OrderedDict( seq ) ).to(self.device)
        model_seq.apply(init_weights_JH)
        return model_seq
    
    def scale_T(self, T):
        return ( T - self.P_PDE['T_min']) / (self.P_PDE['T_max'] - self.P_PDE['T_min'])
    
    def inv_scale_T(self, T):
        return  (self.P_PDE['T_max'] - self.P_PDE['T_min']) * T + self.P_PDE['T_min'] 
    
    def forward_keff(self, T): # 1D tensor (flat)
        T_sc   = self.scale_T(T)
        out_sc = self.net_keff( T_sc ) 
        out    = self.P['k_eff_scale'] * self.inv_scale_T(out_sc) 
        return out.flatten()
    
    def forward_dkeffdT(self, k, T): # 1D tensor (flat)
        dummy  = torch.ones( len(T),1 ).flatten().to(self.device) # 
        out = torch.autograd.grad( k, T, grad_outputs=dummy, create_graph=True, only_inputs=True)[0]
        return out
    
    def forward_rhomCpeff(self, T): # 1D tensor (flat)
        T_sc   = self.scale_T(T)
        out_sc = self.net_rhomCpeff( T_sc ) 
        out    = self.P['rhomCp_eff_scale'] * self.inv_scale_T(out_sc) 
        return out.flatten()
    
    def forward(self, T_current, T_previous, Q): # T:(mini-batch, Nx)
        # Add boundary condition 
        bound = self.P_PDE['T0'] * torch.ones((T_current.shape[0], 1)).to(self.device) # (mini-batch,1)
    
                
        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1) / self.P['CONV1_scale']  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1) / self.P['CONV2_scale']  # 
        
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        T_in = T_current[:,1:-1].flatten().unsqueeze(1) # size( a*b, 1 )
        
        # Material properties: (applied element-by-element)
        keff_T      = self.forward_keff( T_in )            # size( a*b, 1 )
        dkeffdT     = self.forward_dkeffdT( keff_T, T_in ) # size( a*b, 1 )
        rhomCpeff_T = self.forward_rhomCpeff( T_in )       # size( a*b, 1 )

        # Bring back original size
        rhomCpeff_T = rhomCpeff_T.reshape(a,b)
        keff_T      = keff_T.reshape(a,b)
        dkeffdT     = dkeffdT.reshape(a,b)
        
        # Temperature-dependent coefficients
        gamma_T = 2 * self.P_PDE['dt'] / ( rhomCpeff_T )
        A_T = 2 * keff_T * self.P_PDE['dt'] / ( rhomCpeff_T * self.P_PDE['dx']**2 )
        B_T = dkeffdT * self.P_PDE['dt'] / (2 * rhomCpeff_T * self.P_PDE['dx']**2 )

        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)
        out =  ( A_T*conv1_T + B_T*conv2_T**2 + 
                    (1-A_T)*T_previous[:,1:-1] + 
                  gamma_T *Q[:,1:-1] 
                ) / (1+A_T) 

        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 
        
        return out
    
    def fit(self, cube_train, cube_val, DATA, epochs):# Inputs: CUBE, DATA est en DataFrame

        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            Seq_train = cubes2sequences_JH(cube_train, steps) # Quand on parle de séquences, on redistribue les cubes en "ensembles", voir notes
            Seq_val   = cubes2sequences_JH(cube_val, steps) # Stock tous les types nécessaires Tn-2,Tn-1,etc.
                    
            # Convert to Torch tensors 
            Seq_train_Tn_2 = torch.tensor( Seq_train['Tn-1(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_train_Tn_1 = torch.tensor( Seq_train['Tn(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_train_Q_1  = torch.tensor( Seq_train['Q(W/m3)'], dtype=torch.float).to(self.device) 
            Seq_train_Tn   = torch.tensor( Seq_train['Tn+1(K)'],   dtype=torch.float).to(self.device)
       
            Seq_val_Tn_2 = torch.tensor( Seq_val['Tn-1(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_val_Tn_1 = torch.tensor( Seq_val['Tn(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_val_Q_1  = torch.tensor( Seq_val['Q(W/m3)'], dtype=torch.float).to(self.device) 
            Seq_val_Tn   = torch.tensor( Seq_val['Tn+1(K)'], dtype=torch.float).to(self.device) 
        
            # Number of samples
            N = Seq_train_Tn.size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(Seq_train_Tn.size()[0])
            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define minibatch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train_Tn_2[indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train_Tn_1[indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train_Q_1[indices]
                batch_Seq_train_Tn   = Seq_train_Tn[indices]
                 
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)
                
                Tn_recur_previous_val = Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)
               
                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]
                for n in range( 0, int(np.floor(steps-1)) ):  
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:] # size: (K,Nx)
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)
                    
                    Q_val         = Seq_val_Q_1[:,n,:] #
                    Tn_true_val   = Seq_val_Tn[:,n,:]  # 
                    
                    # Predictions
                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train) # size: (K, 2*Nx)
                    Tn_predict_val   = self.forward( Tn_recur_current_val,   Tn_recur_previous_val, Q_val)

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                
                # Error  (K*L,Nx) (for one mini-batch) (Nbre mini batch * (Nbre step-1),Nx)
                difference_Tn_train = torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] )
                difference_Tn_val   = torch.stack(Tn_pred_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), self.P_PDE['Nx'] )
            
                # Mean square error for one mini-batch
                loss_train = torch.mean(torch.square( difference_Tn_train )) #Scalaire
                loss_val   = torch.mean(torch.square( difference_Tn_val )) #Scalaire

                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################

                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters               
                loss_train.backward(retain_graph=True)

                # Adaptive learning rate
                learning_rate_thresholds = [1E-4, 1E-5, 1E-6]

                new_lr = self.P['l_r']
                if steps >= 9:
                    new_lr = self.P['l_r']/6#30
                    for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                elif steps >= 6:
                    new_lr = self.P['l_r']/2
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                #if Loss_train_epoch:
                #    for threshold in learning_rate_thresholds:
                #        if Loss_train_epoch[-1] <= threshold:
                #            # Decrease the learning rate based on the crossed threshold
                #            new_lr = self.P['l_r'] / (10 ** (learning_rate_thresholds.index(threshold) + 1))
                #            for param_group in self.optimizer.param_groups:
                #                param_group['lr'] = new_lr

                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()
                #####################################################################################################
                
                # Save loss per batch
                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch
                loss_val_batch.append( loss_val.item() ) 
                
            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display
            
            # Predict material properties:
            self.keff_150        = self.forward_keff( torch.tensor([[150]]).to(self.device))
            self.rhomCpeff_150   = self.forward_rhomCpeff( torch.tensor([[150]]).to(self.device))
 
            print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, keff(150): %.1f (%.1f), rhomCpeff(150): %.2e (%.2e), lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.keff_150, self.P_PDE['k_eff_150'], self.rhomCpeff_150, self.P_PDE['rhomCp_eff_150'],new_lr))

            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current): # Unscaled input
        # Convert to Torch tensors
        T_current  = torch.tensor( T_current, dtype=torch.float, requires_grad=True ).to(self.device)
        T_previous = torch.tensor( T_previous, dtype=torch.float ).to(self.device)
        Q_current  = torch.tensor( Q_current, dtype=torch.float ).to(self.device) 
        
        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous, Q_current)# size:
        Tn_predict.cpu()

        return Tn_predict
    
    def set_device(self,device):
        self.device = device
     

class PINN_PrefactNet(nn.Module):
    def __init__(self, P, P_PDE, device, labels):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE  = P_PDE

        self.device = device 

        # Architecture pour les neural nets
        #self.net_p = self.neural_net( P['p_dh'] )
        #self.net_m = self.neural_net( P['m_dh'] )
        self.net_g = self.neural_net( P['g_dh'] )
        self.net_q = self.neural_net( P['q_dh'] )

        # Convolution layer pour faire convolution, + général et peut-être modifier
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 = np.array( P['CONV1_scale'] * np.array(P['CONV1_init']) )
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV1 = self.CONV1.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV1.named_parameters():
            param.requires_grad = not(P['CONV1_frozen']) # Freeze weights
        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array( P['CONV2_scale'] * np.array(P['CONV2_init']) ) #
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV2 = self.CONV2.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV2.named_parameters():
            param.requires_grad = not(P['CONV2_frozen']) # Freeze weights  

        # Display name of tensors who require grad
        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)

    def neural_net(self, layers):
            # General Neural network architecture
        seq = []
        for i in range(len(layers)-2):
            seq.append( ('Linear'+str(i), nn.Linear(layers[i], layers[i+1])) ) # (Nom,structure)
            seq.append( ('Relu'+str(i), nn.ReLU()) )
        seq.append( ('Linear'+str(i+1), nn.Linear(layers[i+1], layers[i+2])) )   
        # Create neural network
        model_seq = torch.nn.Sequential( OrderedDict( seq ) ).to(self.device)
        model_seq.apply(init_weights_JH)
        return model_seq
    
    def scale_T(self, T):
        return ( T - self.P_PDE['T_min']) / (self.P_PDE['T_max'] - self.P_PDE['T_min'])
    
    def inv_scale_T(self, T):
        return  (self.P_PDE['T_max'] - self.P_PDE['T_min']) * T + self.P_PDE['T_min'] 
    
    def forward_p(self,T):
        T_sc = self.scale_T(T)
        out_sc = self.net_p(T_sc)
        out = self.P['p_scale']*self.inv_scale_T(out_sc)
        return out.flatten()
    
    def forward_m(self,T):
        T_sc = self.scale_T(T)
        out_sc = self.net_m(T_sc)
        out = self.P['m_scale']*self.inv_scale_T(out_sc)
        return out.flatten()
    
    def forward_g(self,T):
        T_sc = self.scale_T(T)
        out_sc = self.net_g(T_sc)
        out = self.P['g_scale']*self.inv_scale_T(out_sc)
        return out.flatten()
    
    def forward_q(self,T):
        T_sc = self.scale_T(T)
        out_sc = self.net_q(T_sc)
        out = self.P['q_scale']*self.inv_scale_T(out_sc)
        return out.flatten()
    
    def func_keff_forward(self,T):
        T = T.detach().cpu().numpy()
        keff = func_keff(T,self.P_PDE)
        keff = torch.tensor(keff,dtype=torch.float).to(self.device)
        return keff
    
    def func_dkeffdT_forward(self,T):
        T = T.detach().cpu().numpy()
        dkdT = func_dkeffdT(T,self.P_PDE)
        dkdT = torch.tensor(dkdT,dtype=torch.float).to(self.device)
        return dkdT
    
    def func_rhomCpeff_forward(self,T):
        T = T.detach().cpu().numpy()
        rhomCp = func_rhomCpeff(T, self.P_PDE)
        rhomCp = torch.tensor(rhomCp,dtype=torch.float).to(self.device)
        return rhomCp
    
    def forward(self, T_current, T_previous, Q): # T:(mini-batch, Nx)
        # Add boundary condition 
        bound = self.P_PDE['T0'] * torch.ones((T_current.shape[0], 1)).to(self.device) # (mini-batch,1)
    
                
        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1) / self.P['CONV1_scale']  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1) / self.P['CONV2_scale']  # 
        
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        T_in = T_current[:,1:-1].flatten().unsqueeze(1) # size( a*b, 1 )
        
        # Prefacteur prediction: (applied element-by-element)
        #p_T         = self.forward_p(T_in).reshape(a,b)
        #m_T         = self.forward_m(T_in).reshape(a,b)
        g_T         = self.forward_g(T_in).reshape(a,b)
        q_T         = self.forward_q(T_in).reshape(a,b)
        
        p_T         = 1
        m_T         = 1
        #g_T         = 1
        #q_T         = 1
        # Temperature-dependent coefficients
        keff_T = self.func_keff_forward(T_in).reshape(a,b)
        dkeffdT = self.func_dkeffdT_forward(T_in).reshape(a,b)
        rhomCpeff_T = self.func_rhomCpeff_forward(T_in).reshape(a,b)

        gamma_T = 2 * self.P_PDE['dt'] / ( rhomCpeff_T )
        A_T = 2 * keff_T * self.P_PDE['dt'] / ( rhomCpeff_T * self.P_PDE['dx']**2 )
        B_T = dkeffdT * self.P_PDE['dt'] / (2 * rhomCpeff_T * self.P_PDE['dx']**2 )

        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)
        out =  ( A_T*conv1_T*m_T + B_T*conv2_T**2*g_T + 
                    (1-A_T)*T_previous[:,1:-1]*p_T + 
                  gamma_T *Q[:,1:-1]*q_T 
                ) / (1+A_T) 

        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 
        
        return out
    
    def fit(self, cube_train, cube_val, DATA, epochs):# Inputs: CUBE, DATA est en DataFrame

        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            Seq_train = cubes2sequences_JH(cube_train, steps) # Quand on parle de séquences, on redistribue les cubes en "ensembles", voir notes
            Seq_val   = cubes2sequences_JH(cube_val, steps) # Stock tous les types nécessaires Tn-2,Tn-1,etc.
                    
            # Convert to Torch tensors 
            Seq_train_Tn_2 = torch.tensor( Seq_train['Tn-1(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_train_Tn_1 = torch.tensor( Seq_train['Tn(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_train_Q_1  = torch.tensor( Seq_train['Q(W/m3)'], dtype=torch.float).to(self.device) # rajouter grad ici
            Seq_train_Tn   = torch.tensor( Seq_train['Tn+1(K)'],   dtype=torch.float).to(self.device) # rajouter grad ici
       
            Seq_val_Tn_2 = torch.tensor( Seq_val['Tn-1(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_val_Tn_1 = torch.tensor( Seq_val['Tn(K)'], dtype=torch.float, requires_grad=True).to(self.device) 
            Seq_val_Q_1  = torch.tensor( Seq_val['Q(W/m3)'], dtype=torch.float).to(self.device) 
            Seq_val_Tn   = torch.tensor( Seq_val['Tn+1(K)'], dtype=torch.float).to(self.device) 
        
            # Number of samples
            N = Seq_train_Tn.size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(Seq_train_Tn.size()[0])
            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define minibatch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train_Tn_2[indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train_Tn_1[indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train_Q_1[indices]
                batch_Seq_train_Tn   = Seq_train_Tn[indices]
                 
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)
                
                Tn_recur_previous_val = Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)
               
                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]
                for n in range( 0, int(np.floor(steps-1)) ):  
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:] # size: (K,Nx)
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)
                    
                    Q_val         = Seq_val_Q_1[:,n,:] #
                    Tn_true_val   = Seq_val_Tn[:,n,:]  # 
                    
                    # Predictions
                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train) # size: (K, 2*Nx)
                    Tn_predict_val   = self.forward( Tn_recur_current_val,   Tn_recur_previous_val, Q_val)

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                
                # Error  (K*L,Nx) (for one mini-batch) (Nbre mini batch * (Nbre step-1),Nx)
                difference_Tn_train = torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] )
                difference_Tn_val   = torch.stack(Tn_pred_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), self.P_PDE['Nx'] )
            
                # Mean square error for one mini-batch
                loss_train = torch.mean(torch.square( difference_Tn_train )) #Scalaire
                loss_val   = torch.mean(torch.square( difference_Tn_val )) #Scalaire

                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################

                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters               
                loss_train.backward(retain_graph=True)

                # Adaptive learning rate
                learning_rate_thresholds = [5E-6,  1E-7, 5E-8]

                new_lr = self.P['l_r']
                if steps >= 10:
                    new_lr = self.P['l_r']/3#30
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                elif steps >= 8:
                    new_lr = self.P['l_r']/2
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                #if Loss_train_epoch:
                #    for threshold in learning_rate_thresholds:
                #        if Loss_train_epoch[-1] <= threshold:
                #            # Decrease the learning rate based on the crossed threshold
                #            new_lr = self.P['l_r'] / (10 ** (learning_rate_thresholds.index(threshold) + 1))
                #            for param_group in self.optimizer.param_groups:
                #                param_group['lr'] = new_lr

                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()
                #####################################################################################################
                
                # Save loss per batch
                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch
                loss_val_batch.append( loss_val.item() ) 
                
            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display
            
            # Predict material properties:
            #self.p_150          = self.forward_p( torch.tensor([[150]]).to(self.device))
            #self.m_150          = self.forward_m( torch.tensor([[150]]).to(self.device))
            self.p_150 = 1
            self.m_150 = 1
            self.g_150          = self.forward_g( torch.tensor([[150]]).to(self.device))
            self.q_150          = self.forward_q( torch.tensor([[150]]).to(self.device))
            #self.q_150          = 1
 
            print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, p(150): %.3f (%.1f), m(150): %.3f (%.1f), g(150): %.3f (%.1f), q(150): %.3f (%.1f), lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.p_150, self.P['p_init'], self.m_150, self.P['m_init'], self.g_150, self.P['g_init'], self.q_150, self.P['q_init'],new_lr))

            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current): # Unscaled input
        # Convert to Torch tensors
        T_current  = torch.tensor( T_current, dtype=torch.float, requires_grad=True ).to(self.device)
        T_previous = torch.tensor( T_previous, dtype=torch.float ).to(self.device) #requires_grad=True
        Q_current  = torch.tensor( Q_current, dtype=torch.float).to(self.device)  # requires_grad=True
        
        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous, Q_current)# size:
        Tn_predict = Tn_predict.cpu()
        return Tn_predict
    
    def set_device(self,device):
        self.device = device
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob, dtype):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features, dtype=dtype)
        self.activation = nn.ReLU()
        #self.activation = nn.GELU() # nn.ELU()
        #self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(out_features, out_features, dtype=dtype)

        # À RETIRER SI PAS 3 DE PROFONDEUR
        #self.linear3 = nn.Linear(out_features, out_features, dtype=dtype)
        #self.batch_norm3 = nn.BatchNorm1d(out_features)

        # Adjust dimensions if in_features != out_features
        self.adjust_dim = (in_features != out_features)
        if self.adjust_dim:
            self.adjust_layer = nn.Linear(in_features, out_features, dtype=dtype)

        self.layer_norm = nn.LayerNorm(out_features)
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        self.batch_norm2 = nn.BatchNorm1d(out_features)

        self.attention = SelfAttention(out_features, num_heads=1)

    def forward2(self, x):
        identity = x

        out = self.layer_norm(x)
        out = self.linear1(out)
        #  # Apply layer normalization after the first linear layer
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        # à retirer si pas feef forward pas 3 de profondeur
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear3(out)
        # Adjust identity if necessary
        if self.adjust_dim:
            identity = self.adjust_layer(identity)

        out += identity  # Residual connection

        return out
        

    def forward(self, x):
        identity = x

        if x.size(0) > 1:  # Apply BatchNorm only if batch size > 1
            out = self.batch_norm1(x)
        else:
            out = x  # Skip BatchNorm for single sample

        out = self.activation(out)
        out = self.linear1(out)
        out = self.dropout(out)

        if out.size(0) > 1:  # Apply BatchNorm only if batch size > 1
            out = self.batch_norm2(out)

        out = self.activation(out)
        out = self.linear2(out)

         # À RETIRER SI PAS 3 DE PROFONDEUR
        #if out.size(0) > 1:  # Apply BatchNorm only if batch size > 1
        #    out = self.batch_norm3(out)
        #out = self.activation(out)
        #out = self.linear3(out)

        if self.adjust_dim:
            identity = self.adjust_layer(identity)

        out = self.attention(out)
        out += identity  # Residual connection
        return out

import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define query, key, and value linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        x = x.unsqueeze(1)
        batch_size, seq_length, embed_dim = x.size()

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_length, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate attention outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output.squeeze(1)
  
class PGRML_ABCNet_4v(nn.Module):

    def __init__(self, P, P_PDE, device):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE  = P_PDE

        # CPU or GPU
        self.device = device
        self.dtype = P["dtype"]

        # Architecture pour les neural nets
        self.net_a = self.neural_net(P['a_dh'],P['dropout_prob'])
        self.net_b = self.neural_net(P['b_dh'],P['dropout_prob'])
        self.net_c = self.neural_net(P['c_dh'],P['dropout_prob'])
        self.net_d = self.neural_net(P['d_dh'],P['dropout_prob'])
        self.net_e = self.neural_net(P['e_dh'],P['dropout_prob'])

        # Convolution layer pour faire convolution, + général et peut-être modifier
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 = np.array( P['CONV1_scale'] * np.array(P['CONV1_init']) )
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV1 = self.CONV1.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV1.named_parameters():
            param.requires_grad = not(P['CONV1_frozen']) # Freeze weights
        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array( P['CONV2_scale'] * np.array(P['CONV2_init']) ) #
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV2 = self.CONV2.to(self.device)

        # Set requires_grad=False
        for name, param in self.CONV2.named_parameters():
            param.requires_grad = not(P['CONV2_frozen']) # Freeze weights  

        # Display name of tensors who require grad
        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)
        # Define optimizer:
        #self.optimizer_a = torch.optim.Adam(self.net_a.parameters(), lr=P['a_lr'], weight_decay=P['lambda_L2']['a'], amsgrad=False)
        #self.optimizer_b = torch.optim.Adam(self.net_b.parameters(), lr=P['b_lr'], weight_decay=P['lambda_L2']['b'], amsgrad=False)
        #self.optimizer_c = torch.optim.Adam(self.net_c.parameters(), lr=P['c_lr'], weight_decay=P['lambda_L2']['c'], amsgrad=False)
        #self.optimizer_e = torch.optim.Adam(self.net_e.parameters(), lr=P['e_lr'], weight_decay=P['lambda_L2']['e'], amsgrad=False)

    def neural_net2(self, layers, dropout_prob):
            # General Neural network architecture
        seq = []
        for i in range(len(layers)-2):
            seq.append( ('Linear'+str(i), nn.Linear(layers[i], layers[i+1],dtype=self.dtype)) ) # (Nom,structure)
            seq.append( ('Dropout' + str(i), nn.Dropout(dropout_prob)) )
            seq.append( ('Relu'+str(i), nn.ReLU()) )
            #seq.append( ('Dropout' + str(i), nn.Dropout(dropout_prob)) )  # Add dropout layer ## Ramener ce commentaire si problème, lui que j'ai changé pour dropout avant RELU 2/8/2024
            #seq.append( ('Sigmoid'+str(i), nn.Sigmoid()) )
            #seq.append( ('Tanh'+str(i), nn.Tanh()) )
            #seq.append( ('LeakyReLU'+str(i), nn.LeakyReLU()) )
        seq.append( ('Linear'+str(i+1), nn.Linear(layers[i+1], layers[i+2],dtype = self.dtype)) )   
        # Create neural network
        model_seq = torch.nn.Sequential( OrderedDict( seq ) ).to(self.device)
        model_seq.apply(init_weights_JH)

        return model_seq
    
    def neural_net(self, layers, dropout_prob):
        # General Neural network architecture with residual connections
        seq = []

        for i in range(len(layers) - 2):
            # Decide whether to use a residual block or a normal linear layer
            in_features = layers[i]
            out_features = layers[i + 1]

            # Using a residual block
            if in_features == out_features:
                seq.append((f'ResidualBlock{i}', ResidualBlock(in_features, out_features, dropout_prob, self.dtype)))
            else:
                # Fallback to standard linear + dropout + ReLU if dimensions don't match for residual block
                seq.append((f'Linear{i}', nn.Linear(in_features, out_features, dtype=self.dtype)))
                #seq.append((f'LayerNorm{i}', nn.LayerNorm(out_features)))
                seq.append((f'Dropout{i}', nn.Dropout(dropout_prob)))
                seq.append((f'ReLU{i}', nn.ReLU()))

        # Adding the final linear layer
        seq.append((f'Linear{len(layers) - 2}', nn.Linear(layers[-2], layers[-1], dtype=self.dtype)))

        # Create the neural network
        model_seq = nn.Sequential(OrderedDict(seq)).to(self.device)

        # Initialize weights with a custom function if needed
        model_seq.apply(init_weights_JH)

        return model_seq

    def scale_T(self, T):
        return ( T - self.P_PDE['T_min']) / (self.P_PDE['T_max'] - self.P_PDE['T_min'])
    
    def inv_scale_T(self, T):
        return  (self.P_PDE['T_max'] - self.P_PDE['T_min']) * T + self.P_PDE['T_min'] 
    
    def scale_Q(self,Q,min,max): # log et apres min-max
        return ( Q - min) / (max - min)
    
    def inv_scale_Q(self,Q_normalize,min,max):
        return (max - min) * Q_normalize + min
    
    def forward_a(self,T_prev2,T_prev, T, T_next,T_next2): # 1D tensor (flat)
        T_input = torch.cat([T_prev2,T_prev, T, T_next,T_next2], dim=1)
        T_sc   = self.scale_T(T_input)
        out_sc = self.net_a( T_sc ) 
        #out    = self.inv_scale_T(out_sc) 
        return out_sc.flatten()
    
    def forward_a2(self,Q_prev2,Q_prev,Q,Q_next,Q_next2): # 1D tensor (flat)
        Q_input = torch.cat([Q_prev2,Q_prev, Q, Q_next,Q_next2], dim=1)
        Q_sc    = self.scale_Q(Q_input,self.P_PDE['Q_min_ag_supra_inter'],self.P_PDE['Q_max_ag_supra_inter'])
        out_sc  = self.net_a(Q_sc)
        return out_sc
    
    def forward_b(self,T_prev2,T_prev, T, T_next,T_next2): # 1D tensor (flat)
        T_input = torch.cat([T_prev2,T_prev, T, T_next,T_next2], dim=1)
        T_sc   = self.scale_T(T_input)
        out_sc = self.net_b( T_sc ) 
        #out    = self.inv_scale_T(out_sc) 
        return out_sc.flatten()
    
    def forward_b2(self,Q_prev2,Q_prev,Q,Q_next,Q_next2): # 1D tensor (flat)
        Q_input = torch.cat([Q_prev2,Q_prev, Q, Q_next,Q_next2], dim=1)
        Q_sc    = self.scale_Q(Q_input,self.P_PDE['Q_min_ag_supra_inter'],self.P_PDE['Q_max_ag_supra_inter'])
        out_sc  = self.net_b(Q_sc)
        return out_sc
    
    def forward_c(self,T_prev2,T_prev, T, T_next,T_next2): # 1D tensor (flat)
        T_input = torch.cat([T_prev2,T_prev, T, T_next,T_next2], dim=1)
        T_sc   = self.scale_T(T_input)
        out_sc = self.net_c( T_sc ) 
        #out    =  self.inv_scale_T(out_sc) 
        return out_sc.flatten()
    
    def forward_c2(self,Q_prev2,Q_prev,Q,Q_next,Q_next2): # 1D tensor (flat)
        Q_input = torch.cat([Q_prev2,Q_prev, Q, Q_next,Q_next2], dim=1)
        Q_sc    = self.scale_Q(Q_input,self.P_PDE['Q_min_ag_supra_inter'],self.P_PDE['Q_max_ag_supra_inter'])
        out_sc  = self.net_c(Q_sc)
        return out_sc
    
    def forward_e(self,Q_prev2,Q_prev,Q,Q_next,Q_next2):
        Q_input = torch.cat([Q_prev2,Q_prev, Q, Q_next,Q_next2], dim=1)
        Q_sc    = self.scale_Q(Q_input,self.P_PDE['Q_min_cool'],self.P_PDE['Q_max_cool'])
        out_sc  = self.net_e(Q_sc)
        return out_sc
    
    def forward_d(self,T_prev2,T_prev, T, T_next,T_next2):
        T_input = torch.cat([T_prev2,T_prev, T, T_next,T_next2], dim=1)
        T_sc   = self.scale_T(T_input)
        out_sc = self.net_d( T_sc ) 
        #out    = self.inv_scale_T(out_sc) 
        return out_sc.flatten()

    
    def func_keff_forward(self,T):
        T = T.detach().cpu().numpy()
        keff = func_keff(T,self.P_PDE)
        keff = torch.tensor(keff,dtype=self.dtype).to(self.device)
        return keff
    
    def func_dkeffdT_forward(self,T):
        T = T.detach().cpu().numpy()
        dkdT = func_dkeffdT(T,self.P_PDE)
        dkdT = torch.tensor(dkdT,dtype=self.dtype).to(self.device)
        return dkdT
    
    def func_rhomCpeff_forward(self,T):
        T = T.detach().cpu().numpy()
        rhomCp = func_rhomCpeff(T, self.P_PDE)
        rhomCp = torch.tensor(rhomCp,dtype=self.dtype).to(self.device)
        return rhomCp
    
    def keff_tensors(self,T):
        k_ag = torch_interp1d(self.P_PDE['k_ag_data'][:,0],self.P_PDE['k_ag_data'][:,1],T)
        k_ybco = torch_interp1d(self.P_PDE['k_ybco_data'][:,0],self.P_PDE['k_ybco_data'][:,1],T)
        k_hast = torch_interp1d(self.P_PDE['k_hast_data'][:,0],self.P_PDE['k_hast_data'][:,1],T)

        keff = self.P_PDE['ha']*k_ag + self.P_PDE['hs']*k_ybco + self.P_PDE['hhast']*k_hast
        return keff
    
    def rhom_Cpeff_tensors(self,T):
        Cp_ag = torch_interp1d(self.P_PDE['Cp_ag_data'][:,0],self.P_PDE['Cp_ag_data'][:,1],T)
        Cp_ybco = torch_interp1d(self.P_PDE['Cp_ybco_data'][:,0],self.P_PDE['Cp_ybco_data'][:,1],T)
        Cp_hast = torch_interp1d(self.P_PDE['Cp_hast_data'][:,0],self.P_PDE['Cp_hast_data'][:,1],T)

        Cpeff = self.P_PDE['ha']*self.P_PDE['rhom_a']*Cp_ag + self.P_PDE['hs']*self.P_PDE['rhom_s']*Cp_ybco + self.P_PDE['hhast']*self.P_PDE['rhom_hast']*Cp_hast

        return Cpeff
    

    def dkdT_tensors(self,T):
        dkdT_ag = derivative_spline_torch(T,self.P_PDE['k_ag_data'])
        dkdT_ybco = derivative_spline_torch(T,self.P_PDE['k_ybco_data'])
        dkdT_hast = derivative_spline_torch(T,self.P_PDE['k_hast_data'])

        dkdT_eff = self.P_PDE['ha']*dkdT_ag + self.P_PDE['hs']*dkdT_ybco + self.P_PDE['hhast']*dkdT_hast
        return dkdT_eff

    def forward(self, T_current, T_previous, Q): # inputs:(mini-batch, Nx)
        
        #pretime0 = time.time() 
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        
        # Add boundary condition 
        bound = self.P_PDE['T0'] * torch.ones((T_current.shape[0], 1),device=self.device) # (mini-batch,1)
        batch_front = bound*torch.ones((a, 1), dtype=self.dtype,device=self.device)
        batch_zeros = torch.zeros((a, 1), dtype=self.dtype,device=self.device)
        T_current_enhance =  torch.cat([batch_front, T_current, batch_front], dim=1)

        Q_ag = Q[:,:,0]
        Q_supra = Q[:,:,1]
        Q_inter = Q[:,:,2]
        #Q_conv = Q[:,:,3]
        Q_conv = -2*self.P_PDE['h_conv']*(T_current-self.P_PDE['T0']) # Pas de 2x car e_T + d_T = 2

        Q_ag_supra_inter = Q_ag + Q_supra + Q_inter 
        Q_ag_supra_inter_enhance =  torch.cat([batch_zeros, Q_ag_supra_inter, batch_zeros], dim=1)
        Q_conv_enhance = torch.cat([batch_zeros, Q_conv, batch_zeros], dim=1)

        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 

        
        # Shape size

        T_in = T_current[:,1:-1].flatten().unsqueeze(1)     # size( a*b, 1 )
        T_in_p = T_current[:,0:-2].flatten().unsqueeze(1)   # T_in_(i-1) previous, spatial
        T_in_n = T_current[:,2:].flatten().unsqueeze(1)     # T_in_(i+1) next, spatial
        T_in_pp = T_current_enhance[:,0:-4].flatten().unsqueeze(1) # T_in_(i-2) previous, spatial
        T_in_nn = T_current_enhance[:,4:].flatten().unsqueeze(1) # T_in_(i+2) next, spatial



        if self.P['coef_type'] == 'c(Q)':
            Q_conv_middle = Q_conv[:,1:-1].flatten().unsqueeze(1)
            Q_conv_previous_i = Q_conv[:,0:-2].flatten().unsqueeze(1)
            Q_conv_next_i = Q_conv[:,2:].flatten().unsqueeze(1)
            Q_conv_previous_ii = Q_conv_enhance[:,0:-4].flatten().unsqueeze(1)
            Q_conv_next_ii = Q_conv_enhance[:,4:].flatten().unsqueeze(1)

            Q_ag_supra_inter_middle = Q_ag_supra_inter[:,1:-1].flatten().unsqueeze(1)
            Q_ag_supra_inter_previous_i = Q_ag_supra_inter[:,0:-2].flatten().unsqueeze(1)
            Q_ag_supra_inter_next_i = Q_ag_supra_inter[:,2:].flatten().unsqueeze(1)
            Q_ag_supra_inter_previous_ii = Q_ag_supra_inter_enhance[:,0:-4].flatten().unsqueeze(1)
            Q_ag_supra_inter_next_ii = Q_ag_supra_inter_enhance[:,4:].flatten().unsqueeze(1)
            
            c_T      = self.forward_c2(Q_ag_supra_inter_previous_ii, Q_ag_supra_inter_previous_i, Q_ag_supra_inter_middle, Q_ag_supra_inter_next_i,Q_ag_supra_inter_next_ii).reshape(a,b)
            e_T      = self.forward_e(Q_conv_previous_ii, Q_conv_previous_i, Q_conv_middle, Q_conv_next_i,Q_conv_next_ii).reshape(a,b)
        else:
            c_T      = self.forward_c(T_in_pp, T_in_p, T_in, T_in_n,T_in_nn).reshape(a,b)
            e_T      = self.forward_e2(T_in_pp, T_in_p, T_in, T_in_n,T_in_nn).reshape(a,b)


        a_T      = self.forward_a(T_in_pp, T_in_p, T_in, T_in_n,T_in_nn ).reshape(a,b)      
        b_T      = self.forward_b(T_in_pp, T_in_p, T_in, T_in_n,T_in_nn).reshape(a,b)
        d_T      = self.forward_d(T_in_pp, T_in_p, T_in, T_in_n,T_in_nn).reshape(a,b) 
        #a_T      = self.forward_a2(Q_ag_supra_inter_previous_ii, Q_ag_supra_inter_previous_i, Q_ag_supra_inter_middle, Q_ag_supra_inter_next_i,Q_ag_supra_inter_next_ii).reshape(a,b)
        #b_T      = self.forward_b2(Q_ag_supra_inter_previous_ii, Q_ag_supra_inter_previous_i, Q_ag_supra__intermiddle, Q_ag_supra_next_i,Q_ag_supra_next_ii).reshape(a,b)

        #pretime1 = time.time()
        #self.P_PDE['pretime'] += pretime1-pretime0 

        #nn_forward0 = time.time()
        # Coefficients(T): (applied element-by-element)

        #nn_forward1 = time.time()
        #self.P_PDE['nn_forward'] += nn_forward1-nn_forward0      
        #a_T = 1
        #b_T = 1
        #c_T = 1
        #e_T = 1
        #d_T = 1
        # Temperature-dependent coefficients
        #coef0 = time.time()
        #k0 = time.time()
        keff_T = self.keff_tensors(T_in).reshape(a,b)
        #k1 = time.time()
        #self.P_PDE['k'] += k1 - k0
        #keff_T2 = self.func_keff_forward(T_in).reshape(a,b)
        #dkeffdT = self.func_dkeffdT_forward(T_in).reshape(a,b)
        #dk0 = time.time()
        dkeffdT = self.dkdT_tensors(T_in).reshape(a,b)
        #dk1 = time.time()
        #self.P_PDE['dk'] += dk1 - dk0
        #cp0 = time.time()
        #rhomCpeff_T = self.func_rhomCpeff_forward(T_in).reshape(a,b) 
        rhomCpeff_T = self.rhom_Cpeff_tensors(T_in).reshape(a,b)
        #cp1 = time.time()  
        #self.P_PDE['cp'] += cp1 - cp0 
        #coef1 = time.time()
        #self.P_PDE['coef'] += coef1 - coef0

        #DF0 = time.time()
        # Temperature-dependent coefficients
        gamma_T = 2 * self.P_PDE['dt'] / ( rhomCpeff_T )
        A_T = 2 * keff_T * self.P_PDE['dt'] / ( rhomCpeff_T * self.P_PDE['dx']**2 )*a_T
        B_T = dkeffdT * self.P_PDE['dt'] / (2 * rhomCpeff_T * self.P_PDE['dx']**2 )*b_T

        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)
        out =  ( A_T*conv1_T + B_T*conv2_T**2 + (1-A_T)*T_previous[:,1:-1] + 
                  gamma_T *(c_T*Q_ag_supra_inter[:,1:-1]+ d_T*Q_conv[:,1:-1]/2+e_T*Q_conv[:,1:-1]/2)) / (1+A_T)
        
        #out =  ( A_T*conv1_T + B_T*conv2_T**2 + (1-A_T)*T_previous[:,1:-1] + 
        #          gamma_T *(Q_ag_supra[:,1:-1]+Q_conv[:,1:-1]+Q_inter[:,1:-1])) / (1+A_T)

        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 
        #DF1 = time.time()
        #self.P_PDE['DF'] += DF1-DF0
        return out
    
    def fit(self, DATA_train, DATA_val, epochs):# Inputs: CUBE, DATA est en DataFrame

        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        #new_lr = self.P['l_r']
        
        if self.P['steps_max'] == 2: # Steps then won't change, only need to initiate tensor a single time
            Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
            Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)


        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            if self.P['steps_max'] > 2:
                Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
                Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)
        
            # Number of samples
            N = Seq_train['Tn+1(K)'].size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(N)
            #batch_count=0
            #forward_time_total = 0
            #forward_time_total1 = 0
            #loss_time_total = 0
            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define batch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train['Q(W/m2)'][indices]
                batch_Seq_train_Tn   = Seq_train['Tn+1(K)'][indices]
                 
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)
                
                #Tn_recur_previous_val = Seq_val['Tn-1(K)'][:,0,:] # (scaled) size: (K,Nx)
                #Tn_recur_current_val  = Seq_val['Tn(K)'][:,0,:] # (scaled) size: (K,Nx)
               
                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list = [],[]#, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]

                #forward_start_time = time.time()
                for n in range( 0, int(np.floor(steps-1)) ):  
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)
                    
                    #Q_val         = Seq_val['Q(W/m2)'][:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    #Tn_true_val   = Seq_val['Tn+1(K)'][:,n,:]  # 
                    
                    #forward_start_time1 = time.time()
                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train) # size: (K, 2*Nx)
                    #Tn_predict_val   = self.forward( Tn_recur_current_val,   Tn_recur_previous_val, Q_val)
                    #forward_end_time1 = time.time()
                    #forward_time_total1 += forward_end_time1 - forward_start_time1

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    #Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    #Tn_true_val_list.append( Tn_true_val )
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                    #Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    #Tn_recur_current_val  = Tn_predict_val.clone().float()

                #forward_end_time = time.time()  # End timer for forward pass
                #forward_time_total += forward_end_time - forward_start_time
                
                difference_Tn_train = (torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                #difference_Tn_val   = (torch.stack(Tn_pred_val_list,dim=1).reshape( len(Seq_val['Tn(K)'])*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_val_list,dim=1).reshape( len(Seq_val['Tn(K)'])*(steps-1), self.P_PDE['Nx'] ))

                # Mean square error for one mini-batch
                #loss_train = torch.mean(torch.square( difference_Tn_train ))
                #loss_start_time = time.time()
                loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1,p=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #loss_val   = torch.mean(torch.square( difference_Tn_val ))


                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################

                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters     
                #backward_start_time = time.time()          
                #loss_train.backward(retain_graph=True)
                loss_train.backward()

                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()
                
                #####################################################################################################
                #backward_end_time = time.time()
                #backward_time_total += backward_end_time - backward_start_time

                # Save loss per batch
                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch
                #loss_val_batch.append( loss_val.item() ) 
                #batch_count += 1

            # For loop for validation
            M = Seq_val['Tn+1(K)'].size()[0]
            permutation = torch.randperm(M)
            for j in range(0,M,K):
                indices = permutation[j:j+K]
                batch_Seq_val_Tn_2 = Seq_val['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_val_Tn_1 = Seq_val['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_val_Q_1  = Seq_val['Q(W/m2)'][indices]
                batch_Seq_val_Tn   = Seq_val['Tn+1(K)'][indices]
                 
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_val = batch_Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = batch_Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                Tn_pred_val_list, Tn_true_val_list = [],[]
                for n in range( 0, int(np.floor(steps-1)) ):
                    Q_val       = batch_Seq_val_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_val = batch_Seq_val_Tn[:,n,:]
                    Tn_predict_val = self.forward( Tn_recur_current_val, Tn_recur_previous_val, Q_val)
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                difference_Tn_val   = (torch.stack(Tn_pred_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx']) - torch.stack(Tn_true_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                loss_val   = torch.mean(torch.square( difference_Tn_val ))
                loss_val_batch.append( loss_val.item() )     
            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display
            #loss_end_time = time.time()
            #loss_time_total += loss_end_time - loss_start_time
            #print(f"Epoch {epo+1}: Average total forward pass time: {forward_time_total / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average forward pass time: {forward_time_total1 / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average rest of forward pass time: {(forward_time_total-forward_time_total1) / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average loss pass: {loss_time_total / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average NN forward pass time: {self.P_PDE['nn_forward'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average Coeff pass time: {self.P_PDE['coef'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average k pass time: {self.P_PDE['k'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average dk pass time: {self.P_PDE['dk'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average cp pass time: {self.P_PDE['cp'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average DF pass time: {self.P_PDE['DF'] / batch_count:.4f}s")
            #print(f"Epoch {epo+1}: Average backward pass time: {backward_time_total / batch_count:.4f}s")
            ##############################################################################
            # Loss prévue sur t= 0 à Nt
        
            ##############################################################################

            # Predict material properties:
            T_evaluation =  torch.tensor([[150]]).to(self.device)
            Q_evaluation =  torch.tensor([[self.P_PDE['Q_max_cool']/3]],device=self.device)
            Q_eval_supra = torch.tensor([[self.P_PDE['Q_max_ag_supra_inter']/3]],device=self.device)

            self.a_150        = self.forward_a( T_evaluation, T_evaluation,T_evaluation,T_evaluation,T_evaluation)
            self.b_150        = self.forward_b( T_evaluation,T_evaluation,T_evaluation,T_evaluation,T_evaluation)
            self.d_150        = self.forward_d( T_evaluation,T_evaluation,T_evaluation,T_evaluation,T_evaluation)
            if self.P['coef_type' ] == "c(T)":
                self.c_150        = self.forward_c( T_evaluation,T_evaluation,T_evaluation,T_evaluation,T_evaluation)
                self.e_150        = self.forward_e2(T_evaluation,T_evaluation,T_evaluation,T_evaluation,T_evaluation)
                print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, a(150): %.3f (1), b(150): %.3f (1), c(150): %.3f (1), e(150): %.3f (1) lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.a_150, self.b_150,self.c_150,self.e_150,self.P['l_r']))
            else:
                self.c_150        = self.forward_c2(Q_eval_supra,Q_eval_supra,Q_eval_supra,Q_eval_supra, Q_eval_supra)
                self.e_150        = self.forward_e(Q_evaluation,Q_eval_supra,Q_eval_supra, Q_evaluation, Q_evaluation)
                print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, a(150): %.3f (1), b(150): %.3f (1), c(Q_agc/3): %.3f (1), d(150): %.3f (1), e(Qinter/3)): %.3f (1) lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.a_150, self.b_150,self.c_150,self.d_150,self.e_150,self.P['l_r']))
 
            
            #print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, Recursive validation loss: %.3f, a(150): %.3f (1), b(150): %.3f (1), c(Q): %.3f (1), e(Q): %.3f (1) lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1],loss_val_total, self.a_150, self.b_150,self.c_150,self.e_150,new_lr))
            #print("Expected Nt Loss: %.2e" % loss_val_total)

            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current,P_PDE): # Unscaled input

        self.P_PDE['dt'] = P_PDE['dt']
        self.P_PDE['dx'] = P_PDE['dx']
        # Convert to Torch tensors
        T_current  = torch.tensor( T_current, dtype=torch.float ).to(self.device)
        T_previous = torch.tensor( T_previous, dtype=torch.float ).to(self.device)
        Q_current  = torch.tensor( Q_current, dtype=torch.float ).to(self.device)  

        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous, Q_current)
        #Tn_predict, q_out = self.forward( T_current, T_previous, Q_current)# size:

        return Tn_predict#,q_out
    
    def set_device(self,device):
        self.device = device


class PGRML_master(nn.Module):

    def __init__(self, P, P_PDE, device):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE  = P_PDE

        # Forward paramters
        self.dx = P_PDE['dx']
        self.dt = P_PDE['dt']
        self.T0 = P_PDE['T0']
        self.h_conv = P_PDE['h_conv']
        self.Tmin = P_PDE['T_min']
        self.Tmax = P_PDE['T_max']
        self.Q_min_cool = P_PDE['Q_min_cool']
        self.Q_max_cool = P_PDE['Q_max_cool']
        self.Q_min_inter = P_PDE['Q_min_inter']
        self.Q_max_inter = P_PDE['Q_max_inter']
        self.Q_min_ag_supra = P_PDE['Q_min_ag_supra']
        self.Q_max_ag_supra = P_PDE['Q_max_ag_supra']
        #self.Q_min_ag = P_PDE['Q_min_ag']
        #self.Q_max_ag = P_PDE['Q_max_ag']
        #self.Q_min_supra = P_PDE['Q_min_supra']
        #self.Q_max_supra = P_PDE['Q_max_supra']

        self.k_ag_data = P_PDE['k_ag_data']
        self.k_ybco_data = P_PDE['k_ybco_data']
        self.k_hast_data = P_PDE['k_hast_data']

        self.Cp_ag_data = P_PDE['Cp_ag_data']
        self.Cp_ybco_data = P_PDE['Cp_ybco_data']
        self.Cp_hast_data = P_PDE['Cp_hast_data']

        self.rhom_a = P_PDE['rhom_a']
        self.rhom_s = P_PDE['rhom_s']
        self.rhom_hast = P_PDE['rhom_hast']

        self.ha = P_PDE['ha']
        self.hs = P_PDE['hs']
        self.hhast = P_PDE['hhast']
        self.kernel_size = P['kernel_size']

        # CPU or GPU
        self.device = device
        self.dtype = P["dtype"]

        # Architecture pour les neural nets
        self.net_master = self.neural_net(P['master_dh'],P['dropout_prob'])

        # Convolution layer pour faire convolution, + général et peut-être modifier
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 =  np.array(P['CONV1_init'])
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV1 = self.CONV1.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV1.named_parameters():
            param.requires_grad = not(P['CONV1_frozen']) # Freeze weights
        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array(P['CONV2_init']) 
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV2 = self.CONV2.to(self.device)

        # Set requires_grad=False
        for name, param in self.CONV2.named_parameters():
            param.requires_grad = not(P['CONV2_frozen']) # Freeze weights  

        # Display name of tensors who require grad
        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)

        
        # Define Schedule for the learning rate
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=P['lr_gamma'])
    
    def neural_net(self, layers, dropout_prob):
        # General Neural network architecture with residual connections
        seq = []

        for i in range(len(layers) - 2):
            # Decide whether to use a residual block or a normal linear layer
            in_features = layers[i]
            out_features = layers[i + 1]

            # Using a residual block
            #resblock = False
            if in_features == out_features:
            #if resblock:
                seq.append((f'ResidualBlock{i}', ResidualBlock(in_features, out_features, dropout_prob, self.dtype)))
            else:
                # Fallback to standard linear + dropout + ReLU if dimensions don't match for residual block
                seq.append((f'Linear{i}', nn.Linear(in_features, out_features, dtype=self.dtype)))
                #seq.append((f'LayerNorm{i}', nn.LayerNorm(out_features)))  # Layer Normalization
                seq.append((f'Dropout{i}', nn.Dropout(dropout_prob)))
                seq.append((f'ReLU{i}', nn.ReLU()))
            if i == 0:
                seq.append((f'SelfAttention{i}', SelfAttention(out_features, num_heads=1)))
        # Adding the final linear layer
        seq.append((f'Linear{len(layers) - 2}', nn.Linear(layers[-2], layers[-1], dtype=self.dtype)))

        # Create the neural network
        model_seq = nn.Sequential(OrderedDict(seq)).to(self.device)

        # Initialize weights with a custom function if needed
        model_seq.apply(init_weights_JH)

        return model_seq

    def linear_scaler(self,Q,min,max): # log et apres min-max
        return ( Q - min) / (max - min)
    
    def inv_linear_scaler(self,Q_normalize,min,max):
        return (max - min) * Q_normalize + min
    
    def forward_master(self,inputs): # 1D tensor (flat)

        a_T, b_T, c_T, d_T, e_T,f_T = self.net_master( inputs).T
        inputs = inputs.squeeze(1)
        #a_T, b_T, c_T, d_T, e_T,f_T = inputs.T
        #a_T, b_T, c_T, d_T, e_T,f_T = self.net_master( inputs).T
        return a_T, b_T, c_T, d_T, e_T,f_T

    def keff_tensors(self,T):
        k_ag = torch_interp1d(self.k_ag_data[:,0],self.k_ag_data[:,1],T)
        k_ybco = torch_interp1d(self.k_ybco_data[:,0],self.k_ybco_data[:,1],T)
        k_hast = torch_interp1d(self.k_hast_data[:,0],self.k_hast_data[:,1],T)

        keff = self.ha*k_ag + self.hs*k_ybco + self.hhast*k_hast
        return keff
    
    def rhom_Cpeff_tensors(self,T):
        Cp_ag = torch_interp1d(self.Cp_ag_data[:,0],self.Cp_ag_data[:,1],T)
        Cp_ybco = torch_interp1d(self.Cp_ybco_data[:,0],self.Cp_ybco_data[:,1],T)
        Cp_hast = torch_interp1d(self.Cp_hast_data[:,0],self.Cp_hast_data[:,1],T)

        Cpeff = self.ha*self.rhom_a*Cp_ag + self.hs*self.rhom_s*Cp_ybco + self.hhast*self.rhom_hast*Cp_hast

        return Cpeff
    

    def dkdT_tensors(self,T):
        dkdT_ag = derivative_spline_torch(T,self.k_ag_data)
        dkdT_ybco = derivative_spline_torch(T,self.k_ybco_data)
        dkdT_hast = derivative_spline_torch(T,self.k_hast_data)

        dkdT_eff = self.ha*dkdT_ag + self.hs*dkdT_ybco + self.hhast*dkdT_hast
        return dkdT_eff

    def forward(self, T_current:torch.tensor, T_previous:torch.tensor, Q:torch.tensor): # inputs:(mini-batch, Nx)
        
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        
        # Add boundary condition 
        bound = self.T0 * torch.ones((T_current.shape[0], 1),device=self.device) # (mini-batch,1)
        bound_kernelT = self.T0* torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device) # need to have a odd kernel_size
        

        Q_ag = Q[:,:,0]
        Q_supra = Q[:,:,1]
        Q_inter = Q[:,:,2]
        Q_conv = -2*self.h_conv*(T_current-self.T0) # e_T 1 et d_T 1

        Q_ag_supra = Q_ag + Q_supra #+ Q_inter 

        # Add boundary for heat & geometric parameters
        bound_kernelQ_asi = Q_ag_supra[:,0].unsqueeze(1)*torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device)
        #bound_kernelQ_ag = Q_ag[:,0].unsqueeze(1)*torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device)
        #bound_kernelQ_supra = Q_supra[:,0].unsqueeze(1)*torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device)
        bound_kernelQ_inter = Q_inter[:,0].unsqueeze(1)*torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device)
        bound_kernelQ_conv = Q_conv[:,0].unsqueeze(1)*torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device)
        #Rinter_inputs = geo_params.unsqueeze(1).repeat(1,b).flatten().unsqueeze(1)

        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 

        
        # Neural Networks inputs as [Exemples,'kernel_size']
        T_current_inputs    = torch.cat([bound_kernelT, T_current[:,1:-1], bound_kernelT], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        T_previous_inputs   = torch.cat([bound_kernelT, T_previous[:,1:-1], bound_kernelT], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        Q_ag_supra_inputs   = torch.cat([bound_kernelQ_asi, Q_ag_supra[:,1:-1], bound_kernelQ_asi], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        #Q_ag_inputs   = torch.cat([bound_kernelQ_ag, Q_ag[:,1:-1], bound_kernelQ_ag], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        #Q_supra_inputs   = torch.cat([bound_kernelQ_supra, Q_supra[:,1:-1], bound_kernelQ_supra], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        Q_inter_inputs      = torch.cat([bound_kernelQ_inter, Q_inter[:,1:-1], bound_kernelQ_inter], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
        Q_conv_inputs       = torch.cat([bound_kernelQ_conv, Q_conv[:,1:-1], bound_kernelQ_conv], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)

        T_current_inputs    = self.linear_scaler(T_current_inputs, self.Tmin, self.Tmax)
        T_previous_inputs   = self.linear_scaler(T_previous_inputs, self.Tmin, self.Tmax)
        Q_conv_inputs       = self.linear_scaler(Q_conv_inputs,self.Q_min_cool,self.Q_max_cool)
        Q_ag_supra_inputs   = self.linear_scaler(Q_ag_supra_inputs, self.Q_min_ag_supra, self.Q_max_ag_supra)
        #Q_ag_inputs        = self.linear_scaler(Q_ag_inputs, self.Q_min_ag, self.Q_max_ag)
        #Q_supra_inputs      = self.linear_scaler(Q_supra_inputs, self.Q_min_supra, self.Q_max_supra)
        Q_inter_inputs      = self.linear_scaler(Q_inter_inputs, self.Q_min_inter, self.Q_max_inter)
        

        inputs = torch.cat((T_current_inputs,T_previous_inputs,Q_conv_inputs,Q_ag_supra_inputs,Q_inter_inputs), dim=1)

        a_T, b_T, c_T, d_T, e_T,f_T = self.forward_master(inputs)

            
        a_T = a_T.reshape(a,b)
        b_T = b_T.reshape(a,b)
        c_T = c_T.reshape(a,b)
        d_T = d_T.reshape(a,b)
        e_T = e_T.reshape(a,b)
        f_T = f_T.reshape(a,b)
        #g_T = g_T.reshape(a,b)
        #a_T = a_T -a_T  + 1     
        #b_T = b_T -b_T  + 1   
        #c_T = c_T -c_T  + 1   
        #d_T = d_T -d_T  + 1   
        #e_T = e_T -e_T  + 1  

        # Temperature-dependent coefficients
        T_current_contiguous = T_current.contiguous()
        keff_T = self.keff_tensors(T_current_contiguous)[:,1:-1]#.reshape(a,b)
        dkeffdT = self.dkdT_tensors(T_current_contiguous)[:,1:-1]#.reshape(a,b)
        rhomCpeff_T = self.rhom_Cpeff_tensors(T_current_contiguous)[:,1:-1]#.reshape(a,b)

        # Temperature-dependent coefficients
        gamma_T = 2 * self.dt / ( rhomCpeff_T )
        A_T = 2 * keff_T * self.dt / ( rhomCpeff_T * self.dx**2 )*a_T
        B_T = dkeffdT * self.dt / (2 * rhomCpeff_T * self.dx**2 )*b_T

        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)
        #out = ( A_T*conv1_T + B_T*conv2_T**2 + (1-A_T)*T_previous[:,1:-1] + 
        #                            gamma_T*(c_T*Q_ag_supra[:,1:-1]+ d_T*Q_inter[:,1:-1] + e_T*Q_conv[:,1:-1]/2 + f_T*Q_conv[:,1:-1]/2 )) / (1+A_T)
        out = ( A_T*conv1_T + B_T*conv2_T**2 + (1-A_T)*T_previous[:,1:-1] + 
                                    gamma_T*(c_T*Q_ag_supra[:,1:-1]+ d_T*Q_inter[:,1:-1] + e_T*Q_conv[:,1:-1]/2 + f_T*Q_conv[:,1:-1]/2 )) / (1+A_T)
        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 

        return out
    
    def fit(self, DATA_train, DATA_val, epochs):# Inputs: CUBE, DATA est en DataFrame

        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        
        if self.P['steps_max'] == 2: # Steps then won't change, only need to initiate tensor a single time
            Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
            Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)

        #R_inters_normalized = (self.P_PDE['R_inter']-self.P_PDE['R_inter'].min())/(self.P_PDE['R_inter'].max()-self.P_PDE['R_inter'].min())
        #R_inters_normalized = torch.tensor(R_inters_normalized,device=self.device,dtype=self.dtype)

        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            if self.P['steps_max'] > 2:
                Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
                Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)
        
            # Number of samples
            N = Seq_train['Tn+1(K)'].size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(N)

            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define batch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train['Q(W/m2)'][indices]
                batch_Seq_train_Tn   = Seq_train['Tn+1(K)'][indices]
                batch_Seq_train_id   = Seq_train['simu_ids'][indices]
                
                #geo_params = R_inters_normalized[batch_Seq_train_id-1]

                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list = [],[]#, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]

                #forward_start_time = time.time()
                for n in range( 0, int(np.floor(steps-1)) ):  
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)

                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train) # size: (K, 2*Nx)

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                
                difference_Tn_train = (torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                
                # Mean square error for one mini-batch
                #loss_train = torch.mean(torch.square( difference_Tn_train ))
                #loss_start_time = time.time()
                loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1,p=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #loss_val   = torch.mean(torch.square( difference_Tn_val ))


                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################

                
                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters     
                loss_train.backward()
                
                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()

                #####################################################################################################

                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch


            # Update the scheduler learning  rate
            
            current_lr = self.optimizer.param_groups[0]['lr']
            #if self.P['scheduler']:
                #self.scheduler.step()

            # For loop for validation
            M = Seq_val['Tn+1(K)'].size()[0]
            permutation = torch.randperm(M)
            for j in range(0,M,K):
                indices = permutation[j:j+K]
                batch_Seq_val_Tn_2 = Seq_val['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_val_Tn_1 = Seq_val['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_val_Q_1  = Seq_val['Q(W/m2)'][indices]
                batch_Seq_val_Tn   = Seq_val['Tn+1(K)'][indices]
                batch_Seq_val_id   = Seq_val['simu_ids'][indices]

                # Initialize state at a starting point of a sequence
                Tn_recur_previous_val = batch_Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = batch_Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                Tn_pred_val_list, Tn_true_val_list = [],[]
                for n in range( 0, int(np.floor(steps-1)) ):
                    Q_val       = batch_Seq_val_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_val = batch_Seq_val_Tn[:,n,:]
                    Tn_predict_val = self.forward( Tn_recur_current_val, Tn_recur_previous_val, Q_val)
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                difference_Tn_val   = (torch.stack(Tn_pred_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx']) - torch.stack(Tn_true_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                loss_val   = torch.mean(torch.square( difference_Tn_val ))
                loss_val_batch.append( loss_val.item() )    

            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display

            ##############################################################################

            # Predict material properties:
            dim_inputs_T = int(self.kernel_size*2)
            T_evaluation =  self.Tmax/3*torch.ones(1,dim_inputs_T ).to(self.device)
            T_evaluation[:,5:] = 0.9*T_evaluation[:,5:]
            Qas          =  self.Q_max_ag_supra/3*torch.ones(1,self.kernel_size ).to(self.device)
            #Qa          =  self.P_PDE['Q_max_ag']/3*torch.ones(1,self.P['kernel_size'] ).to(self.device)
            #Qs          =  self.P_PDE['Q_max_supra']/3*torch.ones(1,self.P['kernel_size'] ).to(self.device)
            Qinter       =  self.Q_max_inter/3*torch.ones(1,self.kernel_size  ).to(self.device)
            Qcool        =  self.Q_min_cool/3*torch.ones(1,self.kernel_size ).to(self.device)
            T_evaluation = self.linear_scaler(T_evaluation, self.Tmin, self.Tmax)
            Qas          = self.linear_scaler(Qas, self.Q_min_ag_supra, self.Q_max_ag_supra)
            #Qa          = self.scale_Q(Qa,self.P_PDE['Q_min_ag'],self.P_PDE['Q_max_ag'])
            #Qs          = self.scale_Q(Qs,self.P_PDE['Q_min_supra'],self.P_PDE['Q_max_supra'])
            Qinter       = self.linear_scaler(Qinter, self.Q_min_inter, self.Q_max_inter)
            Qcool        = self.linear_scaler(Qcool,self.Q_min_cool,self.Q_max_cool)
            #inputs_val   = torch.cat((T_evaluation,Qas,Qinter,Qcool), dim=1)
            inputs_val   = torch.cat((T_evaluation,Qcool,Qas,Qinter), dim=1)

            self.a_150, self.b_150,self.c_150, self.d_150, self.e_150,self.f_150        = self.forward_master( inputs_val )

            print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, a(150): %.3f , b(150): %.3f , c(150): %.3f , d(150): %.3f , e(150): %.3f. f(150): %.3f,  lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.a_150, self.b_150,self.c_150,self.d_150,self.e_150,self.f_150,current_lr))
            
            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current): # Unscaled input

        # Convert to Torch tensors       
        T_current  = torch.tensor( T_current, dtype=torch.float ).to(self.device)
        T_previous = torch.tensor( T_previous, dtype=torch.float ).to(self.device)
        Q_current  = torch.tensor( Q_current, dtype=torch.float ).to(self.device)  

        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous,Q_current)
        Tn_predict = Tn_predict.cpu()

        return Tn_predict
    
    def set_device(self,device):
        self.device = device


class MCE(nn.Module):
    " MCE : Modèle des Coefficients Effectifs. Modèle créé par Jean-Hughes Fournier-Lupien"

    def __init__(self, P, P_PDE, device):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE  = P_PDE
        self.L = P['seq_len']

        # Forward paramters
        self.dx = P_PDE['dx']
        self.dt = P_PDE['dt']
        self.T0 = P_PDE['T0']
        self.Tc = P_PDE['Tc']
        self.h_conv = P_PDE['h_conv']
        self.Tmin = P_PDE['T_min']
        self.Tmax = P_PDE['T_max']

        self.rhom_a = P_PDE['rhom_a']
        self.rhom_s = P_PDE['rhom_s']
        self.rhom_hast = P_PDE['rhom_hast']

        self.ha = torch.tensor(P_PDE['ha'],device=device, dtype= P["dtype"])
        self.hs = P_PDE['hs']
        self.hhast = P_PDE['hhast']
        self.kernel_size = P['kernel_size']

        # CPU or GPU
        self.device = device
        self.dtype = P["dtype"]

        # Architecture pour les neural nets
        self.net_keff      = self.neural_net2(P['nn_keff'],P['dropout_prob'])
        self.net_Ceff      = self.neural_net(P['nn_C'],P['dropout_prob'])

        # Convolution layer pour faire convolution, + général et peut-être modifier
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 =  np.array(P['CONV1_init'])
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV1 = self.CONV1.to(self.device)
        # Set requires_grad=False
        for name, param in self.CONV1.named_parameters():
            param.requires_grad = not(P['CONV1_frozen']) # Freeze weights
        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array(P['CONV2_init']) 
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)
        self.CONV2 = self.CONV2.to(self.device)

        # Set requires_grad=False
        for name, param in self.CONV2.named_parameters():
            param.requires_grad = not(P['CONV2_frozen']) # Freeze weights  

        print('\nWeights:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print( "%s" % (name) )
        print('\n')

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)
    
    def neural_net(self, layers, dropout_prob):
        # General Neural network architecture with residual connections
        seq = []

        for i in range(len(layers) - 2):
            # Decide whether to use a residual block or a normal linear layer
            in_features = layers[i]
            out_features = layers[i + 1]

            # Using a residual block
            #resblock = False
            if in_features == out_features:
            #if resblock:
                seq.append((f'ResidualBlock{i}', ResidualBlock(in_features, out_features, dropout_prob, self.dtype)))
            else:
                # Fallback to standard linear + dropout + ReLU if dimensions don't match for residual block
                seq.append((f'Linear{i}', nn.Linear(in_features, out_features, dtype=self.dtype)))
                #seq.append((f'LayerNorm{i}', nn.LayerNorm(out_features)))  # Layer Normalization
                seq.append((f'Dropout{i}', nn.Dropout(dropout_prob)))
                seq.append((f'ReLU{i}', nn.ReLU()))
            #if i == 0:
                #seq.append((f'SelfAttention{i}', SelfAttention(out_features, num_heads=1)))
        # Adding the final linear layer
        seq.append((f'Linear{len(layers) - 2}', nn.Linear(layers[-2], layers[-1], dtype=self.dtype)))

        # Create the neural network
        model_seq = nn.Sequential(OrderedDict(seq)).to(self.device)

        # Initialize weights with a custom function if needed
        model_seq.apply(init_weights_JH)

        return model_seq
    
    def neural_net2(self, layers, dropout_prob):
        # General Neural network architecture with residual connections
        seq = []

        for i in range(len(layers) - 2):
            # Decide whether to use a residual block or a normal linear layer
            in_features = layers[i]
            out_features = layers[i + 1]

            # Using a residual block
            #resblock = False
            if in_features == out_features:
            #if resblock:
                seq.append((f'ResidualBlock{i}', ResidualBlock(in_features, out_features, dropout_prob, self.dtype)))
            else:
                # Fallback to standard linear + dropout + ReLU if dimensions don't match for residual block
                seq.append((f'Linear{i}', nn.Linear(in_features, out_features, dtype=self.dtype)))
                #seq.append((f'LayerNorm{i}', nn.LayerNorm(out_features)))  # Layer Normalization
                seq.append((f'Dropout{i}', nn.Dropout(dropout_prob)))
                seq.append((f'Tanh{i}', nn.Tanh()))
            #if i == 0:
                #seq.append((f'SelfAttention{i}', SelfAttention(out_features, num_heads=1)))
        # Adding the final linear layer
        seq.append((f'Linear{len(layers) - 2}', nn.Linear(layers[-2], layers[-1], dtype=self.dtype)))

        # Create the neural network
        model_seq = nn.Sequential(OrderedDict(seq)).to(self.device)

        # Initialize weights with a custom function if needed
        model_seq.apply(init_weights_JH)

        return model_seq
    
    
    def linear_scaler(self,Q,min,max): # log et apres min-max
        return ( Q - min) / (max - min)
    
    def inv_linear_scaler(self,Q_normalize,min,max):
        return (max - min) * Q_normalize + min

    def forward_keff(self, T): # 1D tensor (flat) MCE
        T   = self.linear_scaler(T,self.Tmin,self.Tmax)
        keff = self.net_keff( T ) 
        keff    = self.P['k_eff_scale'] * keff
        return keff
     
    def forward_dkdTeff(self, k, T): # T: 1D tensor (flat) utilisé dans le calcul de k
        T_for_grad = T.clone().detach().requires_grad_(True)
        grad_outputs = torch.ones_like(k)
        dk_dT = torch.autograd.grad(outputs=k, inputs=T_for_grad,
                                    grad_outputs=grad_outputs,
                                    create_graph=True, allow_unused=True)[0]
        if dk_dT is None:
            dk_dT = torch.zeros_like(T_for_grad)
        return dk_dT
    
    def forward_Ceff(self, T): # 1D tensor (flat)
        T   = self.linear_scaler(T,self.Tmin,self.Tmax)
        Ceff = self.net_Ceff( T ) 
        Ceff    = self.P['Ceff_scale'] * Ceff 
        return Ceff
    

    def forward(self, T_current:torch.tensor, T_previous:torch.tensor, Q:torch.tensor,ids:torch.tensor): # inputs:(mini-batch, Nx)
        
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        
        # Add boundary condition 
        bound = self.T0 * torch.ones((T_current.shape[0], 1),device=self.device) # (mini-batch,1)
        bound_kernelT = self.T0* torch.ones((T_current.shape[0], int((self.kernel_size-1)/2)),device=self.device) # need to have a odd kernel_size
        

        Qtot = Q[:,:,0] + Q[:,:,1]+Q[:,:,2]-2*self.h_conv*(T_current-self.T0)
 
        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1)  # 

        # Neural Networks inputs as [Exemples,'kernel_size']
        T_current_inputs    = torch.cat([bound_kernelT, T_current[:,1:-1], bound_kernelT], dim=1).unfold(dimension=1, size=self.kernel_size, step=1).reshape(-1,self.kernel_size)
    
        # Material properties: (applied element-by-element)
        keff_T      = self.forward_keff( T_current_inputs ).reshape(a,b)            
        dkeffdT     = self.forward_dkdTeff( keff_T, T_current_inputs ).reshape(a,b) 
        rhomCpeff_T = self.forward_Ceff( T_current_inputs ).reshape(a,b)       

        # Temperature-dependent coefficients
        gamma_T = 2 * self.dt / ( rhomCpeff_T +1e-8)
        A_T = 2 * keff_T * self.dt / ( rhomCpeff_T * self.dx**2 )
        B_T = dkeffdT * self.dt / (2 * rhomCpeff_T * self.dx**2 )

        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)

        out = ( A_T*conv1_T + B_T*conv2_T**2 + (1-A_T)*T_previous[:,1:-1] + 
                                    gamma_T*Qtot[:,1:-1]) / (1+A_T)

        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 

        return out
    
    def fit(self, DATA_train, DATA_val, epochs):# Inputs: CUBE, DATA est en DataFrame

        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        
        if self.P['steps_max'] == 2: # Steps then won't change, only need to initiate tensor a single time
            #T_train, features_train = DATA_to_sequences_v2(DATA_train,self.L,self.P_PDE,self.P)
            #T_val, features_val = DATA_to_sequences_v2(DATA_train,self.L,self.P_PDE,self.P)
            Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
            Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)

        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            if self.P['steps_max'] > 2:
                #T_train, features_train = DATA_to_sequences_v2(DATA_train,self.L,self.P_PDE,self.P)
                #T_val, features_val = DATA_to_sequences_v2(DATA_train,self.L,self.P_PDE,self.P)
                Seq_train = DATA_to_sequences(DATA_train,steps,self.P_PDE,self.P)
                Seq_val = DATA_to_sequences(DATA_val,steps,self.P_PDE,self.P)
        
            # Number of samples
            N = Seq_train['Tn+1(K)'].size()[0]

            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(N)

            for k in range( 0, N, K): # Par de 0 et fait des bonds de K jusqu'à temps d'obtenir N
                # Define batch
                indices = permutation[k:k+K]

                batch_Seq_train_Tn_2 = Seq_train['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_train_Tn_1 = Seq_train['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_train_Q_1  = Seq_train['Q(W/m2)'][indices]
                batch_Seq_train_Tn   = Seq_train['Tn+1(K)'][indices]
                batch_Seq_train_id   = Seq_train['simu_ids'][indices]-1
                

                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list = [],[]#, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]

                #forward_start_time = time.time()
                for n in range( 0, int(np.floor(steps-1)) ):  
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)

                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train,batch_Seq_train_id) # size: (K, 2*Nx)

                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train ) # Vecteur 1xn, va être comparé avec Tn_true_val_list
                    
                    # Predictions become inputs | T^n-2 ---> T^n-1, T^n-1 ---> T^n, on monte d'une itération dans le temps
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                
                difference_Tn_train = (torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                
                # Mean square error for one mini-batch
                #loss_train = torch.mean(torch.square( difference_Tn_train ))
                #loss_train = torch.mean(torch.square( difference_Tn_train/torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ) ))
                #loss_start_time = time.time()
                loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #loss_train = torch.mean(torch.norm(difference_Tn_train,dim=1,p=1)/torch.norm(torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ),dim=1,p=1))
                #loss_val   = torch.mean(torch.square( difference_Tn_val ))

                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################

                
                # 1) zeroes out the gradients of all parameters in your neural network before the backward pass
                self.optimizer.zero_grad()

                # 2) computes the gradients of the loss with respect to the model parameters     
                loss_train.backward()
                
                # 3) updates the model parameters using the gradients computed in the backward pass
                self.optimizer.step()

                #####################################################################################################

                loss_train_batch.append( loss_train.item() ) # loss_train est un scalaire, mais loss_train_batch est un vecteur car .append plusieurs scalaires de plusieurs batch


            # Update the scheduler learning  rate
            
            current_lr = self.optimizer.param_groups[0]['lr']
            #if self.P['scheduler']:
                #self.scheduler.step()

            # For loop for validation
            M = Seq_val['Tn+1(K)'].size()[0]
            permutation = torch.randperm(M)
            for j in range(0,M,K):
                indices = permutation[j:j+K]
                batch_Seq_val_Tn_2 = Seq_val['Tn-1(K)'][indices] # Mélange en petit nombre (K) pour le training Data
                batch_Seq_val_Tn_1 = Seq_val['Tn(K)'][indices] # Seqtrain[indices] = Seqtrain[indices,:,:]
                batch_Seq_val_Q_1  = Seq_val['Q(W/m2)'][indices]
                batch_Seq_val_Tn   = Seq_val['Tn+1(K)'][indices]
                batch_Seq_val_id   = Seq_val['simu_ids'][indices]

                # Initialize state at a starting point of a sequence
                Tn_recur_previous_val = batch_Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = batch_Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)

                Tn_pred_val_list, Tn_true_val_list = [],[]
                for n in range( 0, int(np.floor(steps-1)) ):
                    Q_val       = batch_Seq_val_Q_1[:,n,:,:] # [Sample, time sequence, Nx, 3] 0:ag, 1:supra, 2:inter
                    Tn_true_val = batch_Seq_val_Tn[:,n,:]
                    Tn_predict_val = self.forward( Tn_recur_current_val, Tn_recur_previous_val, Q_val,batch_Seq_val_id)
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                difference_Tn_val   = (torch.stack(Tn_pred_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx']) - torch.stack(Tn_true_val_list,dim=1).reshape( len(indices)*(steps-1), self.P_PDE['Nx'] ))
                #loss_val   = torch.mean(torch.square( difference_Tn_val ))
                loss_val = torch.mean(torch.norm(difference_Tn_val,dim=1))
                loss_val_batch.append( loss_val.item() )    

            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display

            ##############################################################################

            # Predict material properties:
            self.keff_150        = self.forward_keff( torch.tensor([[150]]).to(self.device))
            self.rhomCpeff_150   = self.forward_Ceff( torch.tensor([[150]]).to(self.device))
 
            print( 'Epoch:%03i, steps:%i, loss: %.2e, validation loss: %.2e, keff(150): %.3e (%.3e), rhomCpeff(150): %.3e (%.3e), lr: %.2e' % (epo+1, steps, Loss_train_epoch[-1], Loss_val_epoch[-1], self.keff_150, self.P_PDE['k_eff_150'], self.rhomCpeff_150, self.P_PDE['rhomCp_eff_150'],current_lr))

            
            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < self.P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current, ids): # Unscaled input

        # Convert to Torch tensors       
        T_current  = torch.tensor( T_current, dtype=torch.float ).to(self.device)
        T_previous = torch.tensor( T_previous, dtype=torch.float ).to(self.device)
        Q_current  = torch.tensor( Q_current, dtype=torch.float ).to(self.device)
        ids        = torch.tensor(ids, dtype = torch.int ).to(self.device)  

        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous,Q_current,ids.unsqueeze(0))
        Tn_predict = Tn_predict.cpu()

        return Tn_predict
    
def init_weights_transformer(module):
    """
    Initialise les poids du module selon leur type.
    - Pour les couches Linéaires, utilise l'initialisation Xavier uniform.
    - Pour les couches de Convolution, utilise l'initialisation Kaiming normal.
    - Pour les couches de Normalisation (LayerNorm), on initialise les poids au 1 et les biais à 0.
    - Les couches d'attention multi-head et autres composants internes du Transformer 
      sont généralement déjà initialisés de façon raisonnable par PyTorch, mais vous 
      pouvez également les personnaliser si nécessaire.
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv1d):
        # Kaiming normal pour convolution
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        # Initialisation standard pour LayerNorm : poids à 1, biais à 0
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
   
