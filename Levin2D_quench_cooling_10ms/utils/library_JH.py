                                                                                       
"""
Created on Tue Nov 10 16:05:12 2020

@author: Jean-Hughes Fournier L.
"""
# Standard modules                    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # 0.23.2 


def load_data_JH(path,time_skip = None):
    # ****************************************************
    # Load dataset  from many csv files (concatenate all simulations)
    #       path: path of the folder containing the results of the COMSOL simulations (.csv files)
    #       labels: features to be load
    # Output: dataframe
    # ****************************************************
    # List of path of .csv files
    files = [f for f in glob.glob(path)]
    files.sort()
    
    #if not path2==None:
    #    fI = pd.read_csv(path2+"truth.csv")
    #    I = fI.values[:,1]

    X = []
    # Loop over COMSOL simulations (.csv files) 
    for file in files:
        df = pd.read_csv(file)
        # Remove empty spaces in header
        df.columns = [c.replace(' ', '') for c in df.columns]
        # Add unique_index column
        Simu_id = int(file.split('\\')[-1].split('.')[-2])

        #if not time_skip == None:
        #    df = reshape_dataframe(df,2)
        df['Simu_index'] = Simu_id
        #if not path2 == None:
        #    df['Current'] = I[Simu_id-1]
        X.append( df ) # List of dataframe
        print( 'Simu_id: %i'%(Simu_id) )
        #if not path2 == None:
        #    print('Courrant associe: '+ str(I[Simu_id-1]) + ' A')
        # Drop columns with all NaN values
    result_df = pd.concat(X)
    result_df = result_df.dropna(axis=1, how='all')
    return result_df # Output: Dataframe


def Predict(X, fML, scaler):
    y_dummy = pd.DataFrame( 1e-1*np.ones((X.shape[0],1)), index=X.index, columns=[scaler.labels[-1]] ) # Pourquoi créer un vecteur de 0.1 ?
    # Transform
    X_sc, y_dummy_sc = scaler.transform(X, y_dummy)
    # Prediction
    y_predict_sc = fML.predict(X_sc)
    # inv_transform
    X_dummy, y_predict = scaler.inv_transform(X_sc, y_predict_sc)
    return y_predict

# My_scaler class!
class My_scaler:
    # ****************************************************
    # Purpose: scale input data to be use in a neural network model 
    # Inputs: 
    #      - labels: list of the column name of predictors and target (target should be the last one)
    #      - log: ordered list of 'log' or 'lin' strings. Choose to apply log on a predictor or target. Ex: ['log','lin','lin','log', 'log'] 
    #      - log_cst: constant used in the definition of the log: log(x + log_cst)
    #      - scaler: ordered list of 'stand' or 'minmax'. Choose to apply standard or minmax scaler on predictors or target.
    # ****************************************************
    def __init__(self, labels, log, log_cst, scaler ):
        self.labels = labels
        self.log = log
        self.log_cst = log_cst
        self.scaler = scaler
        
    def fit(self, X, y ):
        # Fit scalers
        X_pre, y_pre = self.pre_transform(X, y) # Apply log
        self.scaler_X_stand  = StandardScaler().fit( X_pre ) 
        self.scaler_X_minmax = MinMaxScaler(feature_range=(-1,1)).fit( X_pre ) 
        self.scaler_y_stand  = StandardScaler().fit( y_pre )
        self.scaler_y_minmax = MinMaxScaler(feature_range=(-1,1)).fit( y_pre )
        
    def transform(self, X, y): # Inputs: Dataframe
        # Pre-transform
        X_pre, y_pre = self.pre_transform(X, y) #Applique log sur colonnes désirées
        
        X_sc_stand  = self.scaler_X_stand.transform( X_pre )  # output: array
        X_sc_minmax = self.scaler_X_minmax.transform( X_pre ) # output: array
        
        X_sc = []
        for i,sca in enumerate(self.scaler[0:-1]):
            if sca.lower()=='stand':
                X_sc.append( X_sc_stand[:,i].reshape((X.shape[0],1)) )  # 
            elif sca.lower()=='minmax':
                X_sc.append( X_sc_minmax[:,i].reshape((X.shape[0],1)) )  #
            elif sca.lower()=='none':                              # Test
                X_sc.append( X_pre.values[:,i].reshape((X.shape[0],1)) )  
        X_sc = np.hstack(X_sc)
        
        # Scale target
        if self.scaler[-1].lower()=='stand':    
            y_sc = self.scaler_y_stand.transform( y_pre ) #
        elif self.scaler[-1].lower()=='minmax':                             
            y_sc = self.scaler_y_minmax.transform( y_pre ) #
        elif self.scaler[-1].lower()=='none':                     # Test                          
            y_sc = y_pre.values #
        
        # Bring back to dataframe
        X_sc = pd.DataFrame( X_sc, index=X.index, columns=X.columns  )
        y_sc = pd.DataFrame( y_sc, index=y.index, columns=y.columns  )
        return X_sc, y_sc # Outputs: Dataframe
        
    def inv_transform(self, X_sc, y_sc):
        X_pre_stand = self.scaler_X_stand.inverse_transform( X_sc )  
        X_pre_minmax = self.scaler_X_minmax.inverse_transform( X_sc ) 
        
        X_pre = []
        for i,sca in enumerate(self.scaler[0:-1]):
            if sca.lower()=='stand':
                X_pre.append( X_pre_stand[:,i].reshape((X_sc.shape[0],1)) )  # Output array
            elif sca.lower()=='minmax':
                X_pre.append( X_pre_minmax[:,i].reshape((X_sc.shape[0],1)) )  # Output array
            elif sca.lower()=='none':
                X_pre.append( X_sc.values[:,i].reshape((X_sc.shape[0],1)) )  # Output array
        X_pre = np.hstack(X_pre)
        
        # Scale target
        if   self.scaler[-1].lower()=='stand':    
            y_pre = self.scaler_y_stand.inverse_transform( y_sc ) #
        elif self.scaler[-1].lower()=='minmax':                             
            y_pre = self.scaler_y_minmax.inverse_transform( y_sc ) #
        elif self.scaler[-1].lower()=='none':                             
            y_pre = y_sc.values #
            
        # Bring back to dataframe
        X_pre = pd.DataFrame( X_pre, index=X_sc.index, columns=X_sc.columns  )
        y_pre = pd.DataFrame( y_pre, index=y_sc.index, columns=y_sc.columns  )
        # Inverse pre-transform
        X, y = self.pre_inv_transform(X_pre, y_pre)
        return X, y # Outputs: Dataframe    
        
    
    def pre_transform(self, X, y): # Input Dataframes
        # # Apply log scaling
        Xy = pd.concat([X, y], axis=1)
        # Apply log according to log_choice list
        Xy_pre = Xy.copy()
        for i in range(len(self.labels)): # Loop over predictors and target names
            if self.log[i].lower()=='log':
                Xy_pre.loc[:,self.labels[i]] = np.log10( Xy[self.labels[i]] + self.log_cst  ) 
        X_pre = Xy_pre[self.labels[0:-1]]
        y_pre = pd.DataFrame( Xy_pre[self.labels[-1]] )
        return X_pre, y_pre # Outputs: Dataframes
    
    def pre_inv_transform(self, X_pre, y_pre): # Input Dataframes
        # # Apply log scaling
        Xy_pre = pd.concat([X_pre, y_pre], axis=1)
        # Apply log according to log_choice list
        Xy = Xy_pre.copy()
        for i in range(len(self.labels)): # Loop over predictors and target names
            if self.log[i].lower()=='log':
                Xy.loc[:,self.labels[i]] = 10 ** Xy_pre[self.labels[i]] - self.log_cst
        X = Xy[self.labels[0:-1]]
        y = pd.DataFrame( Xy[self.labels[-1]] )
        return X, y # Output Dataframes

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_ (0.01)
        
def Grad_norm(net):
    total_norm = 0
    for p in net.parameters():
        param_norm = p.grad.data.norm(2)**2 # p1**2 + p2**2 + ...
        total_norm += param_norm.item()
        #print(p.grad.data)
        #print(param_norm )
        #print('Next')
    return np.sqrt(total_norm)        

def find_closest(choice, df, column):
    # Purpose: find the closest value from a value choice in the column of a dataframe (df)
    df_unique = np.array(list(set( df[column] ))) # Unique values  
    out  = df_unique[np.argmin((df_unique - choice)**2)]
    return out

def rho_Ag(T, path ):
    data = np.loadtxt(path+'rho_silver.txt')
    f = interp1d( data[:,0], data[:,1], kind='cubic')
    return f(T)

def rho_hast(T, path ):
    data = np.loadtxt(path+'rho_hast.txt')
    f = interp1d( data[:,0], data[:,1], kind='cubic')
    return f(T)

def power_law( I, T,Ic0, P_comsol):
    # Return E(V/m)
    nT  = ((P_comsol['n0']-10)*(P_comsol['Tc']-T)/(P_comsol['Tc']-P_comsol['T0']) + 10)*(T<P_comsol['Tc']) + (T>=P_comsol['Tc'])
    IcT = Ic0*(P_comsol['Tc']-T)/(P_comsol['Tc']-P_comsol['T0']) * (T<P_comsol['Tc']) + (T>=P_comsol['Tc'])     
    J     = I/(P_comsol['w']*P_comsol['h_supra']) 
    E_min = 1/P_comsol['sigma_s_max'] * J
    return np.maximum( P_comsol['Ec']*(I/IcT)**nT, E_min ) 

def E_analytic(Itot, T, Ic0, P_comsol, path):
    sigma_Ag = 1/rho_Ag(T, path=path)
    sigma_hast = 1/rho_hast(T, path=path)
    nT  = ((P_comsol['n0']-10)*(P_comsol['Tc']-T)/(P_comsol['Tc']-P_comsol['T0']) + 10)*(T<P_comsol['Tc']) + (T>=P_comsol['Tc'])
    IcT = Ic0*(P_comsol['Tc']-T)/(P_comsol['Tc']-P_comsol['T0']) * (T<P_comsol['Tc']) #+ (T>=P_comsol['Tc'])     
    E_out = []
    for itot in Itot:
        def func(Es):
            return IcT*(Es/P_comsol['Ec'])**(1/nT) + P_comsol['w']*Es*( P_comsol['sigma_n']*P_comsol['h_supra'] + sigma_Ag*P_comsol['h_silver_top'] + sigma_hast*P_comsol['h_hast'] ) - itot
        
        maxfev = 200
        if itot>IcT:
            root = fsolve(func, 1e-6, maxfev=maxfev)
        elif T>P_comsol['Tc']:
            root = fsolve(func, 1e-10, maxfev=maxfev)
        else:
            root = fsolve(func, power_law( itot, T, Ic0, P_comsol), maxfev=maxfev )
        
        # Limit minimum value of E
        J     = itot/(P_comsol['w']*P_comsol['h_supra']) 
        E_min = 1/P_comsol['sigma_s_max'] * J  
        
        E_out.append( np.maximum( root[0] , E_min )  )
    return E_out

def plot_weights(torch_model, dh):
    layers = []
    for mod in torch_model.named_children():
        if mod[0][0:3]=='Lin': # 'Lin'
            layers.append( mod[0] )
            
    # Plot weight and biasesd
    fig = plt.figure()
    for i in range(len(dh)+1):
        plt.subplot(1,len(dh)+1,i+1)
        Weights = eval( 'torch_model.' + layers[i] + '.weight.detach().numpy()' )
        plt.pcolor( Weights )
    plt.colorbar()
    plt.tight_layout()
    return fig
  
def RMSE(true, predict, scale='lin'):
    if scale.lower() == 'lin':
        return np.sqrt(mean_squared_error( true, predict ))
    if scale.lower() == 'log':
        temp =  ( np.log10(predict.to_numpy()) - np.log10(true.to_numpy()))**2
        return np.sqrt( temp.sum()/len(temp) )

def parameters_grid(*args):
    A=np.meshgrid( *args )
    A_list = []
    for a in A:
        A_list.append( a.reshape( np.size(a), 1) )
    return np.hstack(A_list)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class ADAPT_RELU(torch.nn.Module):
        
    def __init__(self, alpha = None):
        
        super(ADAPT_RELU,self).__init__()
        
        # Initialize "alpha"
        if alpha == None:
            self.alpha = torch.nn.parameter.Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        
        self.alpha.requiresGrad = True # set requiresGrad to true!
        
    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        ZERO = torch.zeros(x.shape)
        return self.alpha*torch.max(ZERO, x)

def DF_to_CUBE(DATA):
    # Purpose: Dataframe to cube [i,n,s], i: position index, time index, simulation index
    # Get the different COMSOL simulations unique identifier
    simu_unique = sorted( set( DATA['Simu_index'].to_numpy()) )
    tn_unique   = sorted( set( DATA['tn(s)'].to_numpy()) )
    # Create CUBE of data
    cube_lists = [ [] for n in range(len(DATA.columns)) ]
    for s in simu_unique: # 
        seq_lists = [ [] for n in range(len(DATA.columns)) ]
        for tn in tn_unique: # 
            crit = np.array( (DATA['Simu_index'] == s) & (DATA['tn(s)'] == tn) ) 
            for k in range(len(DATA.columns)):
                seq_lists[k].append( DATA[crit].sort_values('xi(m)')[DATA.columns[k]].values )
        for k in range(len(DATA.columns)):
            cube_lists[k].append( np.vstack( seq_lists[k] ).T )
    out = {}
    for k in range(len(DATA.columns)):
        out[DATA.columns[k]] = np.stack( cube_lists[k],2 ) 
    return out

    
def cubes2sequences(cube_DATA, steps):
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
        
    
############################################################################
# Plot functions:
############################################################################
def plot_loghist(x, bins, log_cst):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0] + log_cst ), np.log10(bins[-1] + log_cst ),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')
  # hist, bins, _ = plt.hist(x, bins=8)
  # plt.close()
  # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  # plt.hist(x, bins=logbins)


def scatter_plots( var, y_true, y_pred, title, scale='lin', frac=0.5 ):
    # ****************************************************
    # Purpose: plot predicted versus true values
    #       var: name of the feature being plot (string)
    #       y_true: true values (array)
    #       y_pred: predicted values (array)
    #       title: title of the plot (string)
    # ****************************************************
    # Take a fraction of the dataset for display 
    temp1, y_true, temp2, y_pred = train_test_split(y_true, y_pred, test_size=frac)    
    plt.xlabel(var+'_true')
    plt.ylabel(var+'_predicted')
    plt.grid()
    line = np.array( [y_true.min(), y_true.max()] ).reshape((2,1))
    if scale=='lin':
        plt.plot( line,line, 'r' )
    if scale=='log':
        plt.loglog( line,line, 'r' )
    # Scatter plots
    if scale=='lin':
        plt.plot( y_true, y_pred, 'o', markersize=1 )
    if scale=='log':
        plt.loglog( y_true, y_pred, 'o', markersize=1 )
    plt.title(title)
    plt.tight_layout()

def plot_scaling( X, y, f, scaler, Log, labels, bins=10):
    data = pd.concat([X, y], axis=1)
    X_sc, y_sc = scaler.transform( X, y ) 
    data_sc = pd.concat([X_sc, y_sc], axis=1)
    N = len(labels)
    # Before scaling
    fig = plt.figure(figsize=(15,7.5))
    for i,label in enumerate(labels):
        plt.subplot( 2, N, i+1 ) 
        if Log[i].lower()=='log':
            plot_loghist( data[label], bins=bins, log_cst=scaler.log_cst) ; plt.xlabel(label)
        else:
            plt.hist(     data[label], bins=bins) ; plt.xlabel(label)
    # After scaling
    for i,label in enumerate(labels):
        plt.subplot( 2, N, N+i+1 ) 
        plt.hist( data_sc[label], bins=bins) ; plt.xlabel(label)
    plt.tight_layout()
    return fig

def CONV_MAT(a,b,c,N):
    M_diago1 = a * (np.triu( np.ones((N-2,N)), k=0) - np.triu( np.ones((N-2,N)), k=1))
    M_diago2 = b * (np.triu( np.ones((N-2,N)), k=1) - np.triu( np.ones((N-2,N)), k=2))
    M_diago3 = c * (np.triu( np.ones((N-2,N)), k=2) - np.triu( np.ones((N-2,N)), k=3))  
    M  = M_diago1 + M_diago2 + M_diago3
    
    v = np.zeros( (1,M.shape[1]) )
    v[0,0]=1
    v_inverse = np.expand_dims(v[0,::-1], axis=0)
    
    M = np.concatenate( (v,M), axis=0 )
    M = np.concatenate( (M,v_inverse), axis=0 )
    return M

