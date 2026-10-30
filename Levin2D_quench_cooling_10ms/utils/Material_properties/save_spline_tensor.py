
"""
Purpose: PINN 
Data:    2021-03-18
Author:  Jean-Hughes Fournier L.
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as P
import torch 
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import pickle
import pandas as pd


directory  = "C:\\Users\\nbrea\\OneDrive\\Bureau\\Polytechnique\\MAITRISE\\Codes\\Levin2D\\"
material_path = directory + "utils\\Material_properties\\" 

def h_poly_helper(tt):
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]], dtype=tt[-1].dtype)
    return [ sum( A[i, j]*tt[j] for j in range(4) ) for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def interp(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = P.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def finite_diff(x_data, y_data):
    x_slope = []
    slope = []
    for i in range(len(x_data)-1):
        slope.append( (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i]) )
        x_slope.append( (x_data[i+1] + x_data[i])/2 )
    return x_slope, slope


########################################################################
############# sigma_Cu ###################################################
########################################################################

data = np.loadtxt(material_path + "data\\rho_Cu.txt")

x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( 1/data[:,1], dtype=torch.float )

T = torch.linspace( 273, 800, 500 )

spl_sigma_Cu = InterpolatedUnivariateSpline( x_data, y_data, k=3, ext=3)


plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_sigma_Cu(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('sigma_Cu')
plt.savefig(material_path + "data_spline\\sigma_Cu.png",dpi=300)
#plt.show()

# Save spline functions
pickle.dump(spl_sigma_Cu, open(material_path + "data_spline\\spline_sigma_Cu.pkl","wb"))

########################################################################
############# rho_Cu ###################################################
########################################################################

data = np.loadtxt(material_path +"data\\rho_Cu.txt")

x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

T = torch.linspace( 273, 800, 500 )

spl_rho_Cu = InterpolatedUnivariateSpline( x_data, y_data, k=3, ext=3)


plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_rho_Cu(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('rho_Cu')
plt.savefig(material_path +"data_spline\\rho_Cu.png",dpi=300)
#plt.show()

# Save spline functions
pickle.dump(spl_rho_Cu, open(material_path +"data_spline\\spline_rho_Cu.pkl","wb"))


########################################################################
############# Cp_Cu ###################################################
########################################################################

data = np.loadtxt(material_path +"data\\Cp_Cu.txt")

x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

T = torch.linspace( 273, 800, 500 )

spl_Cp_Cu = InterpolatedUnivariateSpline( x_data, y_data, k=1, ext=3)


plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_Cp_Cu(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('Cp_Cu')
plt.savefig(material_path +"data_spline\\Cp_Cu.png",dpi=300)
#plt.show()

# Save spline functions
pickle.dump(spl_Cp_Cu, open(material_path +"data_spline\\spline_Cp_Cu.pkl","wb"))


#####################################################################
############## k_Cu & dk/dT_Cu ######################################
#####################################################################

data = np.loadtxt(material_path +"data\\k_Cu.txt")

x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

T = torch.linspace( 273, 800, 500 )

spl_k_Cu = InterpolatedUnivariateSpline( x_data, y_data, k=3, ext=3)


# Slope finite difference
x_slope, slope = finite_diff(x_data, y_data)

# UnivariateSpline
#spl_k_Cu = InterpolatedUnivariateSpline( x_data, y_data, k=3, ext=3)
spl_dkdT_Cu = spl_k_Cu.derivative()


plt.figure()
plt.subplot(2,1,1)
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_k_Cu(T), label='UnivariateSpline' )
plt.legend()
plt.ylabel('k_Cu')

plt.subplot(2,1,2)
plt.plot( x_slope, slope, 'o', label='Finite difference' )
plt.plot( T, spl_dkdT_Cu(T.numpy()), label='UnivariateSpline' )

plt.legend()
plt.ylabel('dkdT_Cu')
plt.xlabel('T(K)')
plt.savefig( material_path +'data_spline/k_Cu', dpi=300)

# Save spline functions
pickle.dump(spl_k_Cu, open(material_path +"data_spline/spline_k_Cu.pkl","wb"))

pickle.dump(spl_dkdT_Cu, open(material_path +"data_spline/spline_dkdT_Cu.pkl","wb"))

########################################################################
############# rho_ag ###################################################
########################################################################

data = np.loadtxt(material_path +"data\\rho_silver.txt")

x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

T = torch.linspace( 75, 500, 400 )

spl_rho_ag = InterpolatedUnivariateSpline( x_data, y_data, k=1, ext=3)


plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_rho_ag(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('rho_ag')
plt.savefig(material_path +"data_spline\\rho_ag.png",dpi=300)
#plt.show()

# Save spline functions
pickle.dump(spl_rho_ag, open(material_path +"data_spline\\spline_rho_ag.pkl","wb"))

########################################################################
############# k_Ag #####################################################
########################################################################
# Load data 
path= material_path + 'data\\k_silver.txt'
data = np.loadtxt(path)
#data = np.loadtxt('data\\k_silver.txt')
x_data = torch.tensor( data[10:,0], dtype=torch.float )
y_data = torch.tensor( data[10:,1], dtype=torch.float )

T = torch.linspace( 75, 500, 400 )

# Slope finite difference
x_slope, slope = finite_diff(x_data, y_data)

# UnivariateSpline
spl_k_Ag = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)
spl_dkdT_Ag = spl_k_Ag.derivative()


plt.figure()
plt.subplot(2,1,1)
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_k_Ag(T)     , label='UnivariateSpline' )
plt.legend()
plt.ylabel('k_Ag(T)')
plt.axis([0,500,300,5500])

plt.subplot(2,1,2)
plt.plot( x_slope, slope, 'o', label='Finite difference' )
plt.plot( T, spl_dkdT_Ag(T.numpy()), label='UnivariateSpline' )

plt.legend()
plt.ylabel('dkdT_Ag')
plt.axis([0,800,-700,50])
plt.xlabel('T(K)')
plt.savefig( material_path +'data_spline\\k_Ag', dpi=300)

# Save spline functions
# tck1 = spl_k_Ag._eval_args
pickle.dump(spl_k_Ag, open(material_path +"data_spline/spline_k_Ag.pkl","wb"))
# tck2 = spl_dkdT_Ag._eval_args
pickle.dump(spl_dkdT_Ag, open(material_path +"data_spline/spline_dkdT_Ag.pkl","wb"))

#####################################################################
############## C_Ag #################################################
#####################################################################
path= material_path + 'data\\cp_silver.txt'
data = np.loadtxt(path)
x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

# UnivariateSpline
spl_Cp_Ag = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)

plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_Cp_Ag(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('Cp_Ag)')
plt.savefig( material_path +'data_spline/Cp_Ag', dpi=300)

# Save spline functions
pickle.dump(spl_Cp_Ag, open(material_path +"data_spline/spline_Cp_Ag.pkl","wb"))


#####################################################################
############## k_ybco ###############################################
#####################################################################
T = torch.linspace( 75, 500, 400 )

# Load data 
path= material_path + 'data\\k_YBCO.txt'
data = np.loadtxt(path)
x_data = torch.tensor( data[ 2: ,0], dtype=torch.float )
y_data = torch.tensor( data[ 2: ,1], dtype=torch.float )


# Slope finite difference
x_slope, slope = finite_diff(x_data, y_data)

# UnivariateSpline
spl_k_ybco = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)
spl_dkdT_ybco = spl_k_ybco.derivative()


plt.figure()
plt.subplot(2,1,1)
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_k_ybco(T), label='UnivariateSpline' )
plt.legend()
plt.ylabel('k_ybco')

plt.subplot(2,1,2)
plt.plot( x_slope, slope, 'o', label='Finite difference' )
plt.plot( T, spl_dkdT_ybco(T.numpy()), label='UnivariateSpline' )

plt.legend()
plt.ylabel('dkdT_ybco')
plt.xlabel('T(K)')
plt.savefig( material_path +'data_spline/k_ybco', dpi=300)

# Save spline functions
# tck1 = spl_k_ybco._eval_args
pickle.dump(spl_k_ybco, open(material_path +"data_spline/spline_k_ybco.pkl","wb"))
# tck2 = spl_dkdT_ybco._eval_args
pickle.dump(spl_dkdT_ybco, open(material_path +"data_spline/spline_dkdT_ybco.pkl","wb"))


#####################################################################
############## C_ybco #################################################
#####################################################################
T = torch.linspace( 75, 500, 400 )

path= material_path + 'data\\cp_ybco.txt'
data = np.loadtxt(path)
x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

# UnivariateSpline
spl_Cp_ybco = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)


plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_Cp_ybco(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('Cp_ybco')
plt.savefig( material_path +'data_spline/Cp_ybco', dpi=300)

# Save spline functions
# tck1 = spl_Cp_ybco._eval_args
# pickle.dump(tck1, open("data_spline/spline_Cp_ybco.pkl","wb"))
pickle.dump(spl_Cp_ybco, open(material_path +"data_spline/spline_Cp_ybco.pkl","wb"))


#####################################################################
############## k_hast ###############################################
#####################################################################
T = torch.linspace(75, 500, 400 )

# Load data 
path= material_path + 'data\\k_hast.txt'
data = np.loadtxt(path)
x_data = torch.tensor( data[ 30: ,0], dtype=torch.float )
y_data = torch.tensor( data[ 30: ,1], dtype=torch.float )


# Slope finite difference
x_slope, slope = finite_diff(x_data, y_data)

# UnivariateSpline
spl_k_ybco = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)
spl_dkdT_ybco = spl_k_ybco.derivative()


plt.figure()
plt.subplot(2,1,1)
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_k_ybco(T), label='UnivariateSpline' )
plt.legend()
plt.ylabel('k_hast')

plt.subplot(2,1,2)
plt.plot( x_slope, slope, 'o', label='Finite difference' )
plt.plot( T, spl_dkdT_ybco(T.numpy()), label='UnivariateSpline' )

plt.legend()
plt.ylabel('dkdT_hast')
plt.xlabel('T(K)')
plt.savefig( material_path +'data_spline/k_hast', dpi=300)

# Save spline functions
# tck1 = spl_k_ybco._eval_args
pickle.dump(spl_k_ybco, open(material_path +"data_spline/spline_k_hast.pkl","wb"))
# tck2 = spl_dkdT_ybco._eval_args
pickle.dump(spl_dkdT_ybco, open(material_path +"data_spline/spline_dkdT_hast.pkl","wb"))



#####################################################################
############## C_hast #################################################
#####################################################################
T = torch.linspace( 75, 500, 400 )

path= material_path + 'data\\cp_hast.txt'
data = np.loadtxt(path)
x_data = torch.tensor( data[:,0], dtype=torch.float )
y_data = torch.tensor( data[:,1], dtype=torch.float )

# UnivariateSpline
spl_Cp_hast = UnivariateSpline( x_data, y_data, k=1, s=0, ext=3)

plt.figure()
plt.plot( x_data, y_data, 'o', label='data')
plt.plot( T, spl_Cp_hast(T), label='UnivariateSpline' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('Cp_hast')
plt.savefig( material_path +'data_spline/Cp_hast', dpi=300)

# Save spline functions
# tck1 = spl_Cp_hast._eval_args
pickle.dump(spl_Cp_hast, open(material_path +"data_spline/spline_Cp_hast.pkl","wb"))

#####################################################################
############## h_conv #################################################
#####################################################################

T = torch.linspace( 75, 500, 400 )

path = material_path + 'data\\hconv.txt'

data = pd.read_csv( path, sep=" ", header=None)
data = data.to_numpy()

# UnivariateSpline
spl_hconv = UnivariateSpline( data[:,0], data[:,1], k=1, s=0, ext=3)

plt.figure()
plt.plot( data[:,0], data[:,1],'o', label='data' )
plt.plot( T, spl_hconv(T), label='spline' )
plt.legend()
plt.xlabel('Delta T (K)')
plt.ylabel('h (W/(m^2 K))')
plt.axis( [0,50,-500,12000] )
plt.grid()
plt.savefig(material_path +'data_spline/h_conv', dpi=300)

# Save spline in pickle
pickle.dump(spl_hconv, open(material_path +"data_spline/spline_h_conv.pkl","wb"))
#tikz_save( 'data_spline/h_conv.tex' )

plt.figure()
plt.plot( data[:,0], data[:,1],'o', label='data' )
plt.plot( T, spl_hconv(T-77), label='spline' )
plt.legend() 
plt.xlabel('Delta T (K)')
plt.ylabel('h (W/(m^2 K))')
plt.axis( [0,300,-500,12000] )
plt.grid()
plt.savefig(material_path +'data_spline/h_conv', dpi=300)

#tikz_save( 'data_spline/h_conv_fig8.tex' )




