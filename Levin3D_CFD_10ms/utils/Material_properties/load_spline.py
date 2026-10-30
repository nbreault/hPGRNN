
"""
Purpose: PINN 
Data:    2021-03-18
Author:  Jean-Hughes Fournier L.
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pickle
import torch


directory = "/home/jh/Documents/Dropbox/Documents_sync/Travaux/Post-Doc/Codes/"
material_path = directory + "datasets_PINN_COMSOL/Material_properties/data/"
spline_path = directory + "datasets_PINN_COMSOL/Material_properties/data_spline/"

def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2

def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))

def Bspline( X, tck ):
    out = []
    for x in X:
        out.append( bspline(x, *tck) )
    return np.array( out )

# Thickness HTS tape
h_Ag   = 2e-6
h_ybco = 1e-6
h_hast = 50e-6
h_tot = h_Ag + h_ybco + h_hast

################### k_Ag ######################
# Temperature range
T = np.linspace( 20, 400, 400 )

# Load pickled spline 
spl_k_Ag   = pickle.load( open( spline_path + "spline_k_Ag.pkl", "rb" ) )
spl_k_ybco = pickle.load( open( spline_path + "spline_k_ybco.pkl", "rb" ) )
spl_k_hast = pickle.load( open( spline_path + "spline_k_hast.pkl", "rb" ) )

k_eff = (spl_k_Ag(T)*h_Ag + spl_k_ybco(T)*h_ybco + spl_k_hast(T)*h_hast) / h_tot

spl_dkdT_Ag   = pickle.load( open( spline_path + "spline_dkdT_Ag.pkl", "rb" ) )
spl_dkdT_ybco = pickle.load( open( spline_path + "spline_dkdT_ybco.pkl", "rb" ) )
spl_dkdT_hast = pickle.load( open( spline_path + "spline_dkdT_hast.pkl", "rb" ) )

dkdT_eff = (spl_dkdT_Ag(T)*h_Ag + spl_dkdT_ybco(T)*h_ybco + spl_dkdT_hast(T)*h_hast) / h_tot

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot( T, spl_k_Ag(T),   label='k_Ag')
plt.plot( T, spl_k_ybco(T), label='k_ybco')
plt.plot( T, spl_k_hast(T), label='k_hast')
plt.plot( T, k_eff, 'k', linewidth=2.0 , label='k_eff' )
plt.legend(loc=4)
plt.ylabel('k')
plt.axis([0,400,0,1000])

plt.subplot(2,1,2)
plt.plot( T, spl_dkdT_Ag(T),   label='dkdT_Ag')
plt.plot( T, spl_dkdT_ybco(T), label='dkdT_ybco')
plt.plot( T, spl_dkdT_hast(T), label='dkdT_hast')
plt.plot( T, dkdT_eff, 'k', linewidth=2.0 , label='dkdT_eff' )
plt.legend(loc=4)
plt.xlabel('T(K)')
plt.axis([0,400,-50,10])

plt.savefig( 'data_spline/k_eff', dpi=300)

############## C_eff ###################
# Temperature range
T = np.linspace( 20, 400, 400 )

# Load pickled spline 
spl_Cp_Ag   = pickle.load( open( spline_path + "spline_Cp_Ag.pkl", "rb" ) )
spl_Cp_ybco = pickle.load( open( spline_path + "spline_Cp_ybco.pkl", "rb" ) )
spl_Cp_hast = pickle.load( open( spline_path + "spline_Cp_hast.pkl", "rb" ) )

Cp_eff = (spl_Cp_Ag(T)*h_Ag + spl_Cp_ybco(T)*h_ybco + spl_Cp_hast(T)*h_hast) / h_tot

# Figure 
plt.figure()
plt.plot( T, spl_Cp_Ag(T),   label='Ag')
plt.plot( T, spl_Cp_ybco(T), label='ybco')
plt.plot( T, spl_Cp_hast(T), label='hast')
plt.plot( T, Cp_eff, 'k', linewidth=2.0 , label='Cp_eff' )
plt.legend()
plt.xlabel('T(K)')
plt.ylabel('Cp')

plt.savefig( 'data_spline/Cp_eff', dpi=300)
















