#!/usr/bin/env python
# coding: utf-8

# In[1]:

from interaction_matrices import interactions
from decay_matrices import decay_class
from rho_X import rho
import Lambdas



from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.sparse import *
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
from jupyterthemes import jtplot


# MCEq Imports
from MCEq.particlemanager import ParticleManager
import MCEq.core 
from MCEq.core import MCEqRun
from MCEq.data import Decays
import mceq_config as config
#import primary model choices
import crflux.models as pm



config.muon_helicity_dependence=True

config.debug_level = 0

# Launcing mceq
mceq_run = MCEqRun(
    #provide the string of the interaction model
    interaction_model='SIBYLL23CPP',
    #primary cosmic ray flux model
    primary_model = (pm.HillasGaisser2012, "H3a"),
        # Zenith angle in degrees. 0=vertical, 90=horizontal
    theta_deg=0.0
    
)





def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rho_fun_data(rr,rh):
    
    return UnivariateSpline(rr,rh[0:len(ser)],k=1,s=0,ext=0)



jtplot.style(theme="grade3", context="notebook", ticks=True, grid=False)


"""""
    gives back division of matrices with 0 at places where infinity arises

    Parameters:
    ------------------------
    a matrix ,   b  matrix

    Returns:
    ------------------------
    c matrix 
    """""
def zero_division(a, b):
    c = np.divide(a, b)
    c[c == np.infty] = 0.
    return c




# In[29]:
    
    
    
    
    

    


"""""
The RHS function for sol_ivp
"""""
def RHS_ivp( x,y ,c , d , inL_int , inL_dec , rh  ):
#think about the rho factor again in interaction term!!!!!!!!!!!!!!!!!
    ch= (
        ((c-eye(c.shape[0]))*inL_int)
                         +((d-eye(d.shape[0]))*inL_dec/rh(x))
        )*y
                
    return ch




class Solver:
    def __init__(self,list_particles,R_sun,h_list,phi_0):
        self.list_particles=list_particles
        self.R_sun=R_sun
        self.h_list=h_list
        self.phi=phi_0
    
    
    def list_pa_pdg_id(self):
        l=[]
        for i in self.list_particles:
            l.append(i.pdg_id[0])
        return l
    
    def C(self):
        return interactions(self.list_particles,self.list_pa_pdg_id()).store_data()



    def D(self):
        return decay_class(self.list_particles,self.list_pa_pdg_id()).decay_matrix()


    def L_dec(self):
        return Lambdas.inverse_lengths(self.list_particles).inv_L_dec()


    def L_inte(self):
        return Lambdas.inverse_lengths(self.list_particles).inv_L_inte()#####change A for sun. !!!!!!

    
    def H(self):
        return self.h_list
    
    
    
    
    def X_rho(self,r_ratio,rho_data):
        "Returns X and x2rho both are dictionaries , x2rho is UnivariateSpline function"
        x2rho={}
        X={}
        for h in self.H():
            print(h)
            rhh=rho(self.R_sun,h,r_ratio,rho_data)
            x2rho[h]=rhh.X2rho()
            X[h]=rhh.final_X()
        return X,x2rho
    
    def X2R(self,r_ratio,rho_data):
        return UnivariateSpline(self.X_rho(r_ratio,rho_data),
                                np.append(-r_ratio,r_ratio[::-1])
                                          ,k=1,s=0)
    
    
    
    def solver(self,r_ratio,rho_data):
        sol_ivp={}
        C=self.C()
        D=self.D()
        L_inte=self.L_inte()
        L_dec=self.L_dec()
        X,x2rho=self.X_rho(r_ratio,rho_data)
        
        
        for h in self.H():
            start_time=time.time()
            sol_ivp[h]=odeint(RHS_ivp,self.phi,X[h]
                ,args=(C,D,L_inte,L_dec,x2rho[h]),tfirst=True)#,[X[h][0],X[h][-1]],vectorized=True,
            
            end_time=time.time()
            print(end_time-start_time)
        return sol_ivp


# In[ ]:





# In[40]:


