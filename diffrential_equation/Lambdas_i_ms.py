#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
from jupyterthemes import jtplot


# In[2]:


# MCEq Imports
from MCEq.particlemanager import ParticleManager
import MCEq.core 
from MCEq.core import MCEqRun
from MCEq.data import Decays
import mceq_config as config
#import primary model choices
import crflux.models as pm


# In[3]:


# Silincing mceq, set to 1 or higher for output
#config.A_target=1.2
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
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.sparse import *
import time


# In[28]:


class inverse_lengths:
    def __init__(self,list_pa):
        self.particles=list_pa
        self.p_loss_dict= pickle.load(open('/home/kruteesh/Desktop/Solar_neutrinos/Energy_loss_1.pkl','rb'))
        self.emca=np.array([
             mceq_run.pman[(13,0)],   #mu-          (13)
                mceq_run.pman[(-13,0)],  #mu+          (14)
                mceq_run.pman[-11],      #e+           (15)   
                mceq_run.pman[11],  
            mceq_run.pman[22]#e-           (16)
            ])
        
    def inv_L_inte(self):
        """""
        inverse interaction length without atmospheric molecular composition

        Parameter:
        ---------------------
        particles list

        Return:
        ---------------
        inv_L matrix (n,121,121) n=number of particles inclded in particles
        """""
        neutrinos=[mceq_run.pman[12],mceq_run.pman[14]]
        neutrinos_b=[mceq_run.pman[-12],mceq_run.pman[-14]]
        inv_L=np.array([])
        A=1.9465 # average molar mass of sun 
        A_air=14.6567
        N_A=scipy.constants.N_A
        E_nu=mceq_run.e_grid
        #for chaged currents-------
        acnup=5.43e-3 * 1e-36
        acbp=4.59e-3*1e-36
        acnun=1.23e-2*1e-36
        acbn=2.19e-3*1e-36
        bcnup=0.965
        bcbp=0.978
        bcnun=0.929
        bcbn=1.022
        
        #for neutral currents------
        annup=2.48e-3*1e-36
        anbp=1.22e-3*1e-36
        annun=2.83e-2*1e-36
        anbn=1.23e-3*1e-36
        bnnup=0.953
        bnbp=0.989
        bnnun=0.948
        bnbn=0.989
        
        for p in self.particles:
            if p in neutrinos:
                sigma=acnup*E_nu**(bcnup) + acnun*E_nu**(bcnun) + annup*E_nu**(bnnup) + annun*E_nu**(bnnun)
               
            elif p in neutrinos_b:
        
                sigma=acbp*E_nu**(bcbp) + acbn*E_nu**(bcbn) + anbp*E_nu**(bnbp) + anbn*E_nu**(bnbn)
                
            elif p in self.emca:
                sigma=self.p_loss_dict['xsec '+str(p.pdg_id[0])]
            else:
                sigma=p.inel_cross_section(mbarn=False)/(A_air)**(2/3)
            
            inv_L=np.append(inv_L,(N_A*sigma)/A**(1/3))     
        
        return csr_matrix( np.diag(inv_L) ),inv_L# here divide with the A for sun instead of 1.2!!! I dont know about this!!!!!
    
    def inv_L_dec(self):
        """""
        inverse decay length without atmospheric density !!!!need to chcek again forgot!!!!!! 

        Parameter:
        -------------------
        particles list

        Return:
        -------------------
        inv_L matrix(n,121,121) n=number of particles included in particles
        """""
        
        inv_L_dec=np.array([])
        
        for p in self.particles:
            inv_L_dec=np.append(inv_L_dec,p.inverse_decay_length())
        
        
            
    
        return csr_matrix(np.diag(inv_L_dec)),inv_L_dec


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




