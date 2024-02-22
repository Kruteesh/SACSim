#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import MCEq.data 
from MCEq.particlemanager import ParticleManager
import MCEq.core 
from MCEq.core import MCEqRun
from MCEq.data import Decays
import mceq_config as config
#import primary model choices
import crflux.models as pm
# Silincing mceq, set to 1 or higher for output


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



# In[2]:


#list of particles in concern


# In[35]:
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

                                  
class decay_class:
    
    def __init__(self,particles,pdg_id):
        self.particles=particles
        self.particles_pdg_id=pdg_id
        self.handed_leptons_id=[
          str( mceq_run.pman[(13,-1)].pdg_id[0]),
            str(mceq_run.pman[(13,1)].pdg_id[0]),
        str(mceq_run.pman[(-13,-1)].pdg_id[0]),str(mceq_run.pman[(-13,1)].pdg_id[0])]   #mu-l         (9)
            #    mceq_run.pman[(13,-1)],  #mu-r         (10)
            #    mceq_run.pman[(-13,1)],  #mu+l         (11)
            #    mceq_run.pman[(-13,-1)], #mu+r         (12)         
        self.handed_leptons=[mceq_run.pman[(13,-1)],
                                            mceq_run.pman[(13,1)],mceq_run.pman[(-13,1)],mceq_run.pman[(-13,-1)]]
       
       # ]
    def name(self):
        name_pre=np.array([i.name for i in self.particles])       
        name_pre=np.append(name_pre, np.array([i.name for i in self.handed_leptons]))
        
        return name_pre
    """""
    rescaling function gives back rescaling factoru inorder to get rid of the energy and material 
    dependecies of decay matrices 

    Paramters:
    --------------------------
    particles list

    Returns:
    --------------------------
    ratio dict
    """""

    #decay lengths from decay matrices -*-------*-----*-----**------***------
    
    
    def Rescaling(self):
        rescale={}
        decay=[]
        
        for parent_particle in self.particles:

            decay_mat=np.zeros((len(self.particles),len(mceq_run.e_grid),len(mceq_run.e_grid)))

            if parent_particle.is_stable==False:

                    for child_particle in parent_particle.decay_dists:
                        if child_particle.pdg_id[0] in self.particles_pdg_id:
                            if child_particle.name in self.name():
                                
                                if child_particle in self.handed_leptons:
                                   
                                    i=self.particles_pdg_id.index(child_particle.pdg_id[0])
                                    if np.any(decay_mat[i])!=0:
                                       

                                        decay_mat[i]=decay_mat[i]+parent_particle.decay_dists[child_particle]# * self.rescale_dec()[parent_particle.name]                        
                                    else:
                                        #print('why')
                                        decay_mat[i]=(parent_particle.decay_dists[child_particle] )#* self.rescale_dec()[parent_particle.name])

                                else:
                                     i=self.particles_pdg_id.index(child_particle.pdg_id[0])   
                                     decay_mat[i]=(parent_particle.decay_dists[child_particle]) #* self.rescale_dec()[parent_particle.name])
                            
                            
                #print(np.sum(decay_mat,axis=0).shape)
                #print(decay_mat.shape,parent_particle.name)
            t=[]
                
            for i in range(0,len(decay_mat)):
                #print(np.linalg.eig(decay_mat[i])[0])
                t.append(zero_division(np.linalg.eig(decay_mat[i])[0].shape ,np.linalg.eig(decay_mat[i])[0]))
            #print(t)
            tt=np.sum(t,axis=0)
            rescale[parent_particle.name]=np.nan_to_num(zero_division(tt,parent_particle.inverse_decay_length()))          
            
            decay.append(decay_mat)
        return rescale, decay
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    gives us decay matrices in a dictionary with particle name as label for only the list of particles in concern
    Parameters:
    -----------------
    list of parateicles, list of particles pdg id

    Rturns:
    ------------------------
    matrix  
    """     



    def decay_matrix(self):
        decay=self.Rescaling()[1]
        d_part={}
        for i in range(0,len(decay)):
            dd=np.empty((121,121))
            for j in range(0,len(decay[i])):
                if  j==0:
                    dd=decay[i][j]

                else:
                    dd=np.concatenate((dd,decay[i][j]),axis=0)
            d_part[str(i)]=dd
        d_final=np.empty((d_part['0'].shape))

        i=0
        for particle in d_part:
            if i==0:
                d_final=d_part[particle]
                i=1
            else:
                d_final=np.concatenate((d_final,d_part[particle]),axis=1)
    
    
        return csr_matrix(d_final)
# In[ ]:





# In[ ]:





# In[ ]:




