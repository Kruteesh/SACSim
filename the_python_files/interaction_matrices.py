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
from MCEq.particlemanager import ParticleManager
import MCEq.core 
from MCEq.core import MCEqRun
from MCEq.data import Decays
import mceq_config as config
#import primary model choices
import crflux.models as pm
# Silincing mceq, set to 1 or higher for output

config.muon_helicity_dependence=False
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





























class interactions:
    def __init__(self,particles,particles_pdg_id):
        self.particles=particles
        self.particles_pdg_id=particles_pdg_id
        self.p_loss_dict= pickle.load(open('/home/kruteesh/Desktop/Solar_neutrinos/Energy_loss_1.pkl','rb'))

        self.emca_child=np.array([
             mceq_run.pman[(13,0)],   #mu-          (13)
                mceq_run.pman[(-13,0)],  #mu+          (14)
                mceq_run.pman[-11],      #e+           (15)   
                mceq_run.pman[11],  
            mceq_run.pman[22]#e-           (16)
            ])
        self.emca_parent=np.array([
             mceq_run.pman[(13,0)],   #mu-          (13)
                mceq_run.pman[(-13,0)], #mu+          (14)
                mceq_run.pman[-11],      #e+           (15)   
                mceq_run.pman[11],  
            mceq_run.pman[22]
        ])
        
        
        
        
    def E_loss_sec_par(self):
        """""
        A dictionary containing  the secondary particles wise yields of list_particles with losses
        """""

        p_lep={}
        for child in self.emca_child:
            p_lep[child.name,'ds']=np.empty((len(mceq_run.e_grid),len(mceq_run.e_grid)))
            p_lep[child.name,'dn']=np.empty((len(mceq_run.e_grid),len(mceq_run.e_grid)))
            
            for i, parent in enumerate(self.particles):
                if i==0:
                    if child in parent.hadr_secondaries:
                            
                            p_lep[child.name,'ds']=parent.hadr_yields[child]*parent.inel_cross_section(mbarn=False)/ mceq_run.e_widths /np.pi
                            p_lep[child.name,'dn']=parent.hadr_yields[child]/ mceq_run.e_widths /np.pi
                            
                    else:
                            p_lep[child.name,'ds']=np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))
                            p_lep[child.name,'dn']=np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))
                             
                else:
                    if parent in self.emca_parent:

                        s=str('('+str(child.pdg_id[0])+','+str(parent.pdg_id[0])+')')  
                        if s in self.p_loss_dict.keys(): 
                            #print(p_lep[child.name,"ds"].shape,s)
                            p_lep[child.name,'ds']=np.concatenate((p_lep[child.name,'ds'],(np.nan_to_num(self.p_loss_dict[s])
                                                                  *self.p_loss_dict['xsec '+str(child.pdg_id[0])]/mceq_run.e_widths/np.pi )),axis=1)
                            p_lep[child.name,'dn']=np.concatenate((p_lep[child.name,'dn'],(np.nan_to_num(self.p_loss_dict[s])
                                                                  /mceq_run.e_widths/np.pi )),axis=1)

                            
                            #print(p_lep[child.name,"ds"].shape)
                        else:
                            p_lep[child.name,'ds']=np.concatenate((p_lep[child.name,'ds'],np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))),axis=1)
                            p_lep[child.name,'dn']=np.concatenate((p_lep[child.name,'dn'],np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))),axis=1)
                            
                    else:
                        if child in parent.hadr_secondaries:

                            p_lep[child.name,'ds']=np.concatenate((p_lep[child.name,'ds'],parent.hadr_yields[child]*parent.inel_cross_section(mbarn=False) /mceq_run.e_widths / np.pi),axis=1)
                            p_lep[child.name,'dn']=np.concatenate((p_lep[child.name,'dn'],parent.hadr_yields[child]/mceq_run.e_widths / np.pi),axis=1)
                            
                        else:
                            p_lep[child.name,'ds']=np.concatenate((p_lep[child.name,'ds'],np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))),axis=1)
                            p_lep[child.name,'dn']=np.concatenate((p_lep[child.name,'dn'],np.zeros((len(mceq_run.e_grid),len(mceq_run.e_grid)))),axis=1)
                            
            
            p_lep[child.name,'rescale']=self.p_loss_dict['xsec '+str(child.pdg_id[0])]/np.sum(p_lep[child.name,'ds'].T*mceq_run.e_widths,axis=0)


           
        return p_lep


        
        
        
        
        
        
        
    def resclaing_factors(self):
        """ corrects the differential xsec to correspond to the total

        Parameters
        ----------
        particles : list
            List of the particles of interest

        Returns
        -------
        rescale : dic
            Dictionary of rescaling factors for each type
        """
    
        rescale = {}
        for particle in self.particles:
            if particle not in self.emca_child:
                all_dxsec_tmp = np.array(np.nan_to_num([
                    particle.hadr_yields[child] *
                    particle.inel_cross_section(mbarn=False) /  mceq_run.e_widths / np.pi
                    for child in particle.hadr_secondaries
                ]))
                all_xsec_tmp = np.array([
                        np.sum(dxsec.T * mceq_run.e_widths, axis=1)
                        for dxsec in all_dxsec_tmp
                    ])
                #print(np.sum(all_xsec_tmp.T*mceq_run.e_widths, axis=0))
                total_xsec_tmp = np.sum(all_xsec_tmp, axis=0)
               
                rescaling = np.nan_to_num(particle.inel_cross_section(mbarn=False) / total_xsec_tmp)
                rescale[particle.name] = rescaling
            else:
                rescale[particle.name]=self.E_loss_sec_par()[particle.name,'rescale']
        return rescale    
    def store_data(self):
        """
        check the dimensions

        Parameters
        ----------
        particles : list
            List of the particles of interest

        Returns
        -------
        Matrix 
        """
        rescale = self.resclaing_factors()
        final={}     
        for particle in self.particles:
                if particle not in self.emca_child:
                    Data=np.empty((121,121))
                    i=0
                    for parent_particle in self.particles:
                        if particle in parent_particle.hadr_secondaries:
                            if i==0:
                                Data=np.nan_to_num(parent_particle.hadr_yields[particle]  /
                        mceq_run.e_widths / np.pi * rescale[parent_particle.name])
                                i=1
                            else:
                                Data=np.concatenate((Data,parent_particle.hadr_yields[particle]  /
                        mceq_run.e_widths / np.pi * rescale[parent_particle.name]),axis=1)
                        else:
                            if i!=0:
                                Data=np.concatenate((Data,np.zeros((len(mceq_run.e_grid), len(mceq_run.e_grid)))),axis=1)
                            else:
                                Data=np.zeros((len(mceq_run.e_grid), len(mceq_run.e_grid)))
                                i=1


                    final[particle.name]=Data
                    
                    
                else:
                   rescale_lep=np.array([])
                   for i in range(0,len(self.particles)):
                        rescale_lep=np.append(rescale_lep,self.E_loss_sec_par()[particle.name,'rescale'])
                    
                   final[particle.name]=self.E_loss_sec_par()[particle.name,'dn']*rescale_lep
        Matrix=np.empty((121,121))
        i=0
        for p in final.keys():
               

                if i==0:
                    Matrix=final[p]
                    i=1
                else:
                
                    Matrix=np.concatenate((Matrix,final[p]),axis=0)
        return csr_matrix(Matrix)     
            
            #pickle.dump(Data,open('../data/Data'+str(particle.name)+'.pkl', "wb" ))