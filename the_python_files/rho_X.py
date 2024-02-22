#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from jupyterthemes import jtplot
jtplot.style(theme="grade3", context="notebook", ticks=True, grid=False)


# In[48]:




# In[49]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# In[ ]:



def rho_fun_data_out(rr,rh):
    
    return UnivariateSpline(rr,rh[0:len(rr)],k=1,s=0,ext=0)

# In[177]:


class rho():
    def __init__(self,R_sun,H,little_r,density):
        self.h=H
        self.R=R_sun
        self.r_rat=np.array(little_r)
        self.rh=np.array(density)
       
    def j(self):
        return find_nearest(self.r_rat*self.R,self.h)   
    def rho(self):
        return self.rh[:self.j()][::-1]
    def r_ratio(self):
        return self.r_rat[:self.j()][::-1]
    "Now the r_ratio and rho goes from the nearest point from centre to the outer surface!!"
    
    def L(self):
        """
        this is from nearest point to cetre to the surface
        """
        return  self.R * np.sin(np.arccos(self.h / self.R))
    
    def l(self):
        
        return np.linspace(0,self.L(),len(self.r_ratio()))
    
    def little_r(self):
      
        return np.sqrt(self.h**2+self.l()**2)
    
    def r2rho(self):
        return UnivariateSpline(self.little_r(),self.rho(),k=1,s=0)(self.little_r())
    
    def r2l(self):
        return np.sqrt(self.little_r()**2-self.h()**2)
    
    def l2rho(self):
        return UnivariateSpline(self.l(),self.rho(),k=1,s=0)

    
    
    
    def X_tmp(self):
        """""Since upto thispoint everything was from the nearest point from the centre to the surface 
        the final one has to be made"""""
        return np.array([self.l2rho().integral(0., l_step) for l_step in self.l()])
    
    
    
    
    def X_tmp2rho(self):
        return UnivariateSpline(self.X_tmp(),self.rho(),k=1,s=0)
    
    
    
    def final_X(self):
        t=np.append(self.X_tmp()[len(self.X_tmp())-1]-self.X_tmp()[::-1],
                    self.X_tmp()[len(self.X_tmp())-1]+self.X_tmp()[1:]
                    )
        return t
    
    
    def X2R(self):
        return UnivariateSpline(self.final_X(),
                                np.append(-self.r_ratio()[::-1],self.r_ratio()[1:])
                                          ,k=1,s=0)(self.final_X())
    
    def final_rho(self):
        return np.append(self.rho()[::-1],self.rho()[1:])
        
    def X2rho(self):
        "it doesnt work this way need to think anout it diffrently"
        return UnivariateSpline(self.final_X(),self.final_rho(),k=1,s=0)
        
# In[184]:





# In[185]:





# In[188]:





# In[190]:





# In[ ]:





# In[ ]:





# In[7]:





# In[8]:



# In[9]:





# In[ ]:




