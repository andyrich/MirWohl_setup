#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flopy
import os
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geopandas as gpd
import basic
import contextily as ctx
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import os
import matplotlib.pyplot as plt

import warnings


# # create drains wherever SWR, SFR or GHB are not located

# In[2]:


import os
os.getcwd()


# In[209]:


m = basic.load_model()
m.free_format_input = True


# In[ ]:





# In[5]:





# In[ ]:





# In[112]:


pond_grid = gpd.read_file('ponds/ponds.geojson')
pond_grid = pond_grid.query("name=='Two' or name=='Three'")
pond_grid.plot('name')
pond = np.zeros((m.nrow, m.ncol), dtype = bool)

pond[pond_grid.row-1, pond_grid.column-1] = True
plt.figure()
plt.imshow(pond)


# In[ ]:





# In[113]:


ghb = m.ghb.stress_period_data
ghbdf = pd.DataFrame(ghb[0])

ghb_ar = np.zeros((m.nrow, m.ncol), dtype = bool)

ghb_ar[ghbdf.i, ghbdf.j] = True

plt.imshow(ghb_ar)


# In[114]:


# get stream thalwegs to use as drain reach

swr = gpd.read_file('GIS/nhd_hr_demo_sfr_cells.shp')

swr_ar = np.zeros((m.nrow, m.ncol), dtype = bool)

swr_ar[swr.i, swr.j] = True

plt.imshow(swr_ar)


# In[115]:


top = m.dis.top.array
ibnd = m.bas6.ibound.array[0]
ibnd = ibnd ==0


# In[ ]:





# In[175]:


# made df with above
mask = np.stack([ibnd, swr_ar, pond])
mask = mask.any(axis = 0)
i, j = np.indices(mask.shape)
df  = np.hstack([top.reshape((-1,1)), mask.reshape((-1,1)), i.reshape((-1,1)), j.reshape((-1,1))])

df  = pd.DataFrame(df, columns = ['modeltop', 'mask', 'i', 'j'])

df = df.loc[df.loc[:,'mask'] ==0,:]
df.loc[:,'k'] = 0
df.loc[:,'cond'] = 1/84600
stress_period = df.loc[:,['k', 'i' , 'j', 'modeltop', 'cond']]
stress_period.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[214]:


sp = flopy.modflow.ModflowDrn.get_empty(stress_period.shape[0])
sp['i'] = stress_period.loc[:,'i']
sp['j'] = stress_period.loc[:,'j']
sp['elev'] = stress_period.loc[:,'modeltop']
sp['cond'] = stress_period.loc[:,'cond'].astype('<f4')
sp['cond'] = 1/86400
sp


# In[215]:


allsp = {x:(sp if x==0 else -1)  for x  in range(m.dis.nper-1)}
allsp


# In[ ]:





# In[217]:


drn = flopy.modflow.mfdrn.ModflowDrn(m, stress_period_data= allsp)


# In[218]:


drn.write_file()
drn.plot(kper = 0, mflay  = 0, ibnd = 'r')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




