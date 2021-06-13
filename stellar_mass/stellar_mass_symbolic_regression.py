#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from pysr import pysr, best, get_hof
import data
import h5py


# In[2]:


# Load Galaxy Data
data = data.read_data(normalize=True, tensor=False)


# In[4]:


# load input variables

# Load output variable
smass = data[:,0]


# In[ ]:


####################################### INPUT ###################################
property_names   = ['spin', 'total mass', 'gas mass', 'bh mass',
             'velocity', 'gas metallicity', 'Star Metallicity',
             'radius', 'SFR', 'vel_disp', 'vmax']
binary_operators = ["plus", "mult", "sub", "pow", "div"]
unary_operators  = ["exp", "logm", "sin", "cos"]

x = data[:,1:12] # normalized 11 variables
y = smass        # normalized stellar mass
cores = 20       # request more cpu cores on slurm
batch_size = 256

# Initiate Symbolic regression with PYSR
equations = pysr(x, y, niterations=1000, binary_operators= binary_operators,
                 unary_operators= unary_operators, variable_names = property_names, procs = cores,
                batching = True, batchSize=batch_size)


# In[ ]:




