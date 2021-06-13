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
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import data
import h5py


# In[2]:


"""Trial 12 finished with value: 0.03468797908739207 and parameters: 
{'n_layers': 1, 'n_units_0': 249, 'lr': 0.001016956828821507, 'wd': 0.0004497269740382847}. 
Best is trial 12 with value: 0.03468797908739207
"""

################################### INPUT #####################################
# Data parameters
seed         = 4
n_properties = 12

# Training Parameters
batch_size    = 1
learning_rate = 0.001016956828821507
weight_decay  = 0.0004497269740382847

# Architecture parameters
input_size    = 11
n_layers      = 1
out_features  = [249]
f_best_model  = "SMASS_NN_12.pt"


# In[ ]:


#################################### DATA #################################
#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed)

#Create Dataloaders
train_loader = DataLoader(dataset=train_Dataset, 
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset,
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,
                          batch_size=batch_size, shuffle=True)


# In[ ]:


print(test_Dataset.__len__())
print(int(720548*0.1))


# In[ ]:


def neural_net(n_layers, out_features):
    # define container for layers
    layers = []
    
    # Define initial in_features and final output_size
    in_features = 11
    output_size = 1 
    
    for i in range(n_layers):
        # Add to layers container linear layer + activation layer
        
        # Define out_features
        layers.append(nn.Linear(in_features, out_features[i]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Turn in_features to out_features for next layer
        in_features = out_features[i]
        
    # last layer doesnt have activation!
    layers.append(nn.Linear(in_features, output_size))
    
    # return the model
    return nn.Sequential(*layers)

# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Load model
model = neural_net(n_layers, out_features).to(device)
if os.path.exists(f_best_model):
    print("loading model")
    model.load_state_dict(torch.load(f_best_model, map_location=device))
    
from torchsummary import summary
summary(model, (1,11))


# In[ ]:


# Find validation and test loss
print(f_best_model)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)

# Denormalize v_rms predicted 
true_smass = np.zeros((72054, 1), dtype = np.float32)
pred_smass = np.zeros((72054, 1), dtype = np.float32)

# Validation of the model.
model.eval() 
count, loss_valid = 0, 0.0
for input, TRUE in valid_loader:
    input = input.to(device=device)
    TRUE = TRUE.to(device=device)
    output = model(input)
    loss    = criterion(output, TRUE)  
    loss_valid += loss.cpu().detach().numpy()
    count += 1
loss_valid /= count
    
# TEST
i = -1 
count, loss_test = 0, 0.0
for input, TRUE in test_loader:
    i +=1 
    input = input.to(device=device)
    true_smass[i] = TRUE.numpy() # For de-norm
    TRUE = TRUE.to(device=device)
    output  = model(input)
    pred_smass[i] = output.cpu().detach().numpy() # For de-norm
    loss    = criterion(output, TRUE)  
    loss_test += loss.cpu().detach().numpy()
    count += 1
loss_test /= count
    
print('%.4e %.4e'%(loss_valid, loss_test))


# In[ ]:


galaxy_stats_log10 = {
    's_mass'        : [9.255056, 0.7089033],
    'spin'          : [751.5774, 1701.194],
    't_mass'        : [11.058424, 0.7831356],
    'g_mass'        : [8.271089, 3.9434538],
    'bh_mass'       : [5.308265, 2.6688797],
    'vel'           : [276.24274, 228.49133],
    'g_metallicity' : [0.009010918, 0.00847879],
    's_metallicity' : [0.01162953, 0.007478786],
    'radius'        : [35.43549, 35.55443],
    'SFR'           : [0.4310661, 1.8536952],
    'v_disp'        : [67.320076, 37.537144],
    'v_max'         : [131.07162, 68.46787]
}

# Denormalize
mean = galaxy_stats_log10.get('s_mass')[0]
std = galaxy_stats_log10.get('s_mass')[1]
denorm_pred_smass = 10**((pred_smass * std) + mean - 1)
denorm_true_smass = 10**((true_smass * std) + mean - 1)


# In[ ]:


import corr_coef
# Calculate r_squared value with respect to y=x line
r_squared = corr_coef.r_squared(denorm_pred_smass[:,0], denorm_true_smass[:,0])

# Make plot with the r_squared value 
figname = "Predicted_vs_True_Stellar_Mass"
plt.scatter(denorm_pred_smass[:,0], denorm_true_smass[:,0])
plt.xlabel("Predicted Stellar Mass", fontsize=11)
plt.ylabel("True Stellar Mass", fontsize=11)
plt.title(figname, fontsize=11)

# y=x line
min = np.min([np.min(denorm_pred_smass[:,0]), np.min(denorm_true_smass[:,0])])
max = np.max([np.max(denorm_pred_smass[:,0]), np.max(denorm_true_smass[:,0])])
x = np.linspace(min, max, 1000)   
plt.plot(x,x, '-r')

# textbox with r_squared value
textstr = str(r_squared)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(np.median([max,min]), min,textstr, fontsize=12, bbox=props, horizontalalignment="center")

# Save and show figure
plt.savefig(figname)
plt.show()


# In[ ]:


plt.hist(denorm_true_smass[:,0], label="True")
plt.hist(denorm_pred_smass[:,0], label="Predicted")
plt.yscale('log')
plt.legend()


# In[ ]:


# Saliency for test data
saliency_array = np.zeros((72054, input_size), dtype=np.float32)
test_input = np.zeros((72054, input_size), dtype=np.float32)

model.eval()
i = -1
for input, output in test_loader:
    test_input[i] = input.numpy()
    i += 1
    input = input.to(device)
    output = output.to(device)
    
    # Get gradient and send pred through back prop
    input.requires_grad_()
    prediction = model(input)
    prediction.backward()
    
    # Print saliency
    saliency = input.grad.cpu().detach().numpy()
    # print(saliency)
    saliency_array[i] = saliency
    
# take abs value of each column and take average saliency for each property
saliency_avg = np.zeros((1,input_size), dtype=np.float32)
for i in range(input_size):
    saliency_array[:,i] = np.abs(saliency_array[:,i])
    saliency_avg[:,i]   = np.mean(saliency_array[:,i])
    
print(saliency_avg)

properties = ['spin', 'total mass', 'gas mass', 'bh mass',
             'velocity', 'gas metallicity', 'Star Metallicity',
             'radius', 'SFR', 'vel_disp', 'vmax']

plt.figure(figsize=(11,5))
plt.bar(height=saliency_avg.reshape(9,), x=properties)
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: Stellar Mass")
plt.savefig("Saliency_Smass")
plt.show()




