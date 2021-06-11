import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
import sys, os, time

# This function reads all the galaxy catalogues from the LH set
def read_data(normalize = True, tensor = True):
    offset = 0
    for i in range(1000):
        catalogue = '/projects/QUIJOTE/CAMELS/Sims/IllustrisTNG/LH_%d/fof_subhalo_tab_033.hdf5'%i
        n_properties = 12
        total = 720548  # Total number of galaxies in all catalogues

        # Open file
        f = h5py.File(catalogue, 'r')

        # Only look at galaxies with > 20 star particles
        star_particles = f['Subhalo/SubhaloLenType'][:,4] # Number of star particles in each galaxy
        np_mask        = np.array([x>20 for x in star_particles]) # Create array mask

        # Get the number of galaxies for this catalogue
        n_galaxies = np.size(f['Subhalo/SubhaloMass'][:][np_mask])

        # Define size for data container
        data = np.zeros((total, n_properties), dtype=np.float32)
        size = offset + n_galaxies

        ################################################# LOAD PROPERTIES ##############################################
        # Stellar Mass [Msun/h]
        data[offset:size,0] = f['Subhalo/SubhaloMassType'][:,4][np_mask]*1e10

        # Spin [(kpc/h)(km/s)] 
        spin_x    = f['Subhalo/SubhaloSpin'][:,0][np_mask]
        spin_y    = f['Subhalo/SubhaloSpin'][:,1][np_mask]
        spin_z    = f['Subhalo/SubhaloSpin'][:,2][np_mask]
        data[offset:size,1] = np.sqrt((spin_x**2) + (spin_y**2) + (spin_z**2))

        # Total Mass [Msun/h]
        data[offset:size,2] = f['Subhalo/SubhaloMass'][:][np_mask]*1e10 

        # Gas Mass 
        data[offset:size,3] = f['Subhalo/SubhaloMassType'][:,0][np_mask]*1e10

        # Black Hole Mass [Msun/h]
        data[offset:size,4] = f['Subhalo/SubhaloBHMass'][:][np_mask]*1e10     

        # Modulus of Velocity [km/s]
        v_x       = f['Subhalo/SubhaloVel'][:,0][np_mask]
        v_y       = f['Subhalo/SubhaloVel'][:,1][np_mask]
        v_z       = f['Subhalo/SubhaloVel'][:,2][np_mask]
        data[offset:size,5] = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))

        # Gas Metallicity 
        data[offset:size,6] = f['Subhalo/SubhaloGasMetallicity'][:][np_mask]

        # Stars Metallicity
        data[offset:size,7] = f['Subhalo/SubhaloStarMetallicity'][:][np_mask]

        # Radius [ckpc/h] *
        data[offset:size,8] = f['Subhalo/SubhaloHalfmassRad'][:][np_mask]

        # Star Formation Rate [Msun/yr] *
        data[offset:size,9] = f['Subhalo/SubhaloSFR'][:][np_mask]

        # Velocity Dispersion [km/s] 
        data[offset:size,10] = f['Subhalo/SubhaloVelDisp'][:][np_mask]

        # V_max [km/s]
        data[offset:size,11] = f['Subhalo/SubhaloVmax'][:][np_mask]

        # Update offset
        offset = size 

    ############################# NORMALIZE DATA ##############################
    # This function normalizes the input data
    def normalize_data(data):
        # Data_shape: (n_samples, n_features)
        n_galaxies = data.shape[0]    # n_samples
        n_properties = data.shape[1]  # n_features

        # Create container for normalized data
        data_norm = np.zeros((n_galaxies, n_properties), dtype=np.float32)

        # Take log10 of stellar mass, total mass, gas mass, and BH mass
        data[:,0] = np.log10(data[:,0]+1)
        data[:,2] = np.log10(data[:,2]+1)
        data[:,3] = np.log10(data[:,3]+1)
        data[:,4] = np.log10(data[:,4]+1)

        for i in range(n_properties):
            mean = np.mean(data[:,i])
            std  = np.std(data[:,i])
            normalized = (data[:,i] - mean)/std
            data_norm[:,i] = normalized

        return(data_norm)

    # Normalize each property
    if normalize == True:
        data = normalize_data(data)

    # Convert to torch tensor
    if tensor == True:
        data = torch.tensor(data, dtype=torch.float)
        
    return data

###################################### Create Datasets ###################################
class make_Dataset(Dataset):
    
    def __init__(self, name, seed):
        
        # Get the data
        galaxy_data = read_data(normalize = True, tensor = True)
        n_properties = galaxy_data.shape[1]
        n_galaxies = galaxy_data.shape[0] 
        
        # shuffle the number (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        
        np.random.seed(seed)
        indexes = np.arange(n_galaxies)
        np.random.shuffle(indexes)
        
        # Divide the dataset into train, valid, and test sets
        if   name=='train':  size, offset = int(n_galaxies*0.8), int(n_galaxies*0.0)
        elif name=='valid':  size, offset = int(n_galaxies*0.1), int(n_galaxies*0.8)
        elif name=='test' :  size, offset = int(n_galaxies*0.1), int(n_galaxies*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, n_properties-1), dtype=torch.float) # Each input has a shape of (9,) (flattened)
        self.output = torch.zeros((size, 1), dtype=torch.float) # Each output has shape of (1,) 
        
        # do a loop over all elements in the dataset
        for i in range(size):
            j = indexes[i+offset]                             # find the index (shuffled)
            self.input[i] = galaxy_data[:,1:n_properties][j]  # load input
            self.output[i] = galaxy_data[:,0][j]              # Load output (stellar_mass)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    
#This function creates datasets for train, valid, test
def create_datasets(seed):
    
    train_Dataset = make_Dataset('train', seed)
    valid_Dataset = make_Dataset('valid', seed)
    test_Dataset  = make_Dataset('test',  seed)
    
    return train_Dataset, valid_Dataset, test_Dataset
