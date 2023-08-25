import os, pickle
import numpy as np, torch
from . import embed

def load_split_dataset(name, train_prop=0.8, verbose=True):
    ''' load and split a dataset

    ### parameters:
    *   `name`: file name in dataset/
    *   `train_prop`: proportion of data used as training set
    *   `verbose`: set to False to disable printing

    ### returns:
    *   `all_data`, `train_data`, `valid_data`: list of object
    '''
    # load
    with open(os.path.join('../dataset', name), 'rb') as f:
        all_data = pickle.load(f)

    # split
    n_train = int(len(all_data) * train_prop)
    train_data = all_data[:n_train]
    valid_data = all_data[n_train:]
    if verbose:
        print(f'total:          {len(all_data)}')
        print(f'training set:   {len(train_data)}')
        print(f'validation set: {len(valid_data)}')

    return all_data, train_data, valid_data

class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, data, normalize=True):
        self.data = data
        self.len = len(data)
        self.normalize = normalize
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        image = self.data[index][0]
        if self.normalize:
            image /= np.linalg.norm(image)
        return image[np.newaxis, :, :] # the 0th dimension is channel

class AlignDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        image, aff_mat, bias_spec, content = self.data[index]
        image /= np.linalg.norm(image)
        return image[np.newaxis, :, :], embed.embed(aff_mat)
