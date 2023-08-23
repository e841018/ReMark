import os, pickle
import numpy as np, torch
from . import embed

def load_split_dataset(name, test_size=0.2, verbose=True):
    '''
    load and split a dataset

    parameters:
        name: file name in dataset/
        test_size: porportion of the data used as test set, default = 0.2
        verbose: set to False to disable printing
    '''
    # load
    with open(os.path.join('../dataset', name), 'rb') as f:
        all_data = pickle.load(f)

    # split
    test_len = int(np.floor(len(all_data) * test_size))
    train_data = all_data[:-test_len]
    valid_data = all_data[-test_len:]
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
        image, label, bias_spec = self.data[index]
        image = image.astype(np.float32) # TODO: is this needed?
        image /= np.linalg.norm(image)
        return image[np.newaxis, :, :], embed.embed(label)
