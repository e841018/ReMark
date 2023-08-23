import numpy as np
dmd_h, dmd_w = 1080, 1920

class ObsMeta:
    '''
    metadata of observation, including all information needed for reconstruction
    members:
        phi: observation matrix,
            dtype = np.int8, entries in {1, -1} if comp else {1, 0},
            shape = (M, N)
        other members kept without modification in __init__
    '''
    def __init__(self, phi, phi_str, exposure,
        pos=(0, 0), size=(60, 60), shape=(18, 32), comp=True, check=True):
        '''
        phi: dtype = np.uint8, will be casted to np.int8
        other parameters documented below
        '''
        self.phi = phi.astype(np.int8)
        self.phi_str = phi_str # name of recon NN or sensing basis
        self.pos = pos # upper left corner of the observed rectangle
        self.size = size # height and width of a single block
        self.shape = shape # number of blocks in vertical and horizontal direction
        self.comp = comp # True if complementary pattern used
        self.expo = exposure # total exposure time for 1 observation in seconds
        if check:
            assert pos[0] >= 0, f'pos={pos}, size={size}, shape={shape}'
            assert pos[1] >= 0, f'pos={pos}, size={size}, shape={shape}'
            assert pos[0] + size[0]*shape[0] <= dmd_h, f'pos={pos}, size={size}, shape={shape}'
            assert pos[1] + size[1]*shape[1] <= dmd_w, f'pos={pos}, size={size}, shape={shape}'

    def __str__(self):
        M, N = self.phi.shape
        ret = 'ObsMeta:\n'
        ret += f'{"sensing basis":>15s}: {self.phi_str}\n'
        ret += f'{"comp. ratio":>15s}: {M} / {N} = {M/N*100:5.2f}%\n'
        ret += f'{"position":>15s}: {self.pos}\n'
        ret += f'{"block size":>15s}: {self.size}\n'
        ret += f'{"#block":>15s}: {self.shape}\n'
        ret += f'{"complementary":>15s}: {self.comp}\n'
        ret += f'{"exposure [ms]":>15s}: {self.expo * 1000}\n'
        ret += f'{"theo. time [ms]":>15s}: {self.expo * 1000 * M}\n'
        return ret