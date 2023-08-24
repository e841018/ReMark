import numpy as np
dmd_h, dmd_w = 1080, 1920

class ObsMeta:
    def __init__(self, phi, phi_str, exposure,
        pos=(0, 0), size=(60, 60), shape=(18, 32), comp=True, check=True):
        ''' metadata of observation, including all information needed for reconstruction

        ### parameters:
        *   `phi`: observation matrix, dtype=np.int8, shape=(M, N), entries in {-1, 1}
        *   `phi_str`: str, model name of recon NN or sensing basis
        *   `exposure`: float, exposure time for 1 bipolar observation (2 unipolar observations) in seconds
        *   `pos`: (y, x), upper left corner of the observed rectangle
        *   `size`: (y, x), height and width of a single block
        *   `shape`: (y, x), number of blocks in vertical and horizontal direction
        *   `comp`: should be True, using bipolar masks
        *   `check`: if True, makes sure `pos`, `size`, `shape` are within DMD dimension
        '''
        self.phi = phi
        self.phi_str = phi_str
        self.pos = pos
        self.size = size
        self.shape = shape
        assert comp, comp
        self.comp = comp
        self.expo = exposure
        if check:
            assert pos[0] >= 0, f'pos={pos}, size={size}, shape={shape}'
            assert pos[1] >= 0, f'pos={pos}, size={size}, shape={shape}'
            assert pos[0] + size[0]*shape[0] <= dmd_h, f'pos={pos}, size={size}, shape={shape}'
            assert pos[1] + size[1]*shape[1] <= dmd_w, f'pos={pos}, size={size}, shape={shape}'

    def __str__(self):
        M, N = self.phi.shape
        return f'''ObsMeta:
      sensing basis: {self.phi_str}
  compression ratio: {M} / {N} = {M/N*100:5.2f}%
           position: {self.pos}
         block size: {self.size}
             #block: {self.shape}
      complementary: {self.comp}
      exposure [ms]: {self.expo * 1000}
total exposure [ms]: {self.expo * 1000 * M}
'''
