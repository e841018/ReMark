# %% imports
import pickle, matplotlib.pyplot as plt
import ml, marker

# The `.obs` file is a pickled dict object, containing a numpy vector of
# observations and a ObsMeta (defined in meta.py) object recording how the
# observations were made. 

# %% detection stage - reconstruct
with open('../data/20220529detection/0.obs', 'rb') as f:
    data = pickle.load(f)
    meta = data['meta']
    obs = data['obs']
    print(meta)
    print(f'obs: dtype={obs.dtype}, shape={obs.shape}')
model_name = meta.phi_str
phi, decode = ml.load_phi_decode(model_name)
recon = decode(obs)
plt.imshow(recon, cmap='gray')
plt.title('detection stage - reconstruct')

# %% identification stage - reconstruct
M = 256
with open(f'../data/20220515identification/marker=0_tilt=0/id_M={M}/0.obs', 'rb') as f:
    data = pickle.load(f)
    meta = data['meta']
    obs = data['obs']
    print(meta)
    print(f'obs: dtype={obs.dtype}, shape={obs.shape}')
model_name = meta.phi_str
phi, decode = ml.load_phi_decode(model_name)
recon = decode(obs)
plt.imshow(recon, cmap='gray')
plt.title('identification stage - reconstruct')

# %% identification stage - align
fit = ml.load_fit('model=[cnm]__train=[dcan256_tilt=60_snr=15db_sample=100000]__rep=1')
aff_mat = fit(recon)
plt.imshow(recon, cmap='gray')
marker.add_parallelograms(plt.gca(), aff_mat)
plt.title('identification stage - align')

# %% identification stage - identify
cells = marker.extract_cells(recon, aff_mat)
m = marker.identify_polarized(cells, dict_size=1000)
print(f'identification stage - identified Marker {m}')

# %%
