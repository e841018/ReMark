import pickle
import matplotlib.pyplot as plt
import ml, marker

# The `.obs` file is a pickled dict object, containing
# (1) data['obs']: a numpy vector of observations
# (2) data['meta']: an ObsMeta (defined in meta.py) object recording how the observations were made.

print('''
================================================================================
detection stage - reconstruct
================================================================================
''')
with open('../data/20220529detection/0.obs', 'rb') as f:
    data = pickle.load(f)
    meta = data['meta']
    obs = data['obs']
    print(meta)
    print(f'obs: dtype={obs.dtype}, shape={obs.shape}')
model_name = meta.phi_str
phi, decode = ml.load_phi_decode(model_name)
recon = decode(obs)
plt.figure()
plt.imshow(recon, cmap='gray')
plt.title('detection stage - reconstruct')
plt.show()

print('''
================================================================================
identification stage - reconstruct
================================================================================
''')
for M in 64, 128, 256:
    print(f'\n[M = {M}]\n')
    with open(f'../data/20220515identification/marker=0_tilt=0/id_M={M}/0.obs', 'rb') as f:
        data = pickle.load(f)
        meta = data['meta']
        obs = data['obs']
        print(meta)
        print(f'obs: dtype={obs.dtype}, shape={obs.shape}')
    model_name = meta.phi_str
    phi, decode = ml.load_phi_decode(model_name)
    recon = decode(obs)
    plt.figure()
    plt.imshow(recon, cmap='gray')
    plt.title(f'identification stage - reconstruct (M={M})')
    plt.show()

print('''
================================================================================
identification stage - align
================================================================================
''')
model_name = 'model=[cnm]__train=[dcan256_snr=15db_sample=100000]__rep=1'
fit = ml.load_fit(model_name)
aff_mat = fit(recon)
print(f'aligned marker edges with {model_name}')
plt.figure()
plt.imshow(recon, cmap='gray')
marker.add_parallelograms(plt.gca(), aff_mat)
plt.title('identification stage - align')
plt.show()

print('''
================================================================================
identification stage - identify
================================================================================
''')
cells = marker.extract_cells(recon, aff_mat)
m = marker.identify_polarized(cells, dict_size=1000)
print(f'identification stage - identified Marker {m}')
plt.figure()
plt.imshow(cells, cmap='gray')
plt.title('identification stage - identify')
plt.show()
