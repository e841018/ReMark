from synthesize import *

n_draw = 100000
maxR = 1100
max_tilt = 60
res = 32, 32
M = 256
nl_db = 0
test_size = 0.2

# recon model
encoder, decoder = ml.load_enc_dec(f'model=[dcan_M={M}]__train=[clear_identification_sample=100000]__rep={0}')
sample = ml.Sampler(encoder)
decode = ml.Decoder(encoder, decoder)

# contents of validation set
with open('../data/markers_aruco_DICT_4X4_1000.pkl', 'rb') as f:
    markers = pickle.load(f)
n_m = len(markers)
n_valid = int(n_draw * test_size)
assert (n_draw * test_size) % 1 == 0, n_draw
assert n_valid % n_m == 0, n_valid
n_repeat = n_valid // n_m
valid_contents = []
orientation = np.random.randint(4, size=n_valid)
for m, marker in enumerate(markers):
    for i in range(n_repeat):
        valid_contents.append(np.rot90(marker, k=orientation[m * n_repeat + i]))

contents = draw_content(n_draw - n_valid, s=4) + valid_contents
labels = draw_aff_mat(n_draw, res, max_tilt=max_tilt)
bias_specs = draw_bias_spec(n_draw, max_radius=maxR)
observations = []
for i in tqdm.tqdm(range(n_draw)):
    # synthesis
    bias = bias_linear(res, *bias_specs[i])
    image = synth_identification_img(
        res, contents[i],
        labels[i],
        sigma=0.8, bias=bias)
    # noisy sampling
    obs = sample(image) # RMS of signal is about 15 dB
    obs += np.random.normal(scale=10**(nl_db/20), size=obs.shape)
    observations.append(obs)
# reconstruction
recons = decode.batch(observations)
all_data = list(zip(recons, labels, bias_specs, contents))

dataset_name = f'dcan{M}_snr={15-nl_db}db_sample={n_draw}'
dataset_dir = 'dataset/align/'
with open(f'../{dataset_dir}/{dataset_name}.pkl', 'wb') as f:
    pickle.dump(all_data, f)
print(f'Created dataset {dataset_name}.pkl\n    in {dataset_dir}')
