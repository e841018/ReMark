from synthesize import *

n_draw = 100000
res = 32, 32
M = 256
nl_db = 0
train_prop = 0.8

# recon model
encoder, decoder = ml.load_enc_dec(f'model=[dcan_M={M}]__train=[clear_identification_sample=100000]__rep={0}')
sample = ml.Sampler(encoder)
decode = ml.Decoder(encoder, decoder)

# contents of validation set
with open('../data/markers_aruco_DICT_4X4_1000.pkl', 'rb') as f:
    markers = pickle.load(f)
n_marker = len(markers)
assert (n_draw * train_prop) % 1 == 0, n_draw
n_train = int(n_draw * train_prop)
n_valid = n_draw - n_train
assert n_valid % n_marker == 0, n_valid
n_repeat = n_valid // n_marker
valid_contents = []
directions = np.random.randint(4, size=n_valid)
for m, marker in enumerate(markers):
    for i in range(n_repeat):
        valid_contents.append(np.rot90(marker, k=directions[m * n_repeat + i]))

contents = draw_content(n_draw - n_valid) + valid_contents
aff_mats = draw_aff_mat(n_draw, res)
bias_specs = draw_bias_spec(n_draw)
observations = []
for i in tqdm.tqdm(range(n_draw)):
    # synthesis
    img = synth_identification_img(
        res,
        contents[i],
        aff_mats[i],
        bias=render_bias(res, *bias_specs[i]))
    # noisy sampling
    obs = sample(img) # RMS of signal is about 15 dB
    obs += np.random.normal(scale=10**(nl_db/20), size=obs.shape)
    observations.append(obs)
# reconstruction
recons = decode.batch(observations)
all_data = list(zip(recons, aff_mats, bias_specs, contents))

dataset_name = f'dcan{M}_snr={15-nl_db}db_sample={n_draw}'
dataset_dir = 'dataset/align/'
with open(f'../{dataset_dir}/{dataset_name}.pkl', 'wb') as f:
    pickle.dump(all_data, f)
print(f'created dataset {dataset_name}.pkl\n    in {dataset_dir}')
