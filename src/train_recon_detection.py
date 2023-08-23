# %% imports

import os, sys
import numpy as np, torch, torch.nn as nn, tqdm
from ml.dataset import load_split_dataset, ReconDataset
from ml.model_parts import GaussianNoise, Normalize, Reshape

# %% hyperparameters

rep = int(sys.argv[1]) # repetition

nl_db = -7 # noise level expressed in dB, relative to rms=1

# loss = MSE + alpha * regularization
reg_alpha_on = 1.0
n_epoch = 30
n_epoch_before_reg = 10
n_epoch_reg = 5
def reg_alpha(epoch):
    '''
    parameters:
        epoch: 1-indexed
    '''
    if epoch <= n_epoch_before_reg:
        return 0.
    if epoch <= n_epoch_before_reg + n_epoch_reg:
        return reg_alpha_on
    return 0.

H, W = 18, 32
comp_ratio = 0.125
N = H * W
M = int(np.floor(N*comp_ratio))

# %% dataset and model name

dataset_name = f'clear_detection_sample={100000}'
print(f'training set: {dataset_name}')

model_name = f'model=[dcan_M={M}]__train=[{dataset_name}]__rep={rep}'
print(f'model name: {model_name}')

# preprocess
all_data, train_data, valid_data = load_split_dataset(f'recon_detection/{dataset_name}.pkl')
train_set = ReconDataset(train_data, normalize=False)
valid_set = ReconDataset(valid_data, normalize=False)

# %% model definition

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'torch device: {device}')

encoder = nn.Sequential(
    nn.Conv2d(1, M, (H, W), padding=0, bias=False),
    nn.Flatten(),
).to(device)

gaussian_noise = GaussianNoise(nl=10**(nl_db/20))

decoder = nn.Sequential(
    Normalize(),
    nn.Linear(M, N, bias=False),
    Reshape(1, H, W),
    nn.Conv2d(1, 64, 9, padding=4),
    nn.LeakyReLU(),
    nn.Conv2d(64, 32, 1, padding=0),
    nn.LeakyReLU(),
    nn.Conv2d(32, 1, 5, padding=2),
    nn.LeakyReLU(),
).to(device)

# %% loss function

def mse_loss(output, label):
    return (output - label).pow(2).mean(dim=(1, 2, 3)).sum()

def binary_reg(weight):
    return ((weight - 1).pow(2) * (weight + 1).pow(2)).mean()

# %% weight statistics

def weight_max_dev(weight):
    '''
    parameters:
        weight: detached tensor
    '''
    return torch.max(torch.min((weight - 1).abs(), (weight + 1).abs())) # in newer versions, torch.min -> torch.minimum

def binarize(weight):
    '''
    parameters:
        weight: detached tensor
    '''
    return (weight > 0).type(torch.float32) * 2 - 1

# %% things to do in an epoch

def run_epoch(epoch, dataloader, is_training):
    alpha = reg_alpha(epoch)

    # set state of model
    if is_training:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    # initialize statistics
    loss_epoch = 0. # accumulated loss in this epoch
    reg_epoch = 0. # accumulated binary regularization loss
    num_instance = 0
    num_flip = 0

    # iterate over batches
    for batch in dataloader:
        images = batch

        # send to GPU if available
        images = images.to(device)

        # forward propagation
        observations = encoder(images)
        observations = gaussian_noise(observations)
        images_recon = decoder(observations)

        # loss and regularization
        loss_batch = mse_loss(images_recon, images) # sum of loss in this batch
        reg_batch = torch.tensor(0.) if alpha == 0. else len(images) * alpha * binary_reg(encoder[0].weight)

        # clear gradients, backward propagation, and update parameters
        if is_training:
            weight_old = binarize(encoder[0].weight.detach()).type(torch.int64)
            optimizer.zero_grad()
            ((loss_batch + reg_batch) / batch_size).backward()
            optimizer.step()
            weight_new = binarize(encoder[0].weight.detach()).type(torch.int64)
            n_flip = (weight_new - weight_old).abs().sum().item() // 2
        else:
            n_flip = 0

        # update statistics
        loss_epoch += loss_batch.item()
        reg_epoch += reg_batch.item()
        num_instance += len(images)
        num_flip += n_flip

    if is_training:
        scheduler.step()

    avg_loss = loss_epoch / num_instance
    avg_reg = reg_epoch / num_instance
    
    return avg_loss, avg_reg, num_flip
    
# %% preparation for training

# mkdir
model_path = f'../model/recon/{model_name}/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
    print(f'created directory: {model_path}')

# optimizer and scheduler
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': 2e-3},
    {'params': decoder.parameters(), 'lr': 1e-3}])
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda e: 0.95)

# dataloaders
batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size)

# %% train

print('-' * 80, flush=True)

trange = tqdm.tqdm(range(1, n_epoch + 1))
for epoch in trange:

    # train
    loss, reg, num_flip = run_epoch(epoch, train_loader, is_training=True)
    stats_train = f'train: {loss:5.3f}, reg: {reg:5.3f}, flips: {num_flip}'

    # validate
    with torch.no_grad():
        loss, reg, num_flip = run_epoch(epoch, valid_loader, is_training=False)
    stats_valid = f'valid: {loss:5.3f}'

    # display status
    max_dev = weight_max_dev(encoder[0].weight.detach())
    stats_reg = f'max_dev:{max_dev:5.3f}'
    trange.set_postfix_str(f'{stats_train} | {stats_valid} | {stats_reg}')
    
    # binarize and fix encoder after `n_epoch_before_reg + n_epoch_reg` epochs
    if epoch == n_epoch_before_reg + n_epoch_reg:
        encoder[0].weight = torch.nn.Parameter(binarize(encoder[0].weight.detach()))
        encoder[0].weight.requires_grad = False
    
# save model after training
torch.save((encoder, decoder), os.path.join(model_path, f'model.pt'))
