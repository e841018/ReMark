# %% imports

import sys, os
import numpy as np, torch, torch.nn as nn, tqdm
import ml.dataset

# %% hyperparameters

rep = int(sys.argv[1]) # repetition

n_epoch = 50

H, W = 32, 32
dim = 32

# %% dataset and model name

dataset_name = 'dcan256_snr=15db_sample=100000'
print(f'training set: {dataset_name}')

model_name = f'model=[cnm]__train=[{dataset_name}]__rep={rep}'
print(f'model name: {model_name}')

# preprocess
all_data, train_data, valid_data = ml.dataset.load_split_dataset(f'align/{dataset_name}.pkl')
train_set = ml.dataset.AlignDataset(train_data)
valid_set = ml.dataset.AlignDataset(valid_data)

# %% model definition

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'torch device: {device}')

model = nn.Sequential(
    nn.Conv2d(1, 32, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(16, 8, 5, padding=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8*H*W, 1024),
    nn.ReLU(),
    nn.Linear(1024, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 6),
).to(device)

# %% loss function

criterion = torch.nn.L1Loss(reduction='sum')

# %% things to do in an epoch

def run_epoch(epoch, dataloader, is_training):
    # set state of model
    if is_training:
        model.train()
    else:
        model.eval()

    # initialize statistics
    loss_epoch = 0. # accumulated loss in this epoch
    errors_epoch = torch.zeros(6, dtype=torch.float32, device=device)
    n_instance = 0

    # iterate over batches
    for batch in dataloader:
        recons, labels = batch

        # send to GPU if available
        recons = recons.to(device)
        labels = labels.to(device)

        # forward propagation
        outputs = model(recons)

        # loss
        loss_batch = criterion(outputs, labels) # sum of loss in this batch

        # clear gradients, backward propagation, and update parameters
        if is_training:
            optimizer.zero_grad()
            (loss_batch / batch_size).backward()
            optimizer.step()

        # update statistics
        labels = labels.detach()
        outputs = outputs.detach()

        loss_epoch += loss_batch.item()
        errors_epoch += (outputs - labels).abs().sum(axis=0)
        n_instance += len(labels)

    if is_training:
        scheduler.step()

    avg_loss = loss_epoch / n_instance
    avg_errors = (errors_epoch / n_instance).cpu().numpy()

    return (avg_loss, *avg_errors)

# %% preparation for training

# mkdir
model_path = f'../model/align/{model_name}/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
    print(f'created directory: {model_path}')

# optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda e: 0.95)

# dataloaders
batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size)

# %% train

def l2(x, y):
    return np.sqrt(x**2 + y**2)

print('-' * 80, flush=True)

trange = tqdm.tqdm(range(1, n_epoch + 1))
for epoch in trange:

    # train
    loss, Cx, Cy, Nx, Ny, Mx, My = run_epoch(epoch, train_loader, is_training=True)
    stats_train = f'train: {loss:5.3f}, C={l2(Cx, Cy)*dim:5.3f}mpx, N={l2(Nx, Ny)*dim:5.3f}mpx, M={l2(Mx, My)*dim:5.3f}mpx'

    # validate
    with torch.no_grad():
        loss, Cx, Cy, Nx, Ny, Mx, My = run_epoch(epoch, valid_loader, is_training=False)
    stats_valid = f'valid: {loss:5.3f}, C={l2(Cx, Cy)*dim:5.3f}mpx, N={l2(Nx, Ny)*dim:5.3f}mpx, M={l2(Mx, My)*dim:5.3f}mpx'

    # display status
    trange.set_postfix_str(f'{stats_train} | {stats_valid}')

# save model after training
torch.save(model, os.path.join(model_path, f'model.pt'))
