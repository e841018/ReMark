import numpy as np, torch
from . import embed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# reconstruction NN
################################################################################

def load_enc_dec(model_name):
    encoder, decoder = torch.load(f'../model/recon/{model_name}/model.pt', map_location=device)
    return encoder, decoder

def encoder2phi(encoder):
    phi = encoder[0].weight.detach().type(torch.int8).cpu().numpy()
    M, _, H, W = phi.shape
    N = H * W
    phi = phi.reshape((M, N))
    return phi

def load_phi_reconstruct(model_name):
    encoder, decoder = load_enc_dec(model_name)
    phi = encoder2phi(encoder) # {-1, 1}
    reconstruct = Reconstruct(encoder, decoder)
    return phi, reconstruct

class Observe():
    def __init__(self, phi):
        ''' make observation to a single image, simulating SPI hardware

        ### parameters:
        *   `phi`: encoder module or ndarray
        \t  *   encoder module: the weight of the 0th layer has shape (M, 1, H, W), entries in {-1., 1.}
        \t  *   ndarray: shape=(M, N), dtype=np.int8, entries in {-1, 1}

        ### usage:
            ```
            observe = Observe(encoder)
            observation = observe(img)
            ```
        '''
        if isinstance(phi, torch.nn.Module):
            phi = encoder2phi(phi)
        self.phi = phi

    def __call__(self, img):
        '''
        ### parameters:
        *   `img`: ndarray, shape= (N, ) or (H, W)

        ### returns:
        *   `observation`: ndarray, shape=(M, )
        '''
        M, N = self.phi.shape
        img = img.reshape((N, ))
        return (self.phi @ img).ravel()

class Reconstruct():
    def __init__(self, phi, decoder):
        ''' reconstruct a single image or a list of images

        ### parameters:
        *   `phi`: encoder module or ndarray
        \t  *   encoder module: the weight of the 0th layer has shape (M, 1, H, W), entries in {-1., 1.}
        \t  *   ndarray: shape=(M, N), dtype=np.int8, entries in {-1, 1}
        *   `decoder`: decoder module, takes inputs of shape (B, M) and returns reconstructions of shape (B, 1, H, W)

        ### usage:
            ```
            reconstruct = Reconstruct(encoder, decoder)
            reconstruction = reconstruct(observation)
            reconstructions = reconstruct.batch(observations)
            ```
        '''
        if isinstance(phi, torch.nn.Module):
            phi = encoder2phi(phi)
        self.phi = torch.tensor(phi, dtype=torch.float32, device=device)
        self.decoder = decoder

    def __call__(self, observation):
        '''
        ### parameters:
        *   `observation`: ndarray, shape=(M, )

        ### returns:
        *   `reconstruction`: ndarray, shape=(H, W), dtype=np.float32
        '''
        # prepare data
        M, N = self.phi.shape
        observation = torch.tensor(observation, dtype=torch.float32, device=device).view(1, M)

        # infer
        self.decoder.eval()
        with torch.no_grad():
            recon = self.decoder(observation)[0, 0, :, :].detach()

        # set negative to 0
        recon[recon < 0] = 0

        # scale recon s.t. (phi@recon).norm() == obs.norm()
        scale = observation.norm() / torch.mm(self.phi, recon.reshape(N, 1)).norm()
        recon *= scale

        # move data to cpu
        recon = recon.cpu().numpy().astype(np.float32)
        return recon

    def batch(self, observations):
        '''
        ### parameters:
        *   `observations`: list of ndarray, shape=(M, )

        ### returns:
        *   `reconstructions`: list of ndarray, shape=(H, W), dtype=np.float32
        '''
        # prepare data
        M, N = self.phi.shape
        dataloader = torch.utils.data.DataLoader(
            dataset=np.array(observations).astype(np.float32),
            batch_size=256)

        # infer
        self.decoder.eval()
        reconstructions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                recons = self.decoder(batch)[:, 0, :, :].detach()

                # set negative to 0
                recons[recons < 0] = 0

                # scale recons s.t. (phi@recon).norm() == obs.norm()
                scales = batch.norm(dim=1) / torch.matmul(self.phi, recons.reshape(-1, N, 1)).norm(dim=(1, 2))
                recons *= scales.view(-1, 1, 1)

                # move data to cpu
                recons = recons.cpu().numpy().astype(np.float32)
                reconstructions += [im for im in recons]

        return reconstructions

################################################################################
# alignment NN
################################################################################

def load_fit(model_name):
    model = torch.load(f'../model/align/{model_name}/model.pt', map_location=device)
    fit = Fit(model)
    return fit

class Fit():
    def __init__(self, model):
        ''' fit a single image or a list of images

        ### parameters:
        *   `model`: alignment module, takes inputs of shape (B, 1, H, W) and returns alignment of shape (B, 6)

        ### usage:
            ```
            fit = Fit(model)
            aff_mat = fit(recon)
            aff_mats = fit.batch(recons)
            ```
        '''
        self.model = model

    def __call__(self, recon):
        '''
        ### parameters:
        *   `recon`: ndarray, shape=(H, W)

        ### returns:
        *   `aff_mat`: as defined in embed.cnm_embed(), unified
        '''
        # prepare data
        H, W = recon.shape
        recon /= np.linalg.norm(recon)
        recon = torch.tensor(recon, dtype=torch.float32, device=device).view(1, 1, H, W)

        # infer
        self.model.eval()
        with torch.no_grad():
            output = self.model(recon)[0].detach().cpu().numpy().astype(np.float32)
        aff_mat = embed.extract(output)

        return aff_mat

    def batch(self, recons):
        '''
        ### parameters:
        *   `recons`: list of ndarray, shape=(H, W)

        ### returns:
        *   `aff_mats`: list of aff_mat, as defined in embed.cnm_embed(), unified
        '''
        # prepare data
        recons = [recon / np.linalg.norm(recon) for recon in recons]
        dataloader = torch.utils.data.DataLoader(
            dataset=np.array(recons).astype(np.float32),
            batch_size=256)

        # infer
        self.model.eval()
        aff_mats = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[:, None, :, :]
                batch = batch.to(device)
                outputs = self.model(batch).detach().cpu().numpy().astype(np.float32)
                aff_mats += [embed.extract(output) for output in outputs]

        return aff_mats
