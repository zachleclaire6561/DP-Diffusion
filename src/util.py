import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Weighted Uniform Sampling
def sample_index(hyper_parameters):
    weights = hyper_parameters["index sampling weights"]
    index_ranges = hyper_parameters["index ranges"]

    indices = np.array([np.random.randint(*index_ranges[i], size = hyper_parameters["batch size"]) for i in range(len(weights))]).T
    group_choice = np.random.choice(a = np.arange(len(weights)), p = weights, size = hyper_parameters["batch size"])
    indices = indices[np.arange(len(group_choice)), group_choice]
    return torch.from_numpy(indices).to(hyper_parameters["device"])


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu().to(torch.int64))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.rand(x_start.shape)

    x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod = sqrt_alphas_cumprod, \
                       sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# forward diffusion step
def q_sample(x_start, t, sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    print(sqrt_alphas_cumprod_t.shape, x_start.shape, sqrt_one_minus_alphas_cumprod_t.shape, noise.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index, hyper_parameters):
    betas_t = extract(hyper_parameters["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        hyper_parameters["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = extract(hyper_parameters["sqrt_recip_alphas"], t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(hyper_parameters["posterior_variance"], t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, hyper_parameters):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, hyper_parameters["timesteps"])), desc='sampling loop time step', total=hyper_parameters["timesteps"]):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, hyper_parameters)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, hyper_parameters, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), hyper_parameters = hyper_parameters)

##
##
##

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_npy(file):
    with open(file, 'rb') as f:
        arr = np.load(f)
    return arr

class ImageNet(Dataset):
    def __init__(self, data_dir):
        self.samples = None
        for entry in os.listdir(data_dir):
            batch_file = os.path.join(data_dir, entry)
            datafile = unpickle(batch_file)["data"]
            #datafile = unpickle(batch_file)
            data_set_batch = torch.from_numpy(datafile)
            print(f"loaded: {entry}")

            # reshape
            data_set_batch = data_set_batch.view(-1, 3, 32, 32)
            if self.samples == None:
                self.samples = data_set_batch
            else:
                torch.cat((self.samples, data_set_batch), 0)
            break

    def __len__(self):
            return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]
    
