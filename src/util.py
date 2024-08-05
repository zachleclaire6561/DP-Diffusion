# source: [for the sampling stuff]

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

from torchvision.transforms import Compose, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, Normalize

import os
import pickle
import numpy as np

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Weighted Uniform Sampling
def sample_index(sample_count, hyper_parameters):
    weights = hyper_parameters["index sampling weights"]
    index_ranges = hyper_parameters["index ranges"]

    indices = np.array([np.random.randint(*index_ranges[i], size = sample_count) for i in range(len(weights))]).T
    group_choice = np.random.choice(a = np.arange(len(weights)), p = weights, size = sample_count)
    indices = indices[np.arange(len(group_choice)), group_choice]
    return torch.from_numpy(indices)


def p_losses(denoise_model, batch, t, hyper_parameters):

    noise = torch.rand_like(batch, dtype=torch.float32)
    x_noisy = q_sample(batch=batch, t=t, hyper_parameters=hyper_parameters)
    predicted_noise = denoise_model(x_noisy, t)

    loss_norm = hyper_parameters["loss_norm"]
    if loss_norm == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_norm == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_norm == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)

    return loss


# forward diffusion step
def q_sample(batch, t, hyper_parameters):

    device = hyper_parameters['device']
    coef_shape = (len(t), 1,1,1)
    sqrt_alphas_cumprod = hyper_parameters["sqrt_alphas_cumprod"]
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    one_minus_alpha_cumprod = hyper_parameters["one_minus_alpha_cumprod"]
    one_minus_alpha_cumprod = one_minus_alpha_cumprod.to(device)
    noise = torch.rand_like(batch, dtype = torch.float32, device = device)

    #print(sqrt_alphas_cumprod.device, batch.device)
    x = sqrt_alphas_cumprod[t].reshape(coef_shape) * batch
    #print(one_minus_alpha_cumprod.device, t.device, noise.device)
    x += one_minus_alpha_cumprod[t].reshape(coef_shape) * noise
    return x


@torch.no_grad()
def p_sample(model, batch, t, hyper_parameters):

    coef_shape = (len(t), 1,1,1)
    betas = hyper_parameters["betas"]
    recp_sqrt_alphas_cumprod = hyper_parameters["recp_sqrt_alphas_cumprod"]
    beta_over_one_minus_alpha = hyper_parameters["beta_over_one_minus_alpha"]
    
    noise = torch.rand_like(batch)

    model_mean = recp_sqrt_alphas_cumprod[t].reshape(coef_shape) * (batch - beta_over_one_minus_alpha[t].reshape(coef_shape) * model(x, t))
    
    return model_mean + torch.sqrt(betas[t]).reshape(coef_shape) * noise 

@torch.no_grad()
def sample(model, hyper_parameters, sample_count = 10):
    
    sample_shape = (sample_count, hyper_parameters["image_channels"], hyper_parameters["image_size"], hyper_parameters["image_size"])
    device = hyper_parameters["device"]
    img = torch.randn(sample_shape, device = device)
    imgs = []

    for i in reversed(range(0, hyper_parameters["timesteps"])):
        img = p_sample(model, img, torch.full(), i, hyper_parameters)
        imgs.append(img.cpu().numpy())

    return imgs

##
## Data Sampling
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
        #self.transform = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
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
            

    def __len__(self):
            return self.samples.shape[0]

    def __getitem__(self, idx):
        return (2*self.samples[idx].to(torch.float32)/255)-1



def save_file(object, path, file_name):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(object, f"{path}/{file_name}")




# Source: Unlocking High-Accuracy Differentially Private Image Classification through Scale
# Page: 30-31 for the various datasets

def get_augmult_transform(augmult = 0, image_shape = (32,32), random_crop = True, random_flip = True, random_color = False):
    transform_list = []
    
    for _ in range(augmult):
        if random_crop:
            # this needs to be fixed to align with paper's augmult procedure
            transform_list.append(RandomCrop(image_shape, pad_if_needed=True))
        if random_flip:
            transform_list.append(RandomVerticalFlip(p = 0.5))
            transform_list.append(RandomHorizontalFlip(p = 0.5))
        #if random_color:
        #    transform_list.append()
    

    transform_list.append(Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        
    return Compose(transform_list)

