'''
TODO: 
1. linear beta schedule - are the variances correct?

'''
import numpy as np

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision import transforms

from pathlib import Path
import os

#from privacy_accountants import privacy_accountant
from src.util import linear_beta_schedule, sample_index, p_losses, ImageNet
from src.models import UNet
from src.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer


def training_loop(model, optimizer, scheduler, hyperparams, dataloader, save_path):
    
    trainer = GaussianDiffusionTrainer(
                model, hyperparams["beta_1"], hyperparams["beta_T"], hyperparams["timesteps"]).to(device)
    itter = 0
    while True:
        for (step, batch) in enumerate(dataloader):
            if itter >= hyperparams["iterations"]:
                return loss
            optimizer.zero_grad()

            #batch_size = batch["pixel_values"].shape[0]
            batch = batch.to(device)

            # weighted sampling of time index
            t = sample_index(hyperparams).to(torch.int64)
            print(t.shape, batch.shape)

            loss = trainer(batch, t)

            if step % 100 == 0:
                print("Loss:", loss.item())
                
            loss.backward()
            optimizer.step()
            model.update_parameters()

            scheduler.step()
            itter += 1
            # save checkpoints
            if itter != 0 and itter % save_and_sample_every == 0:
                checkpoint_path = f"{save_path}{itter}/model.pt"

                torch.save({
                            'epoch': itter,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, checkpoint_path)


if __name__ == '__main__':

   # arg parse stuff here
    
    timesteps = 1000
    betas = linear_beta_schedule(timesteps= timesteps)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    save_and_sample_every = 1000

    private_training = True

    root =  os.getcwd()
    data_dir = os.path.join(root, "DP-DDPM\\data\\Imagenet32_train")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hyperparams = {
        "beta_1": 0.0001, 
        "beta_T": 0.02,
        "timesteps": 1000,
        "batch size": 100, #1024 #4096, 
        "iterations": 2, #200000,
        "image_size":  32,
        "channels": 192,
        "resblock_per_resolution": 2,
        "ch_mults": (1, 2, 2, 2),
        "is_attn": [False, True, False, False], # Original DDPM paper used them at all layers I think
        "aug mult": 0,
        "learning rate": 1e-3, #5e-4,
        "EMA": 0.999,
        "index ranges": [[0,200], [200,800], [800,1000]],
        "index sampling weights": [0.05, 0.9, 0.05],
        "timesteps": timesteps,
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "max_clip": 1e-3, 
        "epsilon threshold": None, 
        "noise variance": None, 
        "dropout": 0, #  
        "device": device, 
        "PATH": '\checkpoints\\Imagenet-Pretrain\\',
        "root": root,
        "data_dir": data_dir,
        "split": 'train'}

    model = UNet(T = hyperparams["timesteps"], ch = hyperparams["channels"], ch_mult = hyperparams["ch_mults"], attn = hyperparams["is_attn"], num_res_blocks = hyperparams["resblock_per_resolution"], dropout = hyperparams["dropout"])

    model.to(device)

    # EMA model weights
    model = AveragedModel(model)

    # TODO: load pretrained model here for other examples
    optimizer = Adam(model.parameters(), lr=hyperparams["learning rate"])
    scheduler = LinearLR(optimizer, start_factor=0.000001, total_iters=5000)

    print(f"Loaded model. Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # load the dataset
    split = hyperparams["split"]
    dataset = ImageNet(data_dir)
    
    print(f"length of dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch size"], shuffle=True)

    save_path = hyperparams["PATH"]
    iterations = hyperparams["iterations"]
    # training 
    training_loop(model, optimizer, scheduler, hyperparams, dataloader, save_path)
    
    torch.save({
    'epoch': iterations,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()},
                    path = f"{save_path}{iterations}/model.pt"
)
