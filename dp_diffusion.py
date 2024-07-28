import numpy as np

import torch
from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

import os
import json

#from privacy_accountants import privacy_accountant
from src.util import linear_beta_schedule, sample_index, p_losses, save_file, ImageNet
from src.models import UNet


def training_loop(model, optimizer, scheduler, hyperparams, dataloader, save_path):
    device = hyperparams["device"]
    save_and_sample_every = hyperparams["save_and_sample_every"]
    
    losses = []

    itter = 0
    while True:
        for (step, batch) in enumerate(dataloader):
            checkpoint_path = f"{save_path}{itter}"
            
            # break out of loop once we've met termination condition
            if itter >= hyperparams["iterations"]:
                return loss
            optimizer.zero_grad()

            #batch_size = batch["pixel_values"].shape[0]
            batch = batch.to(device)

            # weighted sampling of time index
            t = sample_index(hyperparams).to(torch.int64).to(device)
            # print(t.shape, batch.shape)

            loss = p_losses(model, batch, t, hyperparams)
            losses.append(loss)

            if itter != 0 and itter % save_and_sample_every == 0:
                print("Loss:", loss.item())

                # save losses
                np.savetxt(f"{checkpoint_path}", np.array(losses), delimiter=",")

            loss.backward()
            optimizer.step()
            model.update_parameters(model)

            scheduler.step()
            itter += 1

            # save checkpoints
            if itter % save_and_sample_every == 0:
                file_name = "model.pt"

                save_file({
                    'epoch': iterations,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss},
                    checkpoint_path, 
                    file_name)
                
                # save some samples
 

if __name__ == '__main__':

   # arg parse stuff here
    

    # load and add model parameters

    with open('config\\pretrain_2.json') as f:
        hyperparams = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    private_training = True

    root =  os.getcwd()
    data_dir = os.path.join(root, hyperparams["data_directory"])
    save_path = os.path.join(root, hyperparams["save path"])


    timesteps = hyperparams["timesteps"]
    betas = linear_beta_schedule(timesteps= timesteps)
    alphas = torch.ones_like(betas) - betas
    alphas_cumprod = torch.cumprod(alphas, axis = 0) 
    one_minus_alpha_cumprod = torch.ones_like(alphas_cumprod) - alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    beta_over_one_minus_alpha = betas/torch.sqrt(torch.ones_like(alphas_cumprod) - alphas_cumprod)
    recp_sqrt_alphas_cumprod = 1./torch.sqrt(alphas_cumprod)


    hyperparams.update({
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "one_minus_alpha_cumprod": one_minus_alpha_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "beta_over_one_minus_alpha": beta_over_one_minus_alpha,
        "recp_sqrt_alphas_cumprod": recp_sqrt_alphas_cumprod,
        "device": device,
        "root": root,
        "data_dir": data_dir})

    model = UNet(T = hyperparams["timesteps"], ch = hyperparams["channels"], ch_mult = hyperparams["ch_mults"], 
                 attn = hyperparams["attn"], num_res_blocks= hyperparams["resblocks"], dropout = hyperparams["dropout"])
    
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
    
    print(f"Loaded dataset. length of dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch size"], shuffle=True)

    iterations = hyperparams["iterations"]

    # training 
    training_loop(model, optimizer, scheduler, hyperparams, dataloader, save_path)
    
    itter_path = f"{save_path}step{iterations}"
    file_name = "model.pt"
    save_file({
    'epoch': iterations,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}, 
    itter_path, 
    file_name)