from models.vae import VAE
from models.kvae import KVAE

import torch
import os 

from utils import load_model
from model_config import * 

import matplotlib.pyplot as plt

cwd = os.getcwd()
sep = os.sep

tasks = ["generation"]  # Define the tasks you want to run

if __name__ == "__main__":
    
    models_trained = {}
    n = 5 #generate n samples
    
    rows = int(n // 2) 
    cols = 2
    if n % 2 != 0:
        rows += 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    for dataset_name, files in datapaths.items():
        models_trained[dataset_name] = {}
        for task in tasks:            

            fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
            window_size = files["window_size"]
            models = {
                    "vae": VAE(window_size, hidden_dims["linear_var"], conv=False, beta=beta, act=act),
                    "kvae": KVAE(window_size, hidden_dims["linear_kan_var"], conv=False, window_size = 4096, beta=beta, base_activation=act_kan_linear),
                    "cvae": VAE(1, hidden_dims["conv_var"], conv=True, beta=beta, width = 2048, kernel_sizes=kernel_sizes["conv_var"], strides=strides["conv_var"], paddings=paddings["conv_var"], act=act_conv, batch_size=batch_size_normal, window_size=window_size),                             
                    #"kcvae": KVAE(1, hidden_dims["conv_var"], width = 2048, conv=True, kernel_sizes=kernel_sizes["conv_var"], window_size = 4096, beta=beta, strides=strides["conv_var"], paddings=paddings["conv_var"], base_activation=act_kan)
            }

            models_trained[dataset_name][task] = {}
            
            # Load the models
            models_trained[dataset_name][task]["VAE"] = load_model(models["vae"], models_path + sep + f"vae_{dataset_name}_{task}.pt", variational=True).to(device)
            models_trained[dataset_name][task]["CVAE"] = load_model(models["cvae"], models_path + sep + f"cvae_{dataset_name}_{task}.pt", variational=True).to(device)
            models_trained[dataset_name][task]["KVAE"] = load_model(models["kvae"], models_path + sep + f"kvae_{dataset_name}_{task}.pt", variational=True).to(device)
            #models_trained[dataset_name][task]["KCVAE"] = load_model(models["kcvae"], models_path + sep + f"kcvae_{dataset_name}_{task}.pt", variational=True).to(device)

            # Generate samples
            with torch.no_grad():
                models_trained[dataset_name][task]["VAE"].eval()
                models_trained[dataset_name][task]["CVAE"].eval()
                models_trained[dataset_name][task]["KVAE"].eval()
                #models_trained[dataset_name][task]["KCVAE"].eval()

                kwargs_vqvae = {"prior_model": models_trained[dataset_name][task]["VAE"].eval()}
                samples_vae = models_trained[dataset_name][task]["VAE"].sample(n, device).cpu().detach().numpy()
                samples_cvae = models_trained[dataset_name][task]["CVAE"].sample(n, device).cpu().detach().numpy() 
                samples_kvae = models_trained[dataset_name][task]["KVAE"].sample(n, device).cpu().detach().numpy()
                #samples_kcvae = models_trained[dataset_name][task]["KCVAE"].sample(n, device).cpu().detach().numpy()


            print(f"Generated {n} samples for {dataset_name}")
            print(f"Samples VAE shape: {samples_vae.shape}")
            print(f"Samples CVAE shape: {samples_cvae.shape}")
            print(f"Samples KVAE shape: {samples_kvae.shape}")
            #print(f"Samples KCVAE shape: {samples_kcvae.shape}")

            if samples_vae.ndim == 3:
                samples_vae = samples_vae.squeeze(1)
            if samples_cvae.ndim == 3:
                samples_cvae = samples_cvae.squeeze(1)
            if samples_kvae.ndim == 3:
                samples_kvae = samples_kvae.squeeze(1)
            #if samples_kcvae.ndim == 3:
            #    samples_kcvae = samples_kcvae.squeeze(1)

            # Plot the samples
            for i in range(n):
                row = i // cols
                col = i % cols
                axs[row, col].set_title(f"Sample {i+1}")
                axs[row, col].set_xlabel("Time")
                axs[row, col].set_ylabel("Amplitude")
                axs[row, col].set_xlim(0, 4096)
                axs[row, col].plot(samples_vae[i], label = "vae", color='black', linewidth=0.5)
                axs[row, col].plot(samples_cvae[i], label = "cvae", color='blue', linewidth=0.5)
                axs[row, col].plot(samples_kvae[i], label = "kvae", color='red', linewidth=0.5)
                axs[row, col].legend()
                axs[row, col].grid()

            plt.tight_layout()
            plt.savefig(f"generated_samples_{dataset_name}.png", dpi=600)
            plt.show()