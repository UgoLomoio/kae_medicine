
from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE

from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE

from model_config import *
from utils import read_arff, split_windows, add_noise, mask_input, load_model, detect_anomalies, set_seed
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import random 
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import os 
import torch.nn.functional as F

cwd = os.getcwd()
sep = os.sep 

expl_plot_path = cwd + sep + "exp_plots"
if not os.path.exists(expl_plot_path):
    os.makedirs(expl_plot_path)

dataset_type = "clean"

if "generation" in tasks:
    tasks.remove("generation")

skip_variational = False  # Set to True to skip variational models
    
thresholds = {}
with open("threshold_values.txt") as f:
    lines = f.readlines()
    for line in lines:
        model_name, threshold_str = line.strip().split(": ")
        threshold = float(threshold_str)
        thresholds[model_name] = threshold                            

set_seed(seed)

def explain_window_1d(model, window, pathology="Anomalous", th=None, mult=10, step=50, limit_min=0, limit_max=4096, show_rec=False):
    """
    Explain a 1D signal using model reconstruction and error analysis.
    
    Args:
        model: PyTorch model for signal reconstruction
        window: 1D numpy array containing the signal
        pathology: String describing the pathology/condition
        th: Threshold for normalization (optional)
        mult: Multiplier for reconstruction errors when th is None
        step: Step size for windowed analysis
        limit_min: Start index for analysis window
        limit_max: End index for analysis window
    
    Returns:
        matplotlib figure object
    """
    
    window = torch.tensor(window, dtype=torch.float32).to(device)
    model = model.to(device)

    rec = model(window)
    if "c" in model_name:
        rec = rec.squeeze(0).squeeze(0)
        window = window.squeeze(0).squeeze(0)

    reconstruction_error = F.mse_loss(rec, window).item()

    window = window.cpu().detach().numpy()
    rec = rec.cpu().detach().numpy()

    # Set up single plot for 1D signal
    fig, ax = plt.subplots(figsize=(15, 6))

    # Apply window limits
    window_segment = window[limit_min:limit_max]
    rec_segment = rec[limit_min:limit_max]

    # Calculate reconstruction errors for each segment
    res = []
    segment_length = limit_max - limit_min
    
    window_segment = window_segment.flatten()
    rec_segment = rec_segment.flatten()

    for start in range(0, segment_length, step):
        end = min(start + step, segment_length)
        compute_re = True
                
        if compute_re and end > start:
            re = F.mse_loss(torch.tensor(rec_segment[start:end]), 
                           torch.tensor(window_segment[start:end])).item()
        elif not compute_re:
            re = 0.0
        else:
            re = 0.0
            
        res.append(re)

    # Normalize reconstruction errors
    if th is not None:
        res_array = np.array(res).reshape(-1, 1)
        if np.max(res_array) > th:
            res_norm = MinMaxScaler(clip=True).fit_transform(res_array).flatten()
        else:
            res_norm = res_array.flatten()
    else:
        res_norm = [elem * mult for elem in res]


    # Plot original signal and reconstruction
    ax.plot(window_segment, color='blue', linewidth=1, label='Original Signal')
    if show_rec:
        ax.plot(rec_segment, '--', color='orange', linewidth=1, label='Reconstruction')
    
    ax.set_title(f'1D Signal Analysis - {model_name}', fontsize=14)
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    max_val = max(window_segment)
    min_val = min(window_segment)
    
    # Overlay reconstruction error as colored regions
    for i, alpha in enumerate(res_norm):
        start = i * step
        end = min(start + step, segment_length)
        print(f"Segment {i}: Start={start}, End={end}, Alpha={alpha}")
        if end > start:
            toplot = np.arange(start, end)
            alpha_val = min(alpha, 1.0)  # Ensure alpha doesn't exceed 1
            ax.fill_between(toplot, max_val, min_val, 
                          color='red', alpha=alpha_val, 
                          label='Reconstruction Error' if i == 0 else "")

    # Update legend to include error overlay
    handles, labels = ax.get_legend_handles_labels()
    if 'Reconstruction Error' in labels:
        ax.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":

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
    
    for task in tasks:
        for dataset_name, files in datapaths.items():
            datapath = files["datapath"]
            test_filename = files["test_filename"]
            window_size = files["window_size"]
            
            if task == "generation":
                models = {
                    "vae": VAE(window_size, hidden_dims["linear_var"], conv=False, beta=beta, dropout=0.0, act=act),
                    "kvae": KVAE(window_size, hidden_dims["linear_kan_var"], conv=False, window_size = 4096, beta=beta, base_activation=act_kan_linear),

                    "cvae": VAE(1, hidden_dims["conv_var"], conv=True, beta=beta, width = 2048, kernel_sizes=kernel_sizes["conv_var"], strides=strides["conv_var"], paddings=paddings["conv_var"], dropout=0.0, act=act_conv, batch_size=batch_size_normal, window_size=window_size),                             
                    "kcvae": KVAE(1, hidden_dims["conv_var"], width = 2048, conv=True, kernel_sizes=kernel_sizes["conv_var"], window_size = 4096, beta=beta, strides=strides["conv_var"], paddings=paddings["conv_var"], base_activation=act_kan)
                }
            else:
                models = {
                    "ae": AutoEncoder(window_size, hidden_dims["linear"], dropout=dropout, bn=bn, act=act),
                    "kae": KAE(window_size, hidden_dims["linear_kan"], dropout=dropout, bn=bn, base_activation=act_kan_linear),

                    "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], dropout=0.0, act=act_conv),
                    "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan),
                }

            xs_test, ys_test = read_arff(datapath + sep + test_filename)

            if task == "denoising":
                xs_test = add_noise(xs_test, noise_level=0.05, noise_prob=0.5)
            if task == "reconstruction":
                xs_test = xs_test

            xs_test_c, ys_test = split_windows(xs_test, ys_test, window_size=window_size)
            if task == "inpainting":
                xs_test = mask_input(xs_test_c, min_mask_size=0.1, max_mask_size=0.4, mask_prob=0.5)
            else:
                xs_test = xs_test_c
            
            xs_test_anomalous = xs_test[ys_test == 1]
            idx_random = random.randint(0, len(xs_test_anomalous)-1)
            x_random = xs_test_anomalous[idx_random]
            y_random = 1
            for model_name, model in models.items():
                
                print(model_name)

                with torch.no_grad():
                    model_path = models_path + sep + f"{model_name}_{dataset_name}_{task}.pt"
                    if "v" in model_name:
                        variational = True 
                    else:
                        variational = False

                    model = load_model(model, model_path, variational=variational)
                    model.eval()
                    model.to(device)
                    models[model_name] = model

                    if x_random.ndim == 1:
                        x_random = x_random.unsqueeze(0)
                    
                    if "c" in model_name:
                        x_random = x_random.unsqueeze(0)

                    if x_random.ndim == 4:
                        x_random = x_random.squeeze(0)
                    rec = model(x_random.to(device))
                    loss = F.mse_loss(rec, x_random.to(device)).cpu().numpy()
    
                if task == "anomaly detection":
                    th = thresholds[f"{model_name}_{dataset_name}_{task}"]
                    #y_pred = detect_anomalies([loss], th) 
                    fig = explain_window_1d(model, x_random.cpu().numpy(), pathology=y_random, th=th, mult=10, step=20)
                else:
                    fig = explain_window_1d(model, x_random.cpu().numpy(), pathology=y_random, th=0.0005, mult=10, step=20, show_rec=True)
                
                fig.savefig(f"{expl_plot_path}{sep}{model_name}_{dataset_name}_{task}_explanation.png")