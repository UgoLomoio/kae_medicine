from torch import nn
import os
import torch

#maybe redo kcae, cae training for all, use the commented conv_kan 

cwd = os.getcwd()
sep = os.sep

datapaths = { "Stethoscope":
    {
    "datapath": cwd + sep + "datasets" + sep + "AbnormalHeartbeat",
    "train_filename": "AbnormalHeartbeat_TRAIN.arff",
    "test_filename": "AbnormalHeartbeat_TEST.arff",
    "fs": 4000, 
    "window_size": 4096
    }
}

tasks = ["generation", "reconstruction", "anomaly detection", "denoising", "inpainting"]

hidden_dims = {
    "linear": [1024, 512], #last dimension must be lower than 1024, found outofmemory error when training Kolmogorov AEs
    "linear_var": [2048, 128], 
    "linear_kan": [64, 512],
    "linear_kan_var": [128, 256, 256],
    #"conv_kan": [16, 64],
    "conv_kan": [64, 128],
    "conv": [64, 128, 256, 512],
    "conv_var": [16, 32, 32],
}

n_layers = {
    "conv": len(hidden_dims["conv"]),  
    "conv_kan": len(hidden_dims["conv_kan"]),
    "conv_var": len(hidden_dims["conv_var"])  
}

kernel_sizes = {
    "conv": [4]*n_layers["conv"],
    "conv_kan": [4]*n_layers["conv_kan"],
    "conv_var": [6]*n_layers["conv_var"]
}
strides = {
    "conv": [2]*n_layers["conv"],
    "conv_kan": [2]*n_layers["conv_kan"],
    "conv_var": [4]*n_layers["conv_var"]
}
paddings = {
    "conv": [1]*n_layers["conv"],
    "conv_kan": [1]*n_layers["conv_kan"],
    "conv_var": [1]*n_layers["conv_var"]
}
act_conv = nn.Tanh #nn.Tanh
act = nn.Tanh
act_kan = nn.SiLU
act_kan_linear = nn.LeakyReLU

models_path = cwd + sep + "models_ckpt" 
tables_path = cwd + sep + "tables"
figures_training_path = cwd + sep + "figures_training"

if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
else:
   device = torch.device("cpu")
   accelerator = "cpu"


seed = 42
val_size = 0.2
random_state = 42
max_epochs = 200
lr_normal = 1e-3
lr_conv = 1e-4 # Learning rate for convolutional models
lr_kolmogorov = 1e-3 #Learning rate for Kolmogorov models
weight_decay_kan = 1e-4 # Weight decay for Kolmogorov models
weight_decay_normal = 1e-4
batch_size_normal = 16 # Batch size for normal models
batch_size_kan = 16 # Batch size for Kolmogorov models
latent_dim = None
beta = 1e-4 # Beta parameter for the beta-VAE
plot_training = True

batch_size_anomaly = 4 # Batch size for anomaly detection

dropout = 0.5 #0.5 # Dropout rate
bn = True # Use Batch Normalization

max_mask_size = 0.2 # Range for mask size in inpainting
min_mask_size = 0.05 # Range for mask size in inpainting
mask_prob = 0.5 # Probability of applying a mask in inpainting

noise_level = 0.5 # Noise level for denoising
noise_prob = 0.5 # Probability of adding noise in denoising

lr_inpainting = 1e-3 # Learning rate for inpainting models
weight_decay_inpainting = 1e-4 # Weight decay for inpainting models
noise_level_mask = 0.15 # Noise level for inpainting with masking

lr_denoising = 1e-3 # Learning rate for denoising models
weight_decay_denoising = 1e-5 # Weight decay for denoising models

lr_pixel_shuffle = 1e-2 # Learning rate for pixel shuffle models
weight_decay_pixel_shuffle = 1e-4 # Weight decay for pixel shuffle models
