from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE
from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE
from model_config import *
import torch

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

    window_size = 4096
            
    models = {
                        "ae": AutoEncoder(window_size, hidden_dims["linear"], dropout=dropout, bn=bn, act=act),
                        "kae": KAE(window_size, hidden_dims["linear_kan"], base_activation=act_kan_linear, dropout=dropout, bn=bn),
                        "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=False),
                        "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=False),
                        "cae-ps": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=True),
                        "kcae-ps": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=True),
    }

    for model_name, model in models.items():
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name} has {num_params} trainable parameters.")
