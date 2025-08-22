from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE

from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE

from utils import read_arff, split_windows, add_noise, mask_input, load_model, plot_test, plot_test_plotly
from model_config import *
import torch

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


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

    if "generation" in tasks:
        tasks.remove("generation")

    dfs = {}
    test_times = {}
    test_performance = {}
    threshold_values = {}
    
    for task in tasks:
        dfs[task] = {}
        test_times[task] = {}
        test_performance[task] = {}
        print(f"Test task: {task}")
        for dataset_name, files in datapaths.items():
            dfs[task][dataset_name] = {}
            test_times[task][dataset_name] = {}
            datapath = files["datapath"]
            test_filename = files["test_filename"]
            window_size = files["window_size"]
            
            if task == "generation":
                models = {
                    "vae": VAE(window_size, hidden_dims["linear_var"], conv=False, beta=beta, act=act),
                    "kvae": KVAE(window_size, hidden_dims["linear_kan_var"], conv=False, window_size = 4096, beta=beta, base_activation=act_kan_linear),

                    "cvae": VAE(1, hidden_dims["conv_var"], conv=True, beta=beta, width = 2048, kernel_sizes=kernel_sizes["conv_var"], strides=strides["conv_var"], paddings=paddings["conv_var"], act=act_conv, batch_size=batch_size_normal, window_size=window_size),                             
                    "kcvae": KVAE(1, hidden_dims["conv_var"], width = 2048, conv=True, kernel_sizes=kernel_sizes["conv_var"], window_size = 4096, beta=beta, strides=strides["conv_var"], paddings=paddings["conv_var"], base_activation=act_kan)
                }
            else:
                models = {
                    "ae": AutoEncoder(window_size, hidden_dims["linear"], dropout=dropout, bn=bn, act=act),
                    "kae": KAE(window_size, hidden_dims["linear_kan"], base_activation=act_kan_linear, dropout=dropout, bn=bn),
                    "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv),
                    "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan)
                }


            xs_test, ys_test = read_arff(datapath + sep + test_filename)
            xs_test_c, ys_test = split_windows(xs_test, ys_test, window_size=window_size)

            for model_name, model in models.items():
                model_path = cwd + sep + "models_ckpt" + sep + f"{model_name}_{dataset_name}_{task}.pt"
                model = load_model(model, model_path)
                model = model.to(device)
                model.eval()    
                models[model_name] = model

            plot_test(models, xs_test_c, ys_test, dataset_name, task, device = device, skip_variational = True)
            plot_test_plotly(models, xs_test_c, ys_test, dataset_name, task, device = device, skip_variational = True)