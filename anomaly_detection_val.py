from utils import anomaly_detection, load_model, split_windows
from model_config import * 
import torch
from models.ae import AutoEncoder
from models.kae import KAE
from models.cae import CAE
from models.kcae import KCAE
#ae_Stethoscope_anomaly detection:  0.00047511281445622444
#kae_Stethoscope_anomaly detection:  0.0007296130643226206
if __name__ == "__main__":

    window_size = 4096
    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    models = {
        "ae": AutoEncoder(window_size, hidden_dims["linear"], dropout=dropout, bn=bn, act=act),
        "kae": KAE(window_size, hidden_dims["linear_kan"], base_activation=act_kan_linear, dropout=dropout, bn=bn),
        "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=False),
        "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=False),                
        "cae-ps": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=True),
        "kcae-ps": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=True),
    }
    criterion = torch.nn.MSELoss()

    for dataset_name, files in datapaths.items():

        filename_x_val = cwd + sep + "data" + sep + dataset_name + sep + "x_val_an.pt"
        filename_y_val = cwd + sep + "data" + sep + dataset_name + sep + "y_val_an.pt"
        xs_val = torch.load(filename_x_val)
        ys_val= torch.load(filename_y_val)


    xs_val, ys_val = split_windows(xs_val, ys_val, window_size=window_size)
    print(xs_val.shape, ys_val.shape)

    for model_name, model in models.items():
        model_path = models_path + sep + f"{model_name}_{dataset_name}_anomaly detection.pt"
        model = load_model(model, model_path, variational=False)
        model.eval()
        model.to(device)
        print(f"Evaluating model: {model_name} on dataset: {dataset_name}")
        # Compute the threshold and AUC
        losses = []
        for x in xs_val:
            x = x.to(device)
            model.to(device)
            with torch.no_grad():
                model.eval()
                if "c" in model_name:
                    x = x.unsqueeze(0).unsqueeze(0)
                else:
                    x = x.unsqueeze(0)
                x_reconstructed = model(x)
                loss = criterion(x_reconstructed, x)
            losses.append(loss.item())
        losses = torch.tensor(losses, device=device)
        threshold, auc, fpr, tpr = anomaly_detection(losses, ys_val)