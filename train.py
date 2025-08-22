from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE

from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE

from utils import SignalDataset, read_arff, split_train_val, split_windows, add_noise, mask_input, get_only_normal, plot_roc_curves, plot_losses, print_model, clear_memory, anomaly_detection

from lightning_modules import LightningAutoEncoder
from model_config import *

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt
import time 

from sklearn.model_selection import train_test_split
import os 

torch.set_float32_matmul_precision('medium')

cwd = os.getcwd()
sep = os.sep

#PEnding KCAE training for all

if "generation" in tasks:
    tasks.remove("generation")
if "anomaly detection" in tasks:
    tasks.remove("anomaly detection")

clean_files = False
if clean_files:
    with open("training_times.txt", "w") as f:
        f.write("")
    with open("threshold_values.txt", "w") as f:
        f.write("")

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

    models_trained = {}
    train_losses = {}
    val_losses = {}
    training_times = {}

    for task in tasks:

        print(f"Training task: {task}")
        models_trained[task] = {}
        train_losses[task] = {} 
        val_losses[task] = {}
        training_times[task] = {}

        if task == "generation":
            variational = True
            
        if task == "anomaly detection":
            #save threshold values in a dictionary
            threshold_values = {}
            dict_aucs = {}

        for dataset_name, files in datapaths.items():
            
            datapath = files["datapath"]
            models_trained[task][dataset_name] = {}
            train_losses[task][dataset_name] = {}
            val_losses[task][dataset_name] = {}
            training_times[task][dataset_name] = {}

            train_filename = files["train_filename"]
            window_size = files["window_size"]
            
            xs, ys = read_arff(datapath + sep + train_filename)
            
            if task != "anomaly detection":
                xs_train, ys_train, xs_val, ys_val = split_train_val(xs, ys, val_size=val_size, random_state=random_state)
            else:
                filename_x_train = cwd + sep + "data" + sep + dataset_name + sep + "x_train_an.pt"
                filename_x_val = cwd + sep + "data" + sep + dataset_name + sep + "x_val_an.pt"
                filename_y_train = cwd + sep + "data" + sep + dataset_name + sep + "y_train_an.pt"
                filename_y_val = cwd + sep + "data" + sep + dataset_name + sep + "y_val_an.pt"
                if os.path.exists(filename_x_train) and os.path.exists(filename_x_val) and os.path.exists(filename_y_train) and os.path.exists(filename_y_val):
                    print("Loading pre-split data for anomaly detection...")
                    xs_train = torch.load(filename_x_train)
                    ys_train = torch.load(filename_y_train)
                    xs_val = torch.load(filename_x_val)
                    ys_val = torch.load(filename_y_val)
                else:
                    idxs_normal = ys == 0
                    xs_normal = xs[idxs_normal]
                    ys_normal = ys[idxs_normal]
                    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_normal, ys_normal, test_size=val_size, random_state=random_state)
                    xs_val_other = xs[~idxs_normal]
                    ys_val_other = ys[~idxs_normal]
                    xs_val = torch.cat((xs_val, xs_val_other), dim=0)
                    ys_val = torch.cat((ys_val, ys_val_other), dim=0)
                    # Save the split data for future use
                    os.makedirs(cwd + sep + "data" + sep + dataset_name, exist_ok=True)
                    torch.save(xs_train, filename_x_train)
                    torch.save(ys_train, filename_y_train)
                    torch.save(xs_val, filename_x_val)
                    torch.save(ys_val, filename_y_val)
                    print(torch.unique(ys_train, return_counts=True))
                    print(torch.unique(ys_val, return_counts=True))

            if task in ["inpainting", "denoising"]:
                xs_original_train = xs_train.clone()
                xs_original_val = xs_val.clone()
                ys_original_train = ys_train.clone()
                ys_original_val = ys_val.clone()
                xs_original_train, ys_original_train = split_windows(xs_original_train, ys_original_train, window_size=window_size)
                xs_original_val, ys_original_val = split_windows(xs_original_val, ys_original_val, window_size=window_size)
            else:
                xs_original_train = None
                xs_original_val = None

            if task == "denoising":
                xs_train = add_noise(xs_train, noise_level=noise_level, noise_prob=noise_prob)
                xs_val = add_noise(xs_val, noise_level=noise_level, noise_prob=noise_prob)

            if task == "reconstruction":
                pass

            #if task == "anomaly detection":
            #    x_train, y_train, x_other, y_other = get_only_normal(xs_train, ys_train)
            #    xs_train = x_train
            #    ys_train = y_train

            xs_train, ys_train = split_windows(xs_train, ys_train, window_size=window_size)
            xs_val, ys_val = split_windows(xs_val, ys_val, window_size=window_size)

            print(f"Train set size: {len(xs_train)}")
            print(f"Validation set size: {len(xs_val)}")
            
            if task == "inpainting":
                xs_train, xs_train_clean  = mask_input(xs_train, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=mask_prob, add_noise=True, noise_level = noise_level_mask)
                xs_val, xs_val_clean = mask_input(xs_val, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=mask_prob, add_noise=True, noise_level = noise_level_mask)
            
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
                        "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=False, dropout=0.0),
                        "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, dropout=0.0, pixel_shuffle=False),
                        "cae-ps": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=True, dropout=dropout),
                        "kcae-ps": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=True, dropout=dropout),
                }

            fprs = {}
            tprs = {}

            for model_name, model in models.items():
                
                clear_memory()
                plt.close('all')

                print("Model name: ", model_name)  
                print_model(model)

                if "v" in model_name:
                    variational = True
                else:            
                    variational = False
                if "k" in model_name:
                    kolmogorov = True
                else:
                    kolmogorov = False
                if "c" in model_name or variational:
                    lr = lr_conv
                if kolmogorov:
                    lr = lr_kolmogorov
                else:
                    lr = lr_normal
                
                if "inpainting" in task:
                    lr = lr_inpainting
                    weight_decay = weight_decay_inpainting
                elif "denoising" in task:
                    lr = lr_denoising
                    weight_decay = weight_decay_denoising

                if "ps" in model_name:
                    lr = lr_pixel_shuffle
                    weight_decay = weight_decay_pixel_shuffle

                if kolmogorov:
                    batch_size = batch_size_kan
                    weight_decay = weight_decay_kan
                else:
                    batch_size = batch_size_normal
                    weight_decay = weight_decay_normal
                    
                if task == "anomaly detection":
                    # For anomaly detection, we use a smaller batch size
                    batch_size = batch_size_anomaly

                if xs_original_train is not None and xs_original_train.ndim == 2:
                    xs_original_train = xs_original_train.unsqueeze(1)
                if xs_original_val is not None and xs_original_val.ndim == 2:
                    xs_original_val = xs_original_val.unsqueeze(1)

                train_dataset = SignalDataset(xs_train, ys_train, device=device, signals_original=xs_original_train)
                val_dataset = SignalDataset(xs_val, ys_val, device=device, signals_original=xs_original_val)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                print(f"Training {model_name}...")

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                if variational:
                    criterion = "mse_sum"
                    th = 0.0001 # Threshold for variational models
                else:
                    criterion = "mse_mean"
                    th = 0.00001  # Threshold for non-variational models
                if task == "inpainting":
                    criterion = "mse_mean"

                if kolmogorov:
                    patience = 25
                else:
                    patience = 10

                min_lr = 1e-6  # Minimum learning rate for all models
                if task == "anomaly detection":
                    patience = 25
                elif task == "inpainting" or task == "denoising":
                    patience = 10
                    min_lr = 1e-5

                #if not kolmogorov:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=min_lr, threshold=th)
                #else:
                #    scheduler = None

                model = model.to(device)
                if variational:
                    kld_weight = batch_size / len(xs_train)  # Adjust KLD weight based on batch size
                    lightning_model = LightningAutoEncoder(model, model_name, optimizer, criterion=criterion, task=task, scheduler=scheduler, kld_weight=kld_weight, plot_training=plot_training)
                else:
                    lightning_model = LightningAutoEncoder(model, model_name, optimizer, criterion=criterion, task=task, scheduler=scheduler, plot_training=plot_training)
                trainer = Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=1, log_every_n_steps=100)
                
                if "q" in model_name:
                    quantized = True
                else:
                    quantized = False
                    
                t_start = time.time()
                trainer.fit(lightning_model, train_loader, val_loader)
                t_end = time.time()
                training_time = t_end - t_start
                training_times[task][dataset_name][model_name] = training_time

                print(f"Training time for {model_name}: {training_time:.2f} seconds")
                with open("training_times.txt", "a") as f:
                    f.write(f"{model_name}_{dataset_name}_{task}: {training_time} seconds \n")
                print(f"Training {model_name} finished.")
                
                # Store the trained model
                models_trained[task][dataset_name][model_name] = model      
                train_losses[task][dataset_name][model_name] = lightning_model.train_losses
                val_losses[task][dataset_name][model_name] = lightning_model.val_losses
                
                if task == "anomaly detection":

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
                            #print(model)
                            x_reconstructed = model(x)
                            loss = nn.functional.mse_loss(x_reconstructed, x)
                        losses.append(loss.item())
                    losses = torch.tensor(losses, device=device)
                    threshold, auc, fpr, tpr = anomaly_detection(losses, ys_val)

                    # Save the threshold values to a file
                    with open("threshold_values.txt", "a") as f:
                        f.write(f"{model_name}_{dataset_name}_{task}:  {threshold}\n")

                    dict_aucs[model_name] = auc
                    fprs[model_name] = fpr
                    tprs[model_name] = tpr
                    threshold_values[model_name] = threshold

                # Save the model
                # Ensure the directory exists
                os.makedirs("models_ckpt", exist_ok=True)
                # Save the model state_dict
                model_path = f"models_ckpt/{model_name}_{dataset_name}_{task}.pt"
                torch.save(model.state_dict(), model_path)

                #Save losses to a file 
                print("Saving train and validation losses...")
                train_losses_file = f"train_losses/{model_name}{dataset_name}_{task}_train.txt"
                os.makedirs("train_losses", exist_ok=True)
                with open(train_losses_file, "w") as f:
                    for epoch, loss in enumerate(train_losses[task][dataset_name][model_name]):
                        f.write(f"Epoch {epoch + 1}: {loss}\n")
                val_losses_file = f"train_losses/{model_name}{dataset_name}_{task}_val.txt"
                with open(val_losses_file, "w") as f:
                    for epoch, loss in enumerate(val_losses[task][dataset_name][model_name]):
                        f.write(f"Epoch {epoch + 1}: {loss}\n")
                print("Model losses saved")
            
            clear_memory()

            # Plot training and validation losses
            plot_losses(train_losses[task][dataset_name], val_losses[task][dataset_name], dataset_name, task)
            # Plot AUCs for anomaly detection
            if task == "anomaly detection":
                plot_roc_curves(tprs, fprs, split = "val")

