import matplotlib.pyplot as plt
import numpy as np
import torch 
import pandas as pd

from model_config import *
from utils import load_model
from metrics import count_parameters

from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE

from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE

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
print(f"Using device: {device}")
print(f"Using accelerator: {accelerator}")

dataset = "Stethoscope"

if "generation" in tasks:
    tasks.remove("generation")
if "anomaly detection" in tasks:
    tasks.remove("anomaly detection")


models = {
        "ae": AutoEncoder(window_size, hidden_dims["linear"], dropout=dropout, bn=bn, act=act),
        "kae": KAE(window_size, hidden_dims["linear_kan"], base_activation=act_kan_linear, dropout=dropout, bn=bn),
        "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=False, dropout=0.0),
        "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, dropout=0.0, pixel_shuffle=False),
        "cae-ps": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=True, dropout=dropout),
        "kcae-ps": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=True, dropout=dropout),
}

train_times = {}
with open("training_times.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        model_name, time_str = line.strip().split(": ")
        model_type = model_name.split("_")[0]
        if model_type in ["ae", "kae", "cae", "kcae", "cae-ps", "kcae-ps"]:
            train_times[model_name] = float(time_str.split(" ")[0])

test_times = {}
with open("test_times.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        model_name, time_str = line.strip().split(": ")
        test_times[model_name] = float(time_str.split(" ")[0])    

model_params = {}
for task in tasks:
    for model_name, model in models.items():
        model_path = f"models_ckpt/{model_name}_{dataset}_{task}.pt"
        model = load_model(model, model_path)
        model_params[f"{model_name}_{dataset}_{task}"] = count_parameters(model)

test_performance = {}
with open("test_performance.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        model_name, performance_str = line.strip().split(": ")
        test_performance[model_name] = float(performance_str.split(" ")[0])

if __name__ == "__main__":

    for task in tasks:
        train_times_task = {k: v for k, v in train_times.items() if task in k}
        test_times_task = {k: v for k, v in test_times.items() if task in k}
        model_params_task = {k: v for k, v in model_params.items() if task in k}
        test_performance_task = {k: v for k, v in test_performance.items() if task in k}

        print(f"Task: {task}")
        print("Train times:", train_times_task)
        print("Test times:", test_times_task)
        print("Test performance:", test_performance_task)
        print("Model params:", model_params_task)
        print("")

        print(len(train_times_task.values()), len(test_times_task.values()), len(test_performance_task.values()), len(model_params_task.values()))
        df = pd.DataFrame({
        'model': list(model_params_task.keys()),
        'params': list(model_params_task.values()),
        'train_time': list(train_times_task.values()),
        'test_time': list(test_times_task.values()),
        'test_performance': list(test_performance_task.values())
        })

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = plt.scatter(df['params'], df['train_time'], s=df['test_time']*500, alpha=0.5, c=df.index, cmap='viridis')
        #plt.colorbar(label='Model')
        plt.title(f"Model Efficiency for {task}", fontsize=24, fontweight='bold')
        plt.xlabel("Number of Parameters (Millions)", fontsize=16, fontweight='bold')
        plt.ylabel("Test Time (s)", fontsize=16, fontweight='bold')
        for i, txt in enumerate(df['model']):
            txt = txt.split("_")[0]  # Extract the model name
            txt = txt.upper()
            ax.annotate(txt, (df['params'][i], df['train_time'][i]), fontsize=12)
        handles, labels = scatter.legend_elements()
        custom_labels = models.keys()
        custom_labels = [label.upper() for label in custom_labels]
        legend = ax.legend(handles, custom_labels, title="Models")
        ax.add_artist(legend)
        #plt.show()
        fig.savefig(f"figures_training/model_efficiency_{task}.png", dpi=600)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = plt.scatter(df['params'], df['test_performance'], s=df['test_time']*500, alpha=0.5, c=df.index, cmap='viridis')
        #plt.colorbar(label='Model')
        plt.title(f"Model Performance for {task}", fontsize=24, fontweight='bold')
        plt.xlabel("Number of Parameters (Millions)", fontsize=16, fontweight='bold')

        if task == "anomaly detection":
            plt.ylabel("Test Performance (AUC)", fontsize=16, fontweight='bold')
        else:
            plt.ylabel("Test Performance (MSE)", fontsize=16, fontweight='bold')

        for i, txt in enumerate(df['model']):
            txt = txt.split("_")[0]
            txt = txt.upper()
            ax.annotate(txt, (df['params'][i], df['test_performance'][i]), fontsize=12, ha='right', va='top')
            #ax.text(df['params'][i], df['test_performance'][i], f"{df['test_time'][i]:.4f}", fontsize=8, ha='center', va='bottom')
        
        handles, labels = scatter.legend_elements()
        custom_labels = models.keys()
        custom_labels = [label.upper() for label in custom_labels]
        legend = ax.legend(handles, custom_labels, title="Models")
        ax.add_artist(legend)

        #plt.show()
        fig.savefig(f"figures_training/model_performance_{task}.png", dpi=600)
        print(f"Figures saved for task: {task}")
