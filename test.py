from models.ae import AutoEncoder
from models.cae import CAE
from models.vae import VAE

from models.kae import KAE
from models.kcae import KCAE
from models.kvae import KVAE

from utils import SignalDataset, read_arff, split_windows, add_noise, mask_input, load_model, plot_test, detect_anomalies, plot_test_plotly, plot_roc_curves_test
from metrics import evaluate_models
from model_config import *
from sklearn.metrics import roc_auc_score
from umap import UMAP
import matplotlib.pyplot as plt
from utils import print_model
import torch

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


if "generation" in tasks:
    tasks.remove("generation")
if "anomaly detection" in tasks:
    tasks.remove("anomaly detection")

def plot_drift(losses_train, losses, model_name, task):

    losses_train = losses_train[model_name]
    losses = losses[model_name]

    plt.figure(figsize=(10, 5))
    idx = len(losses_train)
    plt.plot(range(idx), losses_train, label='Training Loss', color='blue')
    plt.plot(range(idx, idx + len(losses)), losses, label='Test Loss', color='orange')
    plt.axvline(x=idx, color='red', linestyle='--', label='Train/Test Split')
    plt.title(f'Autoencoder Loss Drift {model_name.upper()}, Task: {task.capitalize()}', fontsize=24, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=16, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'loss_drift_{model_name}_{task}.png', bbox_inches='tight', dpi=600)
    #plt.show()

def plot_latentspace(model, model_name, task, xs_test, ys_test):
    
    plt.figure()

    model.eval()
    model.to(device)
    map_class = {0: "Normal", 1: "Abnormal"}

    embeddings = {}
    for i, (x, y) in enumerate(zip(xs_test, ys_test)):
        
        y = y.item()
        if y not in embeddings.keys():
            embeddings[y] = []

        x = x.to(device)
        if "c" in model_name:
            if x.ndim == 1:
                x = x.unsqueeze(0)
            if x.ndim == 2:
                x = x.unsqueeze(0)

        with torch.no_grad():
            if "k" in model_name:
                if "c" not in model_name:
                    x = x.unsqueeze(0)  # Add batch dimension for Kolmogorov models
                else:
                    if x.ndim == 4:
                        x = x.squeeze(0)

            if x.ndim == 1:
                x = x.unsqueeze(0)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            embedding = model.encode(x)    
        
        if embedding.ndim == 4:
            embedding = embedding.squeeze(0)
        if embedding.ndim == 3:
            embedding = embedding.squeeze(0)
        if embedding.ndim == 2:
            embedding = embedding.flatten(0)
        
        #print(f"Model: {model_name}")
        #print(f"Embedding shape: {embedding.shape}")
        embeddings[y].append(embedding)

    colors = ["green", "red"]
    for j, (class_, class_embeddings) in enumerate(embeddings.items()):
        
        class_ = map_class[class_]
        print(class_, len(class_embeddings))
        class_embeddings = torch.stack(class_embeddings).cpu().numpy()
        reducer = UMAP(n_components=2, random_state=seed)
        embedding_2d = reducer.fit_transform(class_embeddings)
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], label=f"Class {class_}", color=colors[j])

    #plt.title("Latent Space Embeddings - {}, Task: {}".format(model_name.upper(), task.capitalize()), fontsize=18, fontweight='bold')
    plt.xlabel("UMAP Component 1", fontsize=16, fontweight='bold')
    plt.ylabel("UMAP Component 2", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f"latent_space_{model_name}_{task}.png", bbox_inches='tight', dpi=600)

with open("test_times.txt", "w") as f:
    f.write("")
with open("test_performance.txt", "w") as f:
    f.write("")


skip_variational = False  # Set to True to skip variational models

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
                        "cae": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=False, dropout=0.0),
                        "kcae": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, dropout=0.0, pixel_shuffle=False),
                        "cae-ps": CAE(1, hidden_dims["conv"], kernel_sizes=kernel_sizes["conv"], strides=strides["conv"], paddings=paddings["conv"], act=act_conv, pixel_shuffle=True, dropout=dropout),
                        "kcae-ps": KCAE(1, hidden_dims["conv_kan"], kernel_sizes=kernel_sizes["conv_kan"], strides=strides["conv_kan"], paddings=paddings["conv_kan"], base_activation=act_kan, pixel_shuffle=True, dropout=dropout),
                }



            xs_test, ys_test = read_arff(datapath + sep + test_filename)
            xs_train, ys_train = read_arff(datapath + sep + files["train_filename"])
            xs_test_clean = xs_test.clone()

            if task == "denoising":
                xs_test = add_noise(xs_test, noise_level=noise_level, noise_prob=noise_prob)
                xs_train = add_noise(xs_train, noise_level=noise_level, noise_prob=noise_prob)
                
            if task == "reconstruction":
                xs_test = xs_test
                xs_train = xs_train

            xs_test_clean, ys_test_clean = split_windows(xs_test_clean, ys_test, window_size=window_size)
                
            xs_test_c, ys_test = split_windows(xs_test, ys_test, window_size=window_size)
            xs_train_c, ys_train = split_windows(xs_train, ys_train, window_size=window_size)

            print(f"Test size: {len(xs_test_c)}")
            print(len(xs_test_c[0]))
            if task == "inpainting":
                xs_test, xs_test_clean = mask_input(xs_test_c, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=mask_prob, add_noise=True, noise_level = noise_level_mask)
                xs_train, xs_train_clean = mask_input(xs_train_c, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=mask_prob, add_noise=True, noise_level = noise_level_mask)
            else:
                xs_test = xs_test_c
                xs_train = xs_train_c

            for model_name, model in models.items():
                
                test_dataset = SignalDataset(xs_test, ys_test, device=device)

                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

                model_path = models_path + sep + f"{model_name}_{dataset_name}_{task}.pt"

                if "v" in model_name:
                    variational = True 
                else:
                    variational = False
                model = load_model(model, model_path, variational=variational)
                model.eval()
                model.to(device)
                models[model_name] = model
            
            
            print("Evaluating models on test set...")
            df, test_time_models, losses = evaluate_models(models, xs_test, device=device, return_losses = True)

            
            print("Compute losses for training set...")
            _, _, losses_train = evaluate_models(models, xs_train, device=device, return_losses=True)


            for model_name, model in models.items():
                print(model_name)
                if "c" in model_name or "k" in model_name:
                    input_size = (1, 1, xs_test[0].shape[-1])
                else:
                    input_size = (1, xs_test[0].shape[-1])
                plot_latentspace(model, model_name, task, xs_test, ys_test)
                plot_drift(losses_train, losses, model_name, task)

            dfs[task][dataset_name] = df
            
            if task != "anomaly detection":
                with open("test_performance.txt", "a") as f:
                    for model_name, model in models.items():
                        df_model = df[df["Model"] == model_name]
                        test_performance_model = df_model["MSE"].values[0].split("(")[0]
                        f.write(f"{model_name}_{dataset_name}_{task}: {test_performance_model} \n")
                        test_performance[task][dataset_name] = test_performance_model
            else:
                with open("threshold_values.txt") as f:
                    lines = f.readlines()
                    for line in lines:
                        model_name, threshold_str = line.strip().split(": ")
                        threshold = float(threshold_str)
                        threshold_values[model_name] = threshold
                
                with open("test_performance.txt", "a") as f:
                    for model_name, model in models.items():
                        df_model = df[df["Model"] == model_name]
                        threshold = threshold_values[f"{model_name}_{dataset_name}_{task}"]
                        model_losses = losses[model_name]
                        y_pred = detect_anomalies(model_losses, threshold)
                        auc = roc_auc_score(ys_test, y_pred)
                        f.write(f"{model_name}_{dataset_name}_{task}: {auc} \n")
                        test_performance[task][dataset_name] = auc
                        
            if task == "anomaly detection":
                print("Plotting ROC curves...")
                plot_roc_curves_test(losses, ys_test, threshold_values, dataset_name, task)
               

            test_times[task][dataset_name] = test_time_models

            with open("test_times.txt", "a") as f:
                for model in models:
                    test_time = test_time_models[model]
                    f.write(f"{model}_{dataset_name}_{task}: {test_time} seconds \n")
                

            df.to_csv(tables_path + sep + f"{task}_{dataset_name}.csv", index=False)
            print(f"Results saved to {tables_path + sep + f'{task}_{dataset_name}.csv'}")       

            plot_test(models, xs_test_clean, ys_test_clean, dataset_name, task, device = device, skip_variational = skip_variational)
            plot_test_plotly(models, xs_test_clean, ys_test_clean, dataset_name, task, device = device, skip_variational = skip_variational)