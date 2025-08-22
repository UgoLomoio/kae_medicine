from torch.utils.data import Dataset
from scipy.io import arff
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
import numpy as np
import os 
from collections.abc import Sequence 
from torchinfo import summary
from sklearn.metrics import roc_curve, auc
import gc 
from torchmetrics.classification import BinaryAUROC
from torchmetrics import ROC
from model_config import *

cwd = os.getcwd()
sep = os.sep
figures_training_path = cwd + sep + "figures_training"

auroc = BinaryAUROC()
roc = ROC(task='binary')

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def clear_memory():
    """
    Clear the memory by deleting unused variables and calling garbage collector.
    """
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.")
    
class SignalDataset(Dataset):

    def __init__(self, signals, labels, device = "cpu", signals_original=None):

        self.signals = signals
        self.labels = labels
        self.device = device
        self.signals_original = signals_original

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        
        signal = self.signals[idx].to(self.device)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        label = self.labels[idx].to(self.device)
        if self.signals_original is not None:
            original_signal = self.signals_original[idx].to(self.device)
            if original_signal.ndim == 1:
                original_signal = original_signal.unsqueeze(0)
            if self.signals_original.ndim == 2:
                original_signal = original_signal.unsqueeze(0)
            return (original_signal, signal), label
        return signal, label
    
def read_arff(file_path):

    arff_file = arff.loadarff(file_path)
    df = pd.DataFrame(arff_file[0])
    signals = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    labels = [1 if label == b'Abnormal' else 0 for label in labels]  # Convert byte strings to integers
    signals = torch.tensor(signals, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return signals, labels

def plot_signal(signal, label):

    label_map = {0: 'Normal', 1: 'Abnormal'}
    plt.figure(figsize=(10, 4))
    plt.plot(signal.numpy())
    plt.title(f'Stethoscope Signal - {label_map[label.item()]}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def split_train_val(xs, ys, val_size=0.2, random_state=42):
    
    if not os.path.exists("datasets/AbnormalHeartbeat/idx_train.npy") or not os.path.exists("datasets/AbnormalHeartbeat/idx_val.npy"):
        print("Splitting dataset into train and validation sets...")
        val_size = 0.1
        idxs = list(range(len(xs)))
        idx_train, idx_val = train_test_split(idxs, test_size=val_size, random_state=random_state)
        np.save("datasets/AbnormalHeartbeat/idx_train.npy", idx_train)
        np.save("datasets/AbnormalHeartbeat/idx_val.npy", idx_val)
    else:
        print("Loading train and validation indices...")
        idx_train = np.load("datasets/AbnormalHeartbeat/idx_train.npy")
        idx_val = np.load("datasets/AbnormalHeartbeat/idx_val.npy")
    xs_train, ys_train = xs[idx_train], ys[idx_train]
    xs_val, ys_val = xs[idx_val], ys[idx_val]
    return xs_train, ys_train, xs_val, ys_val


def load_model(model_class, model_path, variational=False):    
    """
    Load a model from a file.
    """
    model = model_class
    if variational:
        fake_input = torch.randn(1, 1, 4096)  # Example input for VAE
        model.forward(fake_input)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def split_windows(xs, ys, window_size=128):
    """
    Split the signals into windows of a given size.
    """
    xs_windows = []
    ys_windows = []
    for signal, label in zip(xs, ys):
        for i in range(0, len(signal) - window_size + 1, window_size):
            xs_windows.append(signal[i:i + window_size])
            ys_windows.append(label)
    return torch.stack(xs_windows), torch.tensor(ys_windows)

def add_noise_single(x, noise_level=0.05, noise_prob=0.5, from_mask=False):
    """
    Add Gaussian noise to a single signal.
    """
    prob = random.random()
    if prob < noise_prob:
        #noise level depends on the signal amplitude
        if not from_mask:
            noise_level = random.uniform(0.1, noise_level)
            noise = torch.randn_like(x) * noise_level * (2*x.std())
        else:
            noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        return x_noisy

    else:
        return x 
def add_noise(xs, noise_level=0.05, noise_prob=0.5):
    """
    Add Gaussian noise to the signals.
    """
    xs_noisy = []
    for x in xs:
        x_noisy = add_noise_single(x, noise_level=noise_level, noise_prob=noise_prob)
        xs_noisy.append(x_noisy)
    return torch.stack(xs_noisy)

def get_only_normal(xs, ys):
    """
    Get only the normal signals from the dataset.
    """
    normal_indices = torch.where(ys == 0)[0]
    xs_normal = xs[normal_indices]
    ys_normal = ys[normal_indices]
    x_other = xs[~normal_indices]
    y_other = ys[~normal_indices]
    return xs_normal, ys_normal, x_other, y_other

def mask_input(xs, max_mask_size=0.4, min_mask_size = 0.1, mask_prob=0.5, add_noise = False, noise_level=0.05):
    """
    Mask a portion of the input signals.
    """
    xs_masked = xs.clone()
    xs_masked_clean = xs.clone()

    for i in range(len(xs_masked)):
        x = xs_masked[i]
        # Randomly decide whether to mask the signal
        p = random.random()
        if p < mask_prob:
            #print(f"Masking signal")
            # Randomly select mask_size
            max_mask_size_ = int(max_mask_size * len(x))
            min_mask_size_ = int(min_mask_size * len(x))
            mask_size = random.randint(min_mask_size_, max_mask_size_)
            # Randomly select a portion of the signal to mask
            end = random.randint(0, len(x)-1)
            start = end - mask_size
            if start < 0:
                start = 0
            if end > len(x):
                end = len(x)
            mask_size = end - start

            xs_masked_clean[i, start:end] = torch.zeros(mask_size)

            if add_noise:
                xs_masked[i, start:end]  = add_noise_single(xs_masked[i, start:end], noise_level=noise_level, noise_prob=1.0, from_mask=True)

    if add_noise:
        return xs_masked, xs_masked_clean
    else:
        return xs_masked

def detect_anomalies(losses, threshold):
    yhats = []
    for i, loss in enumerate(losses):
        if loss > threshold:
            yhats.append(1)
        else:
            yhats.append(0)
    yhats = torch.tensor(yhats)
    return yhats


def plot_losses(train_losses, val_losses, dataset_name, task):

    plt.figure(figsize=(10, 8))
    for model_name, train_loss in train_losses.items():
        val_loss = val_losses[model_name]
        plt.plot(train_loss, label=f'Train Loss - {model_name}')
        plt.plot(val_loss, label=f'Validation Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - {task}')
    plt.legend()
    filename = f"{figures_training_path}{sep}{dataset_name}_{task}_losses.pdf"
    plt.savefig(filename, dpi=600)
    #plt.show()

def plot_test(models, xs_test, ys_test, dataset_name, task, device = "cpu", skip_variational = False):

    n = len(xs_test)
    idx = random.randint(0, n-1)
    x_test, y_test = xs_test[idx], ys_test[idx]
    x_test_original = x_test.clone()

    plt.figure(figsize=(10, 8))

    colors = ['red', 'orange', 'black', 'magenta', 'purple', 'brown', 'pink', 'gray', 'olive']

    x_test = x_test.to(device)        

    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(0)

    if task == "inpainting":
        x_test_visualize = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0, add_noise=False)
    else:
        x_test_visualize = x_test.clone().squeeze(dim=0)

    plt.plot(x_test_visualize.cpu().detach().numpy(), label='Input Signal', color = "blue", linewidth=0.5, alpha=0.5)
    plt.plot(x_test_original.cpu().detach().numpy(), label='Original Signal', linewidth=1, color='green', alpha=0.8)
        

    for i, (model_name, model) in enumerate(models.items()):

        if skip_variational and "v" in model_name:
            continue
        model = model.to(device)
        model.eval()

        x_test = x_test_original.clone().to(device)
        
        if x_test.ndim == 1:
            x_test = x_test.unsqueeze(0)

        if task == "inpainting":
            x_test = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0)
            x_test_visualize = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0, add_noise=False)
        if task == "denoising":
            x_test = add_noise(x_test, noise_level=noise_level, noise_prob=1.0)
            x_test_visualize = x_test.clone()
        else:
            x_test_visualize = x_test.clone()

        if x_test.ndim == 2:
            x_test = x_test.unsqueeze(0)

        with torch.no_grad():
            pred = model(x_test)
        if isinstance(pred, Sequence):
            pred = pred[0]

        x_test = x_test.squeeze(dim=0)
        x_test = x_test.squeeze(dim=0)

        if pred.ndim == 3:
            pred = pred.squeeze(dim=0)
        if pred.ndim == 2:
            pred = pred.squeeze(dim=0)
        
        plt.plot(pred.cpu().detach().numpy(), label='{} Reconstruction'.format(model_name), color = colors[i], linewidth=2)

    plt.title(f'{dataset_name} - {task}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    filename = f"{figures_training_path}{sep}{dataset_name}_{task}_test_visualization.pdf"
    plt.savefig(filename, dpi=600)

def plot_test_plotly(models, xs_test, ys_test, dataset_name, task, device = "cpu", skip_variational = False):

    import plotly.graph_objects as go
    n = len(xs_test)
    idx = random.randint(0, n-1)
    x_test, y_test = xs_test[idx], ys_test[idx]
    x_test_original = x_test.clone()

    fig = go.Figure()
    colors = ['red', 'orange', 'black', 'magenta', 'purple', 'brown', 'pink', 'gray', 'olive']

    fig.add_trace(go.Scatter(y=x_test_original.cpu().detach().numpy(), mode='lines', name='Original Signal', line=dict(width=1.0, color='blue')))

    x_test = x_test.to(device)        
    
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(0)

    if task == "inpainting":
        x_test_visualize = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0, add_noise=False)
    else:
        x_test_visualize = x_test.clone().squeeze(dim=0)
    fig.add_trace(go.Scatter(y=x_test_visualize.cpu().detach().numpy(), mode='lines', name='Input Signal', line=dict(width=1.0, color='green')))

    for model_name, model in models.items():
        
        if skip_variational and "v" in model_name:
            continue
        
        model = model.to(device)
        model.eval()

        x_test = x_test_original.clone().to(device)
        if x_test.ndim == 1:
            x_test = x_test.unsqueeze(0)
        if task == "inpainting":
            x_test = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0)
            x_test_visualize = mask_input(x_test, min_mask_size=min_mask_size, max_mask_size=max_mask_size, mask_prob=1.0, add_noise=False)
        if task == "denoising":
            x_test = add_noise(x_test, noise_level=noise_level, noise_prob=1.0)
            x_test_visualize = x_test.clone()
        else:
            x_test_visualize = x_test.clone()

        if x_test.ndim == 2:
            x_test = x_test.unsqueeze(0)

        with torch.no_grad():
            pred = model(x_test)
        if isinstance(pred, Sequence):
            pred = pred[0]

        x_test = x_test.squeeze(dim=0)
        x_test = x_test.squeeze(dim=0)

        if pred.ndim == 3:
            pred = pred.squeeze(dim=0)
        if pred.ndim == 2:
            pred = pred.squeeze(dim=0)
        
        fig.add_trace(go.Scatter(y=pred.cpu().detach().numpy(), mode='lines', name='{} Reconstruction'.format(model_name), line=dict(width=2)))
 
    fig.update_layout(title=f'{dataset_name} - {task}', xaxis_title='Time', yaxis_title='Amplitude')
    fig.update_layout(template='plotly_white', xaxis_rangeslider_visible=True)
    fig.update_layout(legend=dict(title='Models', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    filename = f"{figures_training_path}{sep}{dataset_name}_{task}_test_visualization.html"
    fig.write_html(filename)
    return fig

def print_model(model, input_size=(1, 1, 4096), device="cpu"):

    print(summary(model, input_size=input_size, device=device))


def plot_roc_curves_test(losses_dict, ys, thresholds, dataset_name, task="anomaly detection"):

    fprs = {}
    tprs = {}
    for model_name, losses in losses_dict.items():
        name = f"{model_name}_{dataset_name}_{task}"
        threshold = thresholds[name]
        yhats = detect_anomalies(losses, threshold=threshold)
        yhats = yhats.to("cpu").detach().numpy()
        fpr, tpr, _ = roc_curve(ys, yhats)
        fprs[model_name] = fpr
        tprs[model_name] = tpr
    plot_roc_curves(fprs, tprs, split="test")

def plot_roc_curves(tprs, fprs, split = "val"):

    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr) in zip(tprs.keys(), zip(fprs.values(), tprs.values())):
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} ROC curve: ({auc_value:.4f} AUC)')
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing (AUC = 0.5)')
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    filename = f"{figures_training_path}{sep}{split}_roc_curves.pdf"
    plt.savefig(filename, dpi=600)


def detect_anomalies(val_losses, threshold):
    yhats = []
    for i in range(len(val_losses)):
        if val_losses[i] > threshold:
            yhats.append(1)
        else:
            yhats.append(0)
    yhats = torch.tensor(yhats)
    return yhats

def plot_distribution(val_losses, ys):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    val_losses_np = val_losses.cpu().numpy()
        
    plt.figure(figsize=(10, 8))
    
    # Create histograms with focus on non-outlier range
    normal_losses = val_losses_np[ys == 0]
    anomalous_losses = val_losses_np[ys == 1]
    
    print(f"Normal losses range: {normal_losses.min()} - {normal_losses.max()}")
    print(f"Anomalous losses range: {anomalous_losses.min()} - {anomalous_losses.max()}")
    print(f"Normal shape: {normal_losses.shape}, Anomalous shape: {anomalous_losses.shape}")

    sns.histplot(normal_losses, color='blue', label='Normal', kde=True, stat='count', bins="auto", alpha=0.7)
    sns.histplot(anomalous_losses, color='red', label='Anomalous', kde=True, stat='count', bins="auto", alpha=0.7)

    # Set reasonable x-axis limits
    upper_bound = max(max(normal_losses), np.quantile(anomalous_losses, 0.95))
    plt.xlim(0, upper_bound) 
    plt.xlabel('Validation Losses')
    plt.ylabel('Count')
    plt.title('Distribution of Validation Losses (IQR-based Range)')
    plt.legend()
    filename = f"{figures_training_path}{sep}val_losses_distribution.pdf"
    plt.savefig(filename, dpi=600)
    plt.show()


def anomaly_detection(val_losses_epoch, ys):

    if len(val_losses_epoch) > 0:

        plot_distribution(val_losses_epoch, ys)

        normal_losses = val_losses_epoch[ys == 0]
        anomalous_losses = val_losses_epoch[ys == 1]

        threshold_min = 0.0
        threshold_max = max(torch.quantile(normal_losses, 0.50), torch.quantile(anomalous_losses, 0.50))

        step = (threshold_max - threshold_min) / 1000
        thresholds = torch.arange(threshold_min, threshold_max, step)

        best_threshold = thresholds[0]
        best_auc = auroc(ys, detect_anomalies(val_losses_epoch, best_threshold))
        fpr, tpr, _ = roc(ys, detect_anomalies(val_losses_epoch, best_threshold))
        
        for threshold in thresholds:
            yhats = detect_anomalies(val_losses_epoch, threshold)
            yhats = yhats.to("cpu")
            auc = auroc(ys, yhats)
            #print(f"Threshold: {threshold}, AUC: {auc}")
            if auc > best_auc:
                print(f"New best threshold: {threshold}, AUC: {auc}")
                best_auc = auc
                best_threshold = threshold
                fpr, tpr, _ = roc(ys, yhats)
        
    return best_threshold, best_auc, fpr, tpr

def plot_auc(fpr, tpr, auroc):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auroc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":

    plot = False
    split_val = True
    filepath = "datasets/AbnormalHeartbeat/AbnormalHeartbeat_TRAIN.arff"
    xs, ys = read_arff(filepath)
    if plot:
        idx = random.randint(0, len(xs)-1)
        signal, label = xs[idx], ys[idx]
        plot_signal(signal, label)

    if split_val:
        split_train_val(xs, ys, val_size=0.2, random_state=42)
        

    
