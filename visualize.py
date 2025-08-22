from utils import read_arff, split_windows, add_noise, mask_input
from matplotlib import legend, pyplot as plt
import os 
import random 
import torch
from model_config import *

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

map_labels = {
    0: "Normal",
    1: "Abnormal"
}

if __name__ == "__main__":

    for dataset_name, files in datapaths.items():
           
        datapath = files["datapath"]
        test_filename = files["test_filename"]
        window_size = files["window_size"]
            
        xs_test, ys_test = read_arff(datapath + sep + test_filename)
        xs_test, ys_test = split_windows(xs_test, ys_test, window_size=window_size)
        idx = random.randint(0, len(xs_test)-1)
        
        x_test, y_test = xs_test[idx], ys_test[idx]
        x_test = torch.tensor(x_test, dtype=torch.float32)
        x_test = x_test.unsqueeze(0)
        x_test_n = add_noise(x_test, noise_level=noise_level, noise_prob=1.0)
        x_test_n = x_test_n.squeeze(0)
        x_test_m = mask_input(x_test, max_mask_size=max_mask_size, min_mask_size=min_mask_size, mask_prob=1.0, add_noise=False)
        x_test_m = x_test_m.squeeze(0)

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        x_test = x_test.squeeze(0)
        axs[0].plot(x_test, label='Original', color='blue')
        axs[0].set_title(f'Original Signal, Class: {map_labels[y_test.item()]}', fontsize = 24, fontweight='bold')
        axs[0].set_ylabel('Amplitude', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        axs[0].grid(True)
        legend = axs[0].legend(fontsize=14)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        axs[1].plot(x_test_n, label='Noisy', color='orange')
        axs[1].set_title('Noisy Signal', fontsize = 24, fontweight='bold')
        axs[1].set_ylabel('Amplitude', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        axs[1].grid(True)
        legend = axs[1].legend(fontsize=14)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        axs[2].plot(x_test_m, label='Masked', color='red')
        axs[2].set_title('Masked Signal', fontsize = 24, fontweight='bold')
        axs[2].set_ylabel('Amplitude', fontsize=14, fontweight='bold')
        axs[2].set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        axs[2].grid(True)
        legend = axs[2].legend(fontsize=14) 
        for text in legend.get_texts():
            text.set_fontweight('bold')
        plt.tight_layout()
        fig.savefig(cwd + sep + "input_example.png", dpi=600, bbox_inches='tight')
        
        plt.show()