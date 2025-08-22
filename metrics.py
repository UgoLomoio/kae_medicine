import torch 
from torch.functional import F
import numpy as np 
import time 
from torch import nn
import pandas as pd 
import warnings

def reshape_tensor(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(dim=0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(dim=0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(dim=0)
    return tensor

def reshape_tensors(tensor1, tensor2):
    tensor1 = reshape_tensor(tensor1)
    tensor2 = reshape_tensor(tensor2)
    return tensor1, tensor2



def evaluate_model(model, xs, device = "cuda", var = False, conv = False, return_losses = False):
    import torch, gc
    gc.collect()

    # Set global rounding precision to 6 decimal places
    np.set_printoptions(precision=6)
    
    # Ensure device is defined, e.g., device = torch.device('cuda')
    model = model.to(device)

    model.eval()  # Set model to evaluation mode

    xs = xs.clone()  # Clone the input tensor to avoid modifying the original data
    
    xs_rec = []
    n = len(xs)

    # No gradient tracking during evaluation
    with torch.no_grad():
        for i, x in enumerate(xs):
            if i % 100 == 0:
                print(f"{i+1} / {n}", end="\r")
            
            # Ensure x has a channel dimension if needed
            if x.ndim == 1:
                x = torch.unsqueeze(x, dim=0)
            # Ensure x has a batch dimension if needed
            if x.ndim == 2:
                x = torch.unsqueeze(x, dim=0)
            

            # Move data to GP
            x = x.to(device)

            # Run model
            if not var:
                x_rec = model(x)
            else:
                x_rec, _, _ = model(x)


            # Move the output back to CPU and store it
            xs_rec.append(x_rec.cpu())
            
            # Clean up GPU memory from temporary tensors
            del x, x_rec
            torch.cuda.empty_cache()
    
    # Compute evaluation metrics on CPU tensors
    if not var:
        xs.unsqueeze_(dim=1)
    elif not conv:
        xs.squeeze_(dim=1)

    xs_rec = torch.stack(xs_rec, dim=0)

    losses, mse, mse_std = compute_mse(xs_rec, xs, device = device)
    _, rmse, rmse_std = compute_rmse(xs_rec, xs, device = device)
    _, ssim, ssim_std = compute_ssim(xs_rec, xs, device = device)
    _, snr, snr_std = compute_snr(xs_rec, xs, device = device)
    _, psnr, psnr_std = compute_psnr(xs, xs_rec, device = device)
    _, mad, mad_std = compute_mads(xs_rec, xs, device = device)
    _, cos_sim, cos_sim_std = compute_cosine_sim(xs_rec, xs, device = device)

    params = count_parameters(model)
    t_test, t_mean, t_std = time_complexity(model, xs, device = device)
    
    metrics = [f"{mse}({mse_std})",f"{rmse}({rmse_std})", f"{ssim}({ssim_std})", f"{snr}({snr_std})", f"{psnr}({psnr_std})", f"{mad}({mad_std})", 
            f"{cos_sim}({cos_sim_std})", f"{params} M", f"{t_mean}({t_std})", t_test]

    if return_losses:
        return metrics, losses
    else:
        return metrics

def evaluate_models(models, xs, device = "cuda", return_losses = False):

    columns = ["Model", "MSE", "RMSE", "SSIM", "SNR", "PSNR", "MAD", "Cosine Sim.", "Params (M)", "Inference time", "Testing time"]
    df = pd.DataFrame(columns = columns)
    test_times = {}
    losses_dict = {}
    for i, (model_name, model) in enumerate(models.items()):
        print("Evaluating model: ", model_name)
        model.eval()
        model = model.to(device)
        if "v" in model_name:
            var = True
        else:
            var = False

        if "c" in model_name:
            conv = True
        else:
            conv = False
        
        xs = xs.to(device)
        if return_losses:
            metrics, losses = evaluate_model(model, xs, device = device, var = var, conv = conv, return_losses = True)
        else: 
            metrics = evaluate_model(model, xs, device = device, var = var, conv = conv)
        t_test = float(metrics[-1])
        test_times[model_name] = t_test
        row_df = [model_name]
        [row_df.append(metric) for metric in metrics] 
        df.loc[i] = row_df
        losses_dict[model_name] = losses if return_losses else None
    if return_losses:
        return df, test_times, losses_dict
    else:
        return df, test_times

def compute_cosine_sim(reconstructeds, originals, device = "cuda"):
    
    cos_sims = []
    for original, reconstructed in zip(originals, reconstructeds):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        original = original.to(device)
        reconstructed = reconstructed.to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        sim = cos(original, reconstructed)
        cos_sims.append(sim.item())
    return cos_sims, np.mean(cos_sims), np.std(cos_sims) 


def mad(reconstructed, original):

    # Compute the absolute differences
    abs_diff = torch.abs(original - reconstructed)

    # Find the maximum absolute difference
    max_abs_distance = torch.max(abs_diff)
    return max_abs_distance

def compute_mads(reconstructeds, originals, device = "cuda"):

    mads = []
    for original, reconstructed in zip(originals, reconstructeds):
        original = original.to(device)
        reconstructed = reconstructed.to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        max_abs_distance = mad(original, reconstructed)
        mads.append(max_abs_distance.item())
    return mads, np.mean(mads), np.std(mads)  


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def compute_rmse(reconstructeds, originals, device = "cuda"):
    losses = []
    for i, reconstructed in enumerate(reconstructeds):
        original = originals[i].to(device)
        reconstructed = reconstructed.to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        rmse_loss = RMSELoss(reconstructed, original)
        losses.append(rmse_loss.item())
    return losses, np.mean(losses), np.std(losses)

def compute_mse(reconstructeds, originals, device = "cuda"):
    losses = []
    for i, reconstructed in enumerate(reconstructeds):
        original = originals[i].to(device)
        reconstructed = reconstructed.to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        mse_loss = F.mse_loss(reconstructed, original)
        losses.append(mse_loss.item())
    return losses, np.mean(losses), np.std(losses)

def ssim_2d(signal1, signal2):
    """
    Compute the Structural Similarity Index (SSIM) for two-dimensional signals.

    Parameters:
    - signal1, signal2: Input signals.

    Returns:
    - ssim_index: Structural Similarity Index between the two signals.
    """

    # Ensure the signals have the same shape
    if signal1.shape != signal2.shape:
        raise ValueError("Input signals must have the same shape, but got {} and {}".format(signal1.shape, signal2.shape))

    # Constants for SSIM calculation
    C1 = (0.01 * np.amax(signal1) - np.amin(signal1))**2
    C2 = (0.01 * np.amax(signal2) - np.amin(signal2))**2

    # Mean and variance
    mu1 = np.mean(signal1)
    mu2 = np.mean(signal2)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.var(signal1)
    sigma2_sq = np.var(signal2)
    sigma12 = np.cov(signal1.flatten(), signal2.flatten())[0, 1]

    # SSIM calculation
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    #print(num, den)
    ssim_index = num / den

    return ssim_index

def compute_ssim(reconstructeds, originals, device = "cuda"):

    ssims = []
    for i, reconstructed in enumerate(reconstructeds):
        reconstructed = reconstructed.to(device)
        original = originals[i].to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        reconstructed = reconstructed.cpu().detach().numpy()
        original = original.cpu().detach().numpy()
        ssim = ssim_2d(original, reconstructed)
        ssims.append(ssim)
    return ssims, np.mean(ssims), np.std(ssims)

def psnr_2d(original, reconstructed):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for two-dimensional signals.

    Parameters:
    - original: Original signal.
    - reconstructed: Reconstructed signal.

    Returns:
    - psnr_value: PSNR value between the original and reconstructed signals.
    """

    # Ensure the signals have the same shape
    if original.shape != reconstructed.shape:
        raise ValueError(f"Input signals must have the same shape, but got {original.shape} and {reconstructed.shape}")

    # Calculate the mean squared error
    mse = np.mean((original - reconstructed)**2)

    # The maximum possible pixel value (assuming the signal is in the range [0, 1])
    max_pixel_value = 1.0

    # Calculate PSNR
    psnr_value = 10 * np.log10((max_pixel_value**2) / mse)

    return psnr_value

def compute_psnr(originals, reconstructeds, device = "cuda"):

    psnrs = []
    for i, reconstructed in enumerate(reconstructeds):
        reconstructed = reconstructed.to(device)
        original = originals[i].to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        original = original.cpu().detach().numpy()
        reconstructed = reconstructed.cpu().detach().numpy()
        psnr = psnr_2d(original, reconstructed)
        psnrs.append(psnr)
    return psnrs, np.mean(psnrs), np.std(psnrs)

def signalPower(x):
    return np.average(x**2)

def SNR(reconstructed, original):
        
    noise = reconstructed-original
    powT = signalPower(original)
    powN = signalPower(noise)
    return 10*np.log10(powT/powN)

def compute_snr(reconstructeds, originals, device = "cuda"):
    
    snrs = []
    for reconstructed, original in zip(reconstructeds, originals):
        original = original.to(device)
        reconstructed = reconstructed.to(device)
        original, reconstructed = reshape_tensors(original, reconstructed)
        reconstructed = reconstructed.cpu().detach().numpy()
        original = original.cpu().detach().numpy()
        snr = SNR(reconstructed, original)
        snrs.append(snr)
    return snrs, np.mean(snrs), np.std(snrs)

def count_parameters(model):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    n_parameters = n_parameters / 1e6 # Convert to millions
    n_parameters = round(n_parameters, 2)
    return n_parameters

def time_complexity(model, originals, device = "cuda"):

    model.eval()
    model = model.to(device)
    # No gradient tracking during evaluation
    with torch.no_grad():
        ts = []
        for x in originals:
            # Ensure data has a batch dimension if needed
            if x.ndim == 1:
                x = torch.unsqueeze(x, dim=0).to(device)
            if x.ndim == 2:
                x = torch.unsqueeze(x, dim=0).to(device)
            if x.ndim == 4:
                x = x.squeeze(dim=0).to(device)
            t_start = time.time()
            _ = model(x)
            ts.append(time.time() - t_start)
    return np.sum(ts), np.mean(ts), np.std(ts)