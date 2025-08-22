import torch 
from torch import nn 
import pytorch_lightning as pl
from collections.abc import Sequence 
import numpy as np
import matplotlib.pyplot as plt
import time 


class LightningAutoEncoder(pl.LightningModule):
    def __init__(self, model, model_name, optimizer, criterion="mse_sum", task = "reconstruction", scheduler=None, kld_weight=0.005, plot_training=False):

        super(LightningAutoEncoder, self).__init__()

        tasks = ["reconstruction", "denoising", "inpainting", "anomaly detection", "generation"]
        if task not in tasks:
            raise ValueError(f"Task must be one of {tasks}, but got {task}")
        else:
            self.task = task

        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        if criterion == "mse_sum":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion == "mse_mean":
            self.criterion = nn.MSELoss(reduction="mean")
        elif criterion == "l1":
            self.criterion = nn.L1Loss(reduction="mean")

        self.scheduler = scheduler
        self.threshold = None  
        self.kld_weight = kld_weight
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

        self.val_losses_epoch = []
        self.train_losses_epoch = []
        self.ys = [] 
        self.tpr = []
        self.fpr = []

        if "v" in self.model_name:
            self.variational = True             
            if "q" in self.model_name:
                self.quantized = True
            else:
                self.quantized = False
        else: 
            self.variational = False

        self.plot_training = plot_training
        if self.plot_training:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.ax = ax
            self.fig = fig
            self.fig.show()
            self.line = None
            self.line_rec = None
            self.line_original = None
            self.x = None
            self.x_original = None
            self.x_hat = None
            self.selected_index = None
            self.is_first = True
            plt.ion()
    

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        self.model.train()
        
        x, y = batch
        
        if self.task in ["inpainting", "denoising"]:
            x_original, x_input = x
        else:
            x_original = x_input = x

        x_hat = self(x_input)
            
        if isinstance(x_hat, Sequence):
            if self.quantized:
                x_hat, x_input, vq_loss = x_hat
            else:
                x_hat, mu, logvar = x_hat

        if x_hat.ndim == 2:
            x_hat = x_hat.unsqueeze(1)

        if self.variational:
            if self.quantized: 
                args = (x_original, x_hat, vq_loss)
                losses = self.model.loss_function(self.criterion, *args)
                self.log('vq_loss', vq_loss, prog_bar=True)
            else:
                args = (x_original, x_hat, mu, logvar)
                losses = self.model.loss_function(self.criterion, *args, M_N = self.kld_weight)
                kld_loss = losses['kld_loss']
                self.log('kld_loss', kld_loss, prog_bar=True)

            rec_loss = losses['recon_loss']
            loss = losses['loss']
            self.log('rec_loss', rec_loss, prog_bar=True)
            self.log('loss', loss, prog_bar=True)
            self.train_losses_epoch.append(rec_loss.item())
        else:
            #if batch_idx == 0:
            #    plt.figure()
            #    plt.plot(x_input[0, 0, :].cpu().detach().numpy().flatten(), label='Original Signal', color = "green", alpha=1.0)
            #    plt.show()
            loss = self.criterion(x_hat, x_original)
            self.train_losses_epoch.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        if self.train_losses_epoch != []:
            self.train_losses_epoch = torch.from_numpy(np.array(self.train_losses_epoch))
            train_loss_epoch = torch.mean(self.train_losses_epoch) 
            self.train_losses.append(train_loss_epoch)
            self.log('train_loss', train_loss_epoch, prog_bar=True) 
            if self.task == "anomaly detection":
                self.log('loss_opt', train_loss_epoch, prog_bar=True)
            self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)  
            self.train_losses_epoch = []   

    def on_validation_epoch_start(self):
        self.val_losses_epoch = []   
        self.is_first = True
        self.ys = []

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        criterion_val = nn.MSELoss(reduction="mean")
        with torch.no_grad():
            x, y = batch
            
            if self.task in ["inpainting", "denoising"]:
                x_original, x_input = x
            else:
                x_original = x_input = x
            
            for elem in y:
                self.ys.append(elem)

            x_hat = self(x_input)

            if isinstance(x_hat, Sequence):
                x_hat, mu, logvar = x_hat

            if x_hat.ndim == 2:
                x_hat = x_hat.unsqueeze(1)        

            if self.plot_training:

                if self.task in ["inpainting", "denoising"]:
                    if self.is_first:
                        if self.x is None and self.x_original is None or torch.equal(self.x, self.x_original):
                            # If x is None or equal to x_original, update with the first element
                            for i, elem in enumerate(x_input):
                                if not torch.equal(elem, x_original[i]):
                                    self.selected_index = i
                                    self.x_hat = x_hat[i][0, :]
                                    self.x = x_input[i][0, :]
                                    self.x_original = x_original[i][0, :]
                                    break
                        self.x_hat = x_hat[self.selected_index][0, :]
                        self.is_first = False
                else:
                    if self.is_first:
                        if self.x is None or torch.equal(self.x, self.x_original):
                            self.selected_index = 0
                            self.x_hat = x_hat[0][0, :]
                            self.x = x_input[0][0, :]
                            self.x_original = x_original[0][0, :]
                        self.x_hat = x_hat[self.selected_index][0, :]
                        self.is_first = False

            loss = criterion_val(x_original, x_hat)
            self.val_losses_epoch.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_losses_epoch) > 0:
            self.val_losses_epoch = torch.from_numpy(np.array(self.val_losses_epoch))
            val_loss_epoch = torch.mean(self.val_losses_epoch) 
            if self.task != "anomaly detection":
               self.log('loss_opt', val_loss_epoch, prog_bar=True)
            self.val_losses.append(val_loss_epoch)
            self.log('val_loss', val_loss_epoch, prog_bar=True)
        
        if self.plot_training:
        
            if self.line_original is None:
                if self.x_original is not None:
                    self.line_original, = self.ax.plot(self.x_original.cpu().numpy().flatten(), label='Original Signal', color = "green", alpha=1.0)
            else:
                self.line_original.set_ydata(self.x_original.cpu().numpy().flatten())
            if self.line is None:
                if self.x is not None:  
                    self.line, = self.ax.plot(self.x.cpu().numpy().flatten(), label='Input Signal', color='blue', alpha=0.5)
            else:
                self.line.set_ydata(self.x.cpu().numpy().flatten())
            if self.line_rec is None:
                if self.x_hat is not None:
                    self.line_rec, = self.ax.plot(self.x_hat.cpu().numpy().flatten(), label='Reconstructed Signal', color='orange', alpha=0.7)
            else:
                self.line_rec.set_ydata(self.x_hat.cpu().numpy().flatten())

            self.ax.set_title(f'Training Step {self.current_epoch}')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Amplitude')
            self.ax.legend()
            if self.x is not None and self.x_hat is not None and self.x_original is not None:
                min_y = min(self.x.cpu().numpy().min(), self.x_hat.cpu().numpy().min(), self.x_original.cpu().numpy().min())
                max_y = max(self.x.cpu().numpy().max(), self.x_hat.cpu().numpy().max(), self.x_original.cpu().numpy().max())
                self.ax.set_ylim(min_y, max_y)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.1)  # Pause to allow the plot to update

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.scheduler,
                    'monitor': 'loss_opt',
                    'frequency': 1
                }
            }
        return self.optimizer