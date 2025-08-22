"""
Convolutional Kolmogorov-Arnold network (KANConv1DLayer) block from: https://github.com/IvanDrokin/torch-conv-kan/blob/main/kan_convs/kan_conv.py
"""

import torch 
from torch import nn

class KANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1],
                 **norm_kwargs):
        super(KANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class((grid_size + spline_order) * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
            x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))
        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y

class KANConv1DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(KANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range,**norm_kwargs)
        

class KCN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        kernel_sizes,
        strides,
        paddings,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KCN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            self.layers.append(
                KANConv1DLayer(
                    in_features,
                    out_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )



class KANConvTransposeNDLayer(nn.Module):
    def __init__(self, conv_transpose_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, output_padding=0,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1],
                 **norm_kwargs):
        super(KANConvTransposeNDLayer, self).__init__()
        
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs
        
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        
        # Base transpose convolution layers
        self.base_conv = nn.ModuleList([conv_transpose_class(input_dim // groups,
                                                           output_dim // groups,
                                                           kernel_size,
                                                           stride,
                                                           padding,
                                                           output_padding,
                                                           groups=1,
                                                           bias=False,
                                                           dilation=dilation) for _ in range(groups)])
        
        # Spline transpose convolution layers
        self.spline_conv = nn.ModuleList([conv_transpose_class((grid_size + spline_order) * input_dim // groups,
                                                             output_dim // groups,
                                                             kernel_size,
                                                             stride,
                                                             padding,
                                                             output_padding,
                                                             groups=1,
                                                             bias=False,
                                                             dilation=dilation) for _ in range(groups)])
        
        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])
        
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        
        # Initialize weights using Kaiming uniform distribution
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
    
    def forward_kan(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations
        
        # Compute the basis for the spline using intervals and input values
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(x.device)
        
        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)
        
        # Compute the spline basis over multiple orders
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                              right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))
        
        return x
    
    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind)
            output.append(y.clone())
        
        y = torch.cat(output, dim=1)
        return y

class KANConvTranspose1DLayer(KANConvTransposeNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, 
                 dilation=1, output_padding=0, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], 
                 norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(KANConvTranspose1DLayer, self).__init__(nn.ConvTranspose1d, norm_layer,
                                                     input_dim, output_dim,
                                                     spline_order, kernel_size,
                                                     groups=groups, padding=padding, stride=stride, 
                                                     dilation=dilation, output_padding=output_padding,
                                                     ndim=1,
                                                     grid_size=grid_size, base_activation=base_activation,
                                                     grid_range=grid_range, **norm_kwargs)

class KCNTranspose(nn.Module):
    def __init__(
        self,
        layers_hidden,
        kernel_sizes,
        strides,
        paddings,
        output_paddings=None,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KCNTranspose, self).__init__()
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        
        # Set default output_paddings if not provided
        if output_paddings is None:
            output_paddings = [0] * (len(layers_hidden) - 1)
        
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            output_padding = output_paddings[i]
            
            self.layers.append(
                KANConvTranspose1DLayer(
                    in_features,
                    out_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

