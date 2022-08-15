import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from .renderer_sdf import NeRFRenderer
from encoding import get_encoder

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 cuda_ray=False,
                 ):
        super().__init__(cuda_ray)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.norm_dim = 3

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )

        self.sdf_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim, #+ self.norm_dim 
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim + 6
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.deviation_net = SingleVarianceNetwork(0.3)

    def forward_sdf(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]
        # sdf
        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder(x)
        sdf_output = self.sdf_net(x)

        return sdf_output

    def forward_color(self, x, d, n, geo_feat, bound):

        # dir
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # color
        h = torch.cat([x, d, n, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color
    
    def forward_variance(self):
        inv_s = self.deviation_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        return inv_s

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder(x)
        sdf_output = self.sdf_net(x)
        sdf = sdf_output[:,:1]

        return sdf

    def gradient(self, x, bound, epsilon = 0.0005):
        x.requires_grad_(True)
        x_ = (x + bound) / (2 * bound) # to [0, 1]
        h = self.encoder(x_)
        y = self.sdf_net(h)[..., 0]
        
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def finite_difference_normals_approximator(self, x, bound, epsilon = 0.0005):
        # finite difference
        # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
        pos_x = x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_pos = self.forward_sdf(pos_x.clamp(-bound, bound), bound)[:,:1]
        pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
        dist_dy_pos = self.forward_sdf(pos_y.clamp(-bound, bound), bound)[:,:1]
        pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
        dist_dz_pos = self.forward_sdf(pos_z.clamp(-bound, bound), bound)[:,:1]

        neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_neg = self.forward_sdf(neg_x.clamp(-bound, bound), bound)[:,:1]
        neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
        dist_dy_neg  = self.forward_sdf(neg_y.clamp(-bound, bound), bound)[:,:1]
        neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
        dist_dz_neg  = self.forward_sdf(neg_z.clamp(-bound, bound), bound)[:,:1]

        pos = torch.cat([pos_x, pos_y, pos_z, neg_x, neg_y, neg_z],dim=0)
        dist_pos = self.forward_sdf(pos.clamp(-bound, bound), bound)[:,:1]
        dist_dx_pos = dist_pos[]

   
        return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)

