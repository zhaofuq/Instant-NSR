from builtins import print
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from .renderer_sdf import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound = 1.0,
                 geometric_init = True,
                 weight_norm = True,
                 cuda_ray=False,
                 include_input = True,
                 curvature_loss = False
                 ):
        super().__init__(cuda_ray, curvature_loss)

        # sdf network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.include_input = include_input

        self.encoder, self.in_dim = get_encoder(encoding)

        sdf_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + 3 if self.include_input else self.in_dim
  
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sdf_net.append(nn.Linear(in_dim, out_dim))

            if geometric_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(sdf_net[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(sdf_net[l].bias, 0)     

                elif l==0:
                    if self.include_input:
                        torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                        torch.nn.init.normal_(sdf_net[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                        torch.nn.init.constant_(sdf_net[l].weight[:, 3:], 0.0)
                    else:
                        torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                        torch.nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

                else:
                    torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                    torch.nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                sdf_net[l] = nn.utils.weight_norm(sdf_net[l])

        self.sdf_net = nn.ModuleList(sdf_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color = self.in_dim_color + self.geo_feat_dim + 6 # hash_feat + dir + geo_feat + normal(sdf gradiant) 32 + 

        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

            if weight_norm:
                color_net[l] = nn.utils.weight_norm(color_net[l])

        self.color_net = nn.ModuleList(color_net)
        self.deviation_net = SingleVarianceNetwork(0.3)

        self.activation = nn.Softplus(beta=100)
    
    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # sigma
        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sdf_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def forward_sdf(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]
        # sdf
        h = self.encoder(x, bound)

        if self.include_input:
            h = torch.cat([x, h], dim=-1)

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                #h = F.relu(h, inplace=True)
        sdf_output = h

        return sdf_output
    
    def forward_color(self, x, d, n, geo_feat, bound):
        # dir
        #d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # color x, 
        h = torch.cat([x, d, n, geo_feat], dim=-1)
    
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color
    
    def forward_variance(self):
        inv_s = self.deviation_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        return inv_s

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]
        h = self.encoder(x, bound)

        if self.include_input:
            h = torch.cat([x, h], dim=-1)

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.activation(h)
        sdf = h[..., 0]
        return sdf

    def gradient(self, x, bound, epsilon=0.0005):
        #not allowed auto gradient, using fd instead
        return self.finite_difference_normals_approximator(x, bound, epsilon)

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

        return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)
