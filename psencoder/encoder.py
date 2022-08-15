# rewrite phasor as an encoder for easy deployment 
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
import numpy as np
import pdb

class phasor(nn.Module):
    def __init__(self, resolutions, dimension, device='cuda', **kwargs) -> None:
        """Paramters: 
            resolutions: diserable resolutions
            num_comps: tiny dimensions
            levels: num of levels, default is 1, the largest 
            var_decay: the variance decay function during training
        """
        super().__init__()
        self.device = device
        self.res = torch.tensor(resolutions).to(self.device)
        self.res = self.res // 16
        s = 2
        num_comp = [math.ceil(np.log(N)/np.log(s))+1 for N in resolutions]
        self.D = dimension
        # self.axis = [torch.tensor([0.]+[s**i for i in torch.arange(d-1)]).to(self.device) for d in num_comp]
        self.axis = [torch.tensor([0.]+[i+1 for i in torch.arange(d-1)]).to(self.device) for d in num_comp]
        
        self.ktraj = self.compute_ktraj(self.axis, self.res)
        # self.iters = 0
        # self.max_iters = 1e4
        # self.decay_ratio = 1e-1
        
        self.alpha_params = nn.Parameter(torch.tensor([1e-3]).to(self.device))
        
        init_func= kwargs.get("init_func",False)
        self.get_params_init_func(init_func)

        self.params = nn.ParameterList(
            self.init_(self.res.long(), ksize=dimension, init_scale=1))
        self.output_dim = dimension

        print(self)

    def get_params_init_func(self,init_func):
        if init_func=="ours":
            self.init_func=False
        elif init_func=="uniform":
            self.init_func=nn.init.uniform_
        elif init_func=="normal":
            self.init_func=nn.init.normal_
        elif init_func=="zeros":
            self.init_func=torch.nn.init.zeros_
        elif init_func=="kaiming_uniform":
            self.init_func=torch.nn.init.kaiming_uniform_
        else:
            self.init_func=None

    def decay_variance(self, iter):
        return self.decay_ratio ** min(iter/self.max_iters, 1)

    def get_parameters(self):
        return [{
                "params": self.params,
                "lr": 1e-1,
                },
                {
                "params": self.alpha_params,
                "lr": 1e-3,
                }
            ]

    @property
    def phasor(self):
        # variance = (self.decay_variance(self.iters) * ( self.init_var /(self.res)**2)).to(self.device)
        # if self.multiplier is None or self.training:
        #   multiplier = self.gauss(variance)
        # else:
        #   multiplier = self.multiplier
        # feature = [feat * gau * self.alpha for feat, gau in zip(self.params, multiplier)]
        feature = [feat * self.alpha for feat in self.params]
        return feature


    def forward(self, inputs, variance=0, bound=1):
        inputs = inputs / bound # map to [-1, 1]
        try:
            assert inputs.max() <= 1 and inputs.min() >= -1
        except:
            pdb.set_trace()
        # pdb.set_trace()
        #self.iters += 1
        feature = self.compute_fft(self.phasor, inputs, interp=False)
        return feature.T

    def gauss(self, variance):
        if (variance == 0).all():
            return [1., 1., 1.]
        gauss = [torch.exp((-2*(np.pi*kk)**2*variance[None]).sum(-1)).reshape(1,1,*kk.shape[:-1]) for kk in self.ktraj]
        return gauss

    def mul_gauss(self, features, variance=0):
        return 

    @property
    def alpha(self):
        return F.softplus(self.alpha_params, beta=10, threshold=1)

    def compute_ktraj(self, axis, res):
        ktraj2d = [torch.fft.fftfreq(i, 1/i).to(self.device) for i in res]
        ktraj1d = [torch.arange(ax).to(torch.float).to(self.device) if type(ax) == int else ax for ax in axis]
        ktrajx = torch.stack(torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1)
        ktrajy = torch.stack(torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1)
        ktrajz = torch.stack(torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1)
        ktraj = [ktrajx, ktrajy, ktrajz]
        return ktraj        

    def tv_loss(self):
        # Parseval Loss
        new_feats = [Fk.reshape(-1, *Fk.shape[2:],1) * 1j * np.pi * wk.reshape(1, *Fk.shape[2:], -1) 
            for Fk, wk in zip(self.phasor, self.ktraj)]
        
        loss = sum([feat.abs().square().mean() for feat in itertools.chain(*new_feats)])
        return loss

    def compute_spatial_volume(self, features):
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in self.res]
        Fx, Fy, Fz = features
        Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx = irfft(torch.fft.ifftn(Fx, dim=(3,4), norm='forward'), xx, ff=kx, T=Nx, dim=2)
        fy = irfft(torch.fft.ifftn(Fy, dim=(2,4), norm='forward'), yy, ff=ky, T=Ny, dim=3)
        fz = irfft(torch.fft.ifftn(Fz, dim=(2,3), norm='forward'), zz, ff=kz, T=Nz, dim=4)
        return (fx, fy, fz)

    def compute_fft(self, features, xyz_sampled, interp=True):
        if interp:
            # Nx: num of samples
            # using interpolation to compute fft = (N*N) log (N) d  + (N*N*d*d) + Nsamples 
            fx, fy, fz = self.compute_spatial_volume(features)
            volume = fx+fy+fz
            points = F.grid_sample(volume, xyz_sampled[None, None, None].flip(-1), align_corners=True).view(-1, *xyz_sampled.shape[:1],)
            # this is somewhat expensive when the xyz_samples is few and a 3D volume stills need computed
        else:
            # this is fast because we did 2d transform and matrix multiplication . (N*N) logN d + Nsamples * d*d + 3 * Nsamples 
            Nx, Ny, Nz = self.res
            Fx, Fy, Fz = features
            d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
            kx, ky, kz = self.axis
            kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
            xs, ys, zs = xyz_sampled.chunk(3, dim=-1)
            Fx = torch.fft.ifftn(Fx, dim=(3,4), norm='forward')
            Fy = torch.fft.ifftn(Fy, dim=(2,4), norm='forward')
            Fz = torch.fft.ifftn(Fz, dim=(2,3), norm='forward')
            fx = grid_sample_cmplx(Fx.transpose(3,3).flatten(1,2), torch.stack([zs, ys], dim=-1)[None]).reshape(Fx.shape[1], Fx.shape[2], -1)
            fy = grid_sample_cmplx(Fy.transpose(2,3).flatten(1,2), torch.stack([zs, xs], dim=-1)[None]).reshape(Fy.shape[1], Fy.shape[3], -1)
            fz = grid_sample_cmplx(Fz.transpose(2,4).flatten(1,2), torch.stack([xs, ys], dim=-1)[None]).reshape(Fz.shape[1], Fz.shape[4], -1)
            fxx = batch_irfft(fx, xs, kx, Nx)
            fyy = batch_irfft(fy, ys, ky, Ny)
            fzz = batch_irfft(fz, zs, kz, Nz)
            return fxx+fyy+fzz

        return points

    def compute_normal(self, xyz_sampled):
        new_feat = [Fk.reshape(*Fk.shape[2:],1) * 1j * np.pi * wk.reshape(*Fk.shape[2:], -1) 
                for Fk, wk in zip(self.density[0], self.ktraj)]
        new_feat = [feat.permute(3,0,1,2).unsqueeze(0) for feat in new_feat]
        normal = self.compute_fft(new_feat, xyz_sampled, interp=False)
        normal = normal / (torch.linalg.norm(normal, dim=0, keepdims=True) + 1e-5)
        return normal


    def init_(self, res, ksize=1, init_scale=1):
        # transform the fourier domain to spatial domain
        # Fx, Fy, Fz = features
        Nx, Ny, Nz = res
        d1, d2, d3 = [len(dim) for dim in self.axis]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (Nx,Ny,Nz)]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        #variance = (self.init_var /(self.res)**2).to(self.device)
        gx, gy, gz = 1,1,1 #self.gauss(variance)
        # norms = torch.stack(torch.meshgrid([2*xx-1, 2*yy-1, 2*zz-1]), dim=-1).norm(dim=-1)
        
        fx = torch.ones(1, ksize, len(xx), Ny, Nz).to(self.device)
        fy = torch.ones(1, ksize, Nx, len(yy), Nz).to(self.device)
        fz = torch.ones(1, ksize, Nx, Ny, len(zz)).to(self.device)
        norms = torch.stack(torch.meshgrid([2*xx-1, 2*yy-1, 2*zz-1]), dim=-1).norm(dim=-1)
        # norms = 0.5 - norms
        fx = fx * norms[None, None] * init_scale / 3 / np.sqrt(self.D) / self.alpha
        fy = fy * norms[None, None] * init_scale / 3 / np.sqrt(self.D) / self.alpha
        fz = fz * norms[None, None] * init_scale / 3 / np.sqrt(self.D) / self.alpha
        # pdb.set_trace()
        fxx = rfft(torch.fft.fftn(fx.transpose(2,4), dim=(2,3), norm='forward'),xx, ff=kx, T=Nx).transpose(2,4)
        fyy = rfft(torch.fft.fftn(fy.transpose(3,4), dim=(2,3), norm='forward'),yy, ff=ky, T=Ny).transpose(3,4)
        fzz = rfft(torch.fft.fftn(fz.transpose(4,4), dim=(2,3), norm='forward'),zz, ff=kz, T=Nz).transpose(4,4)
        fxx = fxx / (gx + 1e-5)
        fyy = fyy / (gy + 1e-5)
        fzz = fzz / (gz + 1e-5)
        if self.init_func:
            self.init_func(fxx)
            self.init_func(fyy)
            self.init_func(fzz)
            return [torch.nn.Parameter(fxx), torch.nn.Parameter(fyy), torch.nn.Parameter(fzz)]
        else:
            # desirable output
            # Fx = irfft(torch.fft.ifftn(fxx, dim=(3,4), norm='forward'), xx, ff=kx, T=Nx, dim=2)
            return [torch.nn.Parameter(fxx), torch.nn.Parameter(fyy), torch.nn.Parameter(fzz)]


def irfft(phasors, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    phasors = phasors.transpose(dim, -1)
    assert phasors.shape[-1] == len(ff) if ff is not None else True
    device = phasors.device
    xx = xx * (T-1) / T                       # to match torch.fft.fft
    N = phasors.shape[-1]
    if ff is None:
        ff = torch.arange(N).to(device)       # positive freq only
    xx = xx.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * xx * ff).to(device)
    # indexing in pytorch is slow
    # pdb.set_trace()
    # M[:, 1:-1] = M[:, 1:-1] * 2                # Hermittion symmetry # inplace operation!!!
    M = M * ((ff>0)+1)[None]
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


def batch_irfft(phasors, xx, ff, T):
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx+1) * 0.5
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j*np.pi*xx * ff)                   # twiddle factor
    twiddle = twiddle * ((ff > 0)+1)[None]
    # twiddle[:,1:-1] = twiddle[:, 1:-1] * 2                    # hermitian # [N, d] # inplace operation
    twiddle = twiddle.transpose(0,1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)


def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    sampled = F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
            1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)
    # pdb.set_trace()
    # return grid_sample(input, grid)
    return sampled





def rfft(spatial, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1/T) # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out



import torch
import torch.nn.functional as F

def grid_sample(image, grid, **kwargs):
    N, C, IH, IW = image.shape
    _, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    # pdb.set_trace()q
    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().reshape(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().reshape(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().reshape(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().reshape(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W) + 
               ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W) +
               sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W) +
               se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W))

    return out_val


def getMask_fft(smallSize, largeSize):
    ph_max = [torch.fft.fftfreq(i, 1/i).max() for i in smallSize]
    ph_min = [torch.fft.fftfreq(i, 1/i).min() for i in smallSize]
    tg_ff = torch.stack(torch.meshgrid([torch.fft.fftfreq(i, 1/i) for i in largeSize]))
    mask = torch.ones(largeSize).to(torch.bool)
    for i in range(len(smallSize)):
        mask &= (tg_ff[i] <= ph_max[i]) & (tg_ff[i] >= ph_min[i])
    assert np.array(smallSize).prod() == mask.sum()
    return mask