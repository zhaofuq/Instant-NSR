import time
import mcubes
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        # TODO: if bound < radius, some rays may not intersect with the bbox. (should set near = far = inf ... or far < near)
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        near = torch.clamp(near, min=0.05)

    return near, far


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

class NeRFRenderer(nn.Module):
    def __init__(self,
                 cuda_ray=False,
                 curvature_loss = False
                 ):
        super().__init__()

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        self.curvature_loss = curvature_loss
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([128 + 1] * 3) # +1 because we save values at grid
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(64, 2, dtype=torch.int32) # 64 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d, bound):
        raise NotImplementedError()
    
    def forward_color(self, x, d, n,geo_feat, bound):
        raise NotImplementedError()
    
    def forward_sdf(self, x, bound):
        raise NotImplementedError()
    
    def finite_difference_normals_approximator(self, x, bound, epsilon):
        raise NotImplementedError()
    
    def forward_variance(self):
        raise NotImplementedError()
    
    def gradient(self, x, bound, epsilon = 0.0005):
        raise NotImplementedError()

    def density(self, x, bound):
        raise NotImplementedError()
    
    def run(self, rays_o, rays_d, num_steps, bound, upsample_steps, bg_color, cos_anneal_ratio = 1.0, normal_epsilon_ratio = 1.0):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)

        # sample steps
        near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
     
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)# [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = near + (far - near) * z_vals # [N, T], in [near, far]

        # perturb z_vals
        sample_dist = (far - near) / num_steps
        if self.training:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 3] -> [N, T, 3]
        pts = pts.clamp(-bound, bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                sdf_nn_output = self.forward_sdf(pts.reshape(-1, 3), bound)
                sdf = sdf_nn_output[:, :1]
                sdf = sdf.reshape(N, num_steps) # [N, T]
                
                for i in range(upsample_steps // 16):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, 16, 64 * 2 **i)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, bound, last=(i + 1 == upsample_steps // 16))
                    

            num_steps += upsample_steps

        ### render core
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # [N, T-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :1])], dim=-1)

        # sample pts on new z_vals
        z_vals_mid = (z_vals[:, :-1] + 0.5 * deltas[:, :-1]) # [N, T-1]
        z_vals_mid = torch.cat([z_vals_mid, z_vals[:,-1:]], dim=-1)

        new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals_mid.unsqueeze(-1) # [N, 1, 3] * [N, t, 3] -> [N, t, 3]
        new_pts = new_pts.clamp(-bound, bound)
 
        # only forward new points to save computation
        new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)

        sdf_nn_output = self.forward_sdf(new_pts.reshape(-1, 3), bound)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradient = self.gradient(new_pts.reshape(-1, 3), bound, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()
        normal =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))

        color = self.forward_color(new_pts.reshape(-1, 3), new_dirs.reshape(-1, 3), normal.reshape(-1, 3), feature_vector, bound)

        inv_s = self.forward_variance()     # Single parameter
        inv_s = inv_s.expand(N * num_steps, 1)

        true_cos = (new_dirs.reshape(-1, 3) * normal).sum(-1, keepdim=True)
    
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        # version relu
        # iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
        #             F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        
        # version Softplus
        activation = nn.Softplus(beta=100)
        iter_cos = -(activation(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    activation(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        # Equation 13 in NeuS
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).reshape(N, num_steps).clip(0.0, 1.0)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([N, 1],device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        weights_sum = weights.sum(dim=-1, keepdim=True)
        # calculate color 
        color = color.reshape(N, num_steps, 3) # [N, T, 3]
        image = (color * weights[:, :, None]).sum(dim=1)

        # calculate normal 
        normal_map = normal.reshape(N, num_steps, 3) # [N, T, 3]
        normal_map = torch.sum(normal_map * weights[:, :, None], dim=1)
        
        # calculate depth 
        ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # TODO:Eikonal loss 
        pts_norm = torch.linalg.norm(new_pts.reshape(-1, 3), ord=2, dim=-1, keepdim=True).reshape(N, num_steps)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        gradient_error = (torch.linalg.norm(gradient.reshape(N, num_steps, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        assert (gradient == gradient).all(), 'Nan or Inf found!'

        if self.curvature_loss:
            # TODO:curvature loss 
            random_vec = 2.0 * torch.randn_like(normal) - 1.0
            random_vec_norm = random_vec / (1e-5 + torch.linalg.norm(random_vec, ord=2, dim=-1,  keepdim = True))

            perturbed_pts = new_pts.reshape(-1, 3) + torch.cross(normal, random_vec_norm) * 0.01 * (1.0 - normal_epsilon_ratio) # naively set perturbed points, 
            perturbed_gradient = self.gradient(perturbed_pts.reshape(-1, 3), bound, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()
            perturbed_normal =  perturbed_gradient / (1e-5 + torch.linalg.norm(perturbed_gradient, ord=2, dim=-1,  keepdim = True))

            curvature_error = (torch.sum(normal * perturbed_normal, dim = -1) - 1.0) ** 2
            curvature_error = (relax_inside_sphere * curvature_error.reshape(N, num_steps)).sum() / (relax_inside_sphere.sum() + 1e-5)
        else:
            curvature_error = 0.0

        # mix background color
        if bg_color is None:
            bg_color = 1
    
        image = image + (1 - weights_sum) * bg_color
        
        depth = depth.reshape(B, N)
        image = image.reshape(B, N, 3)

        return depth, image, normal_map, gradient_error, curvature_error

    def run_cuda(self, rays_o, rays_d, num_steps, bound, upsample_steps, bg_color, cos_anneal_ratio, normal_epsilon_ratio):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if bg_color is None:
            bg_color = 1

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 64]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, self.training, 128, False)

            sdf_nn_output = self.forward_sdf(xyzs, bound).float()
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]
            

            # gradient_ = self.gradient(xyzs, bound).squeeze()
            # normal_ =  gradient_ / (1e-5 + torch.linalg.norm(gradient_, ord=2, dim=-1,  keepdim = True))

            gradient = self.finite_difference_normals_approximator(xyzs, bound, 0.005 * (1.0 - normal_epsilon_ratio)).float() #  
            normal =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))

            rgbs = self.forward_color(xyzs, dirs, normal, feature_vector, bound).float()
            inv_s = self.forward_variance()     # Single parameter

            true_cos = (dirs * normal).sum(-1, keepdim=True)
            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * deltas.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * deltas.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            # Equation 13 in NeuS
            alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

            weights_sum, image = raymarching.composite_rays_train(alpha, rgbs, deltas, rays, bound)

            # composite bg (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = None # currently training do not requires depth

            # TODO:Eikonal loss 
            pts_norm = torch.linalg.norm(xyzs.reshape(-1, 3), ord=2, dim=-1)
            inside_sphere = (pts_norm < 1.0).float().detach()
            relax_inside_sphere = (pts_norm < 1.2).float().detach()

            gradient_error = (torch.linalg.norm(gradient, ord=2,dim=-1) - 1.0) ** 2
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
            #gradient_error = 0.0

            normal_map = torch.zeros_like(image)

        else:
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            
            weights_sum = torch.zeros(B * N, dtype=dtype, device=device)
            depth = torch.zeros(B * N, dtype=dtype, device=device)
            image = torch.zeros(B * N, 3, dtype=dtype, device=device)
            normal_map = torch.zeros(B * N, 3, dtype=dtype, device=device)

            gradient_error = 0.0
            
            n_alive = B * N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
            near = near.view(B * N)
            far = far.view(B * N)

            step = 0
            i = 0
            while step < 1024: # max step

                # count alive rays 
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = near
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(B * N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, bound, self.density_grid, self.mean_density, near, far, 128)
                #sigmas, rgbs = self(xyzs, dirs, bound=bound)
                sdf_nn_output = self.forward_sdf(xyzs, bound).float()
                sdf = sdf_nn_output[:, :1]
                feature_vector = sdf_nn_output[:, 1:]

                # gradient_ = self.gradient(xyzs, bound).squeeze()
                # normal_ =  gradient_ / (1e-5 + torch.linalg.norm(gradient_, ord=2, dim=-1,  keepdim = True))

                gradient = self.finite_difference_normals_approximator(xyzs, bound, 0.005 * (1.0 - normal_epsilon_ratio)).float() #  
                normal =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))

                rgbs = self.forward_color(xyzs, dirs, normal, feature_vector, bound).float()

                inv_s = self.forward_variance() # Single parameter

                true_cos = (dirs * normal).sum(-1, keepdim=True)
                # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
                # the cos value "not dead" at the beginning training iterations, for better convergence.
                iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                            F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

                # Estimate signed distances at section points

                estimated_next_sdf = sdf + iter_cos * deltas[:,:1] * 0.5
                estimated_prev_sdf = sdf - iter_cos * deltas[:,:1] * 0.5

                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

                # Equation 13 in NeuS
                alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], alpha, rgbs, normal, deltas, weights_sum, depth, image, normal_map)
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}')
                step += n_step
                i += 1

            # composite bg & rectify depth (shade_kernel_nerf)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - near, min=0) / (far - near)


        image = image.reshape(B, N, 3)
        if depth is not None:
            depth = depth.reshape(B, N)

        normal_map = normal_map.reshape(B, N, 3)

        return depth, image, normal_map, gradient_error, 0

    def update_extra_state(self, bound, decay=0.95):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid
        resolution = self.density_grid.shape[0]
        
        X = torch.linspace(-bound, bound, resolution).split(128)
        Y = torch.linspace(-bound, bound, resolution).split(128)
        Z = torch.linspace(-bound, bound, resolution).split(128)

        tmp_grid = torch.zeros_like(self.density_grid)

        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)

                        sdf = self.density(pts.to(tmp_grid.device), bound)[:n].detach().float()
                        inv_s = 512.0#self.forward_variance().detach() / 10    # Single parameter

                        #density = -1.0 * sdf
                        mask = sdf > 0
                        density = torch.zeros_like(sdf)
                        density[mask] = inv_s * torch.exp(-inv_s * sdf[mask]) / (1 + torch.exp( - inv_s * sdf[mask]))
                        density[~mask] = inv_s * torch.exp(inv_s * sdf[~mask]) / (1 + torch.exp(inv_s * sdf[~mask]))
                        tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density.reshape(lx, ly, lz)
        
        # smooth by maxpooling
        tmp_grid = F.pad(tmp_grid, (0, 1, 0, 1, 0, 1))
        tmp_grid = F.max_pool3d(tmp_grid.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=1).squeeze(0).squeeze(0)

        # ema update
        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
        self.mean_density = torch.mean(self.density_grid).item()
        self.iter_density += 1

        ### update step counter
        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f} | [step counter] mean={self.mean_count} | [SDF] inv_s={inv_s:.4f}')

    def render(self, rays_o, rays_d, num_steps, bound, upsample_steps, staged=False, max_ray_batch=4096, bg_color=None, cos_anneal_ratio = 1.0, normal_epsilon_ratio = 1.0, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            normal = torch.empty((B, N, 3), device=device)

            gradient_error = 0.0
            curvature_error = 0.0 

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)

                    depth_, image_, normal_, gradient_error_, curvature_error_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, bound, upsample_steps, bg_color, 
                                                                cos_anneal_ratio = cos_anneal_ratio, normal_epsilon_ratio = normal_epsilon_ratio)
      
                    depth[b:b+1, head:tail] = depth_.detach()
                    image[b:b+1, head:tail] = image_.detach()
                    normal[b:b+1, head:tail] = normal_.detach()
                    gradient_error_ = gradient_error_.detach()
                    head += max_ray_batch

                    del depth_, image_, normal_, gradient_error_, curvature_error_
        else:
            depth, image, normal, gradient_error, curvature_error = _run(rays_o, rays_d, num_steps, bound, upsample_steps, bg_color, cos_anneal_ratio, normal_epsilon_ratio)

        results = {}
        results['depth'] = depth
        results['rgb'] = image
        results['normal'] = normal
        results['gradient_error'] = gradient_error
        results['curvature_error'] = curvature_error
        
        return results

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)

        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, bound, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.forward_sdf(pts.reshape(-1, 3), bound)[...,:1].reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
