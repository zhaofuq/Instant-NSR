import torch

from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096) # lower if OOM
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
<<<<<<< HEAD
    
    parser.add_argument('--radius', type=float, default=2, help="assume the camera is located on sphere(0, radius))")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-size, size)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch (unstable now)")
=======
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    # the default setting for fox.
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
>>>>>>> 197fd4d382697c0c67b3d4493706f6a015c111d8

    opt = parser.parse_args()

    print(opt)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
<<<<<<< HEAD
        Network = NeRFNetwork_FF
    else:
        Network = NeRFNetwork
    seed_everything(opt.seed)

    model = Network(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        density_grid_size=128 if opt.cuda_ray else -1,
=======
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork
        
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
>>>>>>> 197fd4d382697c0c67b3d4493706f6a015c111d8
    )

    print(model)
    
    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

    # save mesh
<<<<<<< HEAD
    if opt.save_mesh:
        trainer.save_mesh()

    # render images on test dataset
    test_dataset = NeRFDataset(opt.path, 'test', radius=opt.radius, n_test=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        trainer.test(test_loader)
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
=======
    #trainer.save_mesh()
    test_dataset = NeRFDataset(opt.path, 'test', radius=opt.radius, n_test=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
>>>>>>> 197fd4d382697c0c67b3d4493706f6a015c111d8