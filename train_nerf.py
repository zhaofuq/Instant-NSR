import torch

from nerf.provider import NeRFDataset, RayDataset
from nerf.utils import *

import argparse

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    #Basic Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--mode', type=str, default='train', help="running mode, supports (train, mesh, render)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=64)
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--upsample_steps', type=int, default=64)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    
    #Network Settings
    parser.add_argument('--network', type=str, default='sdf', help="network format, supports ( \
                                                                    sdf: use sdf representation, \
                                                                    phasor: use phasor encoding for sdf representation, \
                                                                    tcnn: use TCNN backend for sdf representation, \
                                                                    enc: use only TCNN encoding for sdf representation, \
                                                                    fp16: use amp mixed precision training for sdf representation,\
                                                                    ff: use fully-fused MLP for sdf representation)")
    
    parser.add_argument('--curvature_loss', '--C', action='store_true', help="use curvature loss term, slower but make surface smoother")

    #Dataset Settings
    parser.add_argument('--format', type=str, default='colmap', help="dataset format, supports (colmap, blender)")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-size, size)")

    #Others
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch (unstable now)")

    opt = parser.parse_args()

    print(opt)

    if opt.network =='ff':
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.network =='tcnn':
        from nerf.network_sdf_tcnn import NeRFNetwork
    elif opt.network =='enc':
        from nerf.network_sdf_enc import NeRFNetwork
    elif opt.network =='sdf':
        from nerf.network_sdf import NeRFNetwork
    elif opt.network =='phasor':
        from nerf.network_sdf_phasor import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)

    #network wwith encoding
    if opt.network =='phasor':
        model = NeRFNetwork(
            encoding="phasor", encoding_dir="sphere_harmonics", 
            num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
            cuda_ray=opt.cuda_ray, curvature_loss = opt.curvature_loss
        )
    else:
        model = NeRFNetwork(
            encoding="hashgrid", encoding_dir="sphere_harmonics", 
            num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
            cuda_ray=opt.cuda_ray, curvature_loss = opt.curvature_loss
        )
        
    #optimizer
    if opt.network in ['tcnn', 'enc', 'sdf', 'phasor']:
        optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sdf_net.parameters()) + list(model.color_net.parameters())+ list(model.deviation_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    else:
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    #scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / 20000, 1))

    #criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.HuberLoss()

    trainer = Trainer('ngp', 
                vars(opt), 
                model, 
                workspace=opt.workspace, 
                optimizer=optimizer, 
                criterion=criterion, 
                ema_decay=0.95, 
                fp16=(opt.network=='fp16'), 
                lr_scheduler=scheduler, 
                scheduler_update_every_step=True, 
                use_checkpoint='latest', 
                eval_interval=5,
                )

    if opt.mode == 'train':
        train_dataset = NeRFDataset(opt.path, type='train', mode=opt.format, bound=opt.bound)
        valid_dataset = NeRFDataset(opt.path, type='valid', mode=opt.format, downscale=opt.downscale, bound=opt.bound)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        trainer.train(train_loader, valid_loader, 200)

    elif opt.mode == 'mesh':
        valid_dataset = NeRFDataset(opt.path, type='valid', mode=opt.format, downscale=opt.downscale, bound=opt.bound)
        trainer.save_mesh(aabb = valid_dataset.aabb, resolution= 512, threshold=0.0, use_sdf=(opt.network=='sdf'))

    elif opt.mode == 'render':
        test_dataset = NeRFDataset(opt.path, type='test', mode=opt.format, downscale=opt.downscale, bound=opt.bound)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        trainer.test(test_loader)

    elif opt.mode == 'fvv':
        test_dataset = NeRFDataset(opt.path, type='fvv', mode='blender', downscale=opt.downscale, bound=opt.bound)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        trainer.test(test_loader)

    else: 
        pass
