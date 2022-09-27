from builtins import print
from operator import index
import os
import time
import glob
from turtle import down
import numpy as np

import cv2
from PIL import Image
from tqdm import tqdm 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json
from .utils import get_rays

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, aabb, bound):

    scale = max(0.000001,max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                                abs(float(aabb[1][1])-float(aabb[0][1]))),
                                abs(float(aabb[1][2])-float(aabb[0][2]))))
    scale =  4.0 * bound / scale

    offset = [((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
                ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) * -scale, 
                ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale]
    
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
        [0, 0, 0, 1],
    ], dtype=pose.dtype)
    return new_pose


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp_scale(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


class NeRFDataset(Dataset):
    def __init__(self, path, type='train', mode='colmap', preload=True, downscale=1, bound=0.33, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = path
        self.type = type # train, val, test
        self.mode = mode # colmap, blender, llff
        self.downscale = downscale
        self.preload = preload # preload data into GPU

        # camera radius bound to make sure camera are inside the bounding box.
        self.bound = bound

        # load nerf-compatible format data.
        if mode == 'colmap':
            transform_path = os.path.join(path, 'transforms.json')
        elif mode == 'blender':
            transform_path = os.path.join(path, f'transforms_{type}.json')
        else:
            raise NotImplementedError(f'unknown dataset mode: {mode}')

        with open(transform_path, 'r') as f:
            transform = json.load(f)
        
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])

        # load image size
        try:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        except:
            f_path = os.path.join(self.root_path, frames[0]['file_path'])
            if f_path[-4:] != '.png' and f_path[-4:] != '.jpg':
                f_path = f_path + '.png'
            sample = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            self.H, self.W = sample.shape[0] // downscale, sample.shape[1] // downscale
             

        # load intrinsics
        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
            self.intrinsic = np.array([[fl_x, 0., cx], [0., fl_y, cy], [0., 0., 1.]])
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
            self.intrinsic = np.array([[fl_x, 0., cx], [0., fl_y, cy], [0., 0., 1.]])
        else:
            self.intrinsic = np.array([[0., 0., cx], [0., 0., cy], [0., 0., 1.]])

        # load bounding bbox
        try:
            self.aabb = transform['aabb']
        except:
            aabb_scale = 1.0 #transform['aabb_scale']
            pts = []
            for f in frames:
                pts.append(np.array(f['transform_matrix'], dtype=np.float32)[:3,3]) # [4, 4]
            pts = np.stack(pts, axis=0).astype(np.float32)

            minxyz=np.min(pts, axis=0) * aabb_scale
            maxxyz=np.max(pts, axis=0) * aabb_scale

            self.aabb = [[minxyz[0], minxyz[1], minxyz[2]],
                        [maxxyz[0], maxxyz[1], maxxyz[2]]]

        if type == 'test':
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), aabb=self.aabb, bound=bound) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), aabb=self.aabb, bound=bound) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.intrinsics = []
            try:
                intrinsic = np.array(f0['K'], dtype=np.float32) # [4, 4]
                intrinsic[:2, :] = intrinsic[:2, :] / downscale
            except:
                intrinsic = self.intrinsic

            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
                self.intrinsics.append(intrinsic)

            self.poses = np.stack(self.poses, axis=0).astype(np.float32)
            self.intrinsics = np.stack(self.intrinsics, axis=0).astype(np.float32)

        elif type == 'fvv':
            self.poses = []
            self.images = []
            self.intrinsics = []
            
            for f in tqdm(frames, unit=" views", desc=f"Loading Render Path"):
                pose = nerf_matrix_to_ngp(np.array(f['transform_matrix'], dtype=np.float32), aabb=self.aabb, bound=bound) # [4, 4]
                try:
                    intrinsic = np.array(f['K'], dtype=np.float32) # [4, 4]
                    intrinsic[:2, :] = intrinsic[:2, :] / downscale
                except:
                    intrinsic = self.intrinsic

                self.poses.append(pose)
                self.intrinsics.append(intrinsic)

            self.poses = np.stack(self.poses, axis=0).astype(np.float32)
            self.intrinsics = np.stack(self.intrinsics, axis=0).astype(np.float32)
            
        else:
            if type == 'train':
                frame = frames[:]
            elif type == 'valid':
                frame = frames[:1]

            self.poses = []
            self.images = []
            self.intrinsics = []
            
            for f in tqdm(frame, unit=" images", desc=f"Loading Images"):
                f_path = os.path.join(self.root_path, f['file_path'])
                if f_path[-4:] != '.png' and f_path[-4:] != '.jpg':
                    f_path = f_path + '.png'

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = nerf_matrix_to_ngp(np.array(f['transform_matrix'], dtype=np.float32), aabb=self.aabb, bound=bound) # [4, 4]

                try:
                    intrinsic = np.array(f['K'], dtype=np.float32) # [4, 4]
                    intrinsic[:2, :] = intrinsic[:2, :] / downscale
                except:
                    intrinsic = self.intrinsic

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
                self.intrinsics.append(intrinsic)
            
            self.poses = np.stack(self.poses, axis=0).astype(np.float32)
            self.images = np.stack(self.images, axis=0)
            self.intrinsics = np.stack(self.intrinsics, axis=0).astype(np.float32)
        
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        if preload:
            if type == 'train':
                self.poses = torch.from_numpy(self.poses).cuda()
                self.intrinsics = torch.from_numpy(self.intrinsics).cuda()
                self.images = torch.from_numpy(self.images).cuda()
            else:
                self.poses = torch.from_numpy(self.poses).cuda()
                self.intrinsics = torch.from_numpy(self.intrinsics).cuda()

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsics[index],
            'index': index,
        }

        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            return results
        else:
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            results['image'] = self.images[index]
            return results

class RayDataset(Dataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__()
        # path: the json file path.

        self.path = path
        self.type = type
        self.downscale = downscale
        self.radius = radius # TODO: generate custom views for test?
        self.NeRFDataset =  NeRFDataset(self.path, self.type, downscale=self.downscale, radius=self.radius)


        self.all_rays_o = []
        self.all_rays_d = []
        self.all_inds = []
        self.all_rgbs = []

        for i in range(len(self.NeRFDataset)):
            meta = self.NeRFDataset.__getitem__(i)

            images = torch.from_numpy(meta["image"]) # [H, W, 3/4]
            poses = torch.from_numpy(meta["pose"]) # [4, 4]
            intrinsics = torch.from_numpy(meta["intrinsic"])  # [3, 3]

            # sample rays 
            H, W, C = images.shape
            rays_o, rays_d, inds = get_rays(poses.unsqueeze(0), intrinsics.unsqueeze(0), H, W, -1) #[1, H*W, 3]
            rgbs = images.reshape(1, -1, C) #[1, H*W, 3/4]

            self.all_rays_o.append(rays_o[0])
            self.all_rays_d.append(rays_d[0])
            self.all_inds.append(inds[0])
            self.all_rgbs.append(rgbs[0])
        
        self.all_rays_o = torch.cat(self.all_rays_o, dim=0)
        self.all_rays_d = torch.cat(self.all_rays_d, dim=0)
        self.all_inds = torch.cat(self.all_inds, dim=0)
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)


    def __len__(self):
        return len(self.all_rays_o)

    def __getitem__(self, index):

        results = {
            'rays_o': self.all_rays_o[index],
            'rays_d': self.all_rays_d[index],
            'rgbs': self.all_rgbs[index],
            'shape' : (self.NeRFDataset.H, self.NeRFDataset.W),
            'index': index,
        }
        return results
     
