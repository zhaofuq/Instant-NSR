# torch-ngp

A pytorch implementation of __Instant-NSR__, as described in __Human Performance Modeling and Rendering via Neural Animated Mesh__.


# Install
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Tested on Ubuntu with torch 1.10 & CUDA 11.4 on RTX 3090.

# Usage

We use the same data format as nerf and instant-ngp, e.g. [Chair](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox), [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `Your Own Path`.

First time running will take some time to compile the CUDA extensions.

```bash
# Instant-NSR Training
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_nerf.py "${INPUTS}/${dir}"  --workspace "${WORKSAPCE}" --downscale 1 --network sdf

# Instant-NSR Mesh extraction
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_nerf.py "${INPUTS}/${dir}"  --workspace "${WORKSAPCE}" --downscale 1 --network sdf -mode mesh

# Instant-NSR Rendering
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_nerf.py "${INPUTS}/${dir}"  --workspace "${WORKSAPCE}" --downscale 1 --network sdf -mode render
```

# Acknowledgement

Our code is implemented on torch-ngp code base:
```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
```
