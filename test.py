from sklearn.decomposition import PCA
import jittor as jt
from jittor import init
from jittor import nn
import os
import argparse
import random
import numpy as np
from PIL import Image
from training.networks.stylegan2 import Generator

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    pca = PCA(n_components=100)
    parser.add_argument('--ckpt', type=str, default='weights/photosketch_standing_cat_noaug.pth', help='checkpoint file for the generator')
    args = parser.parse_args()
    with jt.no_grad():
        netG = Generator(256, 512, 8)
        checkpoint = jt.load(args.ckpt)
        netG.load_parameters(checkpoint)
        z = jt.randn(5000, 512)
        latents = netG.get_latent(z)
        latents = latents.numpy()
        pca = PCA(n_components=100)
        lat_cop = pca.fit_transform(latents.transpose(1,0)).transpose(1,0)
        lat_cop = jt.float32(lat_cop)
        lat_cop = lat_cop.view(100,1,512)
        np.savez(f'./weights/ganspace_fur_standing_cat_test.npz',lat_comp=lat_cop)
    
