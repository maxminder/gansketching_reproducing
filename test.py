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
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file for the generator')
    args = parser.parse_args()
    with jt.no_grad():
        if (not os.path.exists(args.save_dir)):
            os.makedirs(args.save_dir)
        netG = Generator(args.size, 512, 8)
        checkpoint = jt.load(args.ckpt)
        netG.load_parameters(checkpoint)
        z = jt.randn(5000, 512)
        latents = netG.get_latent(z)
        pca = PCA(n_components=100)
        lat_cop = jt.transpose(pca.fit_transform(jt.transpose(latents,(1,0))),(1,0))
        lat_cop = lat_cop.view(100,1,512)
        np.savez(f'output/ganspace_fur_standing_cat_test.npz',lat_comp=lat_cop)
    
