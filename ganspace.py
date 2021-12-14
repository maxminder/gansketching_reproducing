import jittor as jt
from jittor import init
from jittor import nn
import os
import argparse
import random
import numpy as np
from PIL import Image
from training.networks.stylegan2 import Generator

def gen_principal_components(dump_name, device='cpu'):
    'Get principle components from GANSpace.'
    with np.load(dump_name) as data:
        lat_comp = jt.float32(data['lat_comp'])    #to device?
        lat_mean = jt.float32(data['lat_mean'])
        lat_std = data['lat_stdev']
    return (lat_comp, lat_mean, lat_std)

def apply_shift(id, g, mean_latent, latents, w_comp, w_std, s, layers, w_plus=False, trunc=0.5):
    'Apply GANSpace edits.'
    if (not w_plus):
        latents = latents[:, None, :].repeat(1, 18, 1)
    boundaries = g.style_weights()
    boundary = boundaries[id]
    #latents[:, layers, :] = (latents[:, layers, :] + ((w_comp[:, None, :] * s) * w_std))
    latents[:, layers, :] = (latents[:, layers, :] + boundary*s)
    im = g([latents], input_is_latent=True, truncation=trunc, truncation_latent=mean_latent)[0]
    #im = im.cpu().numpy().transpose((0, 2, 3, 1))
    im = im.numpy().transpose((0, 2, 3, 1))
    im = np.clip(((im + 1) / 2), 0, 1)
    return (im * 255).astype(np.uint8)

def save_ims(prefix, ims):
    for (ind, im) in enumerate(ims):
        Image.fromarray(im).save((prefix + f'.png'))
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, choices=['cat', 'horse', 'church'], help='which StyleGAN2 class to use')
    parser.add_argument('--comp_id', type=int, required=True, help='which principle component to use')
    parser.add_argument('--scalar', type=float, required=True, help='strength applied to the latent shift, value can be negative')
    parser.add_argument('--layers', type=str, required=True, help='layers to apply GANSpace (e.g., 3,5 means layer 3 to 5')
    parser.add_argument('--save_dir', type=str, default='./output', help='place to save the output')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file for the generator')
    parser.add_argument('--fixed_z', type=str, default=None, help='expect a .pth file. If given, will use this file as the input noise for the output')
    parser.add_argument('--size', type=int, default=256, help='output size of the generator')
    parser.add_argument('--samples', type=int, default=1, help='number of samples to generate, will be overridden if --fixed_z is given')
    parser.add_argument('--truncation', type=float, default=0.5, help='strength of truncation')
    parser.add_argument('--truncation_mean', type=int, default=4096, help='number of samples to calculate the mean latent for truncation')
    parser.add_argument('--seed', type=int, default=None, help='if specified, use a fixed random seed')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--slice', type=float, default=0.5)
    args = parser.parse_args()
    device = args.device
    with jt.no_grad():
        # if (args.seed is not None):
        #     random.seed(args.seed)
        #     jt.set_seed(args.seed)
        #     #torch.cuda.manual_seed_all(args.seed)
        #     jt.set_global_seed(args.seed)
        if (not os.path.exists(args.save_dir)):
            os.makedirs(args.save_dir)
        netG = Generator(args.size, 512, 8)
        #checkpoint = torch.load(args.ckpt, map_location='cpu')
        checkpoint = jt.load(args.ckpt)
        netG.load_parameters(checkpoint)
        if (args.truncation < 1):
            mean_latent = netG.mean_latent(args.truncation_mean)
        else:
            mean_latent = None
        (k, s) = (args.comp_id, args.scalar)
        (l_start, l_end) = [int(d) for d in args.layers.split(',')]
        layers = range(l_start, (l_end + 1))
        (lat_comp, lat_mean, lat_std) = gen_principal_components(f'./weights/ganspace_{args.obj}.npz')
        w_comp = lat_comp[k]
        w_std = lat_std[k]
        if (args.fixed_z is None):
            #z = jt.randn(args.samples, 512).to(device)
            z = jt.randn(args.samples, 512)
        else:
            #z = jt.load(args.fixed_z, map_location='cpu').to(device)
            z = jt.load(args.fixed_z, map_location='cpu')
        latents = netG.get_latent(z)
        # ims = apply_shift(netG, mean_latent, latents, w_comp, w_std, 0, layers, trunc=args.truncation)
        # save_ims(f'./{args.save_dir}/before_', ims)
        # ims = apply_shift(netG, mean_latent, latents, w_comp, w_std, s, layers, trunc=args.truncation)
        # save_ims(f'./{args.save_dir}/after_', ims)
        slice = args.slice
        num = int(s/slice)
        for i in range(num):
            print(i*slice)
            ims = apply_shift(k, netG, mean_latent, latents, w_comp, w_std, -i*slice, layers, trunc=args.truncation)
            save_ims(f'./{args.save_dir}/neg_'+str(i), ims)
            ims = apply_shift(k, netG, mean_latent, latents, w_comp, w_std, i*slice, layers, trunc=args.truncation)
            save_ims(f'./{args.save_dir}/pos_'+str(i), ims)
