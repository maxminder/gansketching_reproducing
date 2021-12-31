#!/bin/bash
rm -r cache_files
python train.py \
--batch 1 \
--name horse_riders_no_aug \
--dataroot_sketch ./data/sketch/photosketch/horse_riders \
--dataroot_image ./data/image/horse --l_image 0.7 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--disable_eval \
--resume_iter 75000 \
--max_iter 150001

python generate.py --ckpt checkpoint/horse_riders_no_aug/150000_net_G.pth --save_dir output/horse_riders_no_aug --samples 2500

python3 -m pytorch_fid output/horse_riders_no_aug data/eval/horse_riders/image --device cuda
