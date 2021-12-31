#!/bin/bash
rm -r cache_files
python train.py \
--batch 1 \
--name horse_riders_weight_image \
--dataroot_sketch ./data/sketch/photosketch/horse_riders \
--dataroot_image ./data/image/horse --l_image 0.7 --l_weight 0.7 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--disable_eval \

python generate.py --ckpt checkpoint/horse_riders_weight_image/150000_net_G.pth --save_dir output/horse_riders_weight_image --samples 2500

python3 -m pytorch_fid output/horse_riders_weight_image data/eval/horse_riders/image --device cuda
