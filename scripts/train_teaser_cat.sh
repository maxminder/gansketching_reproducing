#!/bin/bash
rm -r cache_files
python train.py \
--name teaser_cat_augment --batch 1 \
--dataroot_sketch ~/jittor/gansketching_reproducing/data/sketch/by_author/cat \
--dataroot_image ~/jittor/gansketching_reproducing/data/image/cat --l_image 0.7 \
--g_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-cat/netG.pth \
--d_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-cat/netD.pth \
--max_iter 150001 --disable_eval --diffaug_policy translation --no_wandb --resume_iter 100000\
