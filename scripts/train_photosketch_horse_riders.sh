#!/bin/bash
rm -r cache_files
python train.py \
--batch 1 \
--name horse_riders_augment \
--dataroot_sketch ./data/sketch/photosketch/horse_riders \
--dataroot_image ./data/image/horse --l_image 0.7 \
--eval_dir ./data/eval/horse_riders \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--disable_eval \
