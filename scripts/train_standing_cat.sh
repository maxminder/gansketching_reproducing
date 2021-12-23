rm -r cache_files
python train.py \
--name standing_cat_augment --batch 1 \
--dataroot_sketch ~/jittor/gansketching_reproducing/data/sketch/photosketch/standing_cat \
--dataroot_image ~/jittor/gansketching_reproducing/data/image/cat --l_image 0.7 \
--g_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-cat/netG.pth \
--d_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-cat/netD.pth \
--max_iter 150001 --disable_eval --no_wandb --diffaug_policy translation --resume_iter 77500\