python train.py \
--name church_augment --batch 1 \
--dataroot_sketch ~/jittor/gansketching_reproducing/data/sketch/photosketch/gabled_church \
--dataroot_image ~/jittor/gansketching_reproducing/data/image/church --l_image 0.7 \
--g_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-church/netG.pth \
--d_pretrained ~/jittor/gansketching_reproducing/pretrained/stylegan2-church/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation --no_wandb \