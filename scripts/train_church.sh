python train.py \
--name church_augment --batch 4 \
--dataroot_sketch /root/lcs/new/gansketching_reproducing/data/sketch/photosketch/gabled_church \
--dataroot_image /root/lcs/new/gansketching_reproducing/data/image/church --l_image 0.7 \
--g_pretrained /root/lcs/new/gansketching_reproducing/pretrained/stylegan2-church/netG.pth \
--d_pretrained /root/lcs/new/gansketching_reproducing/pretrained/stylegan2-church/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation --no_wandb \