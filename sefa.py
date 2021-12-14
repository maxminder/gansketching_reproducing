import os
prefix = 'python ganspace.py --obj cat --scalar 15 --ckpt weights/photosketch_standing_cat_noaug.pth --save_dir output/ganspace_fur_standing_cat --slice 3 --layers '
layers = ['1,4','5,9','10,18']
next = ' --comp_id '
comp_id = range(8,100)
next2 = ' --save_dir output/ganspace_fur_standing_cat_'
for i in comp_id:
    for layer in layers:
        command = prefix + layer + next + str(i) + next2 + str(i) + '/' + layer
        os.system(command)
