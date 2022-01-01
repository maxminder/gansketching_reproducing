# import os
# prefix = 'python editor.py --obj cat --scalar 15 --ckpt /root/jittor/gansketching_reproducing/weights/photosketch_standing_cat_noaug.pth --slice 3 --layers '
# layers = ['1,4','5,9','10,18']
# next = ' --comp_id '
# comp_id = range(50)
# next2 = ' --save_dir output/sefa_standing_cat_'
# for i in comp_id:
#     for layer in layers:
#         command = prefix + layer + next + str(i) + next2 + str(i) + '/' + layer
#         os.system(command)
import os
prefix = 'python editor.py --obj cat --scalar 15 --ckpt /root/jittor/gansketching_reproducing/weights/photosketch_standing_cat_noaug.pth --slice 3 --layers '
layers = ['5,8']
next = ' --eigen_id '
comp_id = 0
samples = range(4)
next2 = ' --save_dir output/test'
for i in samples:
    for layer in layers:
        command = prefix + layer + next + str(comp_id) + next2 + str(i)
        os.system(command)
