import os
prefix = 'python editor.py --obj cat --scalar 15 --ckpt checkpoint/standing_cat_noaugment/75000_net_G.pth --slice 3 --layers '
layers = ['1,4','5,9','10,18']
next = ' --comp_id '
comp_id = range(50)
next2 = ' --save_dir output/sefa_standing_cat_'
for i in comp_id:
    for layer in layers:
        command = prefix + layer + next + str(i) + next2 + str(i) + '/' + layer
        os.system(command)
