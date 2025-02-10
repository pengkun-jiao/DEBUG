config = {}



epoch = 50
batch_size = 32
num_worker = 2
lr = 0.001
lr_decay_rate = 0.1
lr_decay_step = 20

warmup_epoch = 5


edge_statistice = [0.2, 0.37] #default

# dataset
dataset = 'pacs'
dataset_root = '/share/test/jpk/PACS'
domains = ['art_painting', 'cartoon', 'sketch', 'photo']
know_classes = [0, 1, 2, 3]

# source_domain='photo'
# source_domain='sketch'
# source_domain='cartoon'
source_domain='art_painting'

target_domains = []
for i in domains:
    if i != source_domain:
        target_domains.append(i)

# loss weight
loss_weight = {
    'cls': 1,
    'ova_pos': 0.5,
    'ova_neg': 0.5,
    'ova_cons': 0.5,
    'fac': 1,
    'kd_cons': 1
}


# output
# output_root = '/share/ckpt/jpk/OSSDG'
# output_dir = '/share/ckpt/jpk/OSSDG'
output_root = '/share_io03_ssd/ckpt2/jiaopengkun/OSSDG'
output_dir = '/share_io03_ssd/ckpt2/jiaopengkun/OSSDG'
tensorboard = False

import math
threshold = math.log(len(know_classes), 2) / 2

