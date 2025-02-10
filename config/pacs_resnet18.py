import config.pacs_base as base_config


config = {}

encoder_name = 'resnet18'

config["meta"] = "pacs_resnet18"
config["experiment_name"] = "default"
config['experiment_time'] = None
config["comment"] = "open-set_single_domain_generalization."

config['output_root'] = base_config.output_root
config['output_dir'] = base_config.output_dir
config['tensorboard'] = base_config.tensorboard


config['edge_statistice'] = base_config.edge_statistice


warmup_epoch = base_config.warmup_epoch
warmup_type = "sigmoid"
lam_const = 5.0    # loss weight for factorization loss
T = 10.0
k = 308


# training setting
epoch = base_config.epoch
batch_size = base_config.batch_size
num_worker = base_config.num_worker

lr = base_config.lr
lr_decay_rate = base_config.lr_decay_rate
lr_decay_step = base_config.lr_decay_step

# open set reject threshold
config['threshold'] = base_config.threshold


# dataset
config['dataset'] = base_config.dataset
config['dataset_root'] = base_config.dataset_root
config['domains'] = base_config.domains
config['source_domain'] = base_config.source_domain
config['target_domains'] = base_config.target_domains
config['know_classes'] = base_config.know_classes
num_know_classes = len(config['know_classes'])
config["num_classes"] = num_know_classes

config["bs"] = batch_size
config["nw"] = num_worker
config["epoch"] = epoch
config["lam_const"] = lam_const
config["warmup_epoch"] = warmup_epoch
config["warmup_type"] = warmup_type
config["T"] = T
config["k"] = k


#loss weight
config['loss_weight'] = base_config.loss_weight


# network configs
networks = {}

encoder = {
    "name": encoder_name,
}
networks["encoder"] = encoder

classifier = {
    "name": "base",
    "in_dim": 512,
    "num_classes": num_know_classes # num_know_classes
}
networks["classifier"] = classifier

config["networks"] = networks



# optimizer configs
optimizer = {}

encoder_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": lr_decay_step,
    "lr_decay_rate": lr_decay_rate
}
optimizer["encoder_optimizer"] = encoder_optimizer

classifier_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": lr_decay_step,
    "lr_decay_rate": lr_decay_rate
}
optimizer["classifier_optimizer"] = classifier_optimizer

config["optimizer"] = optimizer
