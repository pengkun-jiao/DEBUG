import os
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--gpu", default=0, type=int, help="Gpu ID")
parser.add_argument("--config", default='pacs_uresnet18', type=str, help="Config")
parser.add_argument("--name", default='default', type=str, help="Experiment name")
parser.add_argument("--comment", default='open-set_single_domain_generalization.', type=str, help="Implement info")
parser.add_argument("--times", default=1, type=int, help="Repeat times")
parser.add_argument("--source", default=None, type=str)

args = parser.parse_args()

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python train.py '
              f'--config {args.config} '
              f'--name {args.name} '
              f'--comment {args.comment} '
              f'--source {args.source} ')
