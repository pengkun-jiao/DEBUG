import argparse

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data.DgDataLoader import DgDataLoader
from utils.logger import Logger
from utils.tools import *
from utils.losses import *
from models.classifier import Masker

import torchvision.transforms as transforms



class Trainer():
    def __init__(self, config, device):

        from data.hed import  Network as HED
        self.hed = HED().to(device).eval()

        self.device = device
        self.config = config
 
        # networks
        # print(self.config["networks"]["encoder"])
        # print("==========")
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        self.classifier_bi = get_multi_bi_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_bi_optim, self.classifier_bi_sched = \
            get_optim_and_scheduler(self.classifier_bi, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        dataLoader = DgDataLoader(self.config, self.device)

        self.train_loader = dataLoader.get_train_loader(domain=self.config['source_domain'], ifMask=True)
        self.val_loader = dataLoader.get_val_loader(domain=self.config['source_domain'])
        self.test_loaders = {}
        for target in  self.config['target_domains']:
            self.test_loaders[target] = dataLoader.get_test_loader(domain=target)


        self.edge_mean = 0.1
        self.edge_var = 0.3


    def do_epoch(self, loader):

        # turn on train mode
        self.encoder.train()
        self.classifier.train()
        self.classifier_bi.train()
        
        train_log = {}
        training_loss = {}

        for i, (batch, label, clip_mask) in enumerate(loader):
            loss_dict = {}


            mask_batch = torch.einsum('nchw, nhw->nchw', batch, clip_mask)


            # preprocessing
            batch = batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            label = label.to(self.device)

            # edge
            edge_batch = self.hed(batch).data.clamp(0.0, 1.0)
            
            edge_batch = (0.45 - edge_batch)/0.225
            edge_batch = edge_batch.expand(edge_batch.shape[0],3,edge_batch.shape[2],edge_batch.shape[3])
    


         
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            self.classifier_bi_optim.zero_grad()

            features = self.encoder(batch, label, disturb=True)
            features_kd = self.encoder(mask_batch, label, disturb=True).detach()
            features_ed = self.encoder(edge_batch, label, disturb=False)


            scores = self.classifier(features)
            scores_ova = self.classifier_bi(features)
            scores_ova = scores_ova.view(scores_ova.size(0), 2, -1)

            scores_edge = self.classifier(features_ed)
            scores_edge_ova = self.classifier_bi(features_ed)
            scores_edge_ova = scores_edge_ova.view(scores_edge_ova.size(0), 2, -1)


            # get loss weight
            loss_weight = self.config['loss_weight']

            
            # classification loss
            loss_cls = ce_loss(scores, label)
            loss_cls_ed = ce_loss(scores_edge, label)
            loss_dict['cls-' + str(loss_weight['cls'])] = loss_cls * loss_weight['cls']
            loss_dict['cls-ed'] = loss_cls_ed*0.5



            # ova loss
            loss_ova_pos, loss_ova_neg = ova_loss(scores_ova, label)
            # loss_dict['ova_pos-' + str(loss_weight['ova_pos'])] = loss_ova_pos * loss_weight['ova_pos']
            loss_dict['ova_neg-' + str(loss_weight['ova_neg'])] = loss_ova_neg 

            loss_eva_pos, loss_eva_neg = ova_loss(scores_edge_ova, label)
            loss_dict['eva_pos-' + str(loss_weight['ova_pos'])] = loss_ova_pos 
            loss_dict['eva_neg-' + str(loss_weight['ova_neg'])] = loss_ova_neg 

            

            # knowledge distillation loss
            T = self.config['T']
            loss_kd = kl_loss(features / T, features_kd / T)
            loss_dict['kd'] = loss_kd




            # total loss
            total_loss = 0.
            for k, v in loss_dict.items():
                total_loss += v
                if k in training_loss:
                    training_loss[k] += v.item()
                else:
                    training_loss[k] = v.item()

            # backward
            total_loss.backward()

            # step optimizer
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.classifier_bi_optim.step()

        for k in training_loss.keys():
            training_loss[k] /= i+1

        train_log['loss'] = training_loss
        return train_log

    def do_val_eval(self, loader):
        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        self.classifier_bi.eval()

        num_classes = self.config['num_classes']
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)

        for i, (batch, label) in enumerate(loader):
            batch = batch.to(self.device)
            label = label.to(self.device)
            featrues = self.encoder(batch, disturb=False)
            scores = self.classifier(featrues)

            
            pred = inference(scores)
            for p, gt in zip(pred, label):
                total[gt] += 1
                if p == gt: correct[p] += 1

        each_class_acc = correct / total
        acc = each_class_acc.mean()

        result_dict = {}
        result_dict['acc'] = acc.item()
        return result_dict
    
    def do_test_eval(self, loader):
        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        self.classifier_bi.eval()

        num_classes = self.config['num_classes']
        correct = torch.zeros(num_classes + 1)
        total = torch.zeros(num_classes + 1)

        for i, (batch, label) in enumerate(loader):
            batch = batch.to(self.device)
            label = label.to(self.device)
            featrues = self.encoder(batch, disturb=False)
            scores = self.classifier(featrues)

            pred = openset_inference(scores, threshold=self.config['threshold'])
            for p, gt in zip(pred, label):
                if p == -1: p = num_classes
                if gt == -1: gt = num_classes
                total[gt] += 1
                if p == gt: correct[p] += 1

        each_class_acc = correct / total
        acc_k = each_class_acc[:-1].mean()
        acc_u = each_class_acc[-1]
        acc = each_class_acc.mean()
        hs = 2 * acc_k * acc_u / (acc_k + acc_u)

        result_dict = {}
        result_dict['acc'] = acc.item()
        result_dict['hs'] = hs.item()
        result_dict['acc_k'] = acc_k.item()
        result_dict['acc_u'] = acc_u.item()
        # result_dict['each_class_acc'] = [x.item() for x in each_class_acc]
        
        return result_dict


    def do_training(self):

        self.logger = Logger(self.config)

        for self.current_epoch in range(1, self.config['epoch']+1):
            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])

            if self.current_epoch == 6:
                quit()

            # train epoch
            train_log = self.do_epoch(self.train_loader)

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()
            self.classifier_bi_sched.step()

            # eval on val set
            val_log = self.do_val_eval(self.val_loader)

            # eval on test sets
            test_log = {}
            for target_domain, test_loader in self.test_loaders.items():
                test_log[target_domain] = self.do_test_eval(test_loader)

            # record result
            model_state_dict = {
                'encoder': self.encoder.state_dict(),
                'classifier': self.classifier.state_dict(),
                'classifier_bi': self.classifier_bi.state_dict()
            }
            self.logger.log(train_log, val_log, test_log, model_state_dict)
            self.logger.end_epoch()

        self.logger.save_final_best_result()

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Experiment configs")
    parser.add_argument("--name", default=None, type=str, help="Implement info")
    parser.add_argument("--comment", default=None, type=str, help="Implement info")
    parser.add_argument("--source", default=None, type=str)
    args = parser.parse_args()
    
    config_file = "config." + args.config
    config = __import__(config_file, fromlist=[""]).config
    config['meta'] = args.config
    config['experiment_name'] = args.name
    config['comment'] = args.comment
    print("Using config:", config_file)

    config['source_domain'] = args.source
    targets = []
    for i in config['domains']:
        if i != args.source: targets.append(i)
    config['target_domains'] = targets
    print("Source:", config['source_domain'])
    print("Targets:", config['target_domains'])

    return config

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()