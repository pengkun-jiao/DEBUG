import os
import torch 
import json
from time import time, localtime, strftime


class Logger():
    def __init__(self, config, update_frequency=10):
        

        self.config = config

        self.current_epoch = 0
        self.max_epochs = config['epoch']


        self.best_epoch = 0
        self.best_val_acc = 0.
        self.best_test_acc = 0.
        self.best_test_hs = 0.
        self.best_test_acc_k = 0.
        self.best_test_acc_u = 0.
    
        # self.start_time = time()
        self.last_update_time = None

        self.current_iter = 0
        self.update_f = update_frequency

        self.experiment_name = self.config['experiment_name']
        self.experiment_time = self.get_current_time()

        self.root = self.get_output_root_from_config() 
        self.log_path = self.get_log_path()

        if config['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter 
            self.tensorboard_writer = SummaryWriter(self.get_tensorboard_log_path())

        self.save_config('config.json')
        # self.config['experiment_time'] = self.experiment_time

    def new_epoch(self, lrs):
        self.current_epoch += 1
        self.last_update_time = time()
        print('\n'*1)
        print(' <New epoch: %d / %d, lr: %s>' %(self.current_epoch, self.max_epochs, ", ".join([str(lr) for lr in lrs])))


    def end_epoch(self):
        print('-' * 30)
        print(" Total epoch time: %.3f" % (time() - self.last_update_time))

    def caculate_domain_avg_acc(self, test_log):
        avg_result = {
            'acc_k': 0., 'acc_u': 0., 'acc': 0., 'hs': 0.
        }
        
        for _domain, _log in test_log.items():
            for k in avg_result.keys():
                avg_result[k] += _log[k]

        num_domains = len(test_log.keys())
        for k, v in avg_result.items():
            avg_result[k] /= num_domains

        return avg_result


    def log(self, train_log, val_log, test_log, model_state_dict):

        loss_dict = train_log['loss']
        
        # caculate domain avg acc
        avg_result = self.caculate_domain_avg_acc(test_log)
        record_point = {
            'epoch': self.current_epoch,
            'domain_avg_result': avg_result,
            'val_result': val_log,
            'test_result': test_log
        }

        print('-' * 30)
        print(' Accuracies on val : %.4f' %(val_log['acc']))
        print(' Accuracies on test: acc_k %4f acc_u %4f acc %.4f hs %4f ' %(avg_result['acc_k'], avg_result['acc_u'], avg_result['acc'], avg_result['hs']))
        print(' Training loss: %s' %(','.join('%s %f'%(k, v) for k, v in loss_dict.items())))

    
        self.add_log_to_json_file(log_dict=loss_dict, name='loss.json')
        self.add_log_to_json_file(log_dict=record_point, name='results.json')

        # save lase mode;
        self.save_best_model(model_state_dict, name='last_model')

        # save from best val acc
        if val_log['acc'] >= self.best_val_acc:
            self.best_epoch = self.current_epoch
            self.best_val_acc = val_log['acc']
            
            self.best_test_acc_k = avg_result['acc_k']
            self.best_test_acc_u = avg_result['acc_u']
            self.best_test_acc = avg_result['acc']
            self.best_test_hs = avg_result['hs']

            self.save_best_result_to_json(log_dict=record_point)

            model_state_dict['epoch'] = self.best_epoch
            model_state_dict['val_acc'] = self.best_val_acc
            self.save_best_model(model_state_dict)


        # tensorboard
        if self.config['tensorboard']:
            # Loss
            self.tensorboard_writer.add_scalars('Loss', loss_dict, global_step=self.current_epoch)
            # Overall acc
            avg_result['val_acc'] = val_log['acc']
            self.tensorboard_writer.add_scalars('Accuracy', avg_result, global_step=self.current_epoch)
            # Each domain acc
            for domain, _log in test_log.items():
                self.tensorboard_writer.add_scalars('Accuracy on '+domain, {'acc': _log['acc'],
                                                                            'hs': _log['hs'],
                                                                            'acc_k': _log['acc_k'],
                                                                            'acc_u': _log['acc_u'],
                                                                            }, global_step=self.current_epoch)



    def save_best_result_to_json(self, log_dict, name='best_result.json'):
        with open(os.path.join(self.log_path, name), 'w', encoding='utf-8') as file:
            json.dump(log_dict, file, indent=4)
        

    def save_best_model(self, model_state_dict, name='best_model'):
        torch.save(model_state_dict, os.path.join(self.log_path, f'{name}.tar'))

    def save_config(self, name='config.json'):
        with open(os.path.join(self.log_path, name), 'w', encoding='utf-8') as file:
            json.dump(self.config, file, indent=4)

    def save_final_best_result(self):

  
        s = '%s %s %d %f ' % (self.config['meta'], self.config['source_domain'], self.best_epoch, self.best_val_acc)
        s += '%f %f %f %f ' % (self.best_test_acc_k, self.best_test_acc_u, self.best_test_acc, self.best_test_hs)
        s += self.experiment_time + '\n'
        with open(os.path.join(self.root, 'best_result'), 'a', encoding='utf-8') as file:
            file.write(s)

        print()
        print('-'*30)
        print(' Best Result: ')
        print('   sourec_domain: %s'% self.config['source_domain'])
        print('   experiment_meta: %s  '% self.config['meta'])
        print('   experiment_name: %s  '% self.config['experiment_name'])
        print('   experiment_time: %s  '% self.experiment_time)
        print('   epoch: %d' % self.best_epoch)
        print('   best_val_acc: %.4f' %(self.best_val_acc ))
        print('   test: acc_k %.4f acc_u %.4f acc %.4f hs %.4f ' %(self.best_test_acc_k, self.best_test_acc_u, self.best_test_acc, self.best_test_hs))
        print()
    
    def add_log_to_json_file(self, log_dict, name='record.json'):
        with open(os.path.join(self.log_path, name), 'a', encoding='utf-8') as file:
            json.dump(log_dict, file, indent=4)


    @staticmethod
    def get_current_time():
        time = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        return time

    def get_output_root_from_config(self, name='default'):
        path = os.path.join(self.config['output_dir'], self.config['dataset'], self.config['experiment_name'], self.config['meta'])
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    def get_log_path(self, time=None):
        if time == None: time = self.experiment_time
        # path = os.path.join(self.root, time)
        path = os.path.join(self.root, time, self.config['source_domain'])
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_tensorboard_log_path(self, time=None):
        if time == None: time = self.experiment_time
        return os.path.join(self.config['output_dir'], self.config['dataset'], 'tensorboard', time)