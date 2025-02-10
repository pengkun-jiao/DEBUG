import os
import pickle
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.pacs.PACS import PACS
from data.officehome.officehome import OfficeHome
from data.domainnet126.domainnet126 import DomainNet126
from data.office31.office31 import Office31
from data.DgDataset import DgDataset
from models.denseclip.Masker import Masker

class DgDataLoader():

    def __init__(self, config, device):

        self.device = device

        self.dataset = config['dataset']
        self.dataset_root = config['dataset_root']
        self.know_classes = config['know_classes']
        # self.unknow_classes = config['unknow_classes']


        self.bs = config['bs']
        self.nw = config['nw']

        if self.dataset == 'pacs':
            self.data = PACS(self.dataset_root)
        elif self.dataset == 'officehome':
            self.data = OfficeHome(self.dataset_root)
        elif self.dataset == 'dn126':
            self.data = DomainNet126(self.dataset_root)
        elif self.dataset == 'office31':
            self.data = Office31(self.dataset_root)

        self.classnames = self.data.classesname

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # self.augmentation = [
        #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize
        # ]

    def get_train_loader(self, domain, ifMask=True, generate_mask=False):
        know_path, know_label = [], []
        img_path, label = self.data.get_train_path_and_label(domain)
        # print(label)
        for _path, _label in zip(img_path, label):
            if _label != -1:
                know_path.append(_path)
                know_label.append(_label)
        
        if ifMask:
            train_set_prepared_file = os.path.join('/share_io03_ssd/ckpt2/jiaopengkun/OSSDG/prepared_data', self.dataset+'_'+domain+'_train_prepared.pkl')
            # train_set_prepared_file = os.path.join('/share_io03_ssd/ckpt2/jiaopengkun/OSSDG/prepared_data/sd', self.dataset+'_'+domain+'_train_prepared.pkl')
            #################################
            #create variant content mask
            #################################
            if generate_mask:
                mask = []

                #################################
                # Dense CLIP
                masker = Masker(device=self.device, classnames=self.classnames)
                for _path, _label in zip(know_path, know_label):
                    _mask = masker.get_mask(_path, _label)
                    mask.append(_mask)

                # ################################
                # # Sailent Detection
                # import cv2
                # # saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                # for _path, _label in zip(know_path, know_label):
                #     image = cv2.imread(_path)
                #     image = cv2.resize(image, (224,224))
                #     # print(image.shape)
                #     (success, saliencyMap) = saliency.computeSaliency(image)
                #     mask.append(saliencyMap)
                # #     print(saliencyMap.shape)
                # #     quit()
                # # quit()
                # #################################

                train_set = {'img_path': know_path,
                            'label': know_label,
                            'mask': mask}
                with open(train_set_prepared_file, 'wb') as f:
                    pickle.dump(train_set, f)
                print('Save prepared data to ', train_set_prepared_file)
            

            with open(train_set_prepared_file, 'rb') as f:
                train_set = pickle.load(f)

            know_path, know_label, mask = train_set['img_path'],  train_set['label'],  train_set['mask']
            dataset = DgDataset(img_path=know_path, label=know_label, mask=mask, transform=self.transform)
        else:
            dataset = DgDataset(img_path=know_path, label=know_label, transform=self.transform)  

        dataLoader = DataLoader(dataset, batch_size=self.bs, shuffle=True, num_workers=self.nw)
        return dataLoader


    def get_val_loader(self, domain):
        know_path, know_label = [], []
        img_path, label = self.data.get_val_path_and_label(domain=domain)
        for _path, _label in zip(img_path, label):
            if _label != -1:
                know_path.append(_path)
                know_label.append(_label)
        dataset = DgDataset(know_path, know_label, transform=self.transform)
        dataLoader = DataLoader(dataset, batch_size=self.bs, shuffle=True, num_workers=self.nw)
        return dataLoader


    def get_test_loader(self, domain):
        img_path, label = self.data.get_test_path_and_label(domain=domain)
        # for i in  range(len(label)):
        #     if label[i] not in self.know_classes:
        #         label[i] = -1
        dataset = DgDataset(img_path, label, transform=self.transform)
        dataLoader = DataLoader(dataset, batch_size=self.bs, shuffle=True, num_workers=self.nw)
        return dataLoader



if __name__ == "__main__":

    config = {  'bs': 64,
                'nw': 2,
                'dataset': 'pacs',
                'source': 'photo',
                'target': 'sketch',
                'know_classes': [0, 1, 2, 3],
                'dataset_root': '/share/test/jpk/PACS'
                }

    dataLoader = DgDataLoader(config, device='cuda')
    loader = dataLoader.get_train_loader()
    for i, (batch, label, mask) in enumerate(loader):
        print(mask)
        print(label)
