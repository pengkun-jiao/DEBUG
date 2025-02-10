# Author: Pengkun Jiao
# Date: 2022


import os.path as osp


class PACS():
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    """
    def __init__(self, dataset_root):

        self.root = dataset_root

        self.domain = ['art_painting', 'cartoon', 'photo', 'sketch']
        self.classesname = ['dog', 'elephant', 'giraffe', 'guitar', 'horse','house', 'person']
        
        self.art_path = osp.join(self.root , "kfold/art_painting")
        self.cartoon_path = osp.join(self.root , "kfold/cartoon")
        self.photo_path = osp.join(self.root , "kfold/photo")
        self.sketch_path = osp.join(self.root , "kfold/sketch")

        self.train_val_splits_path = osp.join(self.root , "kfold/data_list")

        self.know_classes = [0, 1, 2, 3]
        self.unknow_classes = [4, 5, 6]

       
    def get_label_name(self, label):
        return self.classes_name[label]

    def get_train_path_and_label(self, domain):
        path, label = self.get_prepared_path_and_label(domain, 'train')
        return path, label

    def get_val_path_and_label(self, domain):
        path, label = self.get_prepared_path_and_label(domain, 'val')
        return path, label

    def get_test_path_and_label(self, domain):
        path, label = self.get_prepared_path_and_label(domain, 'test')
        return path, label
        
    def get_prepared_path_and_label(self, domain, set_type):

        img_path = []
        labels = []

        domain_path = osp.join(self.root, 'kfold')

        data_list = domain + '_' + set_type +'.txt'

        with open(osp.join(self.train_val_splits_path, data_list), 'r') as f:
            lines = f.readlines()
            for line in lines:
                _path, _label = line.strip().split()
                _path = osp.join(domain_path, _path)
                _label = int(_label) -1
                if _label in self.know_classes:   
                    img_path.append(_path)
                    labels.append(_label)
                elif _label in self.unknow_classes:
                    img_path.append(_path)
                    labels.append(-1)
                # print(img_path)
                # print(labels)
                # quit()

        return img_path, labels
            






if __name__ == "__main__":

    data_set_path = "/share/test/jpk/PACS"
    dataset = PACS(data_set_path)
    img_path, label = dataset.get_train_path_and_label('photo', know_classes=[1,2,3])
    print(img_path)
    print(label)

