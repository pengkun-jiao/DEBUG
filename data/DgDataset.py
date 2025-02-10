import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, img):
        np_img = np.array(img)
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np_img + noise * 10  # Adjust noise strength to match range of pixel values
        np_img = np.clip(np_img, 0, 255)  # Ensure values remain in valid range
        return Image.fromarray(np_img.astype('uint8'))  # Convert back to PIL image

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, img):
        np_img = np.array(img)
        salt_mask = np.random.choice([True, False], size=np_img.shape, p=[self.salt_prob, 1-self.salt_prob])
        pepper_mask = np.random.choice([True, False], size=np_img.shape, p=[self.pepper_prob, 1-self.pepper_prob])
        np_img[salt_mask] = 255
        np_img[pepper_mask] = 0
        return Image.fromarray(np_img.astype('uint8'))  # Convert back to PIL image

    def __repr__(self):
        return self.__class__.__name__ + '(salt_prob={0}, pepper_prob={1})'.format(self.salt_prob, self.pepper_prob)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transform_aug = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     # transforms.RandomCrop(196, padding=4),
#     AddGaussianNoise(0., 1.),  # Noise applied to PIL image
#     AddSaltPepperNoise(),  # Noise applied to PIL image
#     transforms.Resize([224, 224]),
#     transforms.ToTensor(), 
#     transforms.Normalize(mean, std)
# ])

transform_aug = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # # transforms.RandomCrop(196, padding=4),
    # AddGaussianNoise(0., 1.),  # Noise applied to PIL image
    # AddSaltPepperNoise(),  # Noise applied to PIL image
    transforms.Resize([224, 224]),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])





class DgDataset(Dataset):
    def __init__(self, img_path, label=None, mask=None, transform=None):
        
        self.img_path = img_path
        self.label=label
        self.mask = mask
        # self.transform=transform
        self.transform = transform_aug

    def __getitem__(self, index):

        img = Image.open(self.img_path[index]).convert('RGB')

        if(self.transform!=None):
                img=self.transform(img)
        
        label =  self.label[index]


        if self.mask != None:
            mask = self.mask[index]
            return img, label, mask
        else:
            return img, label

    def __len__(self):
        return len(self.img_path)