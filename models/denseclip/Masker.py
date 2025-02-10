from .denseclip import DenseClip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

class Masker():
    def __init__(self, device, classnames):

        self.device =device
        templates = ['A photo of a {}.']
        self.model = DenseClip('RN50x16', classnames, templates, device=self.device)
        self.model.eval()

        self.threshold = 0.7

        # self.transform = transforms.Compose([
        #     transforms.Resize([224, 224]),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])


    def get_mask(self, image_path, label, background=None):
        with Image.open(image_path, 'r') as image:
            image = image.convert('RGB')

            image = image.resize((224,224))
            # image_tensor = TF.to_tensor(image).multiply(255).to(torch.uint8)
            # image_tensor = self.transform(image)

            prep_image = self.model.preprocess(image)
            prep_image = prep_image.unsqueeze(dim=0)
            prep_image = prep_image.to(self.device)
            output = self.model(prep_image)
            output = F.interpolate(output, size=224, mode='bilinear')  # [1, C, H, W]
            output = F.softmax(output, dim=1)
            output = output.squeeze(dim=0)  # [C, H, W]
            # output = output[:-1]  # [C-1, H, W]
            # masks = output > threshold

            # masks = torch.zeros_like(output, dtype=torch.int8)
            # for i in range(output.shape[0]):
            #     masks[i] = torch.where(output.argmax(dim=0) == i, 1, 0)
            # masks = masks.cpu()
            mask = torch.zeros_like(output[0], dtype=torch.int8)
            mask = torch.where(output.argmax(dim=0) == label, 1, 0).detach().cpu()

            return mask

if __name__ == "__main__":

    classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse','house', 'person']
    device = 'cuda'
    masker = Masker(device=device, classnames=classnames)
    filename = '/share/test/jpk/PACS/kfold/photo/giraffe/n02439033_16483.jpg'

    label = 2
    mask = masker.get_mask(filename, label) 
    print(mask)
