import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
# from mmseg.core.evaluation import get_palette, get_classes
# from torchvision.utils import draw_segmentation_masks

from libs.models import DenseClip
from libs.visualization import plot_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--filename', type=str)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    bgMix = False
    args.filename = '00046.jpg'
    args.device ='cuda'

    # classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse','house', 'person']
    classnames = ['laptop', 'bike', 'chair', 'TV']

    templates = ['A photo of a {}.']
    model = DenseClip('RN50x16', classnames, templates, device=args.device)
    model.eval()
    threshold = 0.7

    
    filename = args.filename
    with Image.open(filename, 'r') as image:
        image = image.convert('RGB')
        image = image.resize((227,227))
        image_tensor = TF.to_tensor(image).multiply(255).to(torch.uint8)

        prep_image = model.preprocess(image)
        prep_image = prep_image.unsqueeze(dim=0)
        prep_image = prep_image.to(args.device)
        output = model(prep_image)
        output = F.interpolate(output, size=image_tensor.shape[-2:], mode='bilinear')  # [1, C, H, W]
        output = F.softmax(output, dim=1)
        output = output.squeeze(dim=0)  # [C, H, W]
        # output = output[:-1]  # [C-1, H, W]
        # masks = output > threshold
        masks = torch.zeros_like(output, dtype=torch.bool)
        for i in range(output.shape[0]):
            masks[i] = torch.where(output.argmax(dim=0) == i, True, False)

        masks = masks.cpu()

        save_path = './result'
        img_name = filename.split('.')[0]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        bg_filename = 'pic_065.jpg'
        bg = Image.open(bg_filename, 'r')
        bg = bg.convert('RGB')
        bg = np.asarray(bg)


        if bgMix:
            for i in range(masks.shape[0]):
                new_img_array = np.asarray(image)
                for j in range(masks.shape[1]):
                    for k in range(masks.shape[2]):
                        if not masks[i][j][k]:
                            new_img_array[j][k][0] = bg[j][k][0]
                            new_img_array[j][k][1] = bg[j][k][1]
                            new_img_array[j][k][2] = bg[j][k][2]
                new_img = Image.fromarray(np.uint8(new_img_array))
                new_img.save('{}/{}_{}.jpg'.format(save_path, img_name, classnames[i])) 
        else:
            for i in range(masks.shape[0]):
                new_img_array = np.asarray(image)
                for j in range(masks.shape[1]):
                    for k in range(masks.shape[2]):
                        if not masks[i][j][k]:
                            new_img_array[j][k][0] = 0
                            new_img_array[j][k][1] = 0
                            new_img_array[j][k][2] = 0
                new_img = Image.fromarray(np.uint8(new_img_array))
                new_img.save('{}/{}_{}.jpg'.format(save_path, img_name, classnames[i])) 

        # # colors = list(map(tuple, get_palette('pascal_voc')))
        # segmentation_masks = draw_segmentation_masks(image_tensor, masks)
        # segmentation_masks = segmentation_masks.permute(1, 2, 0)

        # images = [np.array(image), segmentation_masks]
        # plot_images(images, f'segment_{filename}')


if __name__ == '__main__':
    main()
