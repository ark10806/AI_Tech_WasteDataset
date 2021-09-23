from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from glob import glob
import os
import albumentations as A
import albumentations.pytorch as AP
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import json

'''
## ToDo List
1. Transforms for both (IMG, LABELS).
    BY [absolute_coords]
'''

dataroot = '/opt/ml/code/Waste/data/TACO/data'
visroot = './results/abs'
if not os.path.isdir(visroot):
    os.makedirs(visroot)

def load_data(isTrain, batch_size):
    if isTrain:
        taco = TacoDataset(dataroot=dataroot, isTrain=isTrain)
        datasets = split_dataset(taco)

        dataloader = {x: DataLoader(
            dataset = datasets[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = True,
        ) for x in ['train', 'test']}

    else:
        taco = TacoDataset(dataroot=dataroot, isTrain=isTrain)

        dataloader = DataLoader(
            dataset = taco,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 0,
            drop_last = False
        )
    return dataloader

def split_dataset(dataset, val_split=0.2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

class TacoDataset(Dataset):
    def __init__(self, dataroot, isTrain, transforms=None, absolute_coords=True, vis_ids=[]):
        self.transforms = transforms
        self.absolute_coords = absolute_coords
        self.dataroot = dataroot
        self.isTrain = isTrain

        labels_path = os.path.join(dataroot, 'annotations.json')
        with open(labels_path, 'r') as lb:
            labels = json.load(lb)

        annotations = labels['annotations']
        images = labels['images']
        self.categories = labels['categories']

        self.labels = {}
        self.get_labels(images, annotations)

        self.images = []
        self.get_images(images)

        # vis_ids = [x for x in range(10)]
        for vis_id in vis_ids:
            img = Image.open(self.images[vis_id])
            lab = self.labels[vis_id]
            self.visualize(img, lab, str(vis_id))

    def get_labels(self, images, annotations):
        for img in tqdm(images):
            annot_list = []
            curr_id = img['id']
            for ann in annotations:
                if ann['image_id'] == curr_id:
                    bbox_tuple = (ann['category_id'], ann['bbox'])
                    annot_list.append(bbox_tuple)
            
            self.labels[curr_id] = annot_list

    def get_images(self, images):
        for img in images:
            abs_path = os.path.join(self.₩dataroot, img['file_name'])
            self.images.append(abs_path)

    def __len__(self):
        return len(self.images)

    def visualize(self, img: Image, labels: list, fname: str):
        draw = ImageDraw.Draw(img)

        for category_id, bbox in labels:
            pt1 = (bbox[0], bbox[1])
            pt2 = (pt1[0]+bbox[2], pt1[1]+bbox[3])
            draw.rectangle((pt1, pt2), outline=(255,0,0), width=3)
            category = self.categories[category_id]['name']
            draw.text(pt1, category, (0,255,255), font=ImageFont.truetype('/opt/ml/code/Waste/AI_Tech_WasteDataset/ObjectDetection/Phase1/SwinDetector/FONTS/ubuntu.regular.ttf', 48))

        img.save(f'{visroot}/vis{fname}.jpg')
        print(f'vis{fname} saved!')

    def __getitem__(self, idx):
        """ Returns:
            X   (Image): PIL.Image
            Y   (List[(int, List)]): Contains cateogory_id(int), bboxs(list), segpoly
        """
        X = Image.open(self.images[idx])

        if self.isTrain:
            Y = self.labels[idx]    # [ (category_id, bboxs), ... ]
            return X, Y
        else:
            return X



if __name__ == '__main__':
    taco = TacoDataset(dataroot=dataroot, isTrain=True, transforms=None, absolute_coords=False, vis_ids=[x for x in range(100)])
    # taco = TacoDataset(dataroot=dataroot, isTrain=True, transforms=None, absolute_coords=False, vis_ids=[])