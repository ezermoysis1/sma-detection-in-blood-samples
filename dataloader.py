import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        self.class_counts = {}

        for class_ in self.class_to_idx.keys():
            class_dir = os.path.join(root_dir, class_)
            count = 0
            for id_dir in os.listdir(class_dir):

                id_dir_path = os.path.join(class_dir, id_dir)
                if os.path.isdir(id_dir_path):  
                    images = []
                    for img_name in os.listdir(id_dir_path):
                        img_path = os.path.join(id_dir_path, img_name)
                        if os.path.isfile(img_path) and (img_name.lower().endswith(('.tiff')) or img_name.lower().endswith(('.png'))):  
                            images.append(img_path)
                    if images:  
                        self.samples.append((images, self.class_to_idx[class_]))
                        count += 1  
            self.class_counts[class_] = count 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_files, label = self.samples[idx]
        images = []
        for image_file in image_files:
            image = Image.open(image_file)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images, label

    def get_class_counts(self):
        return self.class_counts