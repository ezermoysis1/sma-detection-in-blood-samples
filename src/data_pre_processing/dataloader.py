from __future__ import annotations

import os

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, min_number_images, transform=None):
        """
        CustomImageDataset constructor.

        Parameters:
            root_dir (str): Root directory path where the image data is located.
            class_to_idx (dict): A dictionary that maps class names to their corresponding index.
            min_number_images (int): Minimum number of images required per class to be included in the dataset.
            transform (callable, optional): A function that applies transformations to the images.

        Attributes:
            root_dir (str): Root directory path where the image data is located.
            transform (callable): A function that applies transformations to the images.
            class_to_idx (dict): A dictionary that maps class names to their corresponding index.
            samples (list): A list to store tuples of image file paths, class label, and subfolder name.
            class_counts (dict): A dictionary to keep track of the number of samples per class.
            min_number_images (int): Minimum number of images required per class to be included in the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        self.class_counts = {}
        self.min_number_images = min_number_images

        # Loop through each class to load image samples and store their counts
        for class_ in self.class_to_idx.keys():
            class_dir = os.path.join(root_dir, class_)
            count = 0
            for id_dir in os.listdir(class_dir):
                id_dir_path = os.path.join(class_dir, id_dir)
                if os.path.isdir(id_dir_path):
                    images = []
                    for img_name in os.listdir(id_dir_path):
                        img_path = os.path.join(id_dir_path, img_name)
                        if os.path.isfile(img_path) and (img_name.lower().endswith('.tiff') or img_name.lower().endswith('.png')):
                            images.append(img_path)
                    # Include the class if it has enough images
                    if len(images) >= min_number_images:
                        self.samples.append(
                            (images, self.class_to_idx[class_], id_dir),
                        )
                        count += 1
            self.class_counts[class_] = count

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            idx (int): Index of the sample to be retrieved.

        Returns:
            tuple: A tuple containing a list of transformed images, class label, and subfolder name.
        """
        image_files, label, subfolder_name = self.samples[idx]
        images = []
        for image_file in image_files:
            image = Image.open(image_file)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images, label, subfolder_name

    def get_class_counts(self):
        """
        Get the number of samples per class in the dataset.

        Returns:
            dict: A dictionary containing class names as keys and their corresponding sample counts as values.
        """
        return self.class_counts
