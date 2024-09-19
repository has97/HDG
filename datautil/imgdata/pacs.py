import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random

class CustomDatasetPACS(Dataset):
    def __init__(self, root_folder, class_indices,source_name,needed_classes,shots, domain_label,indices= None,transform=None,train=True):
        """
        Args:
        root_folder (str): Path to the root folder containing image files.
        txt_file (str): Path to the txt file containing image paths and labels.
        domain_label (int): Label for the domain.
        shots (int): Number of samples per class. If -1, use all samples.
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_folder = root_folder
        self.domain_label = domain_label
        self.shots = shots
        self.transform = transform
        self.filter_class = needed_classes[domain_label]
        self.train = train
        self.source_name = source_name
        self.class_indices = class_indices

        self.image_paths = []
        self.labels_class = []
        self.labels_domain = []
        self.class_names = set()
        if self.train:
            txt_file = f'/raid/biplab/hassan/pacs_data/image_list/{source_name}_train_kfold.txt'
        else:
            txt_file = f'/raid/biplab/hassan/pacs_data/image_list/{source_name}_test_kfold.txt'

        self._load_data_from_txt(txt_file)
        print(self.image_paths[:2])

    def _load_data_from_txt(self, txt_file):
        data_per_class = {}
        
        # First, organize data by class
        with open(txt_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                class_name = img_path.split("/")[1]
                label = int(label)
                # if label not in self.class_indices:
                #     continue
                if class_name not in data_per_class:
                    data_per_class[class_name] = []
                data_per_class[class_name].append(img_path)

        # Then, sample (or take all) data for each class
        sorted_classes = sorted(data_per_class.keys())
        c=0
        for label in sorted_classes:
            if c not in self.class_indices:
                c+=1
                continue
            paths = data_per_class[label]
            if self.shots == -1 or len(paths) <= self.shots:
                selected_paths = paths
            else:
                selected_paths = random.sample(paths, self.shots)
            
            for path in selected_paths:
                full_path = os.path.join(self.root_folder, path)
                self.image_paths.append(full_path)
                self.labels_class.append(c)
                self.labels_domain.append(self.domain_label)
            
            self.class_names.add(label)
            c+=1

        self.labels = np.array(self.labels_class)
        self.class_names = sorted(list(self.class_names))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label_class = self.labels_class[idx]
        label_domain = self.labels_domain[idx]

        if self.transform:
            image = self.transform(image)

        return image, label_class, label_domain