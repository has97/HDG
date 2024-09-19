# coding=utf-8
import os 
import glob 
import random 
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset 
random.seed(42)
class CustomDataset(Dataset): 
    def __init__(self, root_folder, class_indices,source_name,needed_classes,shots, domain_label,indices= None,transform=None,train=True): 
        """ Args: root_folder (str): Path to the root folder containing domain folders. class_indices (list): List of class indices to include. domain_label (int): Label for the domain. transform (callable, optional): Optional transform to be applied on an image. """ 
        self.root_folder = root_folder 
        self.class_indices = class_indices 
        self.domain_label = domain_label 
        self.shot = shots
        self.filter_class = needed_classes[domain_label]
        self.train = train
        self.source_name = source_name
        self.transform = transform 
        self.image_paths,self.labels_class,self.labels_domain,self.class_names = self._gather_images_from_domain() 
        self.labels = np.array(self.labels_class)
    def _gather_images_from_domain(self): 
        image_paths = [] 
        labels_class = [] 
        labels_domain = [] 
        class_names = [] 
        img_dir = os.path.join(self.root_folder,self.source_name)
        # print(img_dir)
        if self.train:
            img_dir = os.path.join(img_dir,'train')
        else:
            img_dir = os.path.join(img_dir,'val')
        # print(img_dir)
        dirs = sorted(os.listdir(img_dir))
        # dirs = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
        #        'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
        #        'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
        #        'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
        #        'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
        #        'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
        #        'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker'] 
        # print(dirs)
        c = 0 
        for i in dirs: 
            if c in self.class_indices: 
                class_names.append(i) 
                impaths = os.path.join(img_dir, i) 
                # if self.train is False:
                #     print(impaths)
                paths = glob.glob(os.path.join(impaths, '*.jpg')) 
                # print(paths)
                # random.shuffle(paths) # Assuming you want to sample a fixed number of images 
                if self.shot!=-1:
                    shots = self.shot # Define the number of shots 
                    paths = random.sample(paths,shots) # Sample the first `shots` images i
                image_paths.extend(paths) 
                labels_class.extend([c for _ in range(len(paths))]) 
                labels_domain.extend([self.domain_label for _ in range(len(paths))]) 
            c += 1 
        # if self.train is False:
        #     print(labels_class)
        return image_paths, labels_class, labels_domain, class_names 
    def __len__(self): 
        return len(self.image_paths) 
    def __getitem__(self, idx): 
        # index = self.indices[index]
        img_path = self.image_paths[idx] 
        image = Image.open(img_path).convert('RGB') 
        label_class = self.labels_class[idx] 
        label_domain = self.labels_domain[idx] 
        if self.transform: 
            image = self.transform(image) 
        return image, label_class, label_domain 