import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):

    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    CLASSES.sort()

    def __init__(self, root, task, filter_class,shots, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/Art.txt",
                "C": "image_list/Clipart.txt",
                "P": "image_list/Product.txt",
                "R": "image_list/Real_World.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": f"image_list/art_train_{shots}.txt",
                "C": f"image_list/clipart_train_{shots}.txt",
                "P": f"image_list/product_train_{shots}.txt",
                "R": f"image_list/realworld_train_{shots}.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": f"image_list/art_val.txt",
                "C": f"image_list/clipart_val.txt",
                "P": f"image_list/product_val.txt",
                "R": f"image_list/realworld_val.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(OfficeHome, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)

        self.domain = ["art", "clipart", "product", "realwdorld"]

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        domain_name = path.split('/')[-4]
        domain_label = self.domain.index(domain_name)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

