# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

from network import Adver_network
from utils.daml_util import *
from torchvision import models
import math
from utils.daml_util import (
    get_ratio_mixup_Dirichlet,
    get_sample_mixup_random,
    manual_CE,
    weighted_manual_CE
)

import sys
sys.path.append('..')
from clip import clip
import timm


class CLIPBase(Algorithm):
    def __init__(self, args):
        super(CLIPBase, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        self.args = args
        self.device = torch.device(f'cuda:{self.args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # self.model, _ = clip.load('RN50', self.device)
        self.model, _ = clip.load('ViT-B/16', self.device)
        # self.model, _ = clip.load('ViT-L/14', self.device)
        # self.model, _ = clip.load('RN101', self.device)
        self.bz = args.batch_size
        self.num_classes = args.num_classes

    def torch_cosine_similarity(self, features1, features2):
        norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
        norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
        end_norm = torch.mm(norm1.to(torch.double), norm2.to(torch.double))
        cos = torch.mm(features1.to(torch.double), features2.T.to(torch.double)) / end_norm
        return cos

    def compute_similarity(self, features, class_features):
        features_norm = torch.nn.functional.normalize(features, dim=1)  # 归一化样本特征
        class_features_norm = torch.nn.functional.normalize(class_features, dim=2)  # 归一化类特征
        similarity_matrix = torch.bmm(features_norm.unsqueeze(1), class_features_norm.permute(0, 2, 1)).squeeze(1)
        return similarity_matrix

    def update(self, minibatches, opt, sch):

        # data preparation
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])
        thresh = self.args.test_envs[0]
        all_domain = torch.tensor([x.item() - 1 if x > thresh else x.item() for x in all_domain]).to(self.device)
        # original
        opt.zero_grad()


        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        known_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles']
        # known_classnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
        #  'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
        #  'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
        #  'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
        #  'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
        #  'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
        #  'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier',
        #  'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler',
        #  'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher',
        #  'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell',
        #  'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence',
        #  'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops',
        #  'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose',
        #  'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
        #  'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse',
        #  'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane',
        #  'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop',
        #  'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
        #  'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey',
        #  'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
        #  'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can',
        #  'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas',
        #  'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
        #  'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon',
        #  'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster',
        #  'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion',
        #  'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard',
        #  'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman',
        #  'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
        #  'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry',
        #  'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
        #  'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent',
        #  'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
        #  'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone',
        #  'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
        #  'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga',
        #  'zebra', 'zigzag']
        # image_input = self.preprocess(minibatches).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in known_classnames]).to(self.device)

        # Calculate features and generate CLIP logits for the image
        with torch.no_grad():
            image_features = self.model.encode_image(all_x)
            text_features = self.model.encode_text(text_inputs)
            text_sim = self.torch_cosine_similarity(text_features, text_features)
            # soft_text_sim0.1 = torch.softmax(text_sim / 0.1, dim=1)
            soft_text_sim = torch.softmax(text_sim, dim=1)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = torch.tensor(similarity, dtype=torch.float32, requires_grad=False)

        # feature output
        features = self.featurizer(all_x)
        class_logits = self.classifier(features)
        class_loss = manual_CE(class_logits, similarity)
        loss = class_loss

        loss.backward()
        opt.step()

        if sch:
            sch.step()

        return {"class": class_loss.item(),
                }


    def predict(self, x):
        features = self.featurizer(x)
        output = self.classifier(features)
        return output

    def base(self, x):
        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        known_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles']
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in known_classnames]).to(self.device)
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        return similarity






