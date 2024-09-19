# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.imgdata.odg import CustomDataset
from datautil.imgdata.pacs import CustomDatasetPACS
from datautil.mydataloader import InfiniteDataLoader
import dalib.vision.datasets as datasets

def get_odgclip_dataloader(args):
    rate =0.2
    trdatalist,valdatalist,tedatalist = [], [], []
    if args.dataset =='office-home':
        names = ['art','clipart','product','realworld']
        args.domain_num = len(names)

        source_classes: list(list) = [[], [], []]
        needed_classes: list(list) = [[], [], [], []]

        if args.dataset == 'office-home':
            # source_classes[0] = list(range(0, 18))
            # source_classes[1] = list(range(18,36))
            # source_classes[2] = list(range(36,54))
            # all_target_classes = list(range(0,65))
            source_classes[0] = list(range(0, 15)) + list(range(21, 32))
            source_classes[1] = list(range(0, 9)) + list(range(15, 21)) + list(range(32, 43))
            source_classes[2] = list(range(0, 3)) + list(range(9, 21)) + list(range(43, 54))
            all_target_classes = [0,3,4,9,10,15,16,21,22,23,32,33,34,43,44,45] + list(range(54, 65))
    elif args.dataset =='PACS':
        names = ['cartoon','photo','sketch', 'art_painting']
        args.domain_num = len(names)

        source_classes: list(list) = [[], [], []]
        needed_classes: list(list) = [[], [], [], []]

        source_classes[0] = [3, 0, 1]
        source_classes[1] = [4, 0, 2]
        source_classes[2] = [5, 1, 2]
        all_target_classes = [0,1,2,3,4,5,6]
        

    needed_classes[3] = all_target_classes
    indices = list(range(4))
    indices.remove(args.test_envs[0])
    source_domains = []
    for i in range(len(names)):
        if i == args.test_envs[0]:
            continue
        source_domains.append(names[i])
    for k in range(3):
        needed_classes[k] = source_classes[k]
    source_domains.sort()
    i=0
    for s in source_domains:
        if args.dataset == 'PACS':
            trdatalist.append(
                CustomDatasetPACS(args.data_dir,source_classes[i],source_domains[i],needed_classes,args.shot,i,indices=None,transform=imgutil.image_train(args.dataset, args.is_data_aug)
                )
            )
            valdatalist.append(
                CustomDatasetPACS(args.data_dir,source_classes[i],source_domains[i],needed_classes,-1,i,indices=None,transform=imgutil.image_test(args.dataset),train=False
                )
            )

        elif args.dataset == 'office-home':
            trdatalist.append(
                CustomDataset(args.data_dir,source_classes[i],source_domains[i],needed_classes,args.shot,i,indices=None,transform=imgutil.image_train(args.dataset, args.is_data_aug)
                )
            )
            valdatalist.append(
                CustomDataset(args.data_dir,source_classes[i],source_domains[i],needed_classes,-1,i,indices=None,transform=imgutil.image_test(args.dataset),train=False
                )
            )
        i+=1
    if args.dataset =='office-home':
        tedatalist.append(
                CustomDataset(args.data_dir,all_target_classes,names[args.test_envs[0]],needed_classes,-1,3,indices=None,transform=imgutil.image_test(args.dataset),train=False
                )
            )
    elif args.dataset == 'PACS':
        tedatalist.append(
                CustomDatasetPACS(args.data_dir,all_target_classes,names[args.test_envs[0]],needed_classes,-1,3,indices=None,transform=imgutil.image_test(args.dataset),train=False
                )
            )
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in valdatalist + tedatalist
    ]

    return train_loaders, eval_loaders



def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []
    # names: ['Art', 'Clipart', 'Product', 'Real_World']
    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    source_classes: list(list) = [[], [], []]
    needed_classes: list(list) = [[], [], [], []]
    if args.dataset == 'PACS':
        if args.is_different_class_space == 1:
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        else:
            for i in range(3):
                source_classes[i] = list(range(6))

        all_target_classes = [0, 1, 2, 3, 4, 5, 6]
    if args.dataset == 'office-home':
        source_classes[0] = list(range(0, 15)) + list(range(21, 32))
        source_classes[1] = list(range(0, 9)) + list(range(15, 21)) + list(range(32, 43))
        source_classes[2] = list(range(0, 3)) + list(range(9, 21)) + list(range(43, 54))
        all_target_classes = [0,3,4,9,10,15,16,21,22,23,32,33,34,43,44,45] + list(range(54, 65))

    needed_classes[args.test_envs[0]] = all_target_classes
    indices = list(range(4))
    indices.remove(args.test_envs[0])
    for idx_src, idx_needed in enumerate(indices):
        needed_classes[idx_needed] = source_classes[idx_src]

    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,  # 'img_dg'
                    args.data_dir,  # rood_dir
                    names[i],  # domain_name
                    i,  # domain_label
                    needed_classes,
                    transform=imgutil.image_test(args.dataset),
                    test_envs=args.test_envs,  # list
                )
            )
        else:  # training domain
            tmpdatay = ImageDataset(
                args.dataset,
                args.task,
                args.data_dir,
                names[i],
                i,
                needed_classes,
                transform=imgutil.image_train(args.dataset, args.is_data_aug),
                test_envs=args.test_envs,
            ).labels

            l = len(tmpdatay)

            if args.split_style == "strat":
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    needed_classes,
                    transform=imgutil.image_train(args.dataset, args.is_data_aug),
                    indices=indextr,
                    test_envs=args.test_envs,
                )
            )
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    needed_classes,
                    transform=imgutil.image_test(args.dataset),
                    indices=indexte,
                    test_envs=args.test_envs,
                )
            )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return train_loaders, eval_loaders


def get_img_source_unknown_dataloader(args):
    """
    Get source domain with unknown class dataloader
    (Source domain but class space is the same as target domain)
    """
    rate = 0.2
    tedatalist = []
    # names: ['Art', 'Clipart', 'Product', 'Real_World']
    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    source_classes: list(list) = [[], [], []]
    needed_classes: list(list) = [[], [], [], []]
    if args.dataset == 'PACS':
        if args.is_different_class_space == 1:
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        else:
            for i in range(3):
                source_classes[i] = list(range(6))

        all_target_classes = [0, 1, 2, 3, 4, 5, 6]
    if args.dataset == 'office-home':
        if args.is_different_class_space == 1:
            source_classes[0] = [
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                0,
                1,
                2,
            ]
            source_classes[1] = [
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                3,
                4,
                5,
                6,
                7,
                8,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
            source_classes[2] = [
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
        else:
            for i in range(3):
                source_classes[i] = list(range(54))

        all_target_classes = [
            0,
            3,
            4,
            9,
            10,
            15,
            16,
            21,
            22,
            23,
            32,
            33,
            34,
            43,
            44,
            45,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
        ]


    needed_classes[args.test_envs[0]] = all_target_classes
    indices = list(range(4))
    indices.remove(args.test_envs[0])
    for idx_src, idx_needed in enumerate(indices):
        if args.dataset == 'PACS':
            needed_classes[idx_needed] = source_classes[idx_src] + [6]
        elif args.dataset == 'office-home':
            needed_classes[idx_needed] = source_classes[idx_src] + [
                i for i in range(54, 64 + 1)
            ]

    """
    for i in range(4):
        needed_classes[i] = all_target_classes
    """
    for i in range(len(names)):
        if i in args.test_envs:
            pass
        else:  # source domain
            tmpdatay = ImageDataset(
                args.dataset,
                args.task,
                args.data_dir,
                names[i],
                i,
                needed_classes,
                # transform=imgutil.image_train(args.dataset),
                transform=imgutil.image_test(args.dataset),
                test_envs=args.test_envs,
            ).labels

            l = len(tmpdatay)

            if args.split_style == "strat":
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    needed_classes,
                    transform=imgutil.image_test(args.dataset),
                    indices=indexte,
                    test_envs=args.test_envs,
                )
            )

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in tedatalist
    ]

    return eval_loaders


def get_img_daml_dataloader(args):

    trdatalist, tedatalist = [], []
    # names: ['Art', 'Clipart', 'Product', 'Real_World']
    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    if args.dataset == 'DomainNet':
        source_classes: list(list) = [[], [], [], [], []]
        needed_classes: list(list) = [[], [], [], [], [], []]
    else:
        source_classes: list(list) = [[], [], []]
        needed_classes: list(list) = [[], [], [], []]

    if args.dataset == 'PACS':
        initial_names = ['A', 'C', 'P', 'S']
        if args.is_different_class_space == 1: # ratio = 1/6
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        elif args.is_different_class_space == 0: # ratio = 0
            source_classes[0] = [0, 1]
            source_classes[1] = [2, 3]
            source_classes[2] = [4, 5]
        elif args.is_different_class_space == 2: # ratio = 1/3
            source_classes[0] = [0, 1, 2, 3]
            source_classes[1] = [0, 1, 4, 5]
            source_classes[2] = [2, 3, 4, 5]
        else: # ratio = 1
            for i in range(3):
                source_classes[i] = list(range(6))
        all_target_classes = [0, 1, 2, 3, 4, 5, 6]
    elif args.dataset == 'office-home':
        initial_names = ['A', 'C', 'P', 'R']
        if args.is_different_class_space == 1: # ratio = 1/6
            source_classes[0] = [
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                0,
                1,
                2,
            ]
            source_classes[1] = [
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                3,
                4,
                5,
                6,
                7,
                8,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
            source_classes[2] = [
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
        elif args.is_different_class_space == 0: # ratio = 0
            source_classes[0] = list(range(18))
            source_classes[1] = list(range(18, 36))
            source_classes[2] = list(range(36, 54))
        elif args.is_different_class_space == 2: # ratio = 1/3
            source_classes[0] = list(range(36))
            source_classes[1] = list(range(18, 54))
            source_classes[2] = list(range(18)) + list(range(36, 54))
        else: # ratio = 1
            for i in range(3):
                source_classes[i] = list(range(54))
        all_target_classes = list(range(65))

    elif args.dataset == 'mini-office-home':
        initial_names = ['A', 'C', 'P', 'R']
        if args.is_different_class_space == 1:
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        else:
            for i in range(3):
                source_classes[i] = list(range(6))
        all_target_classes = [0, 1, 2, 3, 4, 5, 6]
    elif args.dataset == 'DomainNet':
        initial_names = ['C', 'I', 'P', 'Q', 'R', 'S']
        if args.is_different_class_space == 1:  # ratio = 1/10
            source_classes[0] = list(range(0, 108))
            source_classes[1] = list(range(54, 162))
            source_classes[2] = list(range(108, 216))
            source_classes[3] = list(range(162, 270))
            source_classes[4] = list(range(216, 270)) + list(range(0, 54))
        elif args.is_different_class_space == 0:  # ratio = 0
            source_classes[0] = list(range(0, 54))
            source_classes[1] = list(range(54, 108))
            source_classes[2] = list(range(108, 162))
            source_classes[3] = list(range(162, 216))
            source_classes[4] = list(range(216, 270))
        elif args.is_different_class_space == 2:  # ratio = 1/5
            source_classes[0] = list(range(0, 135))
            source_classes[1] = list(range(45, 180))
            source_classes[2] = list(range(90, 225))
            source_classes[3] = list(range(135, 270))
            source_classes[4] = list(range(180, 270)) + list(range(0, 45))
        else: # ratio = 1
            for i in range(5):
                source_classes[i] = list(range(270))

        all_target_classes = list(range(345))

    needed_classes[args.test_envs[0]] = all_target_classes
    if args.dataset == 'DomainNet':
        indices = list(range(6))
    else:
        indices = list(range(4))
    indices.remove(args.test_envs[0])
    for idx_src, idx_needed in enumerate(indices):
        needed_classes[idx_needed] = source_classes[idx_src]
    if args.dataset == 'office-home' or args.dataset == 'mini-office-home':
        dataset = datasets.__dict__['OfficeHome']
    else:
        dataset = datasets.__dict__[args.dataset]
    # e.g.,
    ##### root='./dataset/OfficeHome'
    ##### args.data_dir='./dataset/OfficeHome/'
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],  # equal to all_target_classes
                    split='all',
                    transform=imgutil.image_test(args.dataset),
                )
            )
        else:  # source domain
            trdatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],
                    split='train',
                    transform=imgutil.image_train(args.dataset, args.is_data_aug),
                )
            )
            tedatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],
                    split='val',
                    transform=imgutil.image_test(args.dataset),
                )
            )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]
    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return train_loaders, eval_loaders


def get_img_daml_source_unknown_dataloader(args):
    trdatalist, tedatalist = [], []
    # names: ['Art', 'Clipart', 'Product', 'Real_World']
    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    if args.dataset == 'DomainNet':
        source_classes: list(list) = [[], [], [], [], []]
        needed_classes: list(list) = [[], [], [], [], [], []]
    else:
        source_classes: list(list) = [[], [], []]
        needed_classes: list(list) = [[], [], [], []]

    # source_classes: list(list) = [[], [], []]
    # needed_classes: list(list) = [[], [], [], []]

    if args.dataset == 'PACS':
        initial_names = ['A', 'C', 'P', 'S']
        if args.is_different_class_space == 1:
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        else:
            for i in range(3):
                source_classes[i] = list(range(6))

        all_target_classes = [0, 1, 2, 3, 4, 5, 6]
    elif args.dataset == 'office-home':
        initial_names = ['A', 'C', 'P', 'R']
        if args.is_different_class_space == 1:
            source_classes[0] = [
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                0,
                1,
                2,
            ]
            source_classes[1] = [
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                3,
                4,
                5,
                6,
                7,
                8,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
            source_classes[2] = [
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                0,
                1,
                2,
            ]
        else:
            for i in range(3):
                source_classes[i] = list(range(54))
        all_target_classes = [
            0,
            3,
            4,
            9,
            10,
            15,
            16,
            21,
            22,
            23,
            32,
            33,
            34,
            43,
            44,
            45,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
        ]

    elif args.dataset == 'mini-office-home':
        initial_names = ['A', 'C', 'P', 'R']
        if args.is_different_class_space == 1:
            source_classes[0] = [0, 1, 3]
            source_classes[1] = [0, 2, 4]
            source_classes[2] = [1, 2, 5]
        else:
            for i in range(3):
                source_classes[i] = list(range(6))
        all_target_classes = [0, 1, 2, 3, 4, 5, 6]

    elif args.dataset == 'DomainNet':
        initial_names = ['C', 'I', 'P', 'Q', 'R', 'S']
        if args.is_different_class_space == 1:  # ratio = 1/6
            source_classes[0] = list(range(0, 90))
            source_classes[1] = list(range(45, 135))
            source_classes[2] = list(range(90, 180))
            source_classes[3] = list(range(135, 225))
            source_classes[4] = list(range(180, 270))
        elif args.is_different_class_space == 2:  # ratio = 0
            source_classes[0] = list(range(0, 54))
            source_classes[1] = list(range(54, 108))
            source_classes[2] = list(range(108, 162))
            source_classes[3] = list(range(162, 216))
            source_classes[4] = list(range(216, 270))
        else:
            for i in range(5):  # ratio = 1
                source_classes[i] = list(range(270))

        all_target_classes = list(range(345))

    needed_classes[args.test_envs[0]] = all_target_classes
    if args.dataset == 'DomainNet':
        indices = list(range(6))
    else:
        indices = list(range(4))
    indices.remove(args.test_envs[0])
    for idx_src, idx_needed in enumerate(indices):
        # needed_classes[idx_needed] = source_classes[idx_src]
        needed_classes[idx_needed] = all_target_classes
    if args.dataset == 'office-home' or args.dataset == 'mini-office-home':
        dataset = datasets.__dict__['OfficeHome']
    else:
        dataset = datasets.__dict__[args.dataset]
    # e.g.,
    ##### root='./dataset/OfficeHome'
    ##### args.data_dir='./dataset/OfficeHome/'
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],  # equal to all_target_classes
                    split='all',
                    transform=imgutil.image_test(args.dataset),
                )
            )

        else:  # source domain

            trdatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],
                    split='train',
                    transform=imgutil.image_train(args.dataset, args.is_data_aug),
                )
            )
            tedatalist.append(
                dataset(
                    root=args.data_dir[:-1],
                    task=initial_names[i],
                    filter_class=needed_classes[i],
                    split='val',
                    transform=imgutil.image_test(args.dataset),
                )
            )

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return eval_loaders


def get_img_daml_multi_dataloader(args):
    assert args.dataset == 'MultiDataSet'
    trdatalist, tedatalist = [], []
    # names: ['Art', 'Clipart', 'Product', 'Real_World']

    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    source_root_list = [None, None, None]
    # source_root_list = [None, None, None, None, None]
    # d_path = './dataset
    source_root_list[0] = f'{args.d_path}/Office31/office31'
    # source_root_list[1] = f'{args.d_path}/Office31/office31'
    # source_root_list[2] = f'{args.d_path}/Office31/office31'
    #
    source_root_list[1] = f'{args.d_path}/Visda2017/train'
    source_root_list[2] = f'{args.d_path}/STL10/img'
    # source_root_list[3] = f'{args.d_path}/Visda2017/train'
    # source_root_list[4] = f'{args.d_path}/STL10/img'

    target_root = f'{args.d_path}/DomainNet'

    # initial_names = ['A', 'hoge', 'S', 'V']  # hoge is target domain
    # A: amazon of office31
    # S: stl10
    # V: visda2017
    source_initinal_names = ['A', 'V', 'S']
    # C (Clipart), R (Real), P (Painting) ,K (sKecth)
    # target_initinal_names = list(args.t_domain)
    target_initinal_names = ['C', 'R', 'P', 'K']
    # target_initinal_names = ['C']  # C (Clipart), R (Real), P (Painting) ,K (sKecth)
    source_classes: list(list) = [[], [], [], [], []]
    source_classes[0] = list(range(0, 30 + 1))
    # source_classes[1] = list(range(0, 30 + 1))
    # source_classes[2] = list(range(0, 30 + 1))
    # source_classes[3] = [1] + list(range(31, 41 + 1))
    # source_classes[4] = [31, 33, 34, 41] + list(range(42, 47 + 1))
    source_classes[1] = [1] + list(range(31, 41 + 1))
    source_classes[2] = [31, 33, 34, 41] + list(range(42, 47 + 1))
    target_classes = (
        [0, 1, 5, 6, 10, 11, 14, 17, 20, 26]
        + list(range(31, 36 + 1))
        + list(range(39, 43 + 1))
        + [45, 46]
        + list(range(48, 67 + 1))  # unknown classes
    )

    dataset = datasets.__dict__['MultiDataSet']

    # e.g.,
    ##### root='./dataset/OfficeHome'
    ##### args.data_dir='./dataset/OfficeHome/'
    for i in range(len(source_root_list)):
        trdatalist.append(
            dataset(
                root=source_root_list[i],
                task=source_initinal_names[i],
                filter_class=source_classes[i],
                split='train',
                transform=imgutil.image_train(args.dataset, args.is_data_aug),
            )
        )
        tedatalist.append(
            dataset(
                root=source_root_list[i],
                task=source_initinal_names[i],
                filter_class=source_classes[i],
                split='val',
                transform=imgutil.image_test(args.dataset),
            )
        )

    tedatalist.append(
        dataset(
            root=target_root,
            task=target_initinal_names[0],
            filter_class=target_classes,  # equal to all_target_classes
            split='all',
            transform=imgutil.image_test(args.dataset),
        )
    )
    tedatalist.append(
        dataset(
            root=target_root,
            task=target_initinal_names[1],
            filter_class=target_classes,  # equal to all_target_classes
            split='all',
            transform=imgutil.image_test(args.dataset),
        )
    )
    tedatalist.append(
        dataset(
            root=target_root,
            task=target_initinal_names[2],
            filter_class=target_classes,  # equal to all_target_classes
            split='all',
            transform=imgutil.image_test(args.dataset),
        )
    )
    tedatalist.append(
        dataset(
            root=target_root,
            task=target_initinal_names[3],
            filter_class=target_classes,  # equal to all_target_classes
            split='all',
            transform=imgutil.image_test(args.dataset),
        )
    )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]
    # breakpoint()
    return train_loaders, eval_loaders


"""
def get_img_daml_multi_source_unknown_dataloader(args):
    assert args.dataset == 'MultiDataSet'
    trdatalist, tedatalist = [], []
    # names: ['A','C','S','V']

    names: list = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    source_root_list = [None, None, None]
    # d_path = './dataset
    source_root_list[0] = f'{args.d_path}/office31'
    source_root_list[1] = f'{args.d_path}/stl10'
    source_root_list[2] = f'{args.d_path}/visda2017/train'
    target_root = f'{args.d_path}/DomainNet'

    # initial_names = ['A', 'C', 'S', 'V']  # C is target domain
    source_initinal_names = ['A', 'S', 'V']
    target_initinal_names = list(
        args.t_domain
    )  # C (Clipart), R (Real), P (Painting) ,K (sKecth)
    source_classes: list(list) = [[], [], []]
    source_classes[0] = list(range(0, 30 + 1))
    source_classes[1] = [1] + list(range(31, 41 + 1))
    source_classes[2] = [31, 33, 34, 41] + list(range(42, 47 + 1))
    target_classes = (
        [0, 1, 5, 6, 10, 11, 14, 17, 20, 26]
        + list(range(31, 36 + 1))
        + list(range(39, 43 + 1))
        + [45, 46]
        + list(range(48, 67 + 1))  # unknown classes
    )

    dataset = datasets.__dict__['MultiDataSet']
    # e.g.,
    ##### root='./dataset/OfficeHome'
    ##### args.data_dir='./dataset/OfficeHome/'
    for i in range(len(source_root_list)):
        trdatalist.append(
            dataset(
                root=source_root_list[i],
                task=source_initinal_names[i],
                filter_class=source_classes[i],
                split='train',
                transform=imgutil.image_train(args.dataset, args.is_data_aug),
            )
        )
        tedatalist.append(
            dataset(
                root=source_root_list[i],
                task=source_initinal_names[i],
                filter_class=source_classes[i],
                split='val',
                transform=imgutil.image_test(args.dataset),
            )
        )

    tedatalist.append(
        dataset(
            root=target_root,
            task=target_initinal_names[0],
            filter_class=target_classes,  # equal to all_target_classes
            split='all',
            transform=imgutil.image_test(args.dataset),
        )
    )

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return eval_loaders
"""
