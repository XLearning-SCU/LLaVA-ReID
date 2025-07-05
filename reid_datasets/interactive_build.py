import json
import os

from .bases import ImageDataset, TextDataset
from .build import build_transforms
from .interactive_dataset import InteractiveReIDDataset
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.iotools import read_image
from utils import misc
from torch.utils.data.distributed import DistributedSampler
import torch


class InteractiveTextTrainingDataset(Dataset):
    components = ['pids', 'image_ids', 'image_path', 'fine-grained_caption', 'initial_query', 'questions', 'answers']

    def __init__(self,
                 dataset,
                 image_folder,
                 transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.dataset = self.load_component(dataset)

    def __len__(self):
        return len(self.dataset)

    def load_component(self, dataset):
        dataset_new = []
        for i in range(len(dataset)):
            sample = {}
            for idx, name in enumerate(self.components):
                sample[name] = dataset[i][idx]
            dataset_new.append(sample)

        return dataset_new

    def __getitem__(self, index):
        ret = self.dataset[index]
        if 'image_path' in ret:
            image = read_image(os.path.join(self.image_folder, ret['image_path']))
            if self.transform:
                image = self.transform(image)
            ret.update({'images': image})

        return ret


class InteractiveTextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 fine_grained_caption,
                 initial_query):
        self.caption_pids = caption_pids
        self.fine_grained_caption = fine_grained_caption
        self.initial_query = initial_query
        assert len(self.caption_pids) == len(
            self.fine_grained_caption), f"{len(self.caption_pids)}!={len(self.fine_grained_caption)}"
        assert len(self.caption_pids) == len(self.initial_query), f"{len(self.caption_pids)}!={len(self.initial_query)}"

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, f_cap, iqy = self.caption_pids[index], self.fine_grained_caption[index], self.initial_query[index]

        return index, pid, f_cap, iqy


class WarmUpDataset(Dataset):
    def __init__(self,
                 file_path):
        dataset = json.load(open(os.path.join(file_path, "description.json"), 'r'))
        self.descriptions = [x["description"] for x in dataset]
        self.round = [x["round"] for x in dataset]
        print(f"Loaded dataset from {file_path} with {len(self.descriptions)} samples")

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        return {"idx": index, "rounds": self.round[index], "descriptions": self.descriptions[index]}


def collate_fn(batch):
    keys = set([key for b in batch for key in b.keys()])
    dict_gather = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_dict = {}
    for k, v in dict_gather.items():
        if torch.is_tensor(v[0]):
            batch_dict.update({k: torch.stack(v)})
        elif isinstance(v[0], int):
            batch_dict.update({k: torch.tensor(v)})
        elif 'cap' in k or k in ['questions', 'answers', 'initial_query']:
            batch_dict.update({k: v})
        elif isinstance(v, list) and isinstance(v[0], str):
            batch_dict.update({k: v})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")
    return batch_dict


def build_interactive_dataloader(args, transforms=False, logger=None):
    num_workers = args.num_workers
    dataset = InteractiveReIDDataset(root=args.data_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        is_training = args.stage != 'prepare_data'
        if args.stage == "warmup_selector":
            file_path = args.output_dir
            train_set = WarmUpDataset(file_path)
        else:
            train_transforms = build_transforms(img_size=args.img_size,
                                                aug=args.img_aug,
                                                is_train=is_training)
            val_transforms = build_transforms(img_size=args.img_size,
                                              is_train=False)

            train_set = InteractiveTextTrainingDataset(dataset.train,
                                                       image_folder=dataset.img_dir,
                                                       transform=train_transforms)
        logger.debug("Number of training examples: {}".format(len(train_set)))
        if args.distributed:
            logger.debug('using distributed random sampler')
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler = DistributedSampler(train_set,
                                         num_replicas=num_tasks,
                                         rank=global_rank,
                                         shuffle=is_training,
                                         drop_last=is_training)
            train_loader = DataLoader(train_set,
                                      sampler=sampler,
                                      batch_size=args.batch_size,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)
        else:
            logger.debug('using random sampler')
            sampler = RandomSampler(train_set)
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn,
                                      drop_last=is_training)

        if args.stage == "warmup_selector":
            return train_loader, None, None, num_classes
        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = InteractiveTextDataset(ds['caption_pids'],
                                             ds['fine-grained_description'],
                                             ds['initial_description'])

        if args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            val_img_sampler = DistributedSampler(val_img_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            val_img_loader = DataLoader(val_img_set,
                                        batch_size=args.batch_size,
                                        sampler=val_img_sampler,
                                        num_workers=num_workers)
            val_txt_sampler = DistributedSampler(val_txt_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            val_txt_loader = DataLoader(val_txt_set,
                                        batch_size=args.batch_size,
                                        sampler=val_txt_sampler,
                                        num_workers=num_workers)
        else:
            val_img_loader = DataLoader(val_img_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)
            val_txt_loader = DataLoader(val_txt_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if transforms:
            test_transforms = transforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
