#!/usr/env/bin python3.6

import io
import re
import pickle
from pathlib import Path
from ast import literal_eval
from itertools import repeat
from random import random, shuffle
from operator import itemgetter, mul
from functools import partial, reduce
from multiprocessing import cpu_count
from typing import Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from skimage.transform import resize
from torch._six import container_abcs
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import map_, class2one_hot, one_hot2dist, id_
from utils import one_hot, depth

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]

import platform

from utils import augment

resizing_fn = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def png_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


def npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda npy: np.array(npy)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


def tensor_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


def unet_loss_weights_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    w_0: float = 10
    sigma: float = 5

    def closure(in_: D) -> Tensor:
        gt: Tensor = gt_transform(resolution, K)(in_)

        signed_dist_map: Tensor = dist_map_transform(resolution, K)(in_)
        dist_map: Tensor = torch.abs(signed_dist_map).type(torch.float32)

        w_c: Tensor = torch.einsum("k...->k", gt) / reduce(mul, gt.shape[1:])
        filled_w_c: Tensor = torch.einsum("k,k...->k...", w_c.type(torch.float32), torch.ones_like(dist_map))

        w: Tensor = filled_w_c + w_0 * torch.exp(- dist_map**2 / (2 * sigma**2))
        assert (K, *in_.shape) == w.shape == gt.shape, (in_.shape, w.shape, gt.shape)

        final: Tensor = torch.einsum("k...,k...->k...", gt.type(torch.float32), w)

        return final

    return closure


def unnormalized_color_transform(size: Tuple[int, int], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: img.convert('RGB'),
        lambda img: np.asarray(img, dtype=np.uint8),
        # lambda arr: np.rollaxis(arr, 2, -2),
        lambda nd: torch.tensor(nd, dtype=torch.uint8)
    ])

def get_loaders(args, data_folder: str,
                batch_size: int, n_class: int,
                debug: bool, in_memory: bool,
                dimensions: int,
                ) -> Tuple[List[DataLoader], List[DataLoader]]:
    losses_list = eval(args.losses)
    if depth(losses_list) == 1:
        losses_list = [losses_list]

    list_folders_list = eval(args.folders)
    if depth(list_folders_list) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
        list_folders_list = [list_folders_list]
    # print(folders_list)

    # Prepare the datasets and dataloaders
    train_loaders = []
    
    train_topfolder, folders_list =args.training_folders[0], list_folders_list[0]
    folders, trans, are_hots = zip(*folders_list)
    train_folders: List[Path] = [Path(data_folder, train_topfolder, f) for f in folders]
    # I assume all files have the same name inside their folder: makes things much easier
    train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*"))
    
    patients = [name.split('_')[0] for name in train_names]
    patients = list(set(patients))
    patients.sort()
    #random.shuffle(patients)    
    client_patients=[]
    for i in range(args.client_num):
        if i != args.client_num-1:
            client_patients.append(patients[i*len(patients)//args.client_num: (i+1)*len(patients)//args.client_num])
        else:
            client_patients.append(patients[i*len(patients)//args.client_num:])
    client_patient_names=[]
    for i in range(args.client_num):
        temp=[]
        for case in client_patients[i]:
            for name in train_names:
                if case in name:
                    temp.append(name)
        client_patient_names.append(temp)

    for i in range(args.client_num):
        print(f">> {i}th training loader: {train_topfolder} with {folders}")
        # Create partial functions: Easier for readability later (see the difference between train and validation)
        gen_dataset = partial(SliceDataset,
                              transforms=trans,
                              are_hots=are_hots,
                              debug=debug,
                              K=n_class,
                              in_memory=in_memory,
                              dimensions=dimensions)
        if platform.system().lower() == 'windows':
            num_workers = 0
        elif platform.system().lower() == 'linux':
            num_workers= 4
        
        data_loader = partial(DataLoader,
                              num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=custom_collate)

        train_folders: List[Path] = [Path(data_folder, train_topfolder, f) for f in folders]
        # I assume all files have the same name inside their folder: makes things much easier     
        #train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*"))
        train_names = client_patient_names[i]        
        train_set = gen_dataset(train_names,
                                train_folders,
                                augment = True)
        if args.group_train:
            train_sampler = PatientSampler(train_set, args.grp_regex, shuffle=True)
            train_loader = data_loader(train_set,
                                       batch_sampler=train_sampler)
        else:
            train_loader = data_loader(train_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)
        train_loaders.append(train_loader)


    print(f">> Validation dataloader: {train_topfolder} with {folders}")
    val_folders: List[Path] = [Path(data_folder, args.validation_folder, f) for f in folders]
    val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*"))
    val_set = gen_dataset(val_names,
                          val_folders,
                          augment = False)
    val_sampler = PatientSampler(val_set, args.grp_regex, shuffle=False) if args.group else None
    val_batch_size = 1 if val_sampler else batch_size
    val_loader = data_loader(val_set,
                             batch_sampler=val_sampler,
                             batch_size=val_batch_size)
    return train_loaders, val_loader


class SliceDataset(Dataset):
    def __init__(self, filenames: List[str], folders: List[Path], are_hots: List[bool],
                 transforms: List[Callable], debug=False, quiet=False,
                 K=4, in_memory: bool = False,
                 augment = None, ignore_norm: bool = False,
                 dimensions: int = 2, debug_size: int = 10) -> None:
        self.folders: List[Path] = folders
        self.transforms: List[Callable[[Tuple, int], Callable[[D], Tensor]]] = transforms
        assert len(self.transforms) == len(self.folders)

        self.are_hots: List[bool] = are_hots
        self.filenames: List[str] = filenames
        self.debug = debug
        self.K: int = K  # Number of classes
        self.in_memory: bool = in_memory
        self.quiet: bool = quiet
        self.augment= augment
        self.ignore_norm: bool = ignore_norm
        self.dimensions: int = dimensions
        if self.debug:
            self.filenames = self.filenames[:debug_size]

        assert self.check_files()  # Make sure all file exists

        if not self.quiet:
            print(f">> Initializing {self.__class__.__name__} with {len(self.filenames)} images")
            if self.augment:
                print("> Will augment data online")

        # Load things in memory if needed
        self.files: List[List[F]] = SliceDataset.load_images(self.folders, self.filenames, self.in_memory)
        assert len(self.files) == len(self.folders)
        for files in self.files:
            assert len(files) == len(self.filenames)

    def check_files(self) -> bool:
        for folder in self.folders:
            if not Path(folder).exists():
                return False

            for f_n in self.filenames:
                if not Path(folder, f_n).exists():
                    return False

        return True

    @staticmethod
    def load_images(folders: List[Path], filenames: List[str], in_memory: bool, quiet=False) -> List[List[F]]:
        def load(folder: Path, filename: str) -> F:
            p: Path = Path(folder, filename)
            if in_memory:
                with open(p, 'rb') as data:
                    res = io.BytesIO(data.read())
                return res
            return p
        if in_memory and not quiet:
            print("> Loading the data in memory...")

        files: List[List[F]] = [[load(f, im) for im in filenames] for f in folders]

        return files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, Union[str,
                                                         Tensor,
                                                         List[Tensor],
                                                         List[Tuple[slice, ...]],
                                                         List[Tuple[Tensor, Tensor]]]]:
        filename: str = self.filenames[index]
        path_name: Path = Path(filename)
        images: List[D]

        if path_name.suffix == ".png":
            images = [Image.open(files[index]) for files in self.files]
        elif path_name.suffix == ".npy":
            images = [np.load(files[index]) for files in self.files]
        else:
            raise ValueError(filename)
            
        if self.augment:  
            images = augment(*images)
        else:
            images = images
            
        resolution: Tuple[float, ...]
        resolution = tuple([1] * self.dimensions)

        # Final transforms and assertions
        assert len(images) == len(self.folders) == len(self.transforms)
        final_tensors: List[Tensor] = [tr(resolution, self.K)(e) for (tr, e) in zip(self.transforms, images)]
        _, *img_shape = final_tensors[0].shape
        # main image is between 0 and 1
        if not self.ignore_norm:
            assert 0 <= final_tensors[0].min() and final_tensors[0].max() <= 1, \
                (final_tensors[0].min(), final_tensors[0].max())

        for ttensor in final_tensors[1:]:  # Things should be one-hot or at least have the shape
            assert ttensor.shape == (self.K, *img_shape), (ttensor.shape, self.K, *img_shape)

        for ttensor, is_hot in zip(final_tensors, self.are_hots):  # All masks (ground truths) are class encoded
            if is_hot:
                assert one_hot(ttensor, axis=0), torch.einsum("k...->...", ttensor)

        img, gt = final_tensors[:2]

        return {'filenames': filename,
                'images': final_tensors[0],
                'gt': final_tensors[1],
                'labels': final_tensors[2:],
                }

_use_shared_memory = True

def custom_collate(batch):
    """Collate function to handle dict from dataset Dict[str, Union[str, Tensor, List[Tensor], List[slice]]]"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # if torch.utils.data.get_worker_info() is not None:
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, str) or isinstance(elem, slice):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list):
        if len(elem) == 0:
            return batch

        if isinstance(elem[0], tuple):  # Handling for spacings
            return batch

        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(elem_type)


class PatientSampler(Sampler):
    def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False, quiet=False) -> None:
        filenames: List[str] = dataset.filenames
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        assert grp_regex is not None
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

        # print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(1) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(filenames)
        if not quiet:
            print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images ; regex: {self.grp_regex}")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        # print(self.idx_map)
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

        # print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)
