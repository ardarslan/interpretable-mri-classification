import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.utils.class_weight import compute_class_weight


def get_img_dataset(
    data_dir: str,
    logger: logging.Logger,
    train_transforms: transforms.transforms.Compose,
    test_transforms: transforms.transforms.Compose,
) -> Tuple[
    torch.utils.data.dataset.Subset,
    torch.utils.data.dataset.Subset,
    torch.utils.data.dataset.Subset,
]:
    # Initialize train/test sets
    data_path = Path(data_dir)
    train_dataset = ImageFolder(data_path, transform=train_transforms)
    test_dataset = ImageFolder(data_path, transform=test_transforms)
    classes = train_dataset.find_classes(data_path)[1]
    logger.info(
        "Loaded samples into dataset with label "
        + f"'no'={classes['no']} and 'yes'={classes['yes']}"
    )

    # Split dataset into train/test sets and stratify over labels to balance
    # datasets with set seed
    # DO NOT CHANGE THE SEED
    train_len = int(0.8 * len(train_dataset))
    test_len = int((len(train_dataset) - train_len) / 2)
    train_dataset = random_split(
        dataset=train_dataset,
        lengths=[train_len, test_len, test_len],
        generator=torch.Generator().manual_seed(390397),
    )[0]
    val_dataset, test_dataset = random_split(
        dataset=test_dataset,
        lengths=[train_len, test_len, test_len],
        generator=torch.Generator().manual_seed(390397),
    )[1:]

    return train_dataset, val_dataset, test_dataset


def get_img_data_loaders(
    cfg: dict,
    data_dir: str,
    logger: logging.Logger,
    train_transforms: transforms.transforms.Compose,
    test_transforms: transforms.transforms.Compose,
    shuffle_train: bool = True,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.Tensor,
]:
    def get_img_data_loaders_helper(
        dataset: torch.utils.data.dataset.Subset, shuffle: bool
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["num_workers"],
            pin_memory=False,
            drop_last=False,
        )

    train_dataset, val_dataset, test_dataset = get_img_dataset(
        data_dir=data_dir,
        logger=logger,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    # compute class weights
    train_labels = np.array([y for X, y in train_dataset])
    class_weights = torch.tensor(
        compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels,
        ),
        dtype=torch.float,
        device=cfg["device"],
    )

    train_dataloader = get_img_data_loaders_helper(
        dataset=train_dataset, shuffle=shuffle_train
    )
    val_dataloader = get_img_data_loaders_helper(dataset=val_dataset, shuffle=False)
    test_dataloader = get_img_data_loaders_helper(dataset=test_dataset, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, class_weights


def get_radiomics_dataset(data_dir: str):
    # Define relevant features
    rel_feat = [
        "diagnostics_Versions_PyRadiomics",
        "diagnostics_Versions_Numpy",
        "diagnostics_Versions_SimpleITK",
        "diagnostics_Versions_PyWavelet",
        "diagnostics_Versions_Python",
        "diagnostics_Configuration_Settings",
        "diagnostics_Configuration_EnabledImageTypes",
        "diagnostics_Image-original_Hash",
        "diagnostics_Image-original_Dimensionality",
        "diagnostics_Image-original_Spacing",
        "diagnostics_Image-original_Size",
        "diagnostics_Image-original_Mean",
        "diagnostics_Image-original_Minimum",
        "diagnostics_Image-original_Maximum",
        "diagnostics_Mask-original_Hash",
        "diagnostics_Mask-original_Spacing",
        "diagnostics_Mask-original_Size",
        "diagnostics_Mask-original_BoundingBox",
        "diagnostics_Mask-original_VoxelNum",
        "diagnostics_Mask-original_VolumeNum",
        "diagnostics_Mask-original_CenterOfMassIndex",
        "diagnostics_Mask-original_CenterOfMass",
    ]

    # Load train/test sets from csvs
    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    train_data.drop(inplace=True, axis=1, labels=rel_feat)
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_data = pd.read_csv(os.path.join(data_dir, "validation_data.csv"))
    val_data.drop(inplace=True, axis=1, labels=rel_feat)
    val_labels = np.load(os.path.join(data_dir, "validation_labels.npy"))
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    test_data.drop(inplace=True, axis=1, labels=rel_feat)
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    return (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
    )
