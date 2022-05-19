import os
import pprint
from argparse import ArgumentParser
import logging
import time

from typing import Dict, Any, Union, Tuple, List

import torch
import shap
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_img_data_loaders


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used in train, val and test data loaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers used in train, val and test data loaders.",
    )
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="Maximum number of epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-7, help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler_patience",
        type=int,
        default=20,
        help="Maximum number of epochs without an improvement on validation loss before the learning rate is reduced.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        type=bool,
        default=True,
        help="Whether to use a learning rate scheduler or not.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baseline_cnn",
        choices=["baseline_cnn", "resnet18_cnn", "resnext50_cnn"],
        help="Model name.",
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="Randomly flips input images horizontally.",
    )
    parser.add_argument(
        "--vertical_flip",
        action="store_true",
        help="Randomly flips input images vertically.",
    )
    parser.add_argument(
        "--rotate", action="store_true", help="Randomly rotates input images.",
    )
    parser.add_argument(
        "--perspective",
        action="store_true",
        help="Applies random perspective to input images.",
    )
    parser.add_argument(
        "--color_jitter", action="store_true", help="Randomly applies color jittering.",
    )
    parser.add_argument(
        "--random_crop", action="store_true", help="Randomly crops input images.",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to normalize input images.",
    )
    return parser


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def get_normalization_mean_and_std() -> Tuple[List[float], List[float]]:
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def unnormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
        Input: Shape: (N, H, W, C). Normalized images. Before normalization, pixel values were between [0, 1].
        Output: Shape: (N, H, W, C). Images with pixel values between [0, 1].
    """

    mean, std = get_normalization_mean_and_std()
    mean = torch.Tensor(mean).unsqueeze(-1)
    std = torch.Tensor(std).unsqueeze(-1)
    images = images.permute(3, 0, 1, 2)  # (C, N, H, W)
    images = (images.reshape(3, -1) * std + mean).reshape(images.shape)  # (C, N, H, W)
    images = images.permute(1, 2, 3, 0)  # (N, H, W, C)
    images = images.clamp(0, 1)
    return images


def train_epoch(
    cfg: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    all_y = []
    all_yhat = []
    for batch in train_data_loader:
        optimizer.zero_grad()
        X, y = batch
        X = X.float().to(cfg["device"])
        y = y.to(cfg["device"])
        all_y.append(y)
        yhat = model(X)
        all_yhat.append(yhat)
        loss = nn.CrossEntropyLoss(weight=class_weights)(input=yhat, target=y)
        loss.backward()
        optimizer.step()
    all_y = torch.hstack(all_y)
    all_yhat = torch.vstack(all_yhat)

    train_loss_dict = evaluate_predictions(
        cfg=cfg,
        all_y=all_y,
        all_yhat=all_yhat,
        class_weights=class_weights,
        split="train",
        save_to_disk=False,
    )
    return train_loss_dict


def evaluation_epoch(
    cfg: Dict[str, Any],
    model: nn.Module,
    evaluation_data_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    split: str,
    save_to_disk: bool = False,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        all_y = []
        all_yhat = []
        for batch in evaluation_data_loader:
            X, y = batch
            X = X.float().to(cfg["device"])
            y = y.to(cfg["device"])
            yhat = model(X)
            all_y.append(y)
            all_yhat.append(yhat)

        all_y = torch.hstack(all_y)
        all_yhat = torch.vstack(all_yhat)

        eval_loss_dict = evaluate_predictions(
            cfg=cfg,
            all_y=all_y,
            all_yhat=all_yhat,
            class_weights=class_weights,
            split=split,
            save_to_disk=save_to_disk,
        )
    return eval_loss_dict


def evaluate_predictions(
    cfg: Dict[str, Any],
    all_y: torch.Tensor,
    all_yhat: torch.Tensor,
    class_weights: torch.Tensor,
    split: str,
    save_to_disk: bool,
) -> Dict[str, float]:
    result_dict = {}
    all_yhat_probs = nn.Softmax(dim=-1)(all_yhat)
    all_yhat_argmaxed = torch.argmax(all_yhat_probs, dim=1)
    result_dict["loss"] = float(
        nn.CrossEntropyLoss(weight=class_weights)(input=all_yhat, target=all_y)
    )
    result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
    result_dict["balanced_acc_score"] = balanced_accuracy_score(
        all_y, all_yhat_argmaxed
    )
    result_dict["roc_auc_score"] = roc_auc_score(
        all_y, to_numpy(all_yhat_probs[:, 1]).ravel()
    )
    result_dict["pr_auc_score"] = average_precision_score(
        all_y, to_numpy(all_yhat_probs[:, 1]).ravel()
    )
    if save_to_disk:
        save_predictions_to_disk(
            cfg=cfg,
            all_y=all_y,
            all_yhat=all_yhat,
            result_dict=result_dict,
            split=split,
        )
    return result_dict


def get_checkpoints_dir(cfg: Dict[str, Any]) -> str:
    model_name = cfg["model_name"]
    checkpoints_dir = os.path.join(
        cfg["checkpoints"], "task2", cfg["experiment_time"] + "_" + cfg["id"]
    )
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


def save_predictions_to_disk(
    cfg: Dict[str, Any],
    all_y: torch.Tensor,
    all_yhat: torch.Tensor,
    result_dict: Dict[str, float],
    split: str,
) -> None:
    checkpoints_dir = get_checkpoints_dir(cfg)
    predictions_path = os.path.join(checkpoints_dir, f"{split}_predictions.txt")

    all_yhat_probs = nn.Softmax(dim=-1)(all_yhat)
    columns = ["prob_0", "prob_1", "label"]
    df = pd.DataFrame(
        to_numpy(torch.hstack([all_yhat_probs, all_y.view(-1, 1)])), columns=columns,
    )
    df.to_csv(predictions_path, index=False)

    scores_path = os.path.join(checkpoints_dir, f"{split}_scores.txt")
    with open(scores_path, "w") as file_writer:
        for key, value in result_dict.items():
            file_writer.write(f"{key}: {np.round(value, 3)}\n")


class BaselineCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            # nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Inputs:
                x: Input images. A torch.Tensor with shape (cfg["batch_size"], 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            Output:
                yhat: Logits. A torch.Tensor with shape (cfg["batch_size"], 2)
        """
        return self.model(x)


class ResNetCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["model_name"] == "resnet18_cnn":
            self.model = models.resnet18(pretrained=True)
        elif cfg["model_name"] == "resnext50_cnn":
            self.model = models.resnext50_32x4d(pretrained=True)
        else:
            raise Exception(f"Not a valid ResNetCNN model_name {model_name}.")

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.cfg["num_classes"])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Inputs:
                x: Input images. A torch.Tensor with shape (cfg["batch_size"], 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            Output:
                yhat: Logits. A torch.Tensor with shape (cfg["batch_size"], 2)
        """
        return self.model(x)


def get_model(cfg: Dict[str, Any]) -> nn.Module:
    if cfg["model_name"] == "baseline_cnn":
        return BaselineCNN(cfg).to(cfg["device"])
    elif cfg["model_name"] in ["resnet18_cnn", "resnext50_cnn"]:
        return ResNetCNN(cfg).to(cfg["device"])
    else:
        raise Exception(f"Not a valid model {model}.")


def get_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )


def get_scheduler(
    cfg: Dict[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["lr_scheduler_patience"], verbose=True
    )


def save_checkpoint(
    cfg: Dict[str, Any], model: nn.Module, logger: logging.Logger
) -> None:
    logger.info("Saving the best checkpoint...", cfg)
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = {"model_state_dict": model.state_dict()}
    torch.save(checkpoint_dict, os.path.join(checkpoints_dir, "best_checkpoint"))


def load_checkpoint(cfg: Dict[str, Any], logger: logging.Logger) -> nn.Module:
    logger.info("Loading the best checkpoint...", cfg)
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = torch.load(os.path.join(checkpoints_dir, "best_checkpoint"))
    model = get_model(cfg)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return model


def get_train_transforms(cfg: Dict[str, Any]) -> transforms.transforms.Compose:
    name_transform_mapping = {
        "horizontal_flip": transforms.RandomHorizontalFlip(p=0.5),
        "vertical_flip": transforms.RandomVerticalFlip(p=0.5),
        "perspective": transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
        "color_jitter": transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4
        ),
        "rotate": transforms.RandomRotation(degrees=30),
    }
    train_transforms = [transforms.ToTensor()]
    for name, transform in name_transform_mapping.items():
        if cfg[name]:
            train_transforms.append(transform)
    if cfg["random_crop"]:
        train_transforms.extend(
            [
                transforms.Resize(
                    int(cfg["image_size"] * 1.10)
                ),  # resize shortest side to 140 pixels
                transforms.CenterCrop(
                    int(cfg["image_size"] * 1.10)
                ),  # crop longest side to 140 pixels at center
                transforms.RandomCrop(cfg["image_size"]),
            ]
        )
    else:
        train_transforms.extend(
            [
                transforms.Resize(cfg["image_size"]),
                transforms.CenterCrop(cfg["image_size"]),
            ]
        )
    if cfg["normalize"]:
        mean, std = get_normalization_mean_and_std()
        train_transforms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(train_transforms)


def get_test_transforms(cfg: Dict[str, Any]) -> transforms.transforms.Compose:
    test_transforms = [transforms.ToTensor()]
    test_transforms.extend(
        [
            transforms.Resize(cfg["image_size"]),
            transforms.CenterCrop(cfg["image_size"]),
        ]
    )
    if cfg["normalize"]:
        mean, std = get_normalization_mean_and_std()
        test_transforms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(test_transforms)


def save_shap_feature_attributions(
    cfg: Dict[str, Any],
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
) -> None:

    train_images = torch.vstack([batch[0] for batch in train_data_loader])
    test_images_and_labels = next(iter(test_data_loader))
    test_images = test_images_and_labels[0][:10]
    test_labels = test_images_and_labels[1][:10]

    e = shap.DeepExplainer(model, train_images)
    shap_values = e.shap_values(test_images)

    if cfg["normalize"]:
        test_images = unnormalize_images(test_images)  # for visualization

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    plt.figure(figsize=(20, 20))
    shap.image_plot(shap_numpy, test_numpy)

    checkpoints_dir = get_checkpoints_dir(cfg)
    shap_feature_attributions_path = os.path.join(
        checkpoints_dir, "shap_feature_attributions.png"
    )
    plt.savefig(shap_feature_attributions_path)

    shap_labels_path = os.path.join(checkpoints_dir, "shap_labels.txt")
    with open(shap_labels_path, "w") as file_writer:
        for label in test_labels:
            file_writer.write(f"{int(label)}\n")


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["experiment_time"] = str(int(time.time()))

    model_name = cfg["model_name"]
    if model_name == "baseline_cnn":
        cfg["image_size"] = 128
    elif model_name in ["resnet18_cnn", "resnext50_cnn"]:
        cfg["image_size"] = 224
    else:
        raise Exception(f"Not a valid model_name {model_name}.")

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    checkpoints_dir = get_checkpoints_dir(cfg)
    log_file_path = os.path.join(checkpoints_dir, "logs.txt")

    logger = init_logger(path_to_log=log_file_path, level=logging.INFO)
    logger.info(pprint.pformat(cfg, indent=4))

    logger.info("Loading MRI image dataset...")
    data_dir = os.path.join(cfg["data"], "images")

    train_transforms = get_train_transforms(cfg=cfg)
    test_transforms = get_test_transforms(cfg=cfg)

    (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        class_weights,
    ) = get_img_data_loaders(
        cfg=cfg,
        data_dir=data_dir,
        logger=logger,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        shuffle_train=True,
    )
    logger.info("MRI image dataset loaded!")

    model = get_model(cfg=cfg)
    optimizer = get_optimizer(cfg=cfg, model=model)

    if cfg["use_lr_scheduler"]:
        scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    best_val_loss = np.inf
    for epoch in range(cfg["max_epochs"]):
        train_loss_dict = train_epoch(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            class_weights=class_weights,
        )
        logger.info(
            f"Train | Epoch: {epoch+1}, "
            + ", ".join(
                [
                    f"{loss_function}: {np.round(loss_value, 3)}"
                    for loss_function, loss_value in train_loss_dict.items()
                ]
            )
        )

        val_loss_dict = evaluation_epoch(
            cfg=cfg,
            model=model,
            evaluation_data_loader=val_data_loader,
            class_weights=class_weights,
            split="val",
            save_to_disk=False,
        )
        current_val_loss = val_loss_dict["loss"]

        logger.info(
            f"Validation | Epoch: {epoch+1}, "
            + ", ".join(
                [
                    f"{loss_function}: {np.round(loss_value, 3)}"
                    for loss_function, loss_value in val_loss_dict.items()
                ]
            )
        )

        if cfg["use_lr_scheduler"]:
            scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            save_checkpoint(cfg=cfg, model=model, logger=logger)
            best_val_loss = current_val_loss

    model = load_checkpoint(cfg=cfg, logger=logger)

    if (
        cfg["model_name"] != "resnext50_cnn"
    ):  # Shap gives an error for resnext architecture.
        save_shap_feature_attributions(
            cfg=cfg,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            model=model,
        )

    (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        class_weights,
    ) = get_img_data_loaders(
        cfg=cfg,
        data_dir=data_dir,
        logger=logger,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        shuffle_train=False,
    )

    evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=train_data_loader,
        class_weights=class_weights,
        split="train",
        save_to_disk=True,
    )
    evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=val_data_loader,
        class_weights=class_weights,
        split="val",
        save_to_disk=True,
    )
    test_loss_dict = evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=test_data_loader,
        class_weights=class_weights,
        split="test",
        save_to_disk=True,
    )

    logger.info(
        f"Test | "
        + ", ".join(
            [
                f"{loss_function}: {np.round(loss_value, 3)}"
                for loss_function, loss_value in test_loss_dict.items()
            ]
        )
    )


if "__main__" == __name__:
    main()
