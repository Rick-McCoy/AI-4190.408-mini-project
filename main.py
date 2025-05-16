#!/usr/bin/env python3
"""CIFAR‑10 classifier – experiment‑ready (v5 - Corrected Shuffle Logic)

Features
--------
* Baseline: **no augmentation** (by default for ExperimentHooks)
* Hook mechanism for label/input perturbations
* Nested **tqdm** progress bars
* TensorBoard → loss, top‑1 & top‑5 accuracy (train/val)
* Matplotlib plots: train/val curves + final test point
* Metrics JSON: test loss/Acc@1/Acc@5, Cohen κ
* DEBUG logs in `logs/debug.log` (INFO still on stdout)
* Explicit train/val/test transforms via hooks.
* Corrected evaluation for Random Label Shuffle to report against original labels.
* Corrected Random Label Shuffle to assign truly random labels per sample.
* Improved variable names for readability.
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, cohen_kappa_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Subset as TorchSubset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

################################################################################
# Logging & reproducibility
################################################################################


def setup_logging(output_path: Path):
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    file_handler = logging.FileHandler(log_dir / "debug.log", "w", "utf‑8")
    file_handler.setFormatter(logging.Formatter(fmt))
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt))
    stream_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def set_seed(seed: int):
    logging.debug("Setting global seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


################################################################################
# Config & hooks
################################################################################


@dataclass
class TrainConfig:
    data_root_dir: Path
    output_dir: Path
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    validation_split_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 16
    warmup_epochs: int = 5


class ExperimentHooks:
    """Base hook – override methods for different experiments."""

    def train_transform(self):
        # CIFAR-10 normalization parameters
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def val_transform(self):
        # By default, validation transform is the same as test transform
        return self.test_transform()

    def test_transform(self):
        # By default, test transform is the same as train transform (basic normalization)
        return self.train_transform()

    def label_map(self, index: int, original_label: int) -> int:
        # By default, labels are unchanged
        return original_label


class RandomLabelShuffleHooks(ExperimentHooks):
    """Section‑2: Assign a completely random class label to each sample."""

    def __init__(
        self, num_classes=10, seed=0
    ):  # seed makes the "random" assignment reproducible for a given run
        super().__init__()
        self.num_classes = num_classes
        self.rng = random.Random(
            seed
        )  # Use an internal RNG for this specific hook's behavior
        logging.debug(
            "RandomLabelShuffleHooks: Labels will be completely randomized per sample using internal hook seed %d.",
            seed,
        )

    def label_map(self, index: int, original_label: int) -> int:
        # Assign a label uniformly at random from all available classes,
        # ignoring the original_label and index.
        return self.rng.randrange(self.num_classes)


class LabelNoiseHooks(ExperimentHooks):
    """Section‑3: flip each label with probability *p* (defaults to 0.2)."""

    def __init__(self, noise_ratio=0.2, num_classes=10, seed=0):
        super().__init__()
        self.noise_probability = noise_ratio
        self.num_classes = num_classes
        self.rng = random.Random(
            seed
        )  # Use an internal RNG for this specific hook's behavior
        logging.debug(
            "LabelNoiseHooks: Approx %.1f%% of labels will be flipped. Internal hook seed %d.",
            noise_ratio * 100,
            seed,
        )

    def label_map(self, index: int, original_label: int) -> int:
        if self.rng.random() < self.noise_probability:
            # Generate a new label different from the original
            offset = self.rng.randrange(
                1, self.num_classes
            )  # Offset from 1 to num_classes-1
            noisy_label = (original_label + offset) % self.num_classes
            return noisy_label
        return original_label


class InputPerturbHooks(ExperimentHooks):
    """Section‑4: heavy input distortions for training and validation. Test set has milder distortions.
    Transforms are now in the correct order, with RandomErasing applied to a normalized tensor.
    """

    def __init__(self):
        super().__init__()
        # CIFAR-10 normalization parameters
        self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        logging.debug(
            "InputPerturbHooks: Applying strong train/val augmentations and mild test augmentations "
            "with correct transform order."
        )

    def train_transform(self):
        # Standard order of operations:
        # 1. Augmentations on PIL Images.
        # 2. Convert to Tensor.
        # 3. Normalize the Tensor.
        # 4. Tensor-based augmentations like RandomErasing (operates on the normalized tensor).
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),  # PIL in, PIL out
                transforms.RandomHorizontalFlip(),  # PIL in, PIL out
                transforms.GaussianBlur(
                    kernel_size=3, sigma=(0.1, 2.0)
                ),  # PIL in, PIL out
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),  # PIL in, PIL out
                transforms.ToTensor(),  # PIL in, Tensor out (scaled to [0,1])
                transforms.Normalize(
                    self.mean, self.std
                ),  # Tensor in, Normalized Tensor out
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.2)
                ),  # Normalized Tensor in, Normalized Tensor out
            ]
        )

    def val_transform(self):
        # As per original intent: "heavy input distortions (train & val)"
        # This will use the same correct transform order as train_transform().
        return self.train_transform()

    def test_transform(self):
        # Evaluate under a specific, milder blur only – simulate covariate shift
        # This transform order is also standard and correct.
        return transforms.Compose(
            [
                transforms.GaussianBlur(kernel_size=3, sigma=1.0),  # PIL in, PIL out
                transforms.ToTensor(),  # PIL in, Tensor out
                transforms.Normalize(
                    self.mean, self.std
                ),  # Tensor in, Normalized Tensor out
            ]
        )


HOOK_REGISTRY: dict[str, Callable[[], ExperimentHooks]] = {
    "baseline": ExperimentHooks,
    "shuffle": RandomLabelShuffleHooks,
    "noise": LabelNoiseHooks,
    "perturb": InputPerturbHooks,
}


################################################################################
# Dataset wrapper
################################################################################


class LabelMappedDataset(Dataset):
    """A Dataset wrapper that applies a mapping function to labels."""

    def __init__(
        self, base_dataset: Dataset, label_mapping_function: Callable[[int, int], int]
    ):
        self.base_dataset = base_dataset
        self.label_mapping_function = label_mapping_function

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, original_label = self.base_dataset[index]
        # The label_mapping_function receives the item's original index (within the base_dataset)
        # and its original label.
        modified_label = self.label_mapping_function(index, original_label)
        return image, modified_label


class _TransformedSubsetDataset(Dataset):
    """A helper Dataset to apply a transform to a subset of another dataset."""

    def __init__(self, subset: TorchSubset, transform_fn: Callable[[Any], Any] | None):
        self.subset = subset
        self.transform_fn = transform_fn

    def __getitem__(self, index: int):
        # self.subset[index] gets data from original CIFAR10 (untransformed PILImage) via Subset
        image_pil, label = self.subset[index]
        if self.transform_fn:
            image_tensor = self.transform_fn(image_pil)
            return image_tensor, label
        return (
            image_pil,
            label,
        )  # Should ideally not happen if transforms are always applied

    def __len__(self):
        return len(self.subset)


def build_dataloaders(config: TrainConfig, hooks: ExperimentHooks):
    # 1. Load base CIFAR-10 training data (as PIL Images, without any specific transforms yet)
    cifar_train_val_untransformed = datasets.CIFAR10(
        config.data_root_dir,
        train=True,
        download=True,
        transform=None,  # Load as PIL
    )

    # 2. Split indices for training and validation sets
    num_train_val_samples = len(cifar_train_val_untransformed)
    num_val_samples = int(num_train_val_samples * config.validation_split_ratio)
    num_train_samples = num_train_val_samples - num_val_samples

    # Ensure reproducible splits
    generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices = random_split(
        range(num_train_val_samples),
        [num_train_samples, num_val_samples],
        generator=generator,
    )

    # 3. Create Subset instances from the untransformed base dataset
    train_subset_untransformed = TorchSubset(
        cifar_train_val_untransformed, train_indices
    )
    val_subset_untransformed = TorchSubset(cifar_train_val_untransformed, val_indices)

    # 4. Apply respective transforms via _TransformedSubsetDataset
    train_dataset_transformed = _TransformedSubsetDataset(
        train_subset_untransformed, hooks.train_transform()
    )
    val_dataset_transformed = _TransformedSubsetDataset(
        val_subset_untransformed, hooks.val_transform()
    )

    # 5. Load test dataset and apply its specific transform
    # Note: test_transform is applied directly by torchvision.datasets.CIFAR10 here
    test_dataset_transformed = datasets.CIFAR10(
        config.data_root_dir,
        train=False,
        download=True,
        transform=hooks.test_transform(),
    )
    # For the special case of RandomLabelShuffle reporting, we need original test labels later
    # So, also load a version of test set with original labels but standard test transforms.
    # This is handled in run_experiment for clarity.

    # 6. Wrap datasets with label mapping and create DataLoaders
    common_dataloader_kwargs = dict(
        batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True
    )

    train_loader = DataLoader(
        LabelMappedDataset(train_dataset_transformed, hooks.label_map),
        shuffle=True,
        **common_dataloader_kwargs,
    )
    validation_loader = DataLoader(
        LabelMappedDataset(val_dataset_transformed, hooks.label_map),
        shuffle=False,
        **common_dataloader_kwargs,
    )
    test_loader = DataLoader(
        LabelMappedDataset(test_dataset_transformed, hooks.label_map),
        shuffle=False,
        **common_dataloader_kwargs,
    )

    return train_loader, validation_loader, test_loader


################################################################################
# Utilities
################################################################################


def get_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    logging.debug(
        "Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6
    )
    return model


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    epoch_total_loss = 0.0
    epoch_correct_top1 = 0
    epoch_correct_top5 = 0
    num_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        epoch_total_loss += loss.item() * batch_size

        _, predicted_top1 = outputs.max(1)
        epoch_correct_top1 += predicted_top1.eq(labels).sum().item()

        _, predicted_top5 = outputs.topk(5, 1, True, True)
        labels_expanded = labels.view(-1, 1).expand_as(predicted_top5)
        epoch_correct_top5 += predicted_top5.eq(labels_expanded).any(1).sum().item()

        num_samples += batch_size

    avg_loss = epoch_total_loss / num_samples if num_samples > 0 else 0.0
    avg_acc1 = epoch_correct_top1 / num_samples if num_samples > 0 else 0.0
    avg_acc5 = epoch_correct_top5 / num_samples if num_samples > 0 else 0.0
    return avg_loss, avg_acc1, avg_acc5


def get_predictions_and_labels(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    all_predictions: list[int] = []
    all_ground_truths: list[int] = []
    for images, labels in dataloader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        all_predictions.extend(outputs.argmax(1).cpu().tolist())
        all_ground_truths.extend(
            labels.tolist()
        )  # labels are already on CPU from DataLoader
    return all_predictions, all_ground_truths


def plot_metric_curves(
    output_path: Path,
    epoch_numbers: list[int],
    train_metric_values: list[float],
    val_metric_values: list[float],
    test_metric_value: float,
    title: str,
    y_label: str,
):
    fig, ax = plt.subplots()
    ax.plot(epoch_numbers, train_metric_values, label="Train")
    ax.plot(epoch_numbers, val_metric_values, label="Validation")
    if epoch_numbers:  # Only plot test point if there are epochs
        ax.scatter(
            [epoch_numbers[-1]],
            [test_metric_value],
            color="red",
            marker="*",
            s=100,
            label="Test",
            zorder=5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


################################################################################
# Runner
################################################################################


def run_experiment(config: TrainConfig, hooks: ExperimentHooks, experiment_name: str):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.output_dir)
    set_seed(config.seed)  # Sets global seed for torch, numpy, random

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info("Using device: %s", device)

    train_loader, validation_loader, test_loader = build_dataloaders(config, hooks)

    model = get_model(num_classes=10).to(device)  # CIFAR-10 has 10 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    warmup_scheduler = None
    if config.warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch_idx: (epoch_idx + 1) / config.warmup_epochs,
        )

    # Cosine scheduler starts after warmup (if any)
    cosine_scheduler_total_epochs = config.epochs - config.warmup_epochs
    # Ensure T_max is at least 1 for CosineAnnealingLR
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, cosine_scheduler_total_epochs)
    )

    summary_writer = SummaryWriter(log_dir=config.output_dir.as_posix())
    history_records = []
    best_val_accuracy = 0.0

    logging.info(f"Starting experiment: {experiment_name}")
    for epoch_idx in tqdm(range(config.epochs), desc="Epochs"):
        model.train()
        current_epoch_sum_loss = 0.0
        current_epoch_sum_correct_top1 = 0
        current_epoch_sum_correct_top5 = 0
        num_train_samples_processed = 0

        batch_pbar_desc = f"Epoch {epoch_idx + 1}/{config.epochs} Training"
        for images, labels in tqdm(train_loader, desc=batch_pbar_desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            current_epoch_sum_loss += loss.item() * batch_size

            _, predicted_top1 = outputs.max(1)
            current_epoch_sum_correct_top1 += predicted_top1.eq(labels).sum().item()

            _, predicted_top5 = outputs.topk(5, 1, True, True)
            labels_expanded = labels.view(-1, 1).expand_as(predicted_top5)
            current_epoch_sum_correct_top5 += (
                predicted_top5.eq(labels_expanded).any(1).sum().item()
            )

            num_train_samples_processed += batch_size

        train_loss = (
            current_epoch_sum_loss / num_train_samples_processed
            if num_train_samples_processed > 0
            else 0.0
        )
        train_acc1 = (
            current_epoch_sum_correct_top1 / num_train_samples_processed
            if num_train_samples_processed > 0
            else 0.0
        )
        train_acc5 = (
            current_epoch_sum_correct_top5 / num_train_samples_processed
            if num_train_samples_processed > 0
            else 0.0
        )

        val_loss, val_acc1, val_acc5 = evaluate(
            model, validation_loader, criterion, device
        )

        history_records.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc1": train_acc1,
                "val_acc1": val_acc1,
                "train_acc5": train_acc5,
                "val_acc5": val_acc5,
            }
        )
        summary_writer.add_scalars(
            "Loss", {"train": train_loss, "validation": val_loss}, epoch_idx
        )
        summary_writer.add_scalars(
            "Accuracy@1", {"train": train_acc1, "validation": val_acc1}, epoch_idx
        )
        summary_writer.add_scalars(
            "Accuracy@5", {"train": train_acc5, "validation": val_acc5}, epoch_idx
        )

        # Learning rate scheduling
        if warmup_scheduler and epoch_idx < config.warmup_epochs:
            warmup_scheduler.step()
        elif cosine_scheduler_total_epochs > 0:  # Apply cosine annealing after warmup
            cosine_scheduler.step()

        summary_writer.add_scalar(
            "LearningRate", optimizer.param_groups[0]["lr"], epoch_idx
        )

        if val_acc1 > best_val_accuracy:
            best_val_accuracy = val_acc1
            torch.save(model.state_dict(), config.output_dir / "best_model.pt")
            logging.debug(
                "Epoch %d: New best validation Acc@1: %.4f",
                epoch_idx + 1,
                best_val_accuracy,
            )

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(config.output_dir / "best_model.pt"))

    # Evaluate on the test set (labels used here are as per the test_loader, potentially modified by hooks)
    test_loss, test_acc1, test_acc5 = evaluate(model, test_loader, criterion, device)
    logging.info(
        "TEST (on test_loader's labels): Loss %.4f, Acc@1 %.4f, Acc@5 %.4f",
        test_loss,
        test_acc1,
        test_acc5,
    )

    # For classification report and Cohen Kappa:
    # If RandomLabelShuffle, use original labels for test set to assess true learning.
    # Otherwise, use the (potentially modified) labels from the test_loader.
    if isinstance(hooks, RandomLabelShuffleHooks):
        logging.info(
            "For RandomLabelShuffle experiment, generating classification report "
            "and Cohen Kappa against ORIGINAL CIFAR-10 test labels."
        )
        # Create a test dataset with original labels but using the hook's test_transform for consistency in input.
        # The images are transformed as they would be for the 'test_loader', but labels are original.
        cifar_test_original_labels_transformed = datasets.CIFAR10(
            config.data_root_dir,
            train=False,
            download=False,  # Assumed downloaded
            transform=hooks.test_transform(),  # Apply the same test image transform
        )
        # Use an identity label map to ensure original labels are used.
        identity_label_map = lambda _idx, original_label: original_label
        report_test_loader = DataLoader(
            LabelMappedDataset(
                cifar_test_original_labels_transformed, identity_label_map
            ),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        final_predictions, final_true_labels_for_report = get_predictions_and_labels(
            model, report_test_loader, device
        )
    else:
        # For other hooks, the report is against the labels provided by the standard test_loader
        # (which might have been modified by LabelNoiseHooks, for example).
        final_predictions, final_true_labels_for_report = get_predictions_and_labels(
            model, test_loader, device
        )

    epoch_sequence = [h["epoch"] for h in history_records]
    plot_metric_curves(
        config.output_dir / "loss_curves.png",
        epoch_sequence,
        [h["train_loss"] for h in history_records],
        [h["val_loss"] for h in history_records],
        test_loss,
        "Cross‑Entropy Loss (on loader labels)",
        "Loss",
    )
    plot_metric_curves(
        config.output_dir / "acc1_curves.png",
        epoch_sequence,
        [h["train_acc1"] for h in history_records],
        [h["val_acc1"] for h in history_records],
        test_acc1,
        "Top‑1 Accuracy (on loader labels)",
        "Accuracy@1",
    )
    plot_metric_curves(
        config.output_dir / "acc5_curves.png",
        epoch_sequence,
        [h["train_acc5"] for h in history_records],
        [h["val_acc5"] for h in history_records],
        test_acc5,
        "Top‑5 Accuracy (on loader labels)",
        "Accuracy@5",
    )

    report_path = config.output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                final_true_labels_for_report,
                final_predictions,
                digits=4,
                zero_division=0,
            )
        )
    logging.info(f"Classification report saved to {report_path}")

    metrics_summary = {
        "test_loss_on_loader_labels": test_loss,
        "test_acc1_on_loader_labels": test_acc1,
        "test_acc5_on_loader_labels": test_acc5,
        "cohen_kappa_on_report_labels": cohen_kappa_score(
            final_true_labels_for_report, final_predictions
        ),
    }
    metrics_path = config.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logging.info(f"Metrics summary saved to {metrics_path}")

    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)
    logging.info(f"Training history saved to {history_path}")

    summary_writer.close()
    logging.info(
        f"Experiment {experiment_name} finished. Results in {config.output_dir}"
    )


################################################################################
# CLI helper
################################################################################


def main():
    parser = argparse.ArgumentParser(description="CIFAR‑10 Experimentation Script")
    parser.add_argument(
        "--data_path", type=Path, required=True, help="Root directory for CIFAR‑10 data"
    )
    parser.add_argument(
        "--output_path_base",
        type=Path,  # Renamed for clarity
        default=Path("runs"),
        help="Base directory to save experiment outputs (default: runs/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for DataLoaders (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--hook",
        choices=HOOK_REGISTRY.keys(),
        default="baseline",
        help="Experiment variant to run (default: baseline)",
    )
    parser.add_argument(
        "--hook_seed",
        type=int,
        default=0,
        help="Seed for hook-specific RNG (e.g., for label shuffling/noise). Default: 0",
    )
    args = parser.parse_args()

    # Instantiate the selected hook, potentially passing the hook_seed
    selected_hook_constructor = HOOK_REGISTRY[args.hook]
    hook_kwargs = {}
    if args.hook in ["shuffle", "noise"]:  # Hooks that accept seed and num_classes
        hook_kwargs["seed"] = args.hook_seed
        # num_classes is defaulted in hook constructors
    experiment_hooks_instance = selected_hook_constructor(**hook_kwargs)

    # Construct a unique output directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_run_name = f"{args.hook}_{timestamp}_seed{args.seed}_hookseed{args.hook_seed if args.hook in ['shuffle', 'noise'] else 'N_A'}"
    output_dir_for_run = args.output_path_base / experiment_run_name

    train_config = TrainConfig(
        data_root_dir=args.data_path,
        output_dir=output_dir_for_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        # Other parameters like lr, momentum, etc., use defaults from TrainConfig
    )

    run_experiment(train_config, experiment_hooks_instance, experiment_name=args.hook)


if __name__ == "__main__":
    main()
