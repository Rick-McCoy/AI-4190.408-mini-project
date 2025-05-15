from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision import datasets  # For the special shuffle case test loader

# --- Important: Assuming your main script is named 'your_ai_project_script.py' ---
# --- and it's in the same directory or accessible in PYTHONPATH             ---
# --- You might need to adjust this import based on your file structure       ---
try:
    from main import (
        HOOK_REGISTRY,  # If you want to dynamically get hooks
        ExperimentHooks,
        InputPerturbHooks,
        LabelMappedDataset,  # For the special shuffle case
        LabelNoiseHooks,
        RandomLabelShuffleHooks,
        TrainConfig,
        build_dataloaders,  # Or at least the logic for creating test_loader
        get_model,
        get_predictions_and_labels,
    )
except ImportError as e:
    print(f"Error importing from your main project script: {e}")
    print(
        "Please ensure 'your_ai_project_script.py' (or its correct name) is accessible,"
    )
    print("and contains the necessary class/function definitions.")
    exit()

# --- Configuration Section ---

CIFAR10_DATA_ROOT = "./cifar"  # ! IMPORTANT: Set your CIFAR-10 data path
OUTPUT_FIGS_DIR = Path("./figs")
OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Define the specific run directories for your four key experiments
# These should be the paths to the folders containing 'best_model.pt'
# Update these paths to match your actual run folder names from aggregated_results.txt
EXPERIMENT_RUN_DIRS = {
    "baseline": Path("runs/baseline_20250516_024204_seed42_hookseedN_A"),
    "shuffle": Path("runs/shuffle_20250516_023627_seed42_hookseed0"),
    "noise": Path("runs/noise_20250516_024702_seed42_hookseed0"),
    "perturb": Path("runs/perturb_20250516_030103_seed42_hookseedN_A"),
}

# If seeds were different for each run, you might need to specify them here,
# especially for TrainConfig if it affects dataloader generation (e.g. num_workers)
# For hook instantiation (RandomLabelShuffle, LabelNoise), the hook_seed is important.
# We'll try to infer hook_seed from folder name or use a default.
DEFAULT_TRAIN_SEED = 42
DEFAULT_HOOK_SEED = 0  # Used by shuffle and noise if not specified otherwise

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Function to Plot ---
def plot_confusion_matrix_custom(y_true, y_pred, display_labels, title, output_path):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(display_labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(
        ax=ax,
        cmap=plt.get_cmap("Blues"),
        xticks_rotation="vertical",
        values_format="d",
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


# --- Main Processing Logic ---
def generate_cms():
    print(f"Using device: {DEVICE}")
    for exp_key, run_dir in EXPERIMENT_RUN_DIRS.items():
        print(f"\nProcessing experiment: {exp_key} from {run_dir}")

        if not run_dir.exists() or not (run_dir / "best_model.pt").exists():
            print(f"Skipping {exp_key}: Run directory or best_model.pt not found.")
            continue

        # Instantiate hooks
        # This is a simplified instantiation. If your hooks take more complex args based on config, adjust.
        hook_seed_for_run = DEFAULT_HOOK_SEED  # Default
        if exp_key == "shuffle":  # Assumes hookseed0 from your example folder name
            hooks_instance = RandomLabelShuffleHooks(seed=0)
            cm_title_suffix = "Shuffle (vs. Original Labels)"
        elif exp_key == "noise":  # Assumes hookseed0
            hooks_instance = LabelNoiseHooks(seed=0)  # noise_ratio default is 0.2
            cm_title_suffix = "Noise (20%)"
        elif exp_key == "perturb":
            hooks_instance = InputPerturbHooks()
            cm_title_suffix = "Input Perturbation"
        elif exp_key == "baseline":
            hooks_instance = ExperimentHooks()
            cm_title_suffix = "Baseline"
        else:
            print(f"Unknown experiment key: {exp_key}. Skipping.")
            continue

        # Load model
        model = get_model(num_classes=10)
        model.load_state_dict(
            torch.load(run_dir / "best_model.pt", map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()

        # Create a minimal TrainConfig. Some values might not be strictly necessary
        # for just building the test loader, but good to have.
        # The seed here is for dataloader's random_split if it were used for test,
        # but for test set it's usually not split. num_workers can be important.
        current_train_config = TrainConfig(
            data_root_dir=Path(CIFAR10_DATA_ROOT),
            output_dir=run_dir,  # Not really used here, but part of config
            seed=DEFAULT_TRAIN_SEED,  # Main training seed
            batch_size=128,  # Typical batch size for evaluation
            num_workers=4,  # Adjust as per your system
        )

        y_true_for_cm = []
        y_pred_for_cm = []

        if exp_key == "shuffle":
            # For SHUFFLE, predictions are from the model trained on shuffled labels,
            # but the CM ground truth should be ORIGINAL CIFAR-10 labels.

            # 1. Get predictions using the test loader as defined by RandomLabelShuffleHooks
            # This test_loader will have RANDOMLY SHUFFLED labels. We only care about its images.
            _tr_loader, _val_loader, test_loader_with_shuffled_labels = (
                build_dataloaders(current_train_config, hooks_instance)
            )
            # The model's predictions will be based on what it learned (fitting random noise)
            y_pred_for_cm, _ = get_predictions_and_labels(
                model, test_loader_with_shuffled_labels, DEVICE
            )

            # 2. Get original true labels using a "clean" test dataset
            # The images should undergo the *same test transform* as the shuffle hook defined,
            # but labels must be original.
            cifar_test_original_labels = datasets.CIFAR10(
                current_train_config.data_root_dir,
                train=False,
                download=True,  # Ensure it's downloaded
                transform=hooks_instance.test_transform(),  # Use the test transform defined by RandomLabelShuffleHooks
            )
            # Use an identity label map to get original labels
            identity_label_map = lambda _idx, original_label: original_label
            original_label_test_dataset = LabelMappedDataset(
                cifar_test_original_labels, identity_label_map
            )

            # Create a DataLoader just to iterate and get original labels in the same order
            # as test_loader_with_shuffled_labels (assuming dataset order is preserved by torchvision.datasets.CIFAR10)
            # Batch size doesn't really matter here, just for iterating.
            original_labels_loader = torch.utils.data.DataLoader(
                original_label_test_dataset,
                batch_size=current_train_config.batch_size,
                shuffle=False,  # CRITICAL: Must be false to match order
                num_workers=current_train_config.num_workers,
            )
            for _, labels_batch in original_labels_loader:
                y_true_for_cm.extend(labels_batch.tolist())

            # Sanity check length
            if len(y_true_for_cm) != len(y_pred_for_cm):
                print(
                    f"Warning for {exp_key}: Mismatch in lengths of true ({len(y_true_for_cm)}) and pred ({len(y_pred_for_cm)}) labels."
                )
                continue

        else:  # For baseline, noise, perturb
            # Build the standard test dataloader using the experiment's hooks
            # The labels from this loader are what the model was evaluated against for its metrics
            _tr_loader, _val_loader, test_loader = build_dataloaders(
                current_train_config, hooks_instance
            )
            y_pred_for_cm, y_true_for_cm = get_predictions_and_labels(
                model, test_loader, DEVICE
            )

        # Plot and save
        output_cm_path = OUTPUT_FIGS_DIR / f"cm_{exp_key}.png"
        plot_confusion_matrix_custom(
            y_true_for_cm,
            y_pred_for_cm,
            CIFAR10_CLASSES,
            cm_title_suffix,
            output_cm_path,
        )


if __name__ == "__main__":
    # Ensure your main script (e.g., your_ai_project_script.py) with all necessary
    # definitions (TrainConfig, ExperimentHooks, build_dataloaders, get_model, etc.)
    # is in the same directory or accessible via PYTHONPATH.
    # You might need to rename 'your_ai_project_script' in the import statement at the top.

    print("Starting confusion matrix generation...")
    generate_cms()
    print("\nConfusion matrix generation complete.")
    print(f"Plots saved in: {OUTPUT_FIGS_DIR.resolve()}")
