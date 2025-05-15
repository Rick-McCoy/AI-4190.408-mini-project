from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

# --- Ensure ExperimentHooks and InputPerturbHooks are defined or imported ---
# Option 1: Define them directly in this script if they are simple enough
# Option 2: Import from your main project script (recommended for consistency)
try:
    # Assuming your main script is 'your_ai_project_script.py'
    from your_ai_project_script import ExperimentHooks, InputPerturbHooks

    print("Successfully imported hooks from your_ai_project_script.py")
except ImportError:
    print("Could not import hooks. Defining them locally for this script.")

    # Fallback: Define necessary hook classes locally if import fails
    # (This should match exactly what's in your main script)
    class ExperimentHooks:
        def __init__(self):  # Add __init__ if your hook constructor needs it
            pass

        def train_transform(self):
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

    class InputPerturbHooks(ExperimentHooks):
        def __init__(self):
            super().__init__()
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # logging.debug("InputPerturbHooks: Applying strong train/val augmentations...") # Can be noisy here

        def train_transform(self):
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
                ]
            )
# --- End of Hook Definitions/Import ---


# --- Configuration ---
CIFAR10_DATA_ROOT = "./cifar"  # ! IMPORTANT: Set your CIFAR-10 data path
OUTPUT_FIGS_DIR = Path("./figs")  # Output directory for LaTeX paper
OUTPUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)
NUM_EXAMPLES = 4  # Number of example images to show (e.g., 4 for a 2x4 or 4x2 grid)


# --- Helper function to unnormalize and show an image ---
def imshow_tensor(tensor_img, ax, title=None):
    """Unnormalizes and displays a PyTorch tensor image."""
    # CIFAR-10 mean and std (must match what was used in Normalize)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    img = tensor_img.cpu().numpy().transpose((1, 2, 0))  # C, H, W -> H, W, C
    img = std * img + mean  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip values to be between 0 and 1

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis("off")


def generate_perturbed_examples():
    print(f"Loading CIFAR-10 data from: {CIFAR10_DATA_ROOT}")
    # Load a few original CIFAR-10 training images (as PIL Images)
    original_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR10_DATA_ROOT,
        train=True,
        download=True,
        transform=None,  # Load as PIL to show original before transforms
    )

    if len(original_dataset) < NUM_EXAMPLES:
        print(f"Error: Dataset has fewer than {NUM_EXAMPLES} images.")
        return

    # Get the perturbation transform
    perturb_hooks = InputPerturbHooks()
    perturb_transform = perturb_hooks.train_transform()

    # Prepare plot: NUM_EXAMPLES rows, 2 columns (Original, Perturbed)
    fig, axes = plt.subplots(
        NUM_EXAMPLES, 2, figsize=(6, 2 * NUM_EXAMPLES + 1)
    )  # Adjusted figsize
    if NUM_EXAMPLES == 1:  # Handle single row case for axes indexing
        axes = np.array([axes])

    fig.suptitle("Examples of Input Perturbations (Training)", fontsize=14, y=0.99)

    for i in range(NUM_EXAMPLES):
        pil_img, _ = original_dataset[
            i + 10
        ]  # Take some arbitrary images, avoid first few

        # --- Display Original Image ---
        # Convert PIL to tensor for imshow_tensor if it expects that, or display PIL directly
        # For consistency with perturbed, let's convert to tensor and normalize for display,
        # then unnormalize via imshow_tensor.
        # Or, simpler: just display PIL directly for the 'original'
        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 0].axis("off")

        # --- Apply Perturbation and Display Perturbed Image ---
        # The perturb_transform includes ToTensor and Normalize.
        # RandomErasing is applied last on the normalized tensor.
        perturbed_tensor_img = perturb_transform(pil_img.copy())  # Apply to a copy

        # imshow_tensor will unnormalize it for display
        imshow_tensor(perturbed_tensor_img, axes[i, 1], title=f"Perturbed {i + 1}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle
    output_path = OUTPUT_FIGS_DIR / "perturbed_examples.png"
    plt.savefig(output_path)
    print(f"Saved perturbed example plot to: {output_path}")
    plt.show()  # Optionally display the plot


if __name__ == "__main__":
    # This script assumes InputPerturbHooks is defined or imported correctly.
    # If you run this standalone, ensure the hook definitions are present.
    # If importing from your main script, ensure that script is accessible.
    generate_perturbed_examples()
