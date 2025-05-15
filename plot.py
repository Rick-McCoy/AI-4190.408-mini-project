import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_learning_curves_from_json(
    history_files_dict,
    metrics_files_dict,  # New: dictionary for metrics.json paths
    output_dir_str="figs",
):
    """
    Plots training/validation loss, Acc@1, and Acc@5 curves from history.json files,
    and adds final test points from metrics.json files.

    Args:
        history_files_dict (dict): Keys are exp names, values are paths to training_history.json.
        metrics_files_dict (dict): Keys are exp names, values are paths to metrics.json.
        output_dir_str (str): Directory to save the plot images.
    """
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_histories = {}
    all_metrics = {}  # To store loaded metrics data

    for exp_name, file_path_str in history_files_dict.items():
        file_path = Path(file_path_str)
        if file_path.exists():
            with open(file_path, "r") as f:
                all_histories[exp_name] = json.load(f)
        else:
            print(f"Warning: History file not found for {exp_name} at {file_path}")

    for exp_name, file_path_str in metrics_files_dict.items():  # Load metrics
        file_path = Path(file_path_str)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    all_metrics[exp_name] = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode metrics.json for {exp_name} at {file_path}"
                )
        else:
            print(f"Warning: Metrics file not found for {exp_name} at {file_path}")

    if not all_histories:
        print("No history data loaded. Exiting plotting function.")
        return

    # --- Plot Loss Curves ---
    plt.figure(figsize=(10, 6))
    for exp_name, history in all_histories.items():
        if not history:
            continue  # Skip if history loading failed
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        (line,) = plt.plot(epochs, train_loss, label=f"{exp_name} Train Loss")
        plt.plot(
            epochs,
            val_loss,
            label=f"{exp_name} Val Loss",
            linestyle="--",
            color=line.get_color(),
        )

        # Add test point for loss
        if (
            exp_name in all_metrics
            and "test_loss_on_loader_labels" in all_metrics[exp_name]
            and epochs
        ):
            test_loss_val = all_metrics[exp_name]["test_loss_on_loader_labels"]
            # For shuffle, the loss scale might be very different, handle appropriately if needed
            # or exclude its test point if it makes the graph unreadable.
            # Here, we plot it, but it might be off-scale for the 'shuffle' experiment.
            plt.scatter(
                [epochs[-1]],
                [test_loss_val],
                color=line.get_color(),
                marker="*",
                s=100,
                label=f"{exp_name} Test Loss",
                zorder=5,
            )

    plt.title("Training, Validation, and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    # Adjust ylim carefully, especially considering shuffle experiment's high loss
    min_loss_vals = []
    max_loss_vals = []
    for history in all_histories.values():
        if history:
            min_loss_vals.extend([h["train_loss"] for h in history])
            min_loss_vals.extend([h["val_loss"] for h in history])
    for metrics_data in all_metrics.values():
        if "test_loss_on_loader_labels" in metrics_data:
            # Exclude shuffle's high loss from y-limit calculation if it skews too much
            if not (
                metrics_data.get("test_loss_on_loader_labels", 0) > 2.2
                and "Shuffle" in exp_name
            ):  # Heuristic
                min_loss_vals.append(metrics_data["test_loss_on_loader_labels"])

    # Sensible y-limits, ensuring 0 is included if all losses are positive
    # and handling cases where min_loss_vals might be empty
    if min_loss_vals:
        plot_min_loss = (
            min(0, min(min_loss_vals)) if any(l < 0 for l in min_loss_vals) else 0
        )
        plot_max_loss = (
            max(min_loss_vals) * 1.1 if max(min_loss_vals) < 2.5 else 2.5
        )  # Cap at 2.5 unless shuffle
        if "Shuffle" in [
            exp_name for exp_name, hist in all_histories.items() if hist
        ]:  # If shuffle is plotted
            plot_max_loss = max(plot_max_loss, 2.5)  # Ensure shuffle can be seen
        plt.ylim(plot_min_loss, plot_max_loss)
    else:
        plt.ylim(0, 2.5)

    plt.savefig(output_dir / "loss_curves.png")
    plt.close()
    print(f"Saved loss_curves.png to {output_dir}")

    # --- Plot Top-1 Accuracy Curves ---
    plt.figure(figsize=(10, 6))
    for exp_name, history in all_histories.items():
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train_acc1 = [h["train_acc1"] for h in history]
        val_acc1 = [h["val_acc1"] for h in history]

        (line,) = plt.plot(epochs, train_acc1, label=f"{exp_name} Train Acc@1")
        plt.plot(
            epochs,
            val_acc1,
            label=f"{exp_name} Val Acc@1",
            linestyle="--",
            color=line.get_color(),
        )

        if (
            exp_name in all_metrics
            and "test_acc1_on_loader_labels" in all_metrics[exp_name]
            and epochs
        ):
            test_acc1_val = all_metrics[exp_name]["test_acc1_on_loader_labels"]
            plt.scatter(
                [epochs[-1]],
                [test_acc1_val],
                color=line.get_color(),
                marker="*",
                s=100,
                label=f"{exp_name} Test Acc@1",
                zorder=5,
            )

    plt.title("Training, Validation, and Test Top-1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy@1")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig(output_dir / "acc1_curves.png")
    plt.close()
    print(f"Saved acc1_curves.png to {output_dir}")

    # --- Plot Top-5 Accuracy Curves ---
    plt.figure(figsize=(10, 6))
    for exp_name, history in all_histories.items():
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train_acc5 = [h["train_acc5"] for h in history]  # Assuming key is 'train_a5'
        val_acc5 = [h["val_acc5"] for h in history]  # Assuming key is 'val_a5'

        (line,) = plt.plot(epochs, train_acc5, label=f"{exp_name} Train Acc@5")
        plt.plot(
            epochs,
            val_acc5,
            label=f"{exp_name} Val Acc@5",
            linestyle="--",
            color=line.get_color(),
        )

        if (
            exp_name in all_metrics
            and "test_acc5_on_loader_labels" in all_metrics[exp_name]
            and epochs
        ):
            test_acc5_val = all_metrics[exp_name]["test_acc5_on_loader_labels"]
            plt.scatter(
                [epochs[-1]],
                [test_acc5_val],
                color=line.get_color(),
                marker="*",
                s=100,
                label=f"{exp_name} Test Acc@5",
                zorder=5,
            )

    plt.title("Training, Validation, and Test Top-5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy@5")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.ylim(0.4, 1.05)  # Acc@5 is usually higher, adjust ylim
    plt.savefig(output_dir / "acc5_curves.png")
    plt.close()
    print(f"Saved acc5_curves.png to {output_dir}")


if __name__ == "__main__":
    history_files = {
        "Baseline": "runs/baseline_20250516_024204_seed42_hookseedN_A/training_history.json",
        "Shuffle": "runs/shuffle_20250516_023627_seed42_hookseed0/training_history.json",
        "Noise (20%)": "runs/noise_20250516_024702_seed42_hookseed0/training_history.json",
        "Perturbation": "runs/perturb_20250516_030103_seed42_hookseedN_A/training_history.json",
    }
    metrics_files = {  # Add paths to your metrics.json files
        "Baseline": "runs/baseline_20250516_024204_seed42_hookseedN_A/metrics.json",
        "Shuffle": "runs/shuffle_20250516_023627_seed42_hookseed0/metrics.json",
        "Noise (20%)": "runs/noise_20250516_024702_seed42_hookseed0/metrics.json",
        "Perturbation": "runs/perturb_20250516_030103_seed42_hookseedN_A/metrics.json",
    }
    plot_learning_curves_from_json(history_files, metrics_files, output_dir_str="figs")
