import json
import re
from pathlib import Path


def collect_experiment_data(runs_dir_str: str, output_file_str: str):
    """
    Collects data from experiment run folders and aggregates it into a single text file.

    Searches for 'training_history.json', 'classification_report.txt', and 'metrics.json'
    in each sub-folder of the specified runs_dir.
    """
    runs_dir = Path(runs_dir_str)
    output_file = Path(output_file_str)
    aggregated_output_lines = []

    if not runs_dir.is_dir():
        print(f"Error: Runs directory '{runs_dir}' not found.")
        aggregated_output_lines.append(
            f"Error: Runs directory '{runs_dir}' not found.\n"
        )
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(aggregated_output_lines)
        return

    # Sort folders to process them in a somewhat predictable order (e.g., by name)
    experiment_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if not experiment_folders:
        print(f"No sub-folders found in '{runs_dir}'.")
        aggregated_output_lines.append(f"No sub-folders found in '{runs_dir}'.\n")
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(aggregated_output_lines)
        return

    aggregated_output_lines.append("=" * 80 + "\n")
    aggregated_output_lines.append("AGGREGATED EXPERIMENT DATA\n")
    aggregated_output_lines.append("=" * 80 + "\n\n")

    for exp_folder in experiment_folders:
        aggregated_output_lines.append("-" * 70 + "\n")
        aggregated_output_lines.append(f"Run Folder: {exp_folder.name}\n")

        # Try to infer experiment type from folder name
        # (Assumes folder name starts with 'baseline_', 'shuffle_', 'noise_', or 'perturb_')
        experiment_type = "unknown"
        match = re.match(
            r"^(baseline|shuffle|noise|perturb).*", exp_folder.name, re.IGNORECASE
        )
        if match:
            experiment_type = match.group(1).lower()
        aggregated_output_lines.append(f"Detected Experiment Type: {experiment_type}\n")
        aggregated_output_lines.append("-" * 70 + "\n\n")

        # 1. Read metrics.json
        metrics_file = exp_folder / "metrics.json"
        aggregated_output_lines.append("--- Content of metrics.json ---\n")
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                aggregated_output_lines.append(json.dumps(metrics_data, indent=2))
            except json.JSONDecodeError:
                aggregated_output_lines.append(
                    f"Error: Could not decode JSON from {metrics_file.name}"
                )
            except Exception as e:
                aggregated_output_lines.append(
                    f"Error reading {metrics_file.name}: {e}"
                )
        else:
            aggregated_output_lines.append("File not found: metrics.json")
        aggregated_output_lines.append("\n\n")

        # 2. Read classification_report.txt
        report_file = exp_folder / "classification_report.txt"
        aggregated_output_lines.append("--- Content of classification_report.txt ---\n")
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    aggregated_output_lines.append(f.read())
            except Exception as e:
                aggregated_output_lines.append(f"Error reading {report_file.name}: {e}")
        else:
            aggregated_output_lines.append("File not found: classification_report.txt")
        aggregated_output_lines.append("\n\n")

        # 3. Read training_history.json
        history_file = exp_folder / "training_history.json"
        aggregated_output_lines.append("--- Content of training_history.json ---\n")
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                # To keep the text file manageable, maybe summarize or take first/last few epochs
                # For now, dumping the whole thing as requested for your processing.
                aggregated_output_lines.append(json.dumps(history_data, indent=2))
            except json.JSONDecodeError:
                aggregated_output_lines.append(
                    f"Error: Could not decode JSON from {history_file.name}"
                )
            except Exception as e:
                aggregated_output_lines.append(
                    f"Error reading {history_file.name}: {e}"
                )
        else:
            aggregated_output_lines.append("File not found: training_history.json")
        aggregated_output_lines.append("\n\n")

        aggregated_output_lines.append("=" * 80 + "\n\n")

    # Write to output file
    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(aggregated_output_lines)
        print(f"Successfully aggregated data to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")


if __name__ == "__main__":
    # Configuration:
    # Set this to the path of your main 'runs' directory
    # e.g., "./runs" if it's in the same directory as this script
    # or "/path/to/your/project/runs"
    runs_directory = "./runs"

    # Set this to your desired output file name
    output_filename = "aggregated_results.txt"

    print(f"Starting data aggregation from: {Path(runs_directory).resolve()}")
    print(f"Output will be saved to: {Path(output_filename).resolve()}")

    collect_experiment_data(runs_directory, output_filename)

    print("\nReminder: This script collects data from existing files.")
    print(
        "For confusion matrices and detailed parameter confirmations, please provide them separately if possible."
    )
