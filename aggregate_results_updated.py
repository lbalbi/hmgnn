import os
import csv
import statistics as stats
import re

BASE_OUTPUT_DIR = "output"  # directory that contains experiment folders
RESULT_TSV_PATH = "experiment_results.tsv"
N_RUNS = 10  # number of runs
FINAL_TEST_FILENAMES = ["final_test.log", "test_global.log"]


def find_final_test_file(run_dir: str) -> str | None:
    """Return the path to the first known test log present in run_dir."""
    for fname in FINAL_TEST_FILENAMES:
        path = os.path.join(run_dir, fname)
        if os.path.isfile(path):
            return path
    return None


def _parse_metrics_from_final_test_style(text: str, file_path: str) -> list[float]:
    """Parse the comma-separated 'Test, final,...' line format."""
    last_test_final_line = None
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("Test, final"):
            last_test_final_line = line_stripped

    if last_test_final_line is None:
        raise ValueError(f"No 'Test, final' line found in {file_path}")

    parts = [p.strip() for p in last_test_final_line.split(",")]
    if len(parts) < 9:
        raise ValueError(f"Unexpected format in {file_path}: {last_test_final_line}")

    try:
        metrics = list(map(float, parts[2:]))
    except ValueError as e:
        raise ValueError(f"Could not convert metrics to float in {file_path}: {last_test_final_line}") from e

    if len(metrics) != 7:
        raise ValueError(f"Expected 7 metrics in {file_path}, got {len(metrics)}")
    return metrics


def _parse_metrics_from_test_global_style(text: str, file_path: str) -> list[float]:
    """Parse key:value metric logs like test_global.log."""
    # Example lines:
    # accuracy: 0.4268
    # f1 score: 0.3264
    # precision (pos): 0.4587
    # ...
    kv_re = re.compile(
        r"^\s*([^:]+?)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )

    metrics_map: dict[str, float] = {}
    for line in text.splitlines():
        m = kv_re.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = float(m.group(2))
        metrics_map[key] = val  # keep last occurrence if repeated

    required_keys = [
        "accuracy",
        "f1 score",
        "precision (pos)",
        "recall (pos)",
        "precision (neg)",
        "recall (neg)",
        "roc auc",
    ]
    missing = [k for k in required_keys if k not in metrics_map]
    if missing:
        raise ValueError(f"Missing metrics in {file_path}: {', '.join(missing)}")

    return [metrics_map[k] for k in required_keys]


def parse_metrics_from_final_test(file_path: str) -> list[float]:
    """Parse either final_test.log style or test_global.log style logs."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Prefer the explicit 'Test, final' format if present.
    if "Test, final" in text:
        return _parse_metrics_from_final_test_style(text, file_path)

    # Otherwise try the key:value format used by test_global.log
    return _parse_metrics_from_test_global_style(text, file_path)


def get_run_directories_for_experiment(experiment_dir: str, experiment_name: str) -> list[str]:
    run_dirs: list[tuple[int, str]] = []
    prefix = f"output_{experiment_name}_run_"

    for name in os.listdir(experiment_dir):
        full_path = os.path.join(experiment_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith(prefix):
            continue

        try:
            run_num = int(name[len(prefix):])
        except ValueError:
            continue
        run_dirs.append((run_num, full_path))

    run_dirs.sort(key=lambda x: x[0])
    run_dirs = run_dirs[:N_RUNS]
    return [d for (_, d) in run_dirs]


def compute_mean_and_std(metrics_per_run: list[list[float]]) -> tuple[list[float], list[float]]:
    metrics_by_column = list(zip(*metrics_per_run))
    means = [stats.mean(col) for col in metrics_by_column]
    stds = [stats.stdev(col) for col in metrics_by_column]
    return means, stds


def ensure_tsv_header(path: str) -> None:
    header = [
        "Experiment",
        "Heuristic",
        "Accuracy",
        "F1 Score (W)",
        "Precision (+)",
        "Recall (+)",
        "Precision (-)",
        "Recall (-)",
        "Roc Auc",
    ]

    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)


def append_experiment_stats_to_tsv(experiment_name: str, means: list[float], stds: list[float], tsv_path: str) -> None:
    ensure_tsv_header(tsv_path)

    fmt = lambda x: f"{x:.6f}"
    mean_row = [experiment_name, "mean"] + [fmt(v) for v in means]
    std_row = [experiment_name, "std. dev"] + [fmt(v) for v in stds]

    with open(tsv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(mean_row)
        writer.writerow(std_row)


def process_all_experiments(base_output_dir: str = BASE_OUTPUT_DIR, result_tsv_path: str = RESULT_TSV_PATH) -> None:
    if not os.path.isdir(base_output_dir):
        raise FileNotFoundError(f"Base output directory not found: {base_output_dir}")

    for experiment_name in os.listdir(base_output_dir):
        experiment_dir = os.path.join(base_output_dir, experiment_name)
        if not os.path.isdir(experiment_dir):
            continue

        run_dirs = get_run_directories_for_experiment(experiment_dir, experiment_name)
        if len(run_dirs) == 0:
            print(f"[WARNING] No run directories found for experiment '{experiment_name}'. Skipping.")
            continue

        if len(run_dirs) < N_RUNS:
            print(
                f"[WARNING] Experiment '{experiment_name}' has only {len(run_dirs)} runs "
                f"(expected {N_RUNS}). Using available runs."
            )

        metrics_per_run: list[list[float]] = []
        for run_dir in run_dirs:
            test_file = find_final_test_file(run_dir)
            if test_file is None:
                print(f"[WARNING] No final_test/test_global file found in '{run_dir}'. Skipping this run.")
                continue

            try:
                metrics = parse_metrics_from_final_test(test_file)
                metrics_per_run.append(metrics)
            except Exception as e:
                print(f"[ERROR] Failed to parse '{test_file}': {e}")
                continue

        if len(metrics_per_run) < 2:
            print(
                f"[WARNING] Not enough valid runs for experiment '{experiment_name}' "
                f"to compute std. dev. (found {len(metrics_per_run)}). Skipping."
            )
            continue

        means, stds = compute_mean_and_std(metrics_per_run)
        append_experiment_stats_to_tsv(experiment_name, means, stds, result_tsv_path)
        print(f"[INFO] Processed experiment '{experiment_name}' with {len(metrics_per_run)} runs.")


if __name__ == "__main__":
    process_all_experiments()
