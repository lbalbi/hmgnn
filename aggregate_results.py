import os,csv
import statistics as stats



BASE_OUTPUT_DIR = "output" # directory that contains experiment folders
RESULT_TSV_PATH = "experiment_results.tsv"
N_RUNS = 10    # number of runs
FINAL_TEST_FILENAMES = ["final_test.log"]



def find_final_test_file(run_dir):

    for fname in FINAL_TEST_FILENAMES:
        path = os.path.join(run_dir, fname)
        if os.path.isfile(path): return path
    return None


def parse_metrics_from_final_test(file_path):

    last_test_final_line = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith("Test, final"): last_test_final_line = line_stripped

    if last_test_final_line is None: raise ValueError(f"No 'Test, final' line found in {file_path}")
    parts = [p.strip() for p in last_test_final_line.split(",")]

    if len(parts) < 9: raise ValueError(f"Unexpected format in {file_path}: {last_test_final_line}")
    metrics = list(map(float, parts[2:]))
    if len(metrics) != 7: raise ValueError(f"Expected 7 metrics in {file_path}, got {len(metrics)}")
    return metrics


def get_run_directories_for_experiment(experiment_dir, experiment_name):
    
    run_dirs = []
    for name in os.listdir(experiment_dir):
        full_path = os.path.join(experiment_dir, name)
        if not os.path.isdir(full_path): continue
        prefix = f"output_{experiment_name}_run_"

        if name.startswith(prefix):
            try:
                run_num_str = name[len(prefix):]
                run_num = int(run_num_str)
                run_dirs.append((run_num, full_path))
            except ValueError: continue

    run_dirs.sort(key=lambda x: x[0])
    run_dirs = run_dirs[:N_RUNS]
    return [d for (_, d) in run_dirs]



def compute_mean_and_std(metrics_per_run):
    metrics_by_column = list(zip(*metrics_per_run))
    means = [stats.mean(col) for col in metrics_by_column]
    stds = [stats.stdev(col) for col in metrics_by_column]
    return means, stds



def ensure_tsv_header(path):
    header = ["Experiment","Heuristic","Accuracy",
        "F1 Score (W)","Precision (+)","Recall (+)",
        "Precision (-)","Recall (-)","Roc Auc"]

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)



def append_experiment_stats_to_tsv(experiment_name, means, stds, tsv_path):

    ensure_tsv_header(tsv_path)
    fmt = lambda x: f"{x:.6f}"
    mean_row = [experiment_name, "mean"] + [fmt(v) for v in means]
    std_row = [experiment_name, "std. dev"] + [fmt(v) for v in stds]

    with open(tsv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(mean_row)
        writer.writerow(std_row)


def process_all_experiments(base_output_dir=BASE_OUTPUT_DIR, result_tsv_path=RESULT_TSV_PATH):

    for experiment_name in os.listdir(base_output_dir):
        experiment_dir = os.path.join(base_output_dir, experiment_name)
        if not os.path.isdir(experiment_dir): continue

        run_dirs = get_run_directories_for_experiment(experiment_dir, experiment_name)
        if len(run_dirs) == 0:
            print(f"[WARNING] No run directories found for experiment '{experiment_name}'. Skipping.")
            continue

        if len(run_dirs) < N_RUNS:
            print(f"[WARNING] Experiment '{experiment_name}' has only {len(run_dirs)} runs "
                f"(expected {N_RUNS}). Using available runs.")

        metrics_per_run = []
        for run_dir in run_dirs:
            final_test_file = find_final_test_file(run_dir)
            if final_test_file is None:
                print(f"[WARNING] No final_test file found in '{run_dir}'. Skipping this run.")
                continue
            try:
                metrics = parse_metrics_from_final_test(final_test_file)
                metrics_per_run.append(metrics)
            except Exception as e:
                print(f"[ERROR] Failed to parse '{final_test_file}': {e}")
                continue
        if len(metrics_per_run) < 2:
            print(f"[WARNING] Not enough valid runs for experiment '{experiment_name}' "
             f"to compute std. dev. (found {len(metrics_per_run)}). Skipping.")
            continue

        means, stds = compute_mean_and_std(metrics_per_run)
        append_experiment_stats_to_tsv(experiment_name, means, stds, result_tsv_path)
        print(f"[INFO] Processed experiment '{experiment_name}' with {len(metrics_per_run)} runs.")


if __name__ == "__main__":
    process_all_experiments()
