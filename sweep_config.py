# sweep.py
import os
import yaml
import itertools
import subprocess
import multiprocessing

# == Define the sweep grid ==
sweep_grid = {
    "selectWorstQ": [True, False],
    "findMaxQ":     [True, False],
    "simMaxQ":      [True, False],
    "testMaxQ":     [True, False],
}

with open("config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

keys = list(sweep_grid.keys())
combinations = list(itertools.product(*sweep_grid.values()))

# Use number of physical cores, leave a couple free for the OS
MAX_PARALLEL = max(1, multiprocessing.cpu_count() - 2)
print(f"Running up to {MAX_PARALLEL} experiments in parallel")
print(f"Launching {len(combinations)} total experiments...\n")

all_processes = []
running = []

for i, combo in enumerate(combinations):
    run_config = {section: dict(params) for section, params in base_config.items()}

    for key, val in zip(keys, combo):
        run_config["adversary"][key] = val

    flag_str = "_".join(k for k, v in zip(keys, combo) if v) or "all_false"
    run_name = f"run_{i:02d}_{flag_str}"
    run_config["file"]["name"] = run_name
    run_config["file"]["outFolder"] = os.path.join("experiments", "sweep")
    run_config["file"]["storeFigure"] = True
    run_config["file"]["plotFigure"] = False

    tmp_config_path = f"tmp_config_{i:02d}.yaml"
    with open(tmp_config_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    log_file = open(f"log_{run_name}.txt", "w")

    print(f"[{i+1}/{len(combinations)}] Launching: {run_name}")
    print(f"  Flags: { {k: v for k, v in zip(keys, combo)} }")

    proc = subprocess.Popen(
        ["python", "sim_new_point_mass.py", "--config", tmp_config_path],
        stdout=log_file,
        stderr=log_file,
    )

    entry = (proc, run_name, tmp_config_path, log_file)
    running.append(entry)
    all_processes.append(entry)

    # When we hit the limit, wait for one to finish before launching more
    while len(running) >= MAX_PARALLEL:
        for entry in running:
            if entry[0].poll() is not None:  # process finished
                proc, run_name, tmp_config_path, log_file = entry
                log_file.close()
                os.remove(tmp_config_path)
                status = "✓ Done" if proc.returncode == 0 else f"✗ FAILED (code {proc.returncode})"
                print(f"\n  {status}: {run_name}  →  log_{run_name}.txt")
                running.remove(entry)
                break

# == Wait for remaining processes to finish ==
print(f"\nAll launched. Waiting for remaining runs to complete...\n")

for proc, run_name, tmp_config_path, log_file in running:
    proc.wait()
    log_file.close()
    os.remove(tmp_config_path)
    status = "✓ Done" if proc.returncode == 0 else f"✗ FAILED (code {proc.returncode})"
    print(f"  {status}: {run_name}  →  log_{run_name}.txt")

print("\nSweep complete.")