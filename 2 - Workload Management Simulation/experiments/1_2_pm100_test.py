#!/usr/bin/env python
# coding: utf-8

# =====================
# Imports
# =====================

import csv
import os
import re
import sys
import pandas as pd

from accasim.base.allocator_class import BestFit
from accasim.base.simulator_class import Simulator
from accasim.base.scheduler_class import (
    ShortestJobFirstExtended,
    EASYBackfillingExtended,
    PriorityRulesBasedExtended,
)

# =====================
# Utils
# =====================

def extract_metrics_to_csv(input_file, output_csv, workload, policy):
    awt = None
    asd = None
    number_pattern = re.compile(r"(\d+(?:\.\d+)?)")

    with open(input_file, "r") as f:
        for line in f:
            if "Avg. waiting times:" in line:
                m = number_pattern.search(line)
                if m:
                    awt = float(m.group(1))
            elif "Avg. slowdown:" in line:
                m = number_pattern.search(line)
                if m:
                    asd = float(m.group(1))

    if awt is None or asd is None:
        raise RuntimeError("Statistics not found in stats file")

    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["workload", "policy", "awt", "asd"])
        writer.writerow([workload, policy, awt, asd])


def load_completed_pairs(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["workload"].astype(str), df["policy"].astype(str)))


# =====================
# Paths & data
# =====================

sys_cfg = "config/pm100_980.config"
output_file = "results/pm100_results_980.csv"

pred_data = "predictions/pm100/predictions.csv"
pred_df = pd.read_csv(pred_data)

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================
# Policy Registry
# =====================

POLICIES = [
    ("SJF+User", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_user")),
    ("SJF+Oracle", ShortestJobFirstExtended, dict(precomputed_column="gt_runtime")),
    ("SJF+H", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_heuristic")),
    ("SJF+DT", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_dt")),
    ("SJF+RNP", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_rnp")),
    ("SJF+KNN", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_knn")),
    ("SJF+LLM", ShortestJobFirstExtended, dict(precomputed_column="pred_runtime_llm")),

    ("EASYBF+User", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_user")),
    ("EASYBF+Oracle", EASYBackfillingExtended, dict(precomputed_column="gt_runtime")),
    ("EASYBF+H", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_heuristic")),
    ("EASYBF+DT", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_dt")),
    ("EASYBF+RNP", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_rnp")),
    ("EASYBF+KNN", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_knn")),
    ("EASYBF+LLM", EASYBackfillingExtended, dict(precomputed_column="pred_runtime_llm")),

    ("PRB+User", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_user", tie_breaker="job_area")),
    ("PRB+Oracle", PriorityRulesBasedExtended, dict(precomputed_column="gt_runtime", tie_breaker="job_area")),
    ("PRB+H", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_heuristic", tie_breaker="job_area")),
    ("PRB+DT", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_dt", tie_breaker="job_area")),
    ("PRB+RNP", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_rnp", tie_breaker="job_area")),
    ("PRB+KNN", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_knn", tie_breaker="job_area")),
    ("PRB+LLM", PriorityRulesBasedExtended, dict(precomputed_column="pred_runtime_llm", tie_breaker="job_area")),
]

# =====================
# Workloads
# =====================

WORKLOAD_IDS = [1, 2, 3, 4, 5]

# =====================
# Main loop
# =====================

completed_pairs = load_completed_pairs(output_file)

for wid in WORKLOAD_IDS:

    workload = f"workloads/pm100/pm100_1800_{wid}.swf"

    stats_file = f"experiments/results/stats-pm100_1800_{wid}.swf"
    pprint_file = f"experiments/results/pprint-pm100_1800_{wid}.swf"

    workload_label = f"Heavy_{wid}"

    print(f"\n====================")
    print(f"[WORKLOAD] {workload_label}")
    print(f"====================")

    for policy_name, scheduler_cls, scheduler_kwargs in POLICIES:

        safe_policy = policy_name.replace("+", "_")

        if (workload_label, policy_name) in completed_pairs:
            print(f"[SKIP] {policy_name} on {workload_label}")
            continue

        print(f"\n[INFO] Policy: {policy_name}")

        allocator = BestFit()

        dispatcher = scheduler_cls(
            _allocator=allocator,
            precomputed_data=pred_df,
            job_id_column="job_id",
            _seed=42,
            **scheduler_kwargs,
        )

        simulator = Simulator(
            workload,
            sys_cfg,
            dispatcher,
            scheduling_output=False,
            pprint_output=True,
            benchmark_output=False,
            statistics_output=True,
            show_statistics=False,
        )

        print(f"[RUN ] {policy_name} on {workload_label}")
        simulator.start_simulation()

        # =====================
        # Copy outputs
        # =====================

        pprint_dst = f"experiments/results/pprint-pm100_1800_{wid}_{safe_policy}.swf"
        stats_dst = f"experiments/results/stats-pm100_1800_{wid}_{safe_policy}.swf"

        if os.path.exists(pprint_file):
            os.replace(pprint_file, pprint_dst)
        else:
            raise FileNotFoundError(f"Missing pprint file for {policy_name} ({workload_label})")

        if os.path.exists(stats_file):
            os.replace(stats_file, stats_dst)
        else:
            raise FileNotFoundError(f"Missing stats file for {policy_name} ({workload_label})")

        # =====================
        # Extract metrics
        # =====================

        extract_metrics_to_csv(
            input_file=stats_dst,
            output_csv=output_file,
            workload=workload_label,
            policy=policy_name,
        )

        print(f"[DONE] {policy_name} on {workload_label}")

        # update resume state immediately
        completed_pairs.add((workload_label, policy_name))

print("\n[ALL DONE]")