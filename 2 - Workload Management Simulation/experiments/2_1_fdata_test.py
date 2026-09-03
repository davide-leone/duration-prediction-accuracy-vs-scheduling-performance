#!/usr/bin/env python
# coding: utf-8

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
    PriorityRulesBasedExtended
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


def load_completed_policies(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(df["policy"].astype(str))


# =====================
# Paths & data
# =====================

workload = "workloads/fdata/fdata.swf"
sys_cfg = "config/fdata_159k.config"

# Simulator fixed outputs (CANNOT CHANGE)
stats_file = "experiments/results/stats-fdata.swf"
pprint_file = "experiments/results/pprint-fdata.swf"

output_file = "results/fdata_results_159k_c.csv"

pred_data = "predictions/fdata/predictions.csv"
pred_df = pd.read_csv(pred_data)

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================
# Policies
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
# Resume logic
# =====================

completed = load_completed_policies(output_file)

print(f"[INFO] Already completed policies: {len(completed)}")

# =====================
# Main loop (SEQUENTIAL)
# =====================

for policy_name, scheduler_cls, scheduler_kwargs in POLICIES:

    safe_policy = policy_name.replace("+", "_")

    if policy_name in completed:
        print(f"[SKIP] {policy_name}")
        continue

    print(f"\n[RUN ] {policy_name}")

    allocator = BestFit()

    if scheduler_cls == PriorityRulesBasedExtended:
        dispatcher = scheduler_cls(
            _allocator=allocator,
            precomputed_data=pred_df,
            job_id_column="job_id",
            _seed=42,
            **scheduler_kwargs,
        )
    else:
        dispatcher = scheduler_cls(
            _allocator=allocator,
            mode="precomputed",
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

    simulator.start_simulation()

    # =====================
    # Save outputs safely
    # =====================

    pprint_dst = f"experiments/results/pprint-fdata_{safe_policy}.swf"
    stats_dst = f"experiments/results/stats-fdata_{safe_policy}.swf"

    if os.path.exists(pprint_file):
        os.replace(pprint_file, pprint_dst)
    else:
        raise FileNotFoundError(f"Missing pprint file for {policy_name}")

    if os.path.exists(stats_file):
        os.replace(stats_file, stats_dst)
    else:
        raise FileNotFoundError(f"Missing stats file for {policy_name}")

    # =====================
    # Extract metrics
    # =====================

    extract_metrics_to_csv(
        input_file=stats_dst,
        output_csv=output_file,
        workload="Normal",
        policy=policy_name,
    )

    print(f"[DONE] {policy_name}")

    # update resume state immediately
    completed.add(policy_name)

print("\n[ALL DONE]")
