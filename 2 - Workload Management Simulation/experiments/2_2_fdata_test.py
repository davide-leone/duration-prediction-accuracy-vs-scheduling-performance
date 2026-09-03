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
# Args
# =====================

if len(sys.argv) != 2:
    print("Usage: python script.py <workload_id>")
    sys.exit(1)

wid = int(sys.argv[1])

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

# =====================
# Paths & data
# =====================

sys_cfg = "config/fdata_159k.config"

# 👇 ONE OUTPUT FILE PER WORKLOAD
output_file = f"results/fdata_results_159k_w{wid}.csv"

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
# Single workload
# =====================

workload = f"workloads/fdata/fdata_18k_{wid}.swf"

stats_file = f"experiments/results/stats-fdata_18k_{wid}.swf"
pprint_file = f"experiments/results/pprint-fdata_18k_{wid}.swf"

workload_label = f"Heavy_{wid}"

print(f"\n[WORKLOAD] {workload_label}")

for policy_name, scheduler_cls, scheduler_kwargs in POLICIES:

    safe_policy = policy_name.replace("+", "_")

    print(f"\n[INFO] Policy: {policy_name}")

    allocator = BestFit()

    common_kwargs = dict(

    )

    # Only SJF/EASY need mode="precomputed"
    if scheduler_cls in {ShortestJobFirstExtended, EASYBackfillingExtended}:
        dispatcher = scheduler_cls(
            _allocator=allocator,
            mode="precomputed",
            precomputed_data=pred_df,
            job_id_column="job_id",
            _seed=42,
            **scheduler_kwargs,
        )
    else:
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

    print(f"[RUN ] {policy_name}")
    simulator.start_simulation()

    pprint_dst = f"experiments/results/pprint-fdata_18k_{wid}_{safe_policy}.swf"
    stats_dst = f"experiments/results/stats-fdata_18k_{wid}_{safe_policy}.swf"

    os.replace(pprint_file, pprint_dst)
    os.replace(stats_file, stats_dst)

    extract_metrics_to_csv(
        input_file=stats_dst,
        output_csv=output_file,
        workload=workload_label,
        policy=policy_name,
    )

    print(f"[DONE] {policy_name}")

print("\n[WORKLOAD DONE]")
