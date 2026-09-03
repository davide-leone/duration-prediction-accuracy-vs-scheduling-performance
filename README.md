# Impact of Job Duration Prediction Accuracy on HPC Scheduling Performance

This repository contains the datasets, source code, simulation framework, experimental results, and analysis notebooks used in the paper on **Impact of Job Duration Prediction Accuracy on HPC Scheduling Performance**.

The repository is organized into three main directories that collectively reproduce the five phases of the experimental workflow presented in the paper:

1. **Job Duration Prediction**, including:

   * Data Preparation
   * Workload Generation
   * Runtime Prediction

2. **Workload Management Simulation**

3. **Metric Analysis**

The complete experimental pipeline is illustrated in Figure 1.

## Experimental Workflow

<p align="center">
  <img src="docs/figures/experimental_pipeline.png"
       alt="Experimental pipeline"
       width="900">
</p>

**Figure 1.** Overview of the experimental workflow. The study consists of five phases: Data Preparation, Workload Generation, Runtime Prediction, Workload Management Simulation, and Metric Analysis.

---

# 1. Job Duration Prediction

This directory contains all resources required to reproduce the **job duration prediction** experiments.

## Datasets

Two production HPC datasets are included:

### PM100 (Marconi100)

The original PM100 dataset is publicly available on Zenodo:

* PM100 Dataset: https://zenodo.org/records/10127767

```text
datasets/pm100/
```

Contains:

* `job_data_train.parquet` – training subset
* `job_data_test.parquet` – test subset

These files correspond to the processed and enriched versions used throughout the experiments, after the chronological 70%/30% train-test split adopted in the paper.

### FDATA (Fugaku)

The original FDATA dataset is publicly available on Zenodo:

* FDATA Dataset: https://zenodo.org/records/11467483

```text
datasets/fdata/
```

Contains:

* `22_04_train.parquet` – training subset
* `22_04_test.parquet` – test subset

These files correspond to the processed and enriched versions used throughout the experiments, after the chronological 70%/30% train-test split adopted in the paper.

---

## Helper Modules

### `data_loader.py`

Responsible for:

* Loading the original datasets
* Computing additional historical features
* Chronologically splitting data into training and test sets
* Saving the resulting parquet files used by downstream experiments

---

### `prediction_models.py`

Implements the online prediction models evaluated in the paper:

* Online Decision Tree Regression (**DT**)
* Online Ridge-Normalized Polynomial Regression (**RNP**) with custom Ridge-based model
* Online k-Nearest Neighbors Regression (**KNN**)

> **Note:** The Online Retrieval-Augmented Language Model (**LLM**) predictor is implemented directly inside notebooks `1_5` and `2_5`.

---

### `utils.py`

Provides utility functions for:

* Prediction metric computation
* Histogram generation
* Scatter plot generation
* Predicted-versus-actual runtime analysis

---

## Experiment Notebooks

For both PM100 and FDATA, the workflow follows the same sequence:

| Notebook                             | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `*_0_0 - extend_dataset.ipynb`       | Feature enrichment and dataset preparation         |
| `*_0_1 - workload_generation.ipynb`  | Generates the workload files for the simulator     |
| `*_1 - heuristic.ipynb`              | Heuristic runtime estimation                       |
| `*_2 - DT.ipynb`                     | Online Decision Tree experiments                   |
| `*_3 - RNP.ipynb`                    | Online Ridge-Normalized Polynomial experiments     |
| `*_4 - KNN.ipynb`                    | Online k-NN experiments                            |
| `*_5 - LLM.ipynb`                    | Retrieval-Augmented LLM predictor                  |
| `*_6_0 - fix_predictions.ipynb`      | Prediction post-processing and export              |
| `*_6_1 - compute_metrics.ipynb`      | Computes runtime prediction accuracy metrics       |

The output of this phase is a set of runtime predictions that are subsequently used during workload management simulations.

---

# 2. Workload Management Simulation

This directory contains all resources required to reproduce the **workload management simulations**.

---

## AccaSim Simulator

```text
accasim/
```

Contains the source code of the AccaSim simulator.

The simulator was modified to support **prediction-informed scheduling policies**.

The main modifications are located in:

```text
accasim/base/scheduler_class.py
```

---

## Configuration Files

```text
config/
```

Contains simulator configuration files:

| File                | System                    |
| ------------------- | ------------------------- |
| `pm100_980.config`  | Marconi100                |
| `fdata_159k.config` | Fugaku                    |

---

## Simulation Scripts

```text
experiments/
```

Contains the scripts used to execute all scheduling experiments.

| Script              | Scenario |
| ------------------- | -------  |
| `1_1_pm100_test.py` | M-LR     |
| `1_2_pm100_test.py` | M-HR     |
| `2_1_fdata_test.py` | F-LR     |
| `2_2_fdata_test.py` | F-HR     |

---

## Scheduling Policies

The simulations evaluate:

* Shortest Job First (**SJF**)
* EASY Backfilling (**EASYBF**)
* Priority Rule-Based (**PRB**)

---

## Runtime Estimates

Each scheduling policy is evaluated using seven runtime estimates:

* User-provided runtime (**User**)
* Actual runtime (**Actual**)
* Heuristic estimate (**Heuristic**)
* Decision Tree prediction (**DT**)
* Ridge-Normalized Polynomial prediction (**RNP**)
* k-Nearest Neighbors prediction (**KNN**)
* Retrieval-Augmented Language Model prediction (**LLM**)

---

## Predictions

```text
predictions/
```

Contains the runtime predictions produced during Phase 1.

---

## Workloads

```text
workloads/
```

Contains the workloads used for simulation.

For each system:

* 1 low-rate workload
* 5 high-rate workloads

Datasets are provided in Standard Workload Format (SWF).

---

# 3. Metric Analysis

This directory contains the analyses performed to understand the relationship between prediction quality and scheduling effectiveness.

---

## Predictions

```text
predictions/
```

Contains the prediction outputs generated during Phase 1.

---

## Workloads

```text
workloads/
```

Contains the same 12 workloads used during simulation:

* PM100: 1 low-rate + 5 high-rate workloads
* FDATA: 1 low-rate + 5 high-rate workloads

---

## Simulation Queues

```text
queues/
```

Contains the queue traces generated by the high-rate simulation scenarios for both systems.

These traces are used for scheduling-aware metric analysis.

---

## Results

```text
results/
```

Contains the aggregated outputs of the study.

### Simulation Results

Files named:

```text
*_results.*
```

contain scheduling outcomes, including:

* Average Waiting Time (AWT)
* Average Slowdown (ASD)

Files ending with:

```text
*_results_h.*
```

refer to the high-rate scenarios.

---

### Prediction Metrics

Files named:

```text
*_metrics.csv
```

contain prediction accuracy metrics computed over the complete test datasets.

Files named:

```text
*_metrics_h.csv
```

contain metrics computed for individual high-rate workloads.

---

## Analysis Notebooks

### `node_utilization_comparison.ipynb`

Produces node utilization plots used in the paper.

---

### `performance_analysis.ipynb`

Evaluates how well traditional prediction accuracy metrics explain scheduling performance metrics.

This notebook investigates the relationship between:

* Prediction accuracy
* Average Waiting Time (AWT)
* Average Slowdown (ASD)

---

### `metrics_vs_errors_SJF.ipynb`

Provides a detailed analysis of the relationship between prediction errors and scheduling performance under SJF scheduling.

It also computes the scheduler-aware metrics introduced in the paper.

---

### `scheduler_aware_metric_analysis.ipynb`

Aggregates scheduler-aware metrics across workloads and systems.

The notebook additionally performs:

* Ablation studies
* Statistical significance analyses
* Cross-workload aggregation

---

# Reproducing the Study

The complete workflow is:

1. Execute the notebooks in **1 - Job Duration Prediction** to generate runtime predictions.
2. Use the generated predictions in **2 - Dispatcher Simulation** to execute workload management simulations.
3. Analyze prediction quality and scheduling effectiveness using the notebooks in **3 - Metric Analysis**.

The repository contains all datasets, workload traces, prediction outputs, simulation results, and analysis scripts necessary to reproduce the experiments reported in the paper.
