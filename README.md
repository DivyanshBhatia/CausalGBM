# CausalGBM: Fairness via Causal Feature Selection for Gradient Boosting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CausalGBM** is a two-stage framework that achieves fairness in tabular classification by automatically identifying and removing proxy features using causal discovery.

> 📄 **Paper**: "CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection" (IJCAI 2026)

## 🎯 Key Results

| Dataset | Baseline EOD | CausalGBM EOD | Reduction | AUC |
|---------|-------------|---------------|-----------|-----|
| ACS Income | 0.073 | 0.009 | **87%** | 0.86 |
| Adult | 0.086 | 0.024 | **72%** | 0.87 |
| Taiwan Credit | 0.034 | 0.019 | **46%** | 0.74 |
| Synthetic Hiring | 0.256 | 0.042 | **84%** | 0.67 |
| Synthetic Loan | 0.276 | 0.060 | **78%** | 0.65 |

CausalGBM achieves **46-87% EOD reduction** while maintaining competitive accuracy, running **33-56× faster** than transformer alternatives.

## 📋 Overview

### The Problem
Machine learning models can perpetuate bias through **proxy features**—variables that correlate with outcomes primarily through their association with protected attributes (e.g., zip code proxying for race in lending).

### Our Solution
CausalGBM uses causal discovery to automatically identify these proxy features:

```
Stage 1: Learn DAG via DAGMA → Identify proxy features (A → X edges)
Stage 2: Train gradient boosting on non-proxy features only
```

### Why It Works
- **DAG-based detection** captures proxies that correlation misses
- **Max-aggregation** combines DAG + correlation for robustness
- **Scoring function** balances proxy removal with predictive value retention

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CausalGBM Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Features  │───▶│  DAGMA      │───▶│  Learned    │     │
│  │   X, A, Y   │    │  Structure  │    │  DAG W      │     │
│  └─────────────┘    │  Learning   │    └──────┬──────┘     │
│                     └─────────────┘           │            │
│                                               ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Correlation │───▶│    Max      │───▶│  Scoring    │     │
│  │   |ρ(X,A)|  │    │ Aggregation │    │  Function   │     │
│  └─────────────┘    │  W' = max   │    │  c = W_XY   │     │
│                     └─────────────┘    │    - αW'    │     │
│                                        └──────┬──────┘     │
│                                               │            │
│                                               ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Selected   │◀───│  Threshold  │◀───│  Feature    │     │
│  │  Features   │    │   τ = 0.2   │    │  Scores     │     │
│  └──────┬──────┘    └─────────────┘    └─────────────┘     │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │  XGBoost    │───▶  Fair Predictions                     │
│  │  Training   │                                           │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DivyanshBhatia/CausalGBM.git
cd CausalGBM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from causalgbm import CausalGBM

# Initialize
model = CausalGBM(
    alpha=0.5,        # Fairness-accuracy tradeoff
    tau=0.2,          # Score threshold
    min_features=3    # Minimum features to retain
)

# Fit (X=features, A=protected attribute, y=target)
model.fit(X_train, A_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get selected features
selected = model.get_selected_features()
print(f"Selected features: {selected}")
print(f"Removed proxies: {model.get_removed_features()}")
```

## 📁 Project Structure

```
CausalGBM/
├── models.py                       # CausalGBM implementation
├── data_utils.py                   # Data loading and preprocessing
├── training.py                     # Training and evaluation utilities
├── causalgbm_experiments_v2.py     # Main experiment runner (reproduces Table 2)
├── run_experiment.py               # Single experiment runner
├── complete_benchmark.py           # Full benchmark suite
├── analysis.py                     # Results analysis and visualization
├── dag_visualization.py            # DAG visualization tools
├── min_features_ablation.py        # Hyperparameter ablation studies
├── demo.py                         # Quick demonstration
├── evaluate_indirect_proxy.py      # Indirect proxy limitation analysis (Appendix L)
├── synthetic_loan_data.csv         # Synthetic loan dataset (direct proxies)
├── synthetic_hiring_data.csv       # Synthetic hiring dataset (direct proxies)
├── synthetic_indirect_proxy_loan.csv # Synthetic dataset with indirect proxies (Appendix L)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Datasets

### Included Synthetic Datasets

| Dataset | n | Description | Purpose |
|---------|---|-------------|---------|
| `synthetic_loan_data.csv` | 10,000 | Loan approval with direct proxies | Validation (Section 4.4) |
| `synthetic_hiring_data.csv` | 10,000 | Hiring decisions with direct proxies | Validation (Section 4.4) |
| `synthetic_indirect_proxy_loan.csv` | 10,000 | Loan approval with **indirect** proxies | Limitation analysis (Appendix L) |

### Indirect Proxy Dataset (Appendix L)

The `synthetic_indirect_proxy_loan.csv` dataset is specifically designed to test Definition 1's limitation—CausalGBM's inability to detect indirect proxies.

**Ground Truth Causal Structure:**
```
Race (A) ──► Zip_Code_Risk ──► Loan_Approved (Y)
   │              │                    ▲
   │              ▼                    │
   │         Property_Value ───────────┤  (indirect proxy, |ρ|=0.06)
   │                                   │
   └───────► School_Rating ────────────┤
                  │                    │
                  ▼                    │
             Branch_Quality ───────────┘  (indirect proxy, |ρ|=0.10)

Legitimate: Annual_Income, Employment_Years, Credit_Score
Spurious:   Name_Pattern (correlated but no path to Y)
```

**Key Finding:** CausalGBM achieves 19% EOD reduction vs Oracle's 63%, with a 70% Indirect Gap. This validates the stated limitation when indirect proxies have low correlation (|ρ| < 0.1).

### Supported Real-World Datasets
| Dataset | n | d | Protected | Domain |
|---------|---|---|-----------|--------|
| Adult | 48,842 | 11 | Sex | Census |
| ACS Income | 195,665 | 10 | Race | Census |
| COMPAS | 5,278 | 6 | Race | Criminal Justice |
| German Credit | 1,000 | 20 | Sex | Finance |
| Taiwan Credit | 30,000 | 22 | Sex | Finance |
| Bank Marketing | 41,188 | 20 | Age | Marketing |
| Online Shoppers | 12,330 | 17 | Weekend | E-commerce |

## 🔧 Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Fairness penalty weight (higher = more fairness) |
| `tau` | 0.2 | Score threshold for feature selection |
| `min_features` | 3 | Minimum features to retain |
| `dag_iterations` | 500 | DAGMA optimization iterations |
| `dag_lambda` | 0.1 | DAG sparsity regularization |

### Robustness
Results are stable across wide hyperparameter ranges:
- EOD varies <0.025 for α ∈ [0.25, 2.0]
- EOD varies <0.02 for τ ∈ [0.1, 0.5]

---

## 📈 Reproducing Paper Results

### Command Line Interface

The main experiment script `causalgbm_experiments_v2.py` supports the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--all` | Run all experiments | False |
| `--main_comparison` | Run main method comparison | False |
| `--dag_recovery` | Run DAG recovery analysis on synthetic data | False |
| `--sensitivity` | Run hyperparameter sensitivity analysis | False |
| `--ablation` | Run ablation study (DAG vs Correlation) | False |
| `--datasets` | List of datasets to use | all datasets |
| `--synthetic_loan_path` | Path to synthetic loan CSV | auto-detect |
| `--synthetic_hiring_path` | Path to synthetic hiring CSV | auto-detect |
| `--output_dir` | Directory for results | `results/` |
| `--device` | Compute device | `cuda` |
| `--quick` | Quick mode (2 seeds instead of 5) | False |
| `--max_samples` | Max samples per dataset | 200000 |

---

### 🏃 Run Commands

#### **1. Full Experiment Suite (All Results)**
Reproduce all paper results including Table 2, ablations, and sensitivity analysis:

```bash
# Full experiments with GPU (recommended)
python causalgbm_experiments_v2.py --all --device cuda --output_dir results/full

# Full experiments with CPU only
python causalgbm_experiments_v2.py --all --device cpu --output_dir results/full
```

#### **2. Main Comparison Only (Table 2)**
Compare CausalGBM against all baselines:

```bash
# All datasets
python causalgbm_experiments_v2.py --main_comparison --output_dir results/main

# Specific datasets only
python causalgbm_experiments_v2.py --main_comparison \
    --datasets adult compas german acs_income \
    --output_dir results/main

# Quick test (2 seeds)
python causalgbm_experiments_v2.py --main_comparison --quick \
    --datasets adult compas \
    --output_dir results/quick_test
```

#### **3. DAG Recovery Analysis (Synthetic Data)**
Evaluate causal structure recovery on synthetic datasets with known ground truth:

```bash
# Run DAG recovery on synthetic datasets
python causalgbm_experiments_v2.py --dag_recovery \
    --datasets synthetic_loan synthetic_hiring \
    --output_dir results/dag_recovery

# With custom data paths
python causalgbm_experiments_v2.py --dag_recovery \
    --datasets synthetic_loan synthetic_hiring \
    --synthetic_loan_path /path/to/synthetic_loan_data.csv \
    --synthetic_hiring_path /path/to/synthetic_hiring_data.csv \
    --output_dir results/dag_recovery
```

#### **4. Ablation Study (DAG vs Correlation)**
Compare different aggregation strategies:

```bash
# Ablation on all datasets
python causalgbm_experiments_v2.py --ablation --output_dir results/ablation

# Ablation on specific datasets
python causalgbm_experiments_v2.py --ablation \
    --datasets adult acs_income synthetic_loan \
    --output_dir results/ablation
```

#### **5. Sensitivity Analysis**
Test hyperparameter sensitivity (α, τ, λ_dag, λ_sp, iterations):

```bash
# Full sensitivity analysis
python causalgbm_experiments_v2.py --sensitivity --output_dir results/sensitivity

# Quick sensitivity test
python causalgbm_experiments_v2.py --sensitivity --quick \
    --datasets adult german \
    --output_dir results/sensitivity_quick
```

#### **6. Combined Experiments**

```bash
# Main comparison + ablation
python causalgbm_experiments_v2.py --main_comparison --ablation \
    --output_dir results/comparison_ablation

# DAG recovery + sensitivity on synthetic data
python causalgbm_experiments_v2.py --dag_recovery --sensitivity \
    --datasets synthetic_loan synthetic_hiring \
    --output_dir results/synthetic_analysis

# Everything except sensitivity (faster)
python causalgbm_experiments_v2.py --main_comparison --dag_recovery --ablation \
    --output_dir results/no_sensitivity
```

#### **7. Resource-Constrained Settings**

```bash
# Low memory (limit samples)
python causalgbm_experiments_v2.py --all --max_samples 10000 \
    --output_dir results/low_mem

# CPU only with reduced load
python causalgbm_experiments_v2.py --main_comparison --quick \
    --device cpu --max_samples 20000 \
    --datasets adult compas german \
    --output_dir results/cpu_quick

# Single dataset quick test
python causalgbm_experiments_v2.py --main_comparison --quick \
    --datasets adult --output_dir results/test_adult
```

---

### 📁 Output Files

After running experiments, the following files are generated in `--output_dir`:

| File | Description |
|------|-------------|
| `main_comparison_results.csv` | Full results for all methods/datasets/seeds |
| `dag_recovery_results.csv` | DAG structure recovery metrics (SHD, F1, etc.) |
| `feature_selection_results.csv` | Causal vs spurious feature selection accuracy |
| `ablation_results.csv` | Aggregation strategy comparison |
| `sensitivity_analysis_results.csv` | Hyperparameter sensitivity results |
| `experiment_summary.txt` | Human-readable summary of all experiments |

---

### 📊 Example: Complete Reproducibility Pipeline

```bash
# Step 1: Quick validation (5 min)
python causalgbm_experiments_v2.py --main_comparison --quick \
    --datasets adult synthetic_loan \
    --output_dir results/validation

# Step 2: Full main results (2-4 hours)
python causalgbm_experiments_v2.py --main_comparison \
    --output_dir results/table2

# Step 3: DAG recovery analysis (30 min)
python causalgbm_experiments_v2.py --dag_recovery \
    --datasets synthetic_loan synthetic_hiring \
    --output_dir results/dag_analysis

# Step 4: Ablation studies (1-2 hours)
python causalgbm_experiments_v2.py --ablation \
    --output_dir results/ablation

# Step 5: Sensitivity analysis (3-4 hours)
python causalgbm_experiments_v2.py --sensitivity \
    --datasets adult acs_income german \
    --output_dir results/sensitivity

# Step 6: View summary
cat results/table2/experiment_summary.txt
```

---

### Indirect Proxy Analysis (Appendix L)

```bash
# Run indirect proxy limitation analysis
python evaluate_indirect_proxy.py
```

## 🔬 Understanding the Results

### Feature Analysis Output

```
Feature Analysis on Adult Dataset:
──────────────────────────────────────────────────────────────
Feature          |ρ|     W_A→X    W_X→Y    Score    Selected
──────────────────────────────────────────────────────────────
relationship    0.65     0.82     0.31    -0.10       ✗
marital-status  0.45     0.56     0.28    +0.00       ✗
education       0.12     0.08     0.45    +0.41       ✓
occupation      0.18     0.15     0.38    +0.30       ✓
hours-per-week  0.08     0.05     0.22    +0.19       ✓
age             0.06     0.04     0.35    +0.33       ✓
──────────────────────────────────────────────────────────────
Selected: 4/6 features | Removed proxies: relationship, marital-status
```

### Interpreting Scores
- **Negative score**: Proxy signal > predictive value → Remove
- **Positive score**: Predictive value > proxy signal → Keep
- **High |ρ| but positive score**: Feature is predictive despite correlation

### DAG Recovery Metrics (Synthetic Data)

| Metric | Description |
|--------|-------------|
| SHD | Structural Hamming Distance (lower is better) |
| Precision | Fraction of learned edges that are true edges |
| Recall | Fraction of true edges that were learned |
| F1 | Harmonic mean of precision and recall |
| Causal F1 | F1 for selecting truly causal features |
| Spurious Rejection | Rate of correctly rejecting spurious features |

## ⚠️ Limitations & When to Use

### ✅ Use CausalGBM When:
- Baseline EOD > 0.03 (meaningful unfairness exists)
- Sample size n > 1,000 (preferably n > 3,000)
- Unfairness stems from proxy features

### ❌ Avoid CausalGBM When:
- EOD < 0.02 (already fair)
- n < 1,000 (DAG learning unreliable)
- Unfairness from label bias (e.g., COMPAS)

### Known Limitations
1. **Indirect proxies**: Only detects direct A→X edges (see Appendix L)
2. **Linear assumption**: DAGMA may miss nonlinear relationships
3. **Scalability**: O(d³) cost prohibitive for d > 100 features

## 📚 Citation

```bibtex
@inproceedings{causalgbm2026,
  title={CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection},
  author={Anonymous},
  booktitle={Submitted to Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## 📖 References

1. **DAGMA**: Bello et al. (2022). "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization"
2. **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
3. **Equalized Odds**: Hardt et al. (2016). "Equality of Opportunity in Supervised Learning"
4. **Counterfactual Fairness**: Kusner et al. (2017). "Counterfactual Fairness"

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Nonlinear DAG learning (NOTEARS-MLP integration)
- [ ] Multi-hop (indirect) proxy detection
- [ ] Multiple protected attributes support
- [ ] Online learning for distribution shift
- [ ] Integration with other tree-based models (LightGBM, CatBoost)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

**Note**: This is the official implementation accompanying the IJCAI 2026 paper submission. Code and samples will be fully released upon acceptance.
