# CausalGBM: Fair Gradient Boosting via Causal Feature Selection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IJCAI 2026](https://img.shields.io/badge/IJCAI-2026-green.svg)](https://ijcai26.org/)

Official implementation of **"CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection"** (IJCAI 2026).

## 📋 Overview

CausalGBM is a novel two-stage approach that combines **causal feature selection** with **gradient boosting** to achieve state-of-the-art fairness while maintaining competitive accuracy on tabular data.

### Key Results

| Dataset | EOD Reduction | Best Method |
|---------|---------------|-------------|
| Adult | **94.7%** | CausalGBM-GA |
| Online Shoppers | **90.5%** | CausalGBM-XGB |
| German Credit | **61.5%** | CausalGBM-LGB |
| COMPAS | **33.6%** | CausalGBM-XGB |
| Synthetic Loan | **100%** | CausalGBM |

### Why CausalGBM?

- **Fairness**: Up to 94.7% reduction in Equalized Odds Difference (EOD)
- **Speed**: 44-56× faster than transformer-based alternatives
- **Interpretability**: Clear visualization of which features are fair vs. unfair
- **Simplicity**: Works with standard GBM libraries (XGBoost, LightGBM)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CausalGBM Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: Causal Feature Selection                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Input Data │───▶│   DAGMA     │───▶│   Causal    │             │
│  │  (X, A, Y)  │    │ DAG Learning│    │ Importance  │             │
│  └─────────────┘    └─────────────┘    │  cⱼ=σ(Wⱼ,Y) │             │
│                                        └──────┬──────┘             │
│                                               │                     │
│                     ┌─────────────────────────▼─────────────────┐  │
│                     │  Feature Selection: Select if cⱼ ≥ τ      │  │
│                     │  (Remove spurious correlations with A)     │  │
│                     └─────────────────────────┬─────────────────┘  │
│                                               │                     │
│  Stage 2: Gradient Boosting                   ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Selected   │───▶│  XGBoost /  │───▶│    Fair     │             │
│  │  Features   │    │  LightGBM   │    │ Predictions │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
│  Optional: Group-Aware Reweighting for enhanced fairness            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DivyanshBhatia/CausalGBM.git
cd CausalGBM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
python>=3.10.12
torch>=2.0
xgboost>=1.7.6
lightgbm>=4.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
dagma>=1.0  # For causal discovery
```

### Run Demo

```bash
# Quick demo with synthetic data
python demo.py

# Full benchmark on all datasets
python complete_benchmark.py

# Reproduce paper results
python run_experiment.py --dataset adult --seeds 42 43 44 45 46
```

## 📊 Datasets

### Real-World Benchmarks

| Dataset | n | d | Protected | Groups | Source |
|---------|---|---|-----------|--------|--------|
| Adult | 32,561 | 14 | Sex | 2 | [UCI](https://archive.ics.uci.edu/ml/datasets/adult) |
| COMPAS | 5,278 | 11 | Race | 3 | [ProPublica](https://github.com/propublica/compas-analysis) |
| German Credit | 1,000 | 20 | Age | 3 | [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) |
| Bank Marketing | 41,188 | 20 | Age | 3 | [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| Online Shoppers | 12,330 | 17 | Weekend | 2 | [UCI](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) |

### Synthetic Datasets (included in repo)

| Dataset | n | d | Protected | Description |
|---------|---|---|-----------|-------------|
| `synthetic_loan_data.csv` | 10,000 | 8 | Gender | Loan approval with known causal structure |
| `synthetic_hiring_data.csv` | 10,000 | 10 | Race | Tech hiring with proxy features |

**Synthetic Loan Features:**
- **Fair** (ρ_A < 0.02): `income`, `credit_score`, `employment_years`
- **Unfair** (ρ_A > 0.55): `works_in_tech`, `has_stem_degree`, `plays_golf`

**Synthetic Hiring Features:**
- **Fair** (ρ_A < 0.02): `coding_score`, `years_experience`, `portfolio_quality`, `education_level`
- **Unfair** (ρ_A > 0.30): `ivy_league`, `unpaid_internships`, `golf_club_member`, `lacrosse_player`

## 📁 Project Structure

```
CausalGBM/
├── models.py                  # CausalGBM, XGBoost, LightGBM implementations
├── data_utils.py              # Data loading and preprocessing
├── training.py                # Training loops and evaluation
├── run_experiment.py          # Single dataset experiments
├── complete_benchmark.py      # Full benchmark across all datasets
├── analysis.py                # Results analysis and visualization
├── dag_visualization.py       # Causal graph visualization
├── min_features_ablation.py   # Minimum features ablation study
├── demo.py                    # Quick demo script
├── synthetic_loan_data.csv    # Synthetic loan dataset
├── synthetic_hiring_data.csv  # Synthetic hiring dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Usage

### Basic Usage

```python
from models import CausalGBM
from data_utils import load_dataset

# Load data
X_train, X_test, y_train, y_test, protected_train, protected_test = load_dataset('adult')

# Initialize CausalGBM
model = CausalGBM(
    base_model='xgboost',      # 'xgboost', 'lightgbm', or 'gbm'
    threshold=0.2,              # Causal importance threshold τ
    min_features=3,             # Minimum features to select
    group_aware=True            # Enable group-aware reweighting
)

# Fit model (learns causal structure + trains GBM)
model.fit(X_train, y_train, protected_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Get selected features
selected_features = model.get_selected_features()
print(f"Selected {len(selected_features)} fair features: {selected_features}")
```

### Causal Feature Analysis

```python
# Get causal importance scores for all features
importance_scores = model.get_causal_importance()

# Visualize feature selection
model.plot_feature_importance(save_path='feature_importance.pdf')

# Get learned DAG structure
dag_matrix = model.get_dag_matrix()
```

### Fairness Evaluation

```python
from training import evaluate_fairness

metrics = evaluate_fairness(
    y_true=y_test,
    y_pred=y_pred,
    protected=protected_test
)

print(f"Worst-Group Accuracy: {metrics['wga']:.3f}")
print(f"Equalized Odds Diff:  {metrics['eod']:.3f}")
print(f"Demographic Parity:   {metrics['dpd']:.3f}")
print(f"AUC:                  {metrics['auc']:.3f}")
```

## ⚙️ Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` (τ) | 0.2 | Causal importance threshold for feature selection |
| `min_features` (m) | max(3, ⌊d/3⌋) | Minimum features to retain |
| `dag_iterations` | 500 | DAGMA optimization iterations |
| `dag_lr` | 0.01 | DAGMA learning rate |
| `lambda_dag` | 0.1 | DAG acyclicity constraint weight |
| `lambda_sp` | 0.01 | Sparsity regularization weight |

### CausalGBM Variants

```python
# Standard CausalGBM (feature selection only)
model = CausalGBM(base_model='xgboost', group_aware=False)

# CausalGBM with Group-Aware reweighting (recommended)
model = CausalGBM(base_model='xgboost', group_aware=True)

# CausalGBM with LightGBM backend
model = CausalGBM(base_model='lightgbm', group_aware=True)
```

## 📈 Reproducing Paper Results

### Main Results (Table 2)

```bash
# Run all datasets with 5 seeds
python complete_benchmark.py --seeds 42 43 44 45 46 --output results/

# Analyze results
python analysis.py --input results/ --output figures/
```

### Ablation Studies

```bash
# Minimum features ablation (Table 6)
python min_features_ablation.py --dataset adult synthetic_loan

# Sensitivity analysis (threshold τ)
python run_experiment.py --dataset adult --threshold 0.1 0.2 0.3 0.4 0.5

# DAG visualization (Figure 4)
python dag_visualization.py --dataset synthetic_loan synthetic_hiring
```

### Expected Results

```
============================================================
                    MAIN RESULTS (5 seeds)
============================================================
Dataset          Method              WGA    EOD↓   DPD↓   AUC
------------------------------------------------------------
Adult            XGBoost            .840   .075   .181   .927
Adult            CausalGBM-GA       .783   .004   .064   .812
                 EOD Reduction:     94.7%

COMPAS           XGBoost            .632   .297   .254   .711
COMPAS           CausalGBM-XGB      .623   .198   .224   .677
                 EOD Reduction:     33.6%

Synthetic Loan   XGBoost            .696   .333   .244   .766
Synthetic Loan   CausalGBM          .637   .000   .000   .706
                 EOD Reduction:     100%
============================================================
```

## 🔬 Understanding CausalGBM

### Why Causal Feature Selection?

Traditional ML models learn **correlations**, not **causation**. This can lead to unfair predictions when features are correlated with protected attributes:

```
Example: Loan Approval

Standard ML finds:
  - "plays_golf" → High approval rate (correlation: 0.62)

CausalGBM discovers:
  - "plays_golf" is correlated with gender (ρ = 0.55)
  - Gender causes both golf-playing AND higher income
  - Golf itself doesn't CAUSE creditworthiness
  → CausalGBM rejects this feature
```

### Causal Importance Metric

For each feature X_j, we compute:

```
c_j = σ(W_{j,Y})
```

Where:
- W is the learned DAG adjacency matrix from DAGMA
- σ is the sigmoid function
- W_{j,Y} measures the direct causal effect of X_j on Y

Features with c_j < τ are removed as likely spurious correlations.

## 📊 Visualizations

### Feature Selection Visualization

```python
from dag_visualization import plot_feature_selection

plot_feature_selection(
    model=causalgbm,
    feature_names=X.columns,
    save_path='feature_selection.pdf'
)
```

Output shows:
- **Green bars**: Fair features (selected, low ρ_A)
- **Red bars**: Unfair proxies (rejected, high ρ_A)
- **Dashed line**: Selection threshold τ

### Pareto Frontier

```python
from analysis import plot_pareto_frontier

plot_pareto_frontier(
    results_df=results,
    dataset='adult',
    save_path='pareto_frontier.pdf'
)
```

## ⚠️ Limitations

1. **Multi-group settings**: CausalGBM is optimized for binary protected attributes. With 8+ groups, performance degrades. Use GroupDRO instead.

2. **Small samples**: Causal discovery requires n > 5,000 for reliable results. On German Credit (n=1,000), AUC drops significantly.

3. **Degenerate solutions**: On some datasets (e.g., Law School), the method achieves perfect EOD but AUC drops below 0.6, indicating trivial predictions.

## 🔮 When to Use CausalGBM

| Scenario | Recommendation |
|----------|----------------|
| Binary protected attribute | ✅ Use CausalGBM |
| 2-3 demographic groups | ✅ Use CausalGBM |
| 8+ demographic groups | ❌ Use GroupDRO |
| n < 1,000 samples | ❌ Use traditional fair ML |
| Need interpretability | ✅ Use CausalGBM |
| Need maximum accuracy | ⚠️ Consider accuracy trade-off |

## 📚 Citation

If you use CausalGBM in your research, please cite:

```bibtex
@inproceedings{bhatia2026causalgbm,
  title={CausalGBM: Achieving Fairness in Tabular Classification via Causal Feature Selection},
  author={Bhatia, Divyansh},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## 📖 References

1. **DAGMA**: Bello et al. (2022). "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization"
2. **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
3. **LightGBM**: Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
4. **GroupDRO**: Sagawa et al. (2020). "Distributionally Robust Neural Networks for Group Shifts"
5. **Counterfactual Fairness**: Kusner et al. (2017). "Counterfactual Fairness"

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-group extension using ANOVA-based correlation
- [ ] Adaptive threshold selection
- [ ] Integration with more GBM libraries (CatBoost)
- [ ] Support for continuous protected attributes
- [ ] Comparison with more causal discovery methods

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or issues, please:
1. Open a GitHub issue
2. Contact: [your-email@example.com]

---

**Reproducibility**: All experiments run on NVIDIA H100 GPU with Python 3.10.12, PyTorch 2.0, XGBoost 1.7.6, LightGBM 4.0. Seeds: 42, 43, 44, 45, 46.
