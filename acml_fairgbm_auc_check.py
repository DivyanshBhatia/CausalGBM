"""
FairGBM ACS per-seed AUC at λ=0.75 and λ=1.0
Confirms whether seeds scatter (supporting the footnote) or are stable.
"""
import sys, warnings, numpy as np
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from causalgbm_experiments_v2 import load_acs_income, compute_metrics
from fairgbm import FairGBMClassifier

ds = load_acs_income()
X, y, s = ds.X, ds.y, ds.sensitive

for lam in [0.5, 0.75, 1.0]:
    aucs = []
    for seed in range(10):
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, s, test_size=0.3, random_state=seed, stratify=y)
        try:
            fm = FairGBMClassifier(
                constraint_type="FNR,FPR", n_estimators=100,
                random_state=seed, multiplier_learning_rate=lam, verbose=-1)
            fm.fit(X_tr, y_tr, constraint_group=s_tr)
            met = compute_metrics(y_te, fm.predict(X_te), fm.predict_proba(X_te)[:, 1], s_te)
            aucs.append(met['auc'])
        except:
            aucs.append(float('nan'))
    arr = np.array(aucs)
    print(f"λ={lam}: AUC mean={np.nanmean(arr):.3f}  "
          f"range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]  "
          f"sd={np.nanstd(arr):.3f}  "
          f"seeds below .859: {sum(1 for a in arr if a < .859)}/10")
