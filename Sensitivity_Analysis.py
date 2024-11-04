import numpy as np
from sklearn.metrics import confusion_matrix

# Sensitivity analysis for thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
for thresh in thresholds:
    y_pred_thresh = (model.predict_proba(X_test)[:, 1] >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    print(f"Threshold: {thresh:.1f}, True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")
