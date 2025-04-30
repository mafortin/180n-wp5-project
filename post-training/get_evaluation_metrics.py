from picai_eval import Metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


path2picai_metrics = "/home/marcantf/Data/nnunet/results/Dataset016_180n_lesion/my_nnUNetTrainer__UNet_ResEncL__3d_fullres/inference/picai_metrics.json"

metrics = Metrics(path2picai_metrics)

# aggregate metrics
AP = metrics.AP
auroc = metrics.auroc
picai_score = metrics.score

# Precision-Recall (PR) curve
precision = metrics.precision
recall = metrics.recall

# Receiver Operating Characteristic (ROC) curve
tpr = metrics.case_TPR
fpr = metrics.case_FPR

# Free-Response Receiver Operating Characteristic (FROC) curve
sensitivity = metrics.lesion_TPR
fp_per_case = metrics.lesion_FPR

### Plotting section ###

# Pecision-Recall (PR) curve
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=AP)
disp.plot()
plt.show()

# Receiver Operating Characteristic (ROC) curve
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc)
disp.plot()
plt.show()

# Free-Response Receiver Operating Characteristic (FROC) curve
f, ax = plt.subplots()
disp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity)
disp.plot(ax=ax)
ax.set_xlim(0.001, 5.0); ax.set_xscale('log')
ax.set_xlabel("False positives per case"); ax.set_ylabel("Sensitivity")
plt.show()