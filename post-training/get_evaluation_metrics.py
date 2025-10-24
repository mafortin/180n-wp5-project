from picai_eval import Metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# Provide the full path to the JSON file containing the PICAI metrics
path2picai_metrics = "/home/marcantf/180n/results/test-lesion-segs/baselr-5e5/picai_eval.json"
show_plots = True #False

# Fetch the metrics
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

### Print metrics ###
print('AUROC:', auroc)
print('AP:', AP)
print('PICAI score:', picai_score)
print('Precision:', precision)
print('Recall:', recall)

### Plotting section ###

# Pecision-Recall (PR) curve
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=AP)
disp.plot()
if show_plots:
    plt.show()

# Receiver Operating Characteristic (ROC) curve
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc)
disp.plot()
if show_plots:
    plt.show()

# Free-Response Receiver Operating Characteristic (FROC) curve
f, ax = plt.subplots()
disp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity)
disp.plot(ax=ax)

#if show_plots:
#    ax.set_xlim(0.001, 5.0); ax.set_xscale('log')
#    ax.set_xlabel("False positives per case"); ax.set_ylabel("Sensitivity")
#    plt.show()