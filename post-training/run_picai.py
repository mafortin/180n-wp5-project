import argparse
from picai_eval import Metrics, evaluate_folder
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PICAI evaluation and generate metrics.")
    parser.add_argument("--path2preds", required=True, help="Path to the predictions folder.")
    parser.add_argument("--path2gt", required=True, help="Path to the ground truth folder.")
    #parser.add_argument("--img_ext", type=str, help="Image file extension (default: .nii.gz).")
    parser.add_argument("--path2json", required=True, help="Path to save the metrics JSON file.")
    args = parser.parse_args()

    # Run PICAI evaluation
    metrics = evaluate_folder(
        y_det_dir=args.path2preds,
        y_true_dir=args.path2gt
    )

    # Save the full metrics to a JSON file
    metrics.save_full(args.path2json)

    # Fetch the metrics
    metrics = Metrics(args.path2json)

    # Aggregate metrics
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

    # Lesion-level metrics
    lesion_trp = metrics.lesion_TPR
    lesion_fpr = metrics.lesion_FPR

    # Print metrics
    print('AUROC:', auroc)
    print('AP:', AP)
    print('PICAI score:', picai_score)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Lesion TPR:', lesion_trp)
    print('Lesion FPR:', lesion_fpr)

    # Plotting section
    # Precision-Recall (PR) curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=AP)
    disp.plot()

    # Receiver Operating Characteristic (ROC) curve
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc)
    disp.plot()

    # Free-Response Receiver Operating Characteristic (FROC) curve
    f, ax = plt.subplots()
    disp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity)
    disp.plot(ax=ax)

if __name__ == "__main__":
    main()
