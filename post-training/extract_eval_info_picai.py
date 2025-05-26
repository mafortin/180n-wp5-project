import json
import argparse

def evaluate_lesions(json_path, threshold=0.5):
    with open(json_path, 'r') as f:
        data = json.load(f)

    lesion_results = data.get("lesion_results", {})
    report = {}
    metrics_accumulator = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "total_predicted": 0,
        "ground_truth_lesions": 0
    }

    num_subjects = len(lesion_results)

    for subject_id, lesions in lesion_results.items():
        actual_detected = 0
        false_positives = 0
        false_negatives = 0
        total_predicted = 0
        ground_truth_total = 0

        for lesion in lesions:
            is_true_lesion = lesion[0] == 1
            predicted_score = lesion[1]

            if is_true_lesion:
                ground_truth_total += 1

            if predicted_score > threshold:
                total_predicted += 1
                if is_true_lesion:
                    actual_detected += 1
                else:
                    false_positives += 1
            else:
                if is_true_lesion:
                    false_negatives += 1

        # Compute subject-level true_lesion_detection_rate
        if ground_truth_total == 0:
            if false_positives > 0:
                detection_rate = 0.0
            else:
                detection_rate = 1.0
        else:
            detection_rate = round(actual_detected / ground_truth_total, 2)

        subject_report = {
            "true_positives": actual_detected,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_predicted": total_predicted,
            "ground_truth_lesions": ground_truth_total,
            "true_lesion_detection_rate": detection_rate
        }

        report[subject_id] = subject_report

        # Accumulate totals for summary
        metrics_accumulator["true_positives"] += actual_detected
        metrics_accumulator["false_positives"] += false_positives
        metrics_accumulator["false_negatives"] += false_negatives
        metrics_accumulator["total_predicted"] += total_predicted
        metrics_accumulator["ground_truth_lesions"] += ground_truth_total

        # Console output
        print(f"Subject: {subject_id}")
        for k, v in subject_report.items():
            print(f"  {k.replace('_', ' ').capitalize()}: {v}")
        print("-" * 60)

    # Compute and add the total summary
    summary = dict(metrics_accumulator)
    if summary["ground_truth_lesions"] == 0:
        if summary["false_positives"] > 0:
            summary["true_lesion_detection_rate"] = 0.0
        else:
            summary["true_lesion_detection_rate"] = 1.0
    else:
        summary["true_lesion_detection_rate"] = round(
            summary["true_positives"] / summary["ground_truth_lesions"], 2
        )

    print("Summary (total values across all subjects):")
    for k, v in summary.items():
        print(f"  {k.replace('_', ' ').capitalize()}: {v}")
    print("=" * 60)

    # Add summary to the beginning of the report
    report = {"summary": summary, **report}
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate lesion detection metrics.")
    parser.add_argument('--input', required=True, help="Path to the input JSON file.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Detection threshold (default: 0.5).")
    parser.add_argument('--output', help="Path to save the evaluation report as JSON.")

    args = parser.parse_args()

    report = evaluate_lesions(args.input, args.threshold)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nEvaluation report saved to {args.output}")

if __name__ == "__main__":
    main()
