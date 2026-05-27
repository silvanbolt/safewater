import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CONFUSION_MATRIX_CSV = "input/confusion matrices/final_conf_matrix.csv"
LABEL_MAP_CSV = "input/pre-processing/label_map.csv"

# Confusion matrix convention:
# rows    = predicted labels
# columns = true labels


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def safe_divide(numerator, denominator):
    """Return numerator / denominator, or 0.0 if denominator is 0."""
    return numerator / denominator if denominator != 0 else 0.0


def normalize_label(label):
    """
    Normalize labels so that labels written slightly differently can match.

    Examples:
    - "Borehole_Tubewell" -> "borehole_tubewell"
    - "Borehole/Tubewell" -> "borehole_tubewell"
    - "Surface Water (River/Stream/Lake/Pond/Dam)" -> "surface_water"
    - "Sand or Sub-surface Dam" -> "sand_or_sub_surface_dam"
    """
    label = str(label).strip().lower()

    # Remove explanatory text in parentheses.
    label = re.sub(r"\([^)]*\)", "", label)

    # Convert all non-alphanumeric runs to underscores.
    label = re.sub(r"[^a-z0-9]+", "_", label)

    # Remove repeated/leading/trailing underscores.
    label = re.sub(r"_+", "_", label).strip("_")

    return label


def normalize_binary_group(value):
    """
    Convert mapping values to exactly:
    - "improved"
    - "non-improved"
    """
    value = str(value).strip().lower()

    if value in {"improved", "safe", "1", "true", "yes"}:
        return "improved"

    if value in {
        "unimproved",
        "non-improved",
        "non improved",
        "not improved",
        "unsafe",
        "0",
        "false",
        "no",
    }:
        return "non-improved"

    raise ValueError(
        f"Unknown binary group value: {value!r}. "
        "Expected something like 'Improved' or 'Unimproved'."
    )


# ---------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------
def load_confusion_matrix(csv_path):
    """
    Load a confusion matrix from CSV.

    Expected format:
    - Rows = predicted labels
    - Columns = true labels
    - First column may contain row labels, as in an exported pandas DataFrame.
    """
    df = pd.read_csv(csv_path)

    # If the first column contains row labels, use it as index.
    first_col = df.columns[0]
    if first_col.startswith("Unnamed") or not pd.api.types.is_numeric_dtype(df[first_col]):
        df = df.set_index(first_col)

    # Make labels strings for safer matching.
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    # Make values numeric.
    df = df.apply(pd.to_numeric)

    # Check that rows and columns contain the same classes.
    if set(df.index) != set(df.columns):
        raise ValueError(
            "Row labels and column labels must contain the same classes. "
            "Remember: rows should be predictions and columns should be true labels."
        )

    # Reorder rows to match columns, so the diagonal is correct.
    df = df.loc[df.columns, df.columns]

    return df


def load_label_map(label_map_csv):
    """
    Load the mapping from multiclass labels to binary labels.

    Expected columns:
    - one column with original class names
    - one column with improved/non-improved labels

    For your uploaded file, these are:
    - water_source_clean
    - improved
    """
    label_map = pd.read_csv(label_map_csv)

    if label_map.shape[1] < 2:
        raise ValueError("The label map CSV must contain at least two columns.")

    original_label_col = label_map.columns[0]
    binary_group_col = label_map.columns[1]

    label_map = label_map[[original_label_col, binary_group_col]].copy()
    label_map.columns = ["original_class", "binary_class"]

    label_map["normalized_original_class"] = label_map["original_class"].apply(normalize_label)
    label_map["binary_class"] = label_map["binary_class"].apply(normalize_binary_group)

    if label_map["normalized_original_class"].duplicated().any():
        duplicates = label_map.loc[
            label_map["normalized_original_class"].duplicated(keep=False),
            ["original_class", "normalized_original_class"]
        ]
        raise ValueError(
            "Some labels in the mapping file become duplicated after normalization:\n"
            f"{duplicates}"
        )

    return dict(zip(label_map["normalized_original_class"], label_map["binary_class"]))


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------
def compute_confusion_matrix_metrics(confusion_matrix):
    """
    Compute classification metrics from a confusion matrix where:

        rows    = predicted labels
        columns = true labels

    Returns:
    - per_class_metrics: one row per class
    - summary_metrics: global/macro/weighted metrics
    """
    cm = confusion_matrix.copy()
    labels = list(cm.columns)
    values = cm.to_numpy(dtype=float)

    total = values.sum()
    diagonal_sum = np.trace(values)

    overall_accuracy = safe_divide(diagonal_sum, total)

    row_sums = values.sum(axis=1)       # predicted count per class
    column_sums = values.sum(axis=0)    # true support per class
    diagonal = np.diag(values)

    records = []

    for i, label in enumerate(labels):
        tp = diagonal[i]

        # Because rows are predictions and columns are true labels:
        fp = row_sums[i] - tp           # predicted as this class, but true label was different
        fn = column_sums[i] - tp        # true label is this class, but predicted differently
        tn = total - tp - fp - fn

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        records.append({
            "class": label,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "support_true": column_sums[i],
            "support_predicted": row_sums[i],
            "accuracy_one_vs_rest": safe_divide(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "F1": f1,
        })

    per_class_metrics = pd.DataFrame(records)

    macro_precision = per_class_metrics["precision"].mean()
    macro_recall = per_class_metrics["recall"].mean()
    macro_f1 = per_class_metrics["F1"].mean()

    weighted_f1 = safe_divide(
        (per_class_metrics["F1"] * per_class_metrics["support_true"]).sum(),
        per_class_metrics["support_true"].sum()
    )

    summary_metrics = pd.DataFrame([{
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_F1": macro_f1,
        "weighted_F1": weighted_f1,
        "total_samples": total,
    }])

    return per_class_metrics, summary_metrics


# ---------------------------------------------------------------------
# Binary reduction: improved vs non-improved
# ---------------------------------------------------------------------
def reduce_confusion_matrix_to_binary(confusion_matrix, label_to_binary):
    """
    Reduce a multiclass confusion matrix to a 2-class confusion matrix.

    Input convention:
        rows    = predicted multiclass labels
        columns = true multiclass labels

    Output convention:
        rows    = predicted binary labels
        columns = true binary labels

    The binary classes are:
        improved
        non-improved
    """
    binary_classes = ["improved", "non-improved"]

    binary_cm = pd.DataFrame(
        0.0,
        index=binary_classes,
        columns=binary_classes,
    )

    # Build mapping from the confusion-matrix labels to binary labels.
    cm_labels = list(confusion_matrix.index)

    missing_labels = []
    cm_label_to_binary = {}

    for label in cm_labels:
        normalized = normalize_label(label)

        if normalized not in label_to_binary:
            missing_labels.append(label)
        else:
            cm_label_to_binary[label] = label_to_binary[normalized]

    if missing_labels:
        raise ValueError(
            "The following confusion-matrix labels were not found in the label map:\n"
            f"{missing_labels}\n\n"
            "Check whether the mapping CSV contains all classes."
        )

    # Sum every original cell into its corresponding binary cell.
    for predicted_label in confusion_matrix.index:
        for true_label in confusion_matrix.columns:
            predicted_binary = cm_label_to_binary[predicted_label]
            true_binary = cm_label_to_binary[true_label]

            binary_cm.loc[predicted_binary, true_binary] += confusion_matrix.loc[
                predicted_label,
                true_label
            ]

    return binary_cm


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load original multiclass confusion matrix.
    cm = load_confusion_matrix(CONFUSION_MATRIX_CSV)

    # 2. Compute metrics for the original multiclass confusion matrix.
    multiclass_per_class_metrics, multiclass_summary_metrics = compute_confusion_matrix_metrics(cm)

    # 3. Load mapping from original labels to improved/non-improved.
    label_to_binary = load_label_map(LABEL_MAP_CSV)

    # 4. Reduce the multiclass confusion matrix to improved/non-improved.
    binary_cm = reduce_confusion_matrix_to_binary(cm, label_to_binary)

    # 5. Compute metrics again for the binary confusion matrix.
    binary_per_class_metrics, binary_summary_metrics = compute_confusion_matrix_metrics(binary_cm)

    # 6. Print results.
    print("\nOriginal multiclass confusion matrix:")
    print(cm)

    print("\nMulticlass per-class metrics:")
    print(multiclass_per_class_metrics)

    print("\nMulticlass summary metrics:")
    print(multiclass_summary_metrics)

    print("\nReduced binary confusion matrix:")
    print(binary_cm)

    print("\nBinary per-class metrics:")
    print(binary_per_class_metrics)

    print("\nBinary summary metrics:")
    print(binary_summary_metrics)

    # 7. Save results.
    multiclass_per_class_metrics.to_csv("output/metrics/multiclass_per_class_metrics.csv", index=False)
    multiclass_summary_metrics.to_csv("output/metrics/multiclass_summary_metrics.csv", index=False)

    binary_cm.to_csv("output/metrics/binary_confusion_matrix.csv")
    binary_per_class_metrics.to_csv("output/metrics/binary_per_class_metrics.csv", index=False)
    binary_summary_metrics.to_csv("output/metrics/binary_summary_metrics.csv", index=False)

    print("\nSaved:")
    print("- multiclass_per_class_metrics.csv")
    print("- multiclass_summary_metrics.csv")
    print("- binary_confusion_matrix.csv")
    print("- binary_per_class_metrics.csv")
    print("- binary_summary_metrics.csv")

