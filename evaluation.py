import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import os

# Read command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <input_file> <prefix>")
    sys.exit(1)

input_file = sys.argv[1]  # Input data file path
prefix = sys.argv[2]      # Prefix for output files

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The input file {input_file} does not exist.")
    sys.exit(1)

# Read the data
df = pd.read_csv(input_file, sep="\t", header=None, names=["Sequence", "True Label", "Predicted Probability"])

# Extract true labels and predicted probabilities
y_true = df["True Label"]
y_pred_prob = df["Predicted Probability"]

# Convert true labels to integer type to ensure consistency with predicted labels
y_true = y_true.astype(int)

# Convert probabilities to binary predictions using threshold 0.5
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate main evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_pred_prob)

# Calculate confusion matrix components: TP, FN, FP, TN
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Print results
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUC": auc_score,
    "True Positives (TP)": tp,
    "False Positives (FP)": fp,
    "False Negatives (FN)": fn,
    "True Negatives (TN)": tn
}

# Write the metrics to a text file
metrics_file = f"{prefix}_metrics.txt"
with open(metrics_file, "w") as f:
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

print(f"Metrics have been saved to {metrics_file}.")

# Compute ROC curve and AUC
fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Save ROC curve data to a CSV file
roc_data = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr,
    "Threshold": thresholds_roc
})
roc_data_file = f"{prefix}_roc_curve_data.csv"
roc_data.to_csv(roc_data_file, index=False)
print(f"ROC curve data has been saved to {roc_data_file}.")

# Compute Precision-Recall curve and AUC
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
pr_auc = auc(recall_vals, precision_vals)

# Ensure lengths match by trimming the extra value from precision_vals and recall_vals
precision_vals = precision_vals[:-1]
recall_vals = recall_vals[:-1]

pr_data = pd.DataFrame({
    "Recall": recall_vals,
    "Precision": precision_vals,
    "Threshold": thresholds_pr
})
pr_data_file = f"{prefix}_pr_curve_data.csv"
pr_data.to_csv(pr_data_file, index=False)
print(f"Precision-Recall curve data has been saved to {pr_data_file}.")

# Plot ROC and PR curves
plt.figure(figsize=(12, 6))

# ROC curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

# PR curve
plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, color="b", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")

# Save the plot as a PDF with 600 DPI resolution
roc_pr_plot_file = f"{prefix}_roc_pr_curve.jpg"
plt.tight_layout()
plt.savefig(roc_pr_plot_file, dpi=600)
plt.close()

print(f"ROC and PR curve plot has been saved to {roc_pr_plot_file}.")

print("Metrics, ROC/PR curve data, and plots have been saved successfully.")
