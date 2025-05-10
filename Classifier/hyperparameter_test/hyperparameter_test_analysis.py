import os
import pandas as pd
import numpy as np

# Define the CSV file path relative to this script.
csv_filepath = os.path.join(os.path.dirname(__file__), "hyper_results.csv")

if not os.path.exists(csv_filepath):
    print(f"CSV file not found at: {csv_filepath}")
    exit(1)

# Load CSV file
df = pd.read_csv(csv_filepath)
print(f"Loaded {len(df)} experiment results from: {csv_filepath}")

# ------------------------------
# Aggregation and reporting across all test subjects
# ------------------------------
# Average results over seeds and test subjects for identical hyperparameter settings (excluding 'seed' and 'test_idx')
group_cols = ['hbr_multiplier', 'hbr_shift', 'window_length', 'add_hbr']
grouped = df.groupby(group_cols)['accuracy'].mean().reset_index()
grouped.rename(columns={'accuracy': 'avg_accuracy'}, inplace=True)

# Sort to get top 5 hyperparameter configurations (by average accuracy)
sorted_results = grouped.sort_values(by='avg_accuracy', ascending=False)
top_5 = sorted_results.head(5)

print("\nTop 5 hyperparameter configurations across all test subjects (averaged over seeds and test subjects):")
for index, row in top_5.iterrows():
    print(f"hbr_multiplier={row['hbr_multiplier']}, hbr_shift={row['hbr_shift']}, " +
          f"window_length={row['window_length']}, add_hbr={row['add_hbr']} -> Average Val Accuracy: {row['avg_accuracy']*100:.2f}%")

# Full summary of aggregated results
# print("\nFull aggregated results (each row corresponds to one hyperparameter configuration across all test subjects):")
# print(grouped.to_string(index=False))

# Filter the original data for patients 4 and 1
filtered_df = df[df["test_idx"].isin([2])]

# Aggregate results over seeds and test subjects for these patients
filtered_grouped = filtered_df.groupby(group_cols)["accuracy"].mean().reset_index()
filtered_grouped.rename(columns={'accuracy': 'avg_accuracy'}, inplace=True)

# Filter aggregated data for window_length of 15, 20 and 30
results_15 = filtered_grouped[filtered_grouped["window_length"] == 15]
results_20 = filtered_grouped[filtered_grouped["window_length"] == 20]
results_30 = filtered_grouped[filtered_grouped["window_length"] == 30]

# Compute the overall average accuracy for window_length 15, 20 and 30
avg_15 = results_15["avg_accuracy"].mean()
avg_20 = results_20["avg_accuracy"].mean()
avg_30 = results_30["avg_accuracy"].mean()

print("\nComparison of average accuracy by window_length for patients 4 and 1:")
print(f"Window Length 15 -> Average Accuracy: {avg_15*100:.2f}% ({len(results_15)} configurations)")
print(f"Window Length 20 -> Average Accuracy: {avg_20*100:.2f}% ({len(results_20)} configurations)")
print(f"Window Length 30 -> Average Accuracy: {avg_30*100:.2f}% ({len(results_30)} configurations)")

import matplotlib.pyplot as plt

# Create a figure for the plot.
plt.figure(figsize=(8, 6))

# Plot the average accuracy (in percentage) against each window length.
plt.scatter(filtered_grouped["window_length"], filtered_grouped["avg_accuracy"] * 100, color='blue', label='Configurations')
plt.xlabel("Window Length")
plt.ylabel("Average Accuracy (%)")
plt.title("Average Accuracy vs Window Length for Patients 4 and 1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Plot for hbr_multiplier and hbr_shift
plt.figure(figsize=(12, 5))

# Separate data based on the add_hbr flag
df_true = filtered_grouped[filtered_grouped["add_hbr"] == True]
df_false = filtered_grouped[filtered_grouped["add_hbr"] == False]

# Plot for hbr_multiplier
plt.subplot(1, 2, 1)
plt.scatter(df_true["hbr_multiplier"], df_true["avg_accuracy"] * 100, color='green', label='add_hbr True')
plt.scatter(df_false["hbr_multiplier"], df_false["avg_accuracy"] * 100, color='red', label='add_hbr False')
plt.xlabel("HBR Multiplier")
plt.ylabel("Average Accuracy (%)")
plt.title("Average Accuracy vs HBR Multiplier")
plt.grid(True)
plt.legend()

# Plot for hbr_shift
plt.subplot(1, 2, 2)
plt.scatter(df_true["hbr_shift"], df_true["avg_accuracy"] * 100, color='green', label='add_hbr True')
plt.scatter(df_false["hbr_shift"], df_false["avg_accuracy"] * 100, color='red', label='add_hbr False')
plt.xlabel("HBR Shift")
plt.title("Average Accuracy vs HBR Shift")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Aggregate average accuracy by window_length and add_hbr flag
line_data = filtered_grouped.groupby(["window_length", "add_hbr"])["avg_accuracy"].mean().reset_index()

# Pivot the data so that each add_hbr flag becomes a separate column
pivot_data = line_data.pivot(index="window_length", columns="add_hbr", values="avg_accuracy")

plt.figure(figsize=(8, 6))
if True in pivot_data.columns:
    plt.plot(pivot_data.index, pivot_data[True] * 100, marker="o", label="add_hbr True")
if False in pivot_data.columns:
    plt.plot(pivot_data.index, pivot_data[False] * 100, marker="o", label="add_hbr False")
plt.xlabel("Window Length")
plt.ylabel("Average Accuracy (%)")
plt.title("Comparison of Average Accuracy by Window Length for add_hbr True vs False")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()