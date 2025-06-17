import os
import pandas as pd
import matplotlib.pyplot as plt

# Increase default font sizes
plt.rcParams.update({'font.size': 14})

CHANNEL_ORIENTATION = "all"

# List of subjects with summary CSV files
subjects = [0, 1, 2, 3, 4]
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Added a fifth color

plt.figure(figsize=(10, 6))

for subj, color in zip(subjects, colors):
    csv_summary = f"PA_pvalues/without_overlap_{CHANNEL_ORIENTATION}ch/subject_{subj}_control_replacement_summary.csv"
    if not os.path.exists(csv_summary):
        print(f"File not found: {csv_summary}")
        continue

    df = pd.read_csv(csv_summary)
    
    # Ensure the relevant columns exist
    x = df["CONTROL_REPLACEMENT_FRAC"]
    y_mean = df["smallest_combined_p_mean"]
    y_std = df["smallest_combined_p_std"]
    
    plt.plot(x, y_mean, marker="o", color=color, label=f"Subject {subj}")
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2, linewidth=0)

# Add a dotted horizontal line at y = 0.05 with a label for significance level
plt.axhline(y=0.05, color='black', linestyle=':', linewidth=3, label='Significance level (p=0.05)')

# Add thick lines at x=0 and y=0
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.xlabel("Control Replacement Fraction", fontsize=16)
plt.ylabel("Mean Combined p-value", fontsize=16)
if CHANNEL_ORIENTATION == "inv":
    plt.title("PA-Classifier: p-value vs control replacement %", fontsize=18)
elif CHANNEL_ORIENTATION == "same":
    plt.title("Same-side channels: p-value vs control replacement %", fontsize=18)
elif CHANNEL_ORIENTATION == "all":
    plt.title("All channels: p-value vs control replacement %", fontsize=18)

plt.xlim(-0.03, 0.47)
plt.ylim(-0.03, 0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Save the plot instead of showing it
plt.savefig(f"p-values_vs_controlfrac_{CHANNEL_ORIENTATION}.png")
plt.close()
print(f"Plot saved as p-values_vs_controlfrac_{CHANNEL_ORIENTATION}.png at {os.getcwd()}")
