import os
import pandas as pd
import matplotlib.pyplot as plt

# List of subjects with summary CSV files
subjects = [0, 1, 2, 3, 4]
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Added a fifth color

plt.figure(figsize=(10, 6))

for subj, color in zip(subjects, colors):
    csv_summary = f"subject_{subj}_control_replacement_summary.csv"
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

# Add a dotted horizontal line at y = 0.05
plt.axhline(y=0.05, color='black', linestyle=':', linewidth=1)

# Add thick lines at x=0 and y=0
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.xlabel("Control Replacement Fraction")
plt.ylabel("Mean Combined p-value")
plt.title("Mean Combined p-value vs Control Replacement Fraction per Subject")
plt.xlim(-0.03, 0.6)
plt.ylim(-0.03, 0.4)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()