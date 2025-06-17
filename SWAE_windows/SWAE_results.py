#!/usr/bin/env python
"""
extract_swae_log_and_binom.py
--------------------------------
Parse a SWAE run log and compute, for each subject and control replacement fraction,
the smallest p‑value from Tapping_Left and Tapping_Right tests.
Also generates a plot of the p‑value vs control replacement fraction with a separate
line for each subject (using the same style as plot.py).
If the CSV file already exists, it is loaded directly.
Edit LOGFILE and OUTFILE below, then Run ▶ in VS Code.
"""
import os
import re
import csv
from collections import defaultdict
from scipy.stats import binomtest
import numpy as np
import matplotlib.pyplot as plt

# Increase default font sizes similar to plot.py
plt.rcParams.update({'font.size': 14})

# ─── USER SETTINGS ────────────────────────────────────────────────
LOGFILE  = r"SWAE_windows/RESULTS_SWAE_ctr_frac_run.log"    # ← change this if needed
OUTFILE  = r"SWAE_windows/binom_results.csv"                  # ← change this if needed
P_NULL   = 0.5                                              # H0 success probability
# ──────────────────────────────────────────────────────────────────

LOG_RE = re.compile(
    r"Subject (\d+), frac ([0-9.]+), (Tapping_Left|Tapping_Right), "
    r"inner seed (\d+): (True|False)"
)

def read_log(path):
    """
    Return a nested dict data[subj][frac][comparison] -> list[bool]
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            subj, frac, comp, seed, flag = m.groups()
            data[int(subj)][float(frac)][comp].append(flag == "True")
    return data

def summarise(data, p0=P_NULL):
    """
    Iterate through the nested dict and yield summary rows,
    using the smallest p_value from Tapping_Left and Tapping_Right for each subject and fraction.
    """
    for subj in sorted(data):
        for frac in sorted(data[subj]):
            best = None
            # Evaluate both comparisons and select the one with the smallest p_value.
            for comp in ("Tapping_Left", "Tapping_Right"):
                if comp not in data[subj][frac]:
                    continue
                passed = data[subj][frac][comp]
                k = sum(passed)
                n = len(passed)
                proportion = k / n
                p_val = binomtest(k, n, p=p0, alternative="greater").pvalue
                if (best is None) or (p_val < best["p_value"]):
                    best = {
                        "subject": subj,
                        "fraction": frac,
                        "comparison": "min",  # no differentiation between left/right
                        "passed": k,
                        "total": n,
                        "proportion": f"{proportion:.3f}",
                        "p_value": p_val,
                    }
            if best is not None:
                # Format p_value for CSV output
                best["p_value"] = f"{best['p_value']:.3g}"
                yield best

def write_csv(rows, outfile):
    fieldnames = [
        "subject", "fraction", "comparison",
        "passed", "total", "proportion", "p_value",
    ]
    with open(outfile, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames)
        wr.writeheader()
        wr.writerows(rows)

if __name__ == "__main__":
    if os.path.exists(OUTFILE):
        print(f"CSV file {OUTFILE} exists, loading it and recomputing p-values...")
        with open(OUTFILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                k = int(row["passed"])
                n = int(row["total"])
                p_val = binomtest(k*2, n*2, p=P_NULL, alternative="greater").pvalue
                row["p_value"] = f"{p_val:.3g}"
                rows.append(row)
    else:
        data = read_log(LOGFILE)
        rows = list(summarise(data))
        write_csv(rows, OUTFILE)
        print(f"Wrote {len(rows)} rows to {OUTFILE}")

    # Print p-values for control replacement fraction (frac) 0 for each subject.
    print("\nP-values for control fraction 0:")
    for row in rows:
        if float(row["fraction"]) == 0:
            print(f"Subject {row['subject']}: p-value {row['p_value']}")

    # Group results per subject (each subject has a line)
    from collections import defaultdict
    subject_data = defaultdict(lambda: {"fractions": [], "p_values": [], "p_std": []})
    for row in rows:
        subj = int(row["subject"])
        frac = float(row["fraction"])
        p_val = float(row["p_value"])
        total = int(row["total"])
        # Recover the observed proportion (note: it was stored as a formatted string)
        proportion = float(row["proportion"])
        # Compute a basic uncertainty (standard error) estimator for a binomial proportion.
        p_std = (proportion * (1 - proportion) / (total)) ** 0.5
        subject_data[subj]["fractions"].append(frac)
        subject_data[subj]["p_values"].append(p_val)
        subject_data[subj]["p_std"].append(p_std)

    subjects = sorted(subject_data.keys())
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    while len(colors) < len(subjects):
        colors += colors

    plt.figure(figsize=(10, 6))
    for subj, color in zip(subjects, colors):
        fractions = np.array(subject_data[subj]["fractions"])
        p_vals = np.array(subject_data[subj]["p_values"])
        p_stds = np.array(subject_data[subj]["p_std"])
        order = np.argsort(fractions)
        fractions = fractions[order]
        p_vals = p_vals[order]
        p_stds = p_stds[order]
        plt.plot(fractions, p_vals, marker="o", color=color, label=f"Subject {subj}")
        plt.fill_between(fractions, p_vals - p_stds, p_vals + p_stds, color=color, alpha=0.2, linewidth=0)

    plt.axhline(y=0.05, color='black', linestyle=':', linewidth=3, label='Significance level (p=0.05)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    plt.xlabel("Control Replacement Fraction", fontsize=16)
    plt.ylabel("Combined p-value", fontsize=16)
    plt.title("SWAE: p-value vs control replacement %", fontsize=18)
    plt.xlim(-0.03, 0.47)
    plt.ylim(-0.03, 0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    plot_filename = "swae_pvalues_vs_controlfrac.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved as {plot_filename} at {os.getcwd()}")