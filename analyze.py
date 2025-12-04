import pandas as pd
import glob
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Find the latest parquet files in data_dump/
data_dir = "data_dump"
essays_files = glob.glob(os.path.join(data_dir, "*_essays.parquet"))
if not essays_files:
    print("No essay parquet files found!")
    exit(1)

# Sort by timestamp (embedded in filename) and pick the latest
latest_essays = sorted(essays_files)[-1]
latest_prompts = latest_essays.replace("_essays.parquet", "_prompts.parquet")
latest_schools = latest_essays.replace("_essays.parquet", "_schools.parquet")

print(f"Loading data from:")
print(f"  Essays: {latest_essays}")
print(f"  Prompts: {latest_prompts}")
print(f"  Schools: {latest_schools}")
print()

# Load the parquet files
essays_df = pd.read_parquet(latest_essays)
prompts_df = pd.read_parquet(latest_prompts)
schools_df = pd.read_parquet(latest_schools)

# Join essays with prompts to get application type
essays_with_prompts = essays_df.merge(
    prompts_df[["prompt_id", "application"]], on="prompt_id", how="left"
)

# Print application type distribution
print("Application type distribution:")
print("=" * 60)
app_counts = (
    essays_with_prompts["application"]
    .fillna("(null/unspecified)")
    .replace("", "(empty string)")
)
app_value_counts = app_counts.value_counts()
for app_type, count in app_value_counts.items():
    print(f"{app_type:30} {count:8} essays")
print("=" * 60)
print()

# Filter for Common App essays only (including unspecified/null applications)
common_app_essays = essays_with_prompts[
    (essays_with_prompts["application"] == "COMMON_APP")
    | (essays_with_prompts["application"] == "COMMON_APP_ASSUMED")
    | (essays_with_prompts["application"].isna())
]

print(f"Found {len(common_app_essays)} Common App essays")
print()

# Explode the school_ids list to count essays per school
# Each essay can be associated with multiple schools
school_counts = Counter()
essays_without_schools = 0
for school_ids in common_app_essays["school_ids"]:
    if school_ids is not None and len(school_ids) > 0:
        for school_id in school_ids:
            school_counts[school_id] += 1
    else:
        essays_without_schools += 1

# Create a mapping from school_id to school_name
school_id_to_name = dict(zip(schools_df["school_id"], schools_df["school_name"]))

# Sort schools by essay count (descending)
sorted_schools = sorted(school_counts.items(), key=lambda x: x[1], reverse=True)

# Print results
print("Schools with the most Common App essays:")
print("=" * 60)
for school_id, count in sorted_schools:
    school_name = school_id_to_name.get(school_id, f"Unknown (ID: {school_id})")
    print(f"{school_name:50} {count:5} essays")

print("=" * 60)
print(f"Total schools: {len(sorted_schools)}")
print(f"Total essay-school associations: {sum(school_counts.values())}")
print(f"Essays without school associations: {essays_without_schools}")
print(f"  (These are typically UC/UCAS-only applications or unspecified)")
print()

# Calculate total essay scores (sum of all esslo_ columns)
esslo_columns = [
    "esslo_writing",
    "esslo_detail",
    "esslo_voice",
    "esslo_character",
    "esslo_iv",
    "esslo_contribution",
    "esslo_why_us",
    "esslo_motivation",
    "esslo_academic",
    "esslo_experiences",
    "esslo_reflection",
]

# Calculate total score for each essay
common_app_essays["total_score"] = common_app_essays[esslo_columns].sum(axis=1)

# Get top 10 schools
top_10_schools = sorted_schools[:10]

# Collect score distributions for top 10 schools
school_scores = {}
for school_id, count in top_10_schools:
    # Find all essays associated with this school
    school_essays = common_app_essays[
        common_app_essays["school_ids"].apply(
            lambda ids: ids is not None and school_id in ids
        )
    ]
    # Get scores (drop NaN values)
    scores = school_essays["total_score"].dropna()
    school_name = school_id_to_name.get(school_id, f"Unknown (ID: {school_id})")
    school_scores[school_name] = scores

# Create visualization
print("Creating score distribution plots for top 10 schools...")
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
fig.suptitle(
    "Essay Score Distributions for Top 10 Schools (Sum of all esslo_ scores)",
    fontsize=16,
    fontweight="bold",
)

axes = axes.flatten()

for idx, (school_name, scores) in enumerate(school_scores.items()):
    ax = axes[idx]

    # Plot histogram
    ax.hist(scores, bins=30, edgecolor="black", alpha=0.7, color="steelblue")

    # Add statistics
    mean_score = scores.mean()
    median_score = scores.median()
    std_score = scores.std()

    ax.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.2f}",
    )
    ax.axvline(
        median_score,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_score:.2f}",
    )

    ax.set_title(
        f"{school_name}\n(n={len(scores)} essays)", fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Total Essay Score", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = f"μ={mean_score:.2f}\nσ={std_score:.2f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=8,
    )

plt.tight_layout()
plt.savefig("top_10_schools_score_distributions.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'top_10_schools_score_distributions.png'")
plt.show()

# Print summary statistics
print()
print("Summary Statistics for Top 10 Schools:")
print("=" * 80)
print(f"{'School':<50} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("=" * 80)
for school_name, scores in school_scores.items():
    print(
        f"{school_name:<50} {len(scores):>6} {scores.mean():>8.2f} {scores.median():>8.2f} {scores.std():>8.2f}"
    )
print("=" * 80)

# Calculate average scores for all schools with >= 10 essays
print()
print("All Schools Ordered by Average Score (minimum 10 essays):")
print("=" * 80)

school_avg_scores = []
for school_id, count in sorted_schools:
    if count >= 10:  # Filter schools with at least 10 essays
        # Find all essays associated with this school
        school_essays = common_app_essays[
            common_app_essays["school_ids"].apply(
                lambda ids: ids is not None and school_id in ids
            )
        ]
        # Get scores (drop NaN values)
        scores = school_essays["total_score"].dropna()

        if len(scores) >= 10:  # Double-check we have at least 10 valid scores
            school_name = school_id_to_name.get(school_id, f"Unknown (ID: {school_id})")
            avg_score = scores.mean()
            school_avg_scores.append(
                (school_name, len(scores), avg_score, scores.std())
            )

# Sort by average score (descending)
school_avg_scores.sort(key=lambda x: x[2], reverse=True)

print(f"{'Rank':<5} {'School':<50} {'Count':>6} {'Avg Score':>10} {'Std Dev':>10}")
print("=" * 80)
for rank, (school_name, count, avg_score, std_dev) in enumerate(school_avg_scores, 1):
    print(f"{rank:<5} {school_name:<50} {count:>6} {avg_score:>10.2f} {std_dev:>10.2f}")
print("=" * 80)
print(f"Total schools with ≥10 essays: {len(school_avg_scores)}")
