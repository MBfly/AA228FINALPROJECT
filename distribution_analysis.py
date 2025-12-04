import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the parquet data
print("Step 1: Loading data...")
print("=" * 60)

data_dir = "data_dump"
essays_files = glob.glob(os.path.join(data_dir, "*_essays.parquet"))
if not essays_files:
    print("No essay parquet files found!")
    exit(1)

# Sort by timestamp and pick the latest
latest_essays = sorted(essays_files)[-1]
latest_prompts = latest_essays.replace("_essays.parquet", "_prompts.parquet")

print(f"Loading data from:")
print(f"  Essays: {latest_essays}")
print(f"  Prompts: {latest_prompts}")
print()

# Load the parquet files
essays_df = pd.read_parquet(latest_essays)
prompts_df = pd.read_parquet(latest_prompts)

print(f"Total essays loaded: {len(essays_df)}")
print(f"Total prompts loaded: {len(prompts_df)}")
print()

# Step 2: Join essays with prompts and filter for Common App essays
print("Step 2: Application type distribution...")
print("=" * 60)

# Join essays with prompts to get application type
essays_with_prompts = essays_df.merge(
    prompts_df[["prompt_id", "application"]], on="prompt_id", how="left"
)

# Print application type distribution
app_counts = (
    essays_with_prompts["application"]
    .fillna("(null/unspecified)")
    .replace("", "(empty string)")
)
app_value_counts = app_counts.value_counts()
print("Essays by application type:")
for app_type, count in app_value_counts.items():
    print(f"  {app_type:30} {count:8,} essays")
print()

print("Filtering for Common App essays (including null/unspecified)...")
# Filter for Common App essays (including COMMON_APP_ASSUMED and null/unspecified)
common_app_essays = essays_with_prompts[
    (essays_with_prompts["application"] == "COMMON_APP")
    | (essays_with_prompts["application"] == "COMMON_APP_ASSUMED")
    | (essays_with_prompts["application"].isna())
    | (essays_with_prompts["application"] == "")
].copy()

print(
    f"Common App essays found (COMMON_APP + COMMON_APP_ASSUMED + null/unspecified): {len(common_app_essays)}"
)
print(f"  - COMMON_APP: {(common_app_essays['application'] == 'COMMON_APP').sum()}")
print(
    f"  - COMMON_APP_ASSUMED: {(common_app_essays['application'] == 'COMMON_APP_ASSUMED').sum()}"
)
print(
    f"  - null/unspecified: {(common_app_essays['application'].isna() | (common_app_essays['application'] == '')).sum()}"
)
print()

# Filter for word count between 600 and 650
min_word_count = 600
max_word_count = 650
essays_before_word_filter = len(common_app_essays)
common_app_essays = common_app_essays[
    (common_app_essays["word_count"] >= min_word_count)
    & (common_app_essays["word_count"] <= max_word_count)
].copy()
essays_filtered_by_words = essays_before_word_filter - len(common_app_essays)

print(f"Filtering for word count {min_word_count}-{max_word_count}...")
print(f"Essays after word count filter: {len(common_app_essays)}")
print(f"Essays excluded by word count: {essays_filtered_by_words}")
print()

# Step 3: Calculate average esslo_ scores
print("Step 3: Calculating average esslo_ scores...")
print("=" * 60)

# Define the 11 esslo_ columns
esslo_columns = [
    "esslo_writing",
    "esslo_detail",
    "esslo_voice",
    "esslo_character",
    "esslo_iv",
    "esslo_contribution",
]

# Replace 0 values with NaN for esslo columns
for col in esslo_columns:
    common_app_essays[col] = common_app_essays[col].replace(0, np.nan)

# Calculate row-wise mean ignoring NaN values
common_app_essays["avg_esslo_score"] = common_app_essays[esslo_columns].mean(
    axis=1, skipna=True
)

# Filter out essays with no valid average score (all columns were null/zero)
essays_before_filter = len(common_app_essays)
essays_with_scores = common_app_essays[
    common_app_essays["avg_esslo_score"].notna()
].copy()
essays_excluded = essays_before_filter - len(essays_with_scores)

print(f"Essays with valid scores: {len(essays_with_scores)}")
print(f"Essays excluded (all null/zero scores): {essays_excluded}")
print()

# Step 4: Create histogram visualization
print("Step 4: Creating histogram visualization...")
print("=" * 60)

# Calculate statistics
n_essays = len(essays_with_scores)
mean_score = essays_with_scores["avg_esslo_score"].mean()
std_score = essays_with_scores["avg_esslo_score"].std()

print(f"Number of essays: {n_essays}")
print(f"Mean score: {mean_score:.4f}")
print(f"Standard deviation: {std_score:.4f}")
print()

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot histogram
ax.hist(
    essays_with_scores["avg_esslo_score"],
    bins=50,
    edgecolor="black",
    alpha=0.7,
    color="steelblue",
)

# Labels and title
ax.set_xlabel("Average Esslo Score", fontsize=12, fontweight="bold")
ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax.set_title(
    "Distribution of Average Esslo Scores for Common App Essays",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

# Add grid
ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

# Add text box with statistics
stats_text = (
    f"Number of essays: {n_essays:,}\n"
    f"Mean: {mean_score:.4f}\n"
    f"Std Dev: {std_score:.4f}"
)
ax.text(
    0.98,
    0.98,
    stats_text,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    fontsize=11,
    fontweight="bold",
)

plt.tight_layout()
output_filename = "common_app_score_distribution.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight")
print(f"Plot saved as '{output_filename}'")

# Step 5: Print summary
print()
print("Step 5: Summary")
print("=" * 60)
print(f"Total Common App essays analyzed: {n_essays:,}")
print(f"Mean average esslo score: {mean_score:.4f}")
print(f"Standard deviation: {std_score:.4f}")
print(f"Min score: {essays_with_scores['avg_esslo_score'].min():.4f}")
print(f"Max score: {essays_with_scores['avg_esslo_score'].max():.4f}")
print(f"Median score: {essays_with_scores['avg_esslo_score'].median():.4f}")
print("=" * 60)
print()
print("Analysis complete!")

plt.show()
