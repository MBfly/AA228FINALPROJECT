import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the latest parquet files
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
latest_schools = latest_essays.replace("_essays.parquet", "_schools.parquet")

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

# Join essays with prompts to get application type
essays_with_prompts = essays_df.merge(
    prompts_df[["prompt_id", "application"]], on="prompt_id", how="left"
)

# Filter for Common App essays only (including COMMON_APP_ASSUMED which lack prompt_id)
common_app_essays = essays_with_prompts[
    (essays_with_prompts["application"] == "COMMON_APP")
    | (essays_with_prompts["application"] == "COMMON_APP_ASSUMED")
].copy()

print(
    f"Common App essays found (COMMON_APP + COMMON_APP_ASSUMED): {len(common_app_essays)}"
)
print(f"  - COMMON_APP: {(common_app_essays['application'] == 'COMMON_APP').sum()}")
print(
    f"  - COMMON_APP_ASSUMED: {(common_app_essays['application'] == 'COMMON_APP_ASSUMED').sum()}"
)

# Filter out essays where last_modified is more than 3 days after created_date
max_edit_window_days = 3
common_app_essays["days_to_modify"] = (
    common_app_essays["last_modified"] - common_app_essays["created_date"]
).dt.days

essays_before_edit_filter = len(common_app_essays)
common_app_essays = common_app_essays[
    common_app_essays["days_to_modify"] <= max_edit_window_days
].copy()

essays_filtered_by_edit = essays_before_edit_filter - len(common_app_essays)
print(
    f"Filtered out essays with >3 days between created and modified: {essays_filtered_by_edit} essays"
)
print(f"Common App essays remaining: {len(common_app_essays)}")
print()

# Filter for essays between 600 and 650 words
min_word_count = 600
max_word_count = 650
common_app_essays = common_app_essays[
    (common_app_essays["word_count"] >= min_word_count)
    & (common_app_essays["word_count"] <= max_word_count)
].copy()

print(
    f"Common App essays with {min_word_count}-{max_word_count} words: {len(common_app_essays)}"
)
print()

# Step 2: Calculate per-user baselines
print("Step 2: Calculating per-user baselines...")
print("=" * 60)

# For each user, find their first Common App essay's created_date date
user_first_dates = common_app_essays.groupby("author_id")["created_date"].min()

print(f"Number of unique users: {len(user_first_dates)}")
print()

# Calculate days_since_first for each essay
common_app_essays["user_first_date"] = common_app_essays["author_id"].map(
    user_first_dates
)
common_app_essays["days_since_first"] = (
    common_app_essays["created_date"] - common_app_essays["user_first_date"]
).dt.days

# Step 3: Calculate average esslo_ score for each essay
print("Step 3: Calculating average scores...")
print("=" * 60)

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

# Calculate mean of all esslo_ columns for each essay (ignoring NaN values)
common_app_essays["avg_esslo_score"] = common_app_essays[esslo_columns].mean(axis=1)

# Filter out essays without scores
essays_with_scores = common_app_essays[
    common_app_essays["avg_esslo_score"].notna()
].copy()

print(f"Essays with scores: {len(essays_with_scores)}")
print(
    f"Essays without scores (excluded): {len(common_app_essays) - len(essays_with_scores)}"
)
print()

# Calculate improvement relative to each user's first essay
print("Calculating improvement relative to first essay...")
# Get each user's first essay score
user_first_scores = (
    essays_with_scores.loc[
        essays_with_scores.groupby("author_id")["created_date"].idxmin()
    ][["author_id", "avg_esslo_score"]]
    .set_index("author_id")["avg_esslo_score"]
    .to_dict()
)

# Map first scores to all essays
essays_with_scores["first_essay_score"] = essays_with_scores["author_id"].map(
    user_first_scores
)

# Calculate improvement (current score - first score)
essays_with_scores["score_improvement"] = (
    essays_with_scores["avg_esslo_score"] - essays_with_scores["first_essay_score"]
)

print(f"Calculated baseline scores for {len(user_first_scores)} users")
print()

# Remove outliers using IQR method (excluding first essays which are always 0)
print("Removing outliers using IQR method...")
# Calculate outlier bounds based only on non-first essays
non_first_essays = essays_with_scores[essays_with_scores["days_since_first"] > 0]

if len(non_first_essays) > 0:
    q1 = non_first_essays["score_improvement"].quantile(0.25)
    q3 = non_first_essays["score_improvement"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    essays_before_outlier_removal = len(essays_with_scores)

    # Keep all first essays (day 0) + non-first essays within bounds
    essays_with_scores = essays_with_scores[
        (essays_with_scores["days_since_first"] == 0)
        | (
            (essays_with_scores["score_improvement"] >= lower_bound)
            & (essays_with_scores["score_improvement"] <= upper_bound)
        )
    ].copy()

    outliers_removed = essays_before_outlier_removal - len(essays_with_scores)
    print(
        f"Outliers removed: {outliers_removed} essays ({100*outliers_removed/essays_before_outlier_removal:.1f}%)"
    )
    print(f"Essays remaining: {len(essays_with_scores)}")
    print(
        f"Improvement range for non-first essays: [{lower_bound:.3f}, {upper_bound:.3f}]"
    )
else:
    print("No non-first essays found - skipping outlier removal")
print()

# Step 4: Aggregate across users
print("Step 4: Aggregating statistics by days since first essay...")
print("=" * 60)

# Group by days_since_first and calculate statistics including unique user count
time_stats = (
    essays_with_scores.groupby("days_since_first")
    .agg(
        median=("score_improvement", "median"),
        q1=("score_improvement", lambda x: x.quantile(0.25)),
        q3=("score_improvement", lambda x: x.quantile(0.75)),
        count=("score_improvement", "count"),
        unique_users=("author_id", "nunique"),
    )
    .reset_index()
)

print(f"Time points (days) before filtering: {len(time_stats)}")
print(
    f"Days range before filtering: {time_stats['days_since_first'].min()} to {time_stats['days_since_first'].max()}"
)
print()

# Filter to only include time points with at least 15 unique users
min_users = 15
time_stats_filtered = time_stats[time_stats["unique_users"] >= min_users].copy()

print(
    f"Time points (days) after filtering (>={min_users} users): {len(time_stats_filtered)}"
)
if len(time_stats_filtered) > 0:
    print(
        f"Days range after filtering: {time_stats_filtered['days_since_first'].min()} to {time_stats_filtered['days_since_first'].max()}"
    )
    print(
        f"Users range: {time_stats_filtered['unique_users'].min()} to {time_stats_filtered['unique_users'].max()}"
    )
else:
    print("WARNING: No time points have at least 15 unique users!")
print()

# Display first few rows of statistics
print("Sample statistics (filtered):")
print(time_stats_filtered.head(10))
print()

# Step 5: Create visualization
print("Step 5: Creating visualization...")
print("=" * 60)

if len(time_stats_filtered) == 0:
    print("ERROR: Cannot create plot - no data points with sufficient users!")
    exit(1)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot median line
ax.plot(
    time_stats_filtered["days_since_first"],
    time_stats_filtered["median"],
    color="steelblue",
    linewidth=2,
    label="Median",
    zorder=3,
)

# Fill between Q1 and Q3
ax.fill_between(
    time_stats_filtered["days_since_first"],
    time_stats_filtered["q1"],
    time_stats_filtered["q3"],
    alpha=0.3,
    color="steelblue",
    label="25th-75th Percentile",
    zorder=2,
)

# Labels and title
ax.set_xlabel("Days Since First Common App Essay", fontsize=12, fontweight="bold")
ax.set_ylabel(
    "Average Score Improvement\n(Relative to First Essay)",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    "Essay Score Improvement Over Time\n(Common App Essays 600-650 words, Aggregated Across All Users, Min 15 Users per Time Point)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

# Add horizontal line at y=0 (baseline - first essay)
ax.axhline(
    y=0,
    color="black",
    linestyle="-",
    linewidth=1,
    alpha=0.5,
    zorder=2,
    label="Baseline (First Essay)",
)

# Add grid
ax.grid(True, alpha=0.3, linestyle="--", zorder=1)

# Legend
ax.legend(fontsize=10, loc="best")

# Add text box with summary statistics
summary_text = (
    f"Users: {len(user_first_dates)}\n"
    f"Essays: {len(essays_with_scores)}\n"
    f"Days: {time_stats_filtered['days_since_first'].min()}-{time_stats_filtered['days_since_first'].max()}\n"
    f"Min users/point: {min_users}"
)
ax.text(
    0.02,
    0.98,
    summary_text,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    fontsize=9,
)

plt.tight_layout()
plt.savefig("score_improvement_analysis.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'score_improvement_analysis.png'")

# Create second plot with individual user lines
print()
print("Creating plot with individual user trajectories...")
print("=" * 60)

# Filter users who have at least 2 essays for meaningful trajectories
user_essay_counts = essays_with_scores.groupby("author_id").size()
users_with_multiple_essays = user_essay_counts[user_essay_counts >= 2].index

# Filter to users with multiple essays and within our filtered time range
min_day = time_stats_filtered["days_since_first"].min()
max_day = time_stats_filtered["days_since_first"].max()

essays_for_user_plot = essays_with_scores[
    (essays_with_scores["author_id"].isin(users_with_multiple_essays))
    & (essays_with_scores["days_since_first"] >= min_day)
    & (essays_with_scores["days_since_first"] <= max_day)
].copy()

print(
    f"Users with 2+ essays in time range: {essays_for_user_plot['author_id'].nunique()}"
)
print(f"Total essays for user trajectories: {len(essays_for_user_plot)}")

fig2, ax2 = plt.subplots(figsize=(14, 8))

# Plot individual user trajectories with transparency
for author_id in essays_for_user_plot["author_id"].unique():
    user_data = essays_for_user_plot[
        essays_for_user_plot["author_id"] == author_id
    ].sort_values("days_since_first")
    ax2.plot(
        user_data["days_since_first"],
        user_data["score_improvement"],
        color="lightgray",
        alpha=0.4,
        linewidth=0.7,
        zorder=1,
    )

# Overlay the aggregate statistics on top
ax2.plot(
    time_stats_filtered["days_since_first"],
    time_stats_filtered["median"],
    color="darkblue",
    linewidth=3,
    label="Median (All Users)",
    zorder=3,
)

ax2.fill_between(
    time_stats_filtered["days_since_first"],
    time_stats_filtered["q1"],
    time_stats_filtered["q3"],
    alpha=0.4,
    color="steelblue",
    label="25th-75th Percentile",
    zorder=2,
)

# Labels and title
ax2.set_xlabel("Days Since First Common App Essay", fontsize=12, fontweight="bold")
ax2.set_ylabel(
    "Score Improvement\n(Relative to First Essay)", fontsize=12, fontweight="bold"
)
ax2.set_title(
    "Individual User Essay Score Improvement Trajectories Over Time\n(Common App Essays 600-650 words with Aggregate Statistics)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

# Add horizontal line at y=0 (baseline - first essay)
ax2.axhline(
    y=0,
    color="black",
    linestyle="-",
    linewidth=1,
    alpha=0.5,
    zorder=2,
    label="Baseline (First Essay)",
)

# Add grid
ax2.grid(True, alpha=0.3, linestyle="--", zorder=0)

# Legend
ax2.legend(fontsize=10, loc="best")

# Add text box with summary statistics
user_count = essays_for_user_plot["author_id"].nunique()
summary_text2 = (
    f"Users shown: {user_count}\n"
    f"Essays: {len(essays_for_user_plot)}\n"
    f"Days: {min_day}-{max_day}\n"
    f"Min essays/user: 2"
)
ax2.text(
    0.02,
    0.98,
    summary_text2,
    transform=ax2.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    fontsize=9,
)

plt.tight_layout()
plt.savefig(
    "score_improvement_analysis_individual_users.png", dpi=300, bbox_inches="tight"
)
print("Plot saved as 'score_improvement_analysis_individual_users.png'")

# Step 6: Print summary statistics
print()
print("Step 6: Summary Statistics")
print("=" * 60)
print(f"Total users analyzed: {len(user_first_dates)}")
print(f"Total essays analyzed: {len(essays_with_scores)}")
print(
    f"Time range (filtered): {time_stats_filtered['days_since_first'].min()} to {time_stats_filtered['days_since_first'].max()} days"
)
print(f"Minimum users per time point: {min_users}")
print()

# Calculate overall improvement trend using filtered data
min_day = time_stats_filtered["days_since_first"].min()
max_day = time_stats_filtered["days_since_first"].max()

# Use the first 7 days of filtered data for "beginning"
first_week_end = min(min_day + 7, max_day)
first_week_improvement = essays_with_scores[
    (essays_with_scores["days_since_first"] >= min_day)
    & (essays_with_scores["days_since_first"] <= first_week_end)
]["score_improvement"].median()

# Use the last 7 days of filtered data for "end"
last_week_start = max(max_day - 7, min_day)
last_week_improvement = essays_with_scores[
    (essays_with_scores["days_since_first"] >= last_week_start)
    & (essays_with_scores["days_since_first"] <= max_day)
]["score_improvement"].median()

if pd.notna(first_week_improvement) and pd.notna(last_week_improvement):
    additional_improvement = last_week_improvement - first_week_improvement
    print(
        f"Median improvement at beginning (days {min_day}-{first_week_end}): {first_week_improvement:.3f}"
    )
    print(
        f"Median improvement at end (days {last_week_start}-{max_day}): {last_week_improvement:.3f}"
    )
    print(f"Additional improvement from beginning to end: {additional_improvement:.3f}")
else:
    print("Not enough data to calculate improvement trend")

print()
print("Analysis complete!")
print("=" * 60)

plt.show()
