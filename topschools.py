import pandas as pd
import glob
import os
from collections import Counter

# Find the latest parquet files in data_dump/
data_dir = "data_dump"
essays_files = glob.glob(os.path.join(data_dir, "*_essays.parquet"))
if not essays_files:
    print("No essay parquet files found!")
    exit(1)

# Sort by timestamp (embedded in filename) and pick the latest
latest_essays = sorted(essays_files)[-1]
latest_schools = latest_essays.replace("_essays.parquet", "_schools.parquet")

print(f"Loading data from:")
print(f"  Essays: {latest_essays}")
print(f"  Schools: {latest_schools}")
print()

# Load the parquet files
essays_df = pd.read_parquet(latest_essays)
schools_df = pd.read_parquet(latest_schools)

# Count essays per school by exploding school_ids lists
# Each essay can be associated with multiple schools
school_counts = Counter()
for school_ids in essays_df["school_ids"]:
    if school_ids is not None and len(school_ids) > 0:
        for school_id in school_ids:
            school_counts[school_id] += 1

# Create a mapping from school_id to school_name
school_id_to_name = dict(zip(schools_df["school_id"], schools_df["school_name"]))

# Sort schools by essay count (descending)
sorted_schools = sorted(school_counts.items(), key=lambda x: x[1], reverse=True)

# Print results with ranking
print("Schools ranked by number of essays:")
print("=" * 70)
print(f"{'Rank':<6} {'School Name':<50} {'Essays':>8}")
print("=" * 70)

for rank, (school_id, count) in enumerate(sorted_schools, 1):
    school_name = school_id_to_name.get(school_id, f"Unknown (ID: {school_id})")
    print(f"{rank:<6} {school_name:<50} {count:>8}")

print("=" * 70)
print(f"Total schools: {len(sorted_schools)}")
print(f"Total essay-school associations: {sum(school_counts.values())}")
