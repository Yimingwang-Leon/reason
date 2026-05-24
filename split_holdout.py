"""Stratified 80/20 split. Locked: do NOT re-run after committing."""
import pandas as pd

SEED = 42
df = pd.read_csv("data/train_cat.csv")
# Stratified per category: take 20% per group as holdout
holdout = df.groupby("cat", group_keys=False).sample(frac=0.2, random_state=SEED)
train = df.drop(holdout.index)
holdout.sort_values("id").to_csv("data/holdout.csv", index=False)
train.sort_values("id").to_csv("data/train_split.csv", index=False)
print(f"train_split: {len(train)} rows | holdout: {len(holdout)} rows")
print("\nholdout per-category:")
print(holdout["cat"].value_counts().sort_index())
