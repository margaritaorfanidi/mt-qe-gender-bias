import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--fixes", default="../data/train_data/gambit_gendered_fixes_all.csv")
parser.add_argument("--parquet", default="../data/train_data/gambit_all_sources_all_targets_cometkiwi.parquet")
parser.add_argument("--output", default="../data/train_data/gambit_all_sources_all_targets_cometkiwi_gendered.parquet")
args = parser.parse_args()

fixes = pd.read_csv(args.fixes)
gambit = pd.read_parquet(args.parquet)

mapping = dict(zip(fixes["old_source"], fixes["new_source"]))
changed = gambit["source"].isin(mapping)
gambit.loc[changed, "source"] = gambit.loc[changed, "source"].map(mapping)
gambit["changed"] = changed

gambit.to_parquet(args.output, index=False)

print("Changed per lang pair:", changed.sum() / gambit.loc[changed, "lang_pair"].nunique())
print("Rows in fixes:", len(fixes))
print(gambit.loc[changed, "lang_pair"].value_counts())