"""
Μαζεύει όλα τα gambit_gendered_sources_*.csv από τους φακέλους 1..32
και φτιάχνει ένα ενιαίο CSV μόνο με τις γραμμές που έχουν status == "fix",
με το παλιό source και το καινούργιο gendered_source.

Χρήση:
    python merge_gendered_fixes.py --root <root_folder> [--output out.csv] [--start 1] [--end 32]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Ενώνει τις γραμμές με status=fix από όλα τα gambit_gendered_sources CSV."
)
parser.add_argument("--root", type=Path, default=Path("."),
                    help="Φάκελος που περιέχει τους υποφακέλους 1..32 (default: τρέχων φάκελος)")
parser.add_argument("--output", type=Path, default=None,
                    help="Αρχείο εξόδου (default: ένα επίπεδο πάνω από το root, π.χ. train_data/gambit_gendered_fixes_all.csv)")
parser.add_argument("--start", type=int, default=1, help="Πρώτος φάκελος (default: 1)")
parser.add_argument("--end", type=int, default=32, help="Τελευταίος φάκελος (default: 32)")
args = parser.parse_args()

root = args.root
out_path = args.output or root.resolve().parent / "gambit_gendered_fixes_all.csv"

frames = []
missing = []

for folder_num in range(args.start, args.end + 1):
    folder = root / str(folder_num)
    if not folder.is_dir():
        missing.append(folder_num)
        continue

    # ψάχνει και σε υποφακέλους, για σιγουριά
    files = sorted(folder.rglob("*gendered_sources*.csv"))
    if not files:
        missing.append(folder_num)
        continue

    for f in files:
        # utf-8-sig: αφαιρεί το BOM αν υπάρχει στο header
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        fixed = df[df["status"].astype(str).str.strip().str.lower() == "fix"].copy()
        fixed = fixed.rename(columns={"source": "old_source", "gendered_source": "new_source"})
        fixed.insert(0, "folder", folder_num)
        fixed.insert(1, "file", f.name)
        fixed.insert(2, "row_in_file", fixed.index)

        frames.append(fixed[["folder", "file", "row_in_file", "gender", "old_source", "new_source"]])
        print(f"[{folder_num:>2}] {f.name}: {len(df)} γραμμές, {len(fixed)} fix")

if not frames:
    sys.exit("Δεν βρέθηκε κανένα gendered_sources CSV.")

result = pd.concat(frames, ignore_index=True)
result.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\nΣύνολο: {len(result)} γραμμές με fix -> {out_path}")
print(result["gender"].value_counts().to_string())
if missing:
    print(f"\nΠροσοχή: δεν βρέθηκε αρχείο στους φακέλους: {missing}")