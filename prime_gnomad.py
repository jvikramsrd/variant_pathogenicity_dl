
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.gnomad import load_or_fetch_gene, make_session

panel = json.loads(Path("data/raw/uniprot/expanded_panel.json").read_text())
raw, sess = Path("data/raw"), make_session()
missing = [g for g in panel if not (raw / "gnomad" / f"{g.upper()}_gnomad_v4.csv").exists()]
print(len(missing), "to fetch:", missing)

for i, g in enumerate(missing, 1):
    for attempt in range(1, 7):
        try:
            df = load_or_fetch_gene(g, raw, sequence=panel[g]["sequence"], session=sess)
            print(f"[{i}/{len(missing)}] {g}: {len(df)} rows")
            break
        except Exception as exc:
            wait = 15 * attempt
            print(f"[{i}/{len(missing)}] {g} attempt {attempt}: {exc} -- sleeping {wait}s")
            time.sleep(wait)
    else:
        print(f"[{i}/{len(missing)}] {g}: GAVE UP")
    time.sleep(3)
