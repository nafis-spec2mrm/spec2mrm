#!/usr/bin/env python3
"""
export_masshunter.py -- write a MassHunter dMRM import file
===========================================================
Turns your final transition table into a CSV with the EXACT columns Agilent
MassHunter expects for a dynamic MRM (dMRM) method:

    Cpd Name, Prec Ion, Prod Ion, Frag (V), CE (V), Polarity[, Ret Time (min)]

CE comes from your transition file (measured or model-predicted, from
predict_ce.py). Fragmentor voltage is carried through if present, else set to a
sensible per-polarity default from your instrument data (optimize on the 6470).
Retention time can be carried from the calibrated RT model. It is specific to
the two LC methods recorded in the RT model bundle and must not be treated as
portable to a different column or gradient without recalibration.

Input : a transitions CSV with name, precursor, product, polarity, and a CE
        column (e.g. transitions_with_CE.csv from predict_ce.py).
Output: masshunter_dMRM.csv   (import into MassHunter)

Run:
    python export_masshunter.py --input transitions_with_CE.csv
    python export_masshunter.py --input transitions_with_CE.csv --add-rt
"""
import argparse
import sys
import pandas as pd

FRAG_DEFAULT = {"Positive": 125, "Negative": 110}   # median from Agilent data


def find_col(cols, *keys):
    low = {c.strip().lower(): c for c in cols}
    for k in keys:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def load_any(path):
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin-1", engine="python")


def norm_polarity(v):
    s = str(v).strip().lower()
    if s.startswith("neg") or s in ("-", "n"):
        return "Negative"
    return "Positive"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--add-rt", action="store_true",
                    help="include predicted/measured retention time when present")
    ap.add_argument("--out", default="masshunter_dMRM.csv")
    args = ap.parse_args()

    df = load_any(args.input)
    cols = df.columns
    c_name = find_col(cols, "cpd name", "name", "compound")
    c_prec = find_col(cols, "prec")
    c_prod = find_col(cols, "prod")
    c_ce = find_col(cols, "collision_energy", "ce (", "ce_ev", "optimal_ce", "ce")
    c_frag = find_col(cols, "frag")
    c_pol = find_col(cols, "polar")
    # Prefer the production prediction explicitly. A broad 'rt' search could
    # otherwise select metadata such as rt_method_id.
    c_rt = next((c for c in cols if c.strip().lower() == "predicted_rt_min"), None)
    if c_rt is None:
        c_rt = find_col(cols, "ret time", "retention time")

    if not (c_prec and c_prod and c_ce):
        sys.exit(f"Input must have precursor, product, and CE columns. "
                 f"Found: {list(cols)}")

    out = pd.DataFrame()
    out["Cpd Name"] = df[c_name] if c_name else ""
    out["Prec Ion"] = pd.to_numeric(df[c_prec], errors="coerce").round(1)
    out["Prod Ion"] = pd.to_numeric(df[c_prod], errors="coerce").round(1)
    pol = df[c_pol].map(norm_polarity) if c_pol else "Positive"
    out["Polarity_tmp"] = pol

    # fragmentor: carry through if present, else per-polarity default
    if c_frag:
        frag = pd.to_numeric(df[c_frag], errors="coerce")
        frag = frag.fillna(out["Polarity_tmp"].map(FRAG_DEFAULT))
    else:
        frag = out["Polarity_tmp"].map(FRAG_DEFAULT)
    out["Frag (V)"] = frag.round().astype("Int64")

    out["CE (V)"] = pd.to_numeric(df[c_ce], errors="coerce").round().astype("Int64")
    out["Polarity"] = out["Polarity_tmp"]
    out = out.drop(columns=["Polarity_tmp"])

    if args.add_rt:
        out["Ret Time (min)"] = (pd.to_numeric(df[c_rt], errors="coerce")
                                 if c_rt else "")

    # drop rows missing essential numbers
    before = len(out)
    out = out.dropna(subset=["Prec Ion", "Prod Ion", "CE (V)"])
    dropped = before - len(out)

    # order columns exactly as MassHunter wants
    order = ["Cpd Name", "Prec Ion", "Prod Ion", "Frag (V)", "CE (V)", "Polarity"]
    if args.add_rt:
        order.append("Ret Time (min)")
    out = out[order]
    out.to_csv(args.out, index=False)

    print(f"Wrote {args.out}: {len(out)} transitions "
          f"({out['Cpd Name'].nunique()} compounds)")
    if dropped:
        print(f"  dropped {dropped} rows missing precursor/product/CE")
    print(f"  positive: {(out['Polarity']=='Positive').sum()}   "
          f"negative: {(out['Polarity']=='Negative').sum()}")
    if args.add_rt:
        print("\nRT-centre reference export for MassHunter. Establish acquisition windows before using scheduled dMRM.")
    else:
        print("\nImport into MassHunter as an unscheduled MRM method (no RT scheduling fields).")
    print("Reminder: predicted CE and RT are STARTING values. Verify them on the "
          "target instrument; RT is valid only for the calibrated LC method.")


if __name__ == "__main__":
    main()
