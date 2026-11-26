#!/usr/bin/env python3
"""
predict_ce.py -- assign collision energy
========================================
Two modes, chosen with --ce-mode:

  model           (DEFAULT) every transition gets a MODEL-PREDICTED CE, even if
                  the same transition already exists in Agilent_CE.csv. Use this
                  whenever you are BENCHMARKING the tool against experimental
                  data -- copying measured CE into the output would make the
                  comparison circular, because those same transitions are what
                  the model was trained on.

  measured-first  optional legacy mode that substitutes an exact Agilent
                  measurement when available. The production pipeline does not
                  use this mode; it uses CatBoost predictions for every row.

Either way, a `measured_ce_if_known` column records the measured value when one
exists, so you can compare model vs measured side by side without re-running.

PRODUCTION MODEL:
  CatBoost uses precursor/product/loss m/z, polarity, fragmentor voltage,
  adduct, molecular descriptors, formula counts, neutral-loss features, and a
  256-bit Morgan fingerprint. ce_applicability flags transitions whose supplied
  structure does not explain the precursor under the supported adduct rules.

  The older random-forest structure/mass-only bundles remain loadable so past
  analyses are reproducible, but run_pipeline.py selects CatBoost.

NOTE ON CIRCULARITY: --ce-mode model removes the lookup, but the model was still
TRAINED on Agilent_CE.csv. Compounds in that training set will still be
predicted better than genuinely novel ones. For a fully clean external
comparison you need a held-out model -- see validate_ce_model.py.

Input : a transitions CSV with precursor_mz, product_mz, polarity
        (e.g. final_transitions.csv)
        ce_catboost_model.joblib (from train_ce_catboost.py)
        Agilent_CE.csv           (your measured data)
Output: transitions_with_CE.csv  (adds collision_energy_eV + ce_source)

Run:
    python predict_ce.py --input final_transitions.csv
    python predict_ce.py --input final_transitions.csv --ce-mode measured-first
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


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


def build_measured_lookup(path, mz_tol):
    """dict: rounded (prec, prod, pol_neg) -> measured CE, for fast matching."""
    df = load_any(path)
    c_prec = find_col(df.columns, "prec")
    c_prod = find_col(df.columns, "prod")
    c_ce = find_col(df.columns, "ce (", "collision", "ce")
    c_pol = find_col(df.columns, "polar")
    lookup = []
    for _, r in df.iterrows():
        try:
            prec = float(r[c_prec]); prod = float(r[c_prod]); ce = float(r[c_ce])
        except (ValueError, TypeError):
            continue
        pol_neg = (str(r[c_pol]).lower().startswith("neg")) if c_pol else False
        lookup.append((prec, prod, pol_neg, ce))
    return lookup


def measured_ce(lookup, prec, prod, pol_neg, mz_tol):
    best, bestd = None, mz_tol * 2 + 1
    for lp, ld, lpol, lce in lookup:
        if lpol != pol_neg:
            continue
        d = abs(lp - prec) + abs(ld - prod)
        if abs(lp - prec) <= mz_tol and abs(ld - prod) <= mz_tol and d < bestd:
            best, bestd = lce, d
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="transitions CSV")
    ap.add_argument("--model", default="ce_catboost_model.joblib")
    ap.add_argument("--measured", default="Agilent_CE.csv")
    ap.add_argument("--mz-tol", type=float, default=0.5,
                    help="m/z tolerance to match a measured transition (default 0.5)")
    ap.add_argument("--ce-mode", choices=["model", "measured-first"], default="model",
                    help="'model' (default): predict every CE, even where a "
                         "measured value exists -- use for benchmarking. "
                         "'measured-first': use real measured CE where available "
                         "-- use for building an acquisition method.")
    ap.add_argument("--structures", nargs="+",
                    default=["compounds.csv", "cfmid_id_map_all.csv"],
                    help="compound files with name + smiles, used to look up each "
                         "transition's structure (default: compounds.csv, "
                         "cfmid_id_map_all.csv)")
    ap.add_argument("--out", default="transitions_with_CE.csv")
    args = ap.parse_args()

    import joblib
    if not os.path.exists(args.model):
        sys.exit(f"{args.model} not found. Run train_ce_model.py first.")
    bundle = joblib.load(args.model)

    # Production v3 bundle: the selected CatBoost model and the exact feature
    # schema used in the grouped benchmark.  Legacy RF bundles remain supported
    # below so older analyses stay reproducible.
    if bundle.get("model_type") == "catboost_ce":
        from ce_catboost_features import build_prediction_matrix

        df = load_any(args.input)
        matrix, feature_audit = build_prediction_matrix(
            df, bundle["features"],
            [p for p in args.structures if os.path.exists(p)],
            bundle["fragmentor_defaults"],
        )
        predicted = np.asarray(bundle["model"].predict(matrix), dtype=float)
        c_prec = find_col(df.columns, "precursor", "prec")
        c_prod = find_col(df.columns, "product", "prod")
        c_pol = find_col(df.columns, "polar")
        lookup = build_measured_lookup(args.measured, args.mz_tol) if os.path.exists(args.measured) else []
        measured, final, sources = [], [], []
        for row_number, (_, row) in enumerate(df.iterrows()):
            prec, prod = float(row[c_prec]), float(row[c_prod])
            negative = str(row[c_pol]).lower().startswith("neg") if c_pol else False
            known = measured_ce(lookup, prec, prod, negative, args.mz_tol)
            measured.append(round(known, 1) if known is not None else None)
            if known is not None and args.ce_mode == "measured-first":
                final.append(round(known, 1)); sources.append("measured (Agilent)")
            else:
                final.append(round(float(predicted[row_number]), 1)); sources.append("predicted (CatBoost)")
        df["collision_energy_eV"] = final
        df["ce_source"] = sources
        df["ce_model"] = "CatBoost"
        for column in feature_audit.columns:
            df[column] = feature_audit[column].to_numpy()
        df["ce_applicability"] = np.where(
            feature_audit["ce_structure_consistent"].to_numpy(bool),
            "High (structure matches precursor)",
            "Low (structure/precursor mismatch; verify CE)",
        )
        df["measured_ce_if_known"] = measured
        df.to_csv(args.out, index=False)
        nested = bundle.get("nested_grouped_validation", {})
        r2 = nested.get("R2", nested.get("R2_mean", "not recorded"))
        mae = nested.get("MAE_eV", "not recorded")
        print(f"Wrote {args.out}: {len(df)} transitions [CatBoost; --ce-mode {args.ce_mode}]")
        print(f"  model-predicted: {sum(s.startswith('predicted') for s in sources)}")
        print(f"  measured used:   {sum(s.startswith('measured') for s in sources)}")
        print(f"  SMILES found:    {int(feature_audit.ce_smiles_found.sum())}/{len(df)}")
        print(f"  structure/Q1 consistent: {int(feature_audit.ce_structure_consistent.sum())}/{len(df)}")
        print(f"  nested grouped validation: R2={r2}, MAE={mae} eV")
        print("Predicted CE values are starting values and require instrument verification.")
        return

    # version 2 bundles hold a STRUCTURE model plus a MASS-ONLY fallback;
    # an older bundle has only the mass-only model
    full = bundle.get("full") if bundle.get("version", 1) >= 2 else None
    fallback = (bundle["fallback"] if bundle.get("version", 1) >= 2 else
                {"model": bundle["model"], "features": bundle["features"],
                 "cv_mae": bundle["cv_mae"]})
    structures = {}
    if full:
        import ce_features as cf
        structures, used = cf.load_structures(
            [p for p in args.structures if os.path.exists(p)])
        print(f"  structure model active (trained {bundle.get('trained', '?')}, "
              f"n={full['n_train']}, CV MAE ~{full['cv_mae']:.1f} eV)")
        print("  structures from: " + ", ".join(f"{p} ({n})" for p, n in used))
    else:
        print("  (!) model file has no structure model -- using mass-only. "
              "Re-run train_ce_model.py to get one.")

    lookup = build_measured_lookup(args.measured, args.mz_tol) \
        if os.path.exists(args.measured) else []
    if not lookup:
        print(f"  (!) no measured data at {args.measured}; all CE will be predicted.")

    df = load_any(args.input)
    c_prec = find_col(df.columns, "precursor", "prec")
    c_prod = find_col(df.columns, "product", "prod")
    c_pol = find_col(df.columns, "polar")
    c_name = next((c for c in df.columns if c.strip().lower()
                   in ("name", "cpd name", "compound")), None)
    c_smi = next((c for c in df.columns if c.strip().lower() == "smiles"), None)
    if not (c_prec and c_prod):
        sys.exit(f"Input needs precursor & product columns; found {list(df.columns)}")

    ces, sources, known, which, lossf = [], [], [], [], []
    n_meas = n_full = n_fb = 0
    no_struct, inconsistent = set(), set()
    for _, r in df.iterrows():
        try:
            prec = float(r[c_prec]); prod = float(r[c_prod])
        except (ValueError, TypeError):
            ces.append(None); sources.append("no m/z"); known.append(None)
            which.append(""); lossf.append(""); continue
        pol_neg = (str(r[c_pol]).lower().startswith("neg")) if c_pol else False

        # always record the measured value if we have one, for side-by-side comparison
        m = measured_ce(lookup, prec, prod, pol_neg, args.mz_tol)
        known.append(round(m, 1) if m is not None else None)

        if m is not None and args.ce_mode == "measured-first":
            ces.append(round(m, 1)); sources.append("measured (Agilent)")
            which.append("measured"); lossf.append(""); n_meas += 1
            continue

        # STRUCTURE model when the SMILES is known and explains this precursor
        use_full = False
        if full:
            import ce_features as cf
            smi = None
            if c_smi and pd.notna(r.get(c_smi)) and str(r[c_smi]).strip():
                smi = str(r[c_smi]).strip()
            elif c_name and pd.notna(r.get(c_name)):
                smi = cf.smiles_for(str(r[c_name]), structures)
            feat, info = cf.build_row(prec, prod, pol_neg, smi)
            nm = str(r[c_name]) if c_name else "?"
            if not info["has_structure"]:
                no_struct.add(nm)
            elif not info["consistent"]:
                inconsistent.add(nm)
            else:
                x = np.array([[feat[f] for f in full["features"]]], float)
                pred = float(full["model"].predict(x)[0])
                ces.append(round(pred, 1))
                sources.append(f"predicted (structure model, ~{full['cv_mae']:.0f} eV err)")
                which.append("structure"); lossf.append(info["loss_formula"])
                n_full += 1
                use_full = True

        if not use_full:
            x = np.array([[prec, prod, prec - prod, int(pol_neg)]], float)
            pred = float(fallback["model"].predict(x)[0])
            ces.append(round(pred, 1))
            sources.append(f"predicted (mass-only model, ~{fallback['cv_mae']:.0f} eV err)")
            which.append("mass-only"); lossf.append(""); n_fb += 1

    df["collision_energy_eV"] = ces
    df["ce_source"] = sources
    df["ce_model"] = which
    df["neutral_loss_formula"] = lossf
    df["measured_ce_if_known"] = known
    df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}: {len(df)} transitions   [--ce-mode {args.ce_mode}]")
    print(f"  measured (used real Agilent CE) : {n_meas}")
    print(f"  predicted, STRUCTURE model      : {n_full}")
    print(f"  predicted, MASS-ONLY fallback   : {n_fb}")
    if no_struct:
        print(f"    {len(no_struct)} compounds had no SMILES: "
              f"{', '.join(sorted(no_struct)[:6])}{' ...' if len(no_struct) > 6 else ''}")
    if inconsistent:
        print(f"    {len(inconsistent)} compounds' SMILES don't match their precursor: "
              f"{', '.join(sorted(inconsistent)[:6])}{' ...' if len(inconsistent) > 6 else ''}")

    # where a measured value exists, show how far the model actually is from it
    d = df.dropna(subset=["measured_ce_if_known"])
    if len(d):
        err = (pd.to_numeric(d["collision_energy_eV"], errors="coerce")
               - pd.to_numeric(d["measured_ce_if_known"], errors="coerce")).abs().dropna()
        print(f"\n  {len(d)} transitions also exist in {args.measured} "
              f"(kept in 'measured_ce_if_known' for comparison).")
        if args.ce_mode == "model" and len(err):
            print(f"  Model vs measured on those: MAE {err.mean():.1f} eV, "
                  f"median {err.median():.1f} eV, "
                  f"{100*(err <= 10).mean():.0f}% within +/-10 eV.")
            print("  (!) Optimistic -- these transitions were in the model's "
                  "TRAINING data. Not an external validation.")

    if args.ce_mode == "model":
        print("\nAll CE model-predicted (no measured values copied in) -- "
              "suitable for benchmarking.")
        print("For an acquisition method, re-run with --ce-mode measured-first.")
    print("\nPredicted values are STARTING points -- optimize on the 6470.")


if __name__ == "__main__":
    main()
