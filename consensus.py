#!/usr/bin/env python3
"""
consensus.py -- pick MRM transitions from CONSENSUS across all spectra
======================================================================
Instead of trusting one spectrum, this looks at EVERY spectrum a compound has
in spectra.db and finds the fragments that show up most consistently and most
intensely. A fragment that is strong across many collision energies and
instruments is a far safer MRM transition than one tall in a single record.

For each compound (per polarity) it reports:
  - quantifier + qualifier chosen by a consensus score
  - detection frequency (in what fraction of spectra the fragment appears)
  - mean relative intensity, and how many spectra support it
  - a confidence tier (High / Moderate / Verify-optimize)

USAGE (same filters as search.py):
  python consensus.py --input compounds.csv
  python consensus.py --input compounds.csv --instrument QQQ QTOF
  python consensus.py --input compounds.csv --polarity negative
  python consensus.py --input compounds.csv --instrument ORBITRAP --mz-tol 0.005

OUTPUT:
  consensus_transitions.csv    one row per compound (per polarity)
  consensus_transitions.xlsx   Consensus MRM | Fragment detail | Not found
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys
import pandas as pd

DB_PATH = "spectra.db"
INSTR = ["QQQ","QTOF","ORBITRAP","ION_TRAP","FT_ICR","OTHER","UNKNOWN","ANY"]


def to_skeleton(smiles=None, inchikey=None, name=None):
    if inchikey and isinstance(inchikey, str) and len(inchikey) >= 14:
        return inchikey.strip()[:14]
    if smiles and isinstance(smiles, str) and smiles.strip():
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
            m = Chem.MolFromSmiles(smiles.strip())
            if m:
                return Chem.MolToInchiKey(m)[:14]
        except Exception:
            pass
    if name and isinstance(name, str) and name.strip():
        try:
            import pubchempy as pcp
            h = pcp.get_compounds(name.strip(), "name")
            if h and h[0].inchikey:
                return h[0].inchikey[:14]
        except Exception:
            pass
    return None


def norm_polarity(v):
    """'ESI-POS' / 'Positive' / '+' / 'pos'  ->  'positive'. Unknown -> None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if not s or s in ("nan", "any", "both"):
        return None
    if "neg" in s or s in ("-", "n", "esi-", "esi_neg"):
        return "negative"
    if "pos" in s or s in ("+", "p", "esi+", "esi_pos"):
        return "positive"
    return None


def pick(df, *names):
    """First column present, matched case-insensitively."""
    low = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_queries(path):
    df = pd.read_csv(path)
    c_name = pick(df, "name", "names", "compound", "compounds", "cpd name")
    c_smi = pick(df, "smiles")
    c_ik = pick(df, "inchikey", "inchi_key")
    c_pol = pick(df, "polarity", "ion_mode", "ion mode", "mode")
    out = []
    for _, r in df.iterrows():
        skel = to_skeleton(
            smiles=r.get(c_smi) if c_smi else None,
            inchikey=r.get(c_ik) if c_ik else None,
            name=r.get(c_name) if c_name else None)
        label = None
        for c in (c_name, c_smi, c_ik):
            if c and pd.notna(r.get(c)):
                label = r.get(c)
                break
        smi = r.get(c_smi) if c_smi else None
        out.append({"query": label, "skeleton": skel,
                    "smiles": smi.strip() if isinstance(smi, str) and smi.strip() else None,
                    "want_polarity": norm_polarity(r.get(c_pol)) if c_pol else None})
    return pd.DataFrame(out), (c_pol is not None)


UNIT_TOL = 0.7   # Da: a QQQ quadrupole cannot separate product ions closer than this
PRECURSOR_MATCH_TOL = 0.5   # Da: library precursor -> calculated adduct m/z
ACCURATE_TOL = 0.01         # Da: library precursor recorded at accurate mass


def dedupe_spectra(spectra, min_peaks=3):
    """Drop spectra whose peak list is identical to one already kept.

    The same record is often mirrored between GNPS, MoNA and MassBank, and
    counting it twice inflates detection frequency and confidence. Spectra with
    fewer than `min_peaks` peaks carry too little content to call duplicates and
    are always kept. Returns (unique_spectra, n_removed)."""
    seen, unique = set(), []
    for s in spectra:
        peaks = s["peaks"]
        if len(peaks) >= min_peaks:
            top = max(i for _, i in peaks) or 1.0
            key = tuple(sorted((round(m, 2), round(i / top, 2)) for m, i in peaks))
            if key in seen:
                continue
            seen.add(key)
        unique.append(s)
    return unique, len(spectra) - len(unique)


def precursor_candidates(smiles, negative):
    """Calculated precursor ions [(label, m/z)] for a structure, most common
    first. Labels follow ce_features.ion_composition so the CE model sees the
    same adducts. Empty if the structure cannot be used."""
    if not smiles or not isinstance(smiles, str):
        return []
    try:
        import ce_features as cf
    except ImportError:
        return []
    mol = cf._mol(smiles)
    if mol is None:
        return []
    comp = cf._composition(mol)
    if any(element not in cf.MASS for element in comp):
        return []
    mass = cf._mass(comp)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge:
        if abs(charge) != 1 or (charge < 0) != negative:
            return []
        return [("[M]-" if negative else "[M]+", mass - charge * cf.ELECTRON)]
    if negative:
        return [("[M-H]-", mass - cf.PROTON), ("[M+HCOO]-", mass + 44.998201),
                ("[M+CH3COO]-", mass + 59.013853), ("[M+Cl]-", mass + 34.969402)]
    return [("[M+H]+", mass + cf.PROTON), ("[M+NH4]+", mass + 18.033823),
            ("[M+Na]+", mass + 22.989218)]


def candidate_key(cands):
    """Text key of a precursor candidate list, shared by the public table builder and the web app."""
    return ';'.join(f'{label}@{mz:.3f}' for label, mz in cands)


# Heavy-isotope spacings (13C, 34S, 37Cl, 81Br). A fragment at quantifier + one
# of these is the isotopologue of the quantifier, not a different ion: with Q1
# set on the monoisotopic precursor it cannot form, so it is never a qualifier.
ISOTOPE_SPACINGS = (1.00336, 1.99580, 1.99705, 1.99795)


def _is_isotope_of(mz, base, tol=0.01):
    return any(abs(mz - base - d) <= tol for d in ISOTOPE_SPACINGS)


def pick_quant_qual(frags, unit_tol=UNIT_TOL):
    """Quantifier = best fragment; qualifier = next one Q3 can tell apart that
    is not a heavy-isotope copy of the quantifier."""
    if not frags:
        return None, None
    quant = frags[0]
    qual = next((f for f in frags[1:] if abs(f["fragment_mz"] - quant["fragment_mz"]) >= unit_tol
                 and not _is_isotope_of(f["fragment_mz"], quant["fragment_mz"])), None)
    return quant, qual


def library_consensus(spectra, smiles=None, negative=False, mz_tol=0.01,
                      precursor_guard=5.0, min_freq=0.1, unit_tol=UNIT_TOL):
    """Best-supported precursor group (see library_consensus_groups), or None."""
    return next(library_consensus_groups(spectra, smiles, negative, mz_tol, precursor_guard,
                                         min_freq, unit_tol), None)


def library_consensus_groups(spectra, smiles=None, negative=False, mz_tol=0.01,
                             precursor_guard=5.0, min_freq=0.1, unit_tol=UNIT_TOL):
    """Consensus fragments for one compound in one polarity.

    1. Remove duplicate spectra.
    2. If the structure is known, assign each spectrum to a calculated precursor
       ion (within 0.5 Da) and keep the best-supported one, so e.g. sodium-adduct
       fragments are never mixed with protonated-precursor fragments. The
       reported precursor is the CALCULATED m/z, not the library's rounded one.
       Without a structure, spectra are grouped by library adduct + precursor.
    3. Cluster fragments; merge clusters Q3 cannot separate (unit_tol) and take
       their m/z from accurate-mass spectra where available.
    Yields one dict per precursor group with usable fragments, best-supported
    first (library_consensus takes the first)."""
    spectra, n_dup = dedupe_spectra(spectra)
    cands = precursor_candidates(smiles, negative)
    groups, theo = {}, {}
    if cands:
        order = {label: i for i, (label, _) in enumerate(cands)}
        theo = dict(cands)
        for s in spectra:
            prec = s.get("precursor_mz")
            if not prec:
                continue
            label, mz = min(cands, key=lambda c: abs(c[1] - prec))
            if abs(mz - prec) <= PRECURSOR_MATCH_TOL:
                groups.setdefault(label, []).append(s)
        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), order[kv[0]]))
    else:
        for s in spectra:
            prec = s.get("precursor_mz")
            if not prec:
                continue
            adduct = "".join(str(s.get("adduct") or "").split())
            key = next((k for k in groups if k[0] == adduct and abs(k[1] - prec) <= 0.05), (adduct, prec))
            groups.setdefault(key, []).append(s)
        ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for key, group in ranked:
        precs = [s["precursor_mz"] for s in group]
        if cands:
            precursor, ion, basis = theo[key], key, "calculated from structure"
            accurate = {s["spec_id"] for s in group if abs(s["precursor_mz"] - precursor) <= ACCURATE_TOL}
        else:
            precursor, basis, accurate = statistics.median(precs), "library median", None
            labels = [s["adduct"] for s in group if s.get("adduct")]
            ion = max(set(labels), key=labels.count) if labels else None
        frags, n_total = cluster_fragments(group, mz_tol, precursor_guard, unit_tol, accurate)
        frags = [f for f in frags if f["detection_freq"] >= min_freq] or frags
        if frags:
            yield dict(precursor_mz=round(precursor, 4), precursor_ion=ion, precursor_basis=basis,
                       library_precursor_mz=round(statistics.median(precs), 4), n_spectra=n_total,
                       duplicates_removed=n_dup, n_accurate=len(accurate) if accurate is not None else None,
                       spectra=group, frags=frags)


def cluster_fragments(spectra, mz_tol, precursor_guard, unit_tol=None, accurate_ids=None):
    """spectra: list of dicts {spec_id, precursor_mz, peaks:[(mz,relint)]}.
    Returns ranked consensus fragments across all spectra.

    unit_tol: also merge fragment clusters closer than this (Da). Low-resolution
    library spectra record the same ion up to ~0.25 Da off, which would
    otherwise split one ion into several "different" fragments.
    accurate_ids: spectrum ids whose m/z values are accurate; a merged
    fragment's m/z is taken from these when any contribute."""
    n_total = len(spectra)
    entries = []
    for s in spectra:
        prec = s["precursor_mz"]
        for mz, ri in s["peaks"]:
            if prec and abs(mz - prec) < precursor_guard:
                continue
            entries.append((mz, ri, s["spec_id"]))
    if not entries:
        return [], n_total
    entries.sort(key=lambda x: x[0])

    clusters, cur = [], None
    for mz, ri, sid in entries:
        if (cur is not None
                and abs(mz - cur["wmean"]) <= mz_tol
                and (mz - cur["min_mz"]) <= 2 * mz_tol):
            cur["peaks"].append((mz, ri, sid))
            cur["wsum"] += mz * ri
            cur["isum"] += ri
            cur["wmean"] = cur["wsum"] / cur["isum"]
            cur["min_mz"] = min(cur["min_mz"], mz)
        else:
            if cur:
                clusters.append(cur)
            cur = {"peaks": [(mz, ri, sid)], "wsum": mz * ri, "isum": ri,
                   "wmean": mz, "min_mz": mz}
    if cur:
        clusters.append(cur)

    for c in clusters:
        c["by_spec"] = {}
        for mz, ri, sid in c["peaks"]:
            c["by_spec"][sid] = max(c["by_spec"].get(sid, 0.0), ri)  # one value per spectrum
        c["score"] = sum(c["by_spec"].values()) / n_total
        c["n_merged"] = 1
    if unit_tol:
        merged = []
        for c in sorted(clusters, key=lambda d: d["score"], reverse=True):
            home = next((m for m in merged if abs(c["wmean"] - m["wmean"]) <= unit_tol), None)
            if home is None:
                merged.append(c)
                continue
            home["peaks"] = home["peaks"] + c["peaks"]
            for sid, ri in c["by_spec"].items():
                home["by_spec"][sid] = max(home["by_spec"].get(sid, 0.0), ri)
            home["n_merged"] += 1
        clusters = merged

    out = []
    for c in clusters:
        by_spec = c["by_spec"]
        support = len(by_spec)
        intensities = list(by_spec.values())
        mean_all = sum(intensities) / n_total          # combines freq + intensity
        mean_present = sum(intensities) / support
        mz_value, mz_basis = c["wmean"], "consensus"
        if c["n_merged"] > 1:
            accurate = [(m, r) for m, r, sid in c["peaks"] if accurate_ids and sid in accurate_ids]
            use = accurate or [(m, r) for m, r, _ in c["peaks"]]
            mz_value = sum(m * r for m, r in use) / sum(r for _, r in use)
            mz_basis = "accurate-mass spectra" if accurate else "mixed-resolution average"
        out.append({
            "fragment_mz": round(mz_value, 4),
            "mz_basis": mz_basis,
            "n_merged": c["n_merged"],
            "support": support,
            "n_total": n_total,
            "detection_freq": round(support / n_total, 3),
            "consensus_score": round(mean_all, 4),
            "mean_relint": round(mean_present, 3),
            "median_relint": round(statistics.median(intensities), 3),
        })
    out.sort(key=lambda d: d["consensus_score"], reverse=True)
    return out, n_total


def confidence(n_total, quant_freq):
    if n_total >= 5 and quant_freq >= 0.6:
        return "High"
    if n_total >= 3 and quant_freq >= 0.4:
        return "Moderate"
    return "Verify-optimize"


def fetch(con, skel, args, want_pol=None):
    """want_pol: the polarity to restrict to, or None for all polarities."""
    where, params = ["skeleton = ?"], [skel]
    if args.instrument and "ANY" not in args.instrument:
        where.append("instrument_class IN (%s)" % ",".join("?"*len(args.instrument)))
        params += args.instrument
    if want_pol:
        where.append("polarity = ?")
        params.append(want_pol)
    if args.adduct:
        where.append("REPLACE(LOWER(adduct),' ','') = ?")
        params.append(args.adduct.replace(" ", "").lower())
    if args.ce_min is not None:
        where.append("ce_value >= ?"); params.append(args.ce_min)
    if args.ce_max is not None:
        where.append("ce_value <= ?"); params.append(args.ce_max)
    sql = ("SELECT id, name, precursor_mz, adduct, polarity, instrument_class, "
           "ce_value, peaks_json FROM spectra WHERE " + " AND ".join(where))
    return con.execute(sql, params).fetchall()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--instrument", nargs="+", default=["ANY"], choices=INSTR)
    p.add_argument("--polarity", default="per-compound",
                   choices=["per-compound", "positive", "negative", "any"],
                   help="'per-compound' (default): use each compound's own "
                        "polarity column from --input, so a compound you listed "
                        "as Positive never produces negative-mode transitions. "
                        "'positive'/'negative': force one mode for everything. "
                        "'any': every polarity the library has (old behaviour).")
    p.add_argument("--adduct")
    p.add_argument("--ce-min", type=float)
    p.add_argument("--ce-max", type=float)
    p.add_argument("--mz-tol", type=float, default=0.01,
                   help="m/z tolerance to treat fragments as the same (Da, default 0.01)")
    p.add_argument("--unit-tol", type=float, default=UNIT_TOL,
                   help="merge fragments closer than this, which a QQQ cannot separate; "
                        "also the minimum quantifier-qualifier spacing (Da, default 0.7; 0 = off)")
    p.add_argument("--precursor-guard", type=float, default=5.0)
    p.add_argument("--top-frags", type=int, default=8,
                   help="consensus fragments to list per compound (default 8)")
    p.add_argument("--min-freq", type=float, default=0.1,
                   help="ignore fragments seen in < this fraction of spectra (default 0.1)")
    p.add_argument("--out", default="consensus_transitions")
    args = p.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"{DB_PATH} not found. Run build_index.py first.")

    q, has_pol_col = load_queries(args.input)
    print(f"Queries: {len(q)}   resolved: {q['skeleton'].notna().sum()}")

    if args.polarity == "per-compound":
        if not has_pol_col:
            print(f"  (!) {args.input} has no 'polarity' column -- falling back to "
                  f"'any' (both modes will be reported).")
        else:
            n_blank = q["want_polarity"].isna().sum()
            print(f"  polarity: per-compound from {args.input}"
                  + (f"  ({n_blank} rows blank/unrecognised -> both modes)"
                     if n_blank else ""))
    else:
        print(f"  polarity: {args.polarity} (forced for every compound)")

    con = sqlite3.connect(DB_PATH)

    rows, detail_rows, missing = [], [], []
    n_wrong_pol = 0
    for _, r in q.iterrows():
        skel = r["skeleton"]
        if not skel:
            missing.append({"query": r["query"], "reason": "no InChIKey"})
            continue

        # which polarity do we want for THIS compound?
        if args.polarity == "per-compound":
            want_pol = r["want_polarity"]          # None -> both modes
        elif args.polarity == "any":
            want_pol = None
        else:
            want_pol = args.polarity

        recs = fetch(con, skel, args, want_pol)
        if not recs:
            # distinguish "nothing at all" from "only the other polarity"
            if want_pol:
                other = fetch(con, skel, args, None)
                if other:
                    have = sorted({rec[4] for rec in other if rec[4]})
                    n_wrong_pol += 1
                    missing.append({
                        "query": r["query"],
                        "reason": f"no {want_pol} spectra (library has "
                                  f"{'/'.join(have)} only)"})
                    continue
            missing.append({"query": r["query"], "reason": "no spectra (try CFM-ID)"})
            continue

        # group by polarity so + and - modes are separate transitions
        by_pol = {}
        for _id, name, prec, adduct, pol, instr, ce, pj in recs:
            by_pol.setdefault(pol, []).append(
                {"spec_id": _id, "precursor_mz": prec, "adduct": adduct,
                 "ce": ce, "instr": instr,
                 "peaks": [(m, i) for m, i in json.loads(pj)]})

        disp_name = recs[0][1] or r["query"]
        for pol, spectra in by_pol.items():
            result = library_consensus(spectra, r.get("smiles"), pol == "negative",
                                       args.mz_tol, args.precursor_guard, args.min_freq,
                                       args.unit_tol or None)
            if result is None:
                continue
            frags, n_total = result["frags"], result["n_spectra"]
            ces = [s["ce"] for s in result["spectra"] if s["ce"] is not None]
            ce_lo = min(ces) if ces else None
            ce_hi = max(ces) if ces else None

            quant, qual = pick_quant_qual(frags, args.unit_tol or 0.0)
            conf = confidence(n_total, quant["detection_freq"])

            rows.append({
                "name": disp_name, "inchikey": skel, "polarity": pol,
                "n_spectra": n_total, "duplicates_removed": result["duplicates_removed"],
                "top_adduct": result["precursor_ion"],
                "precursor_mz": result["precursor_mz"],
                "precursor_basis": result["precursor_basis"],
                "library_precursor_mz": result["library_precursor_mz"],
                "quantifier_mz": quant["fragment_mz"],
                "quantifier_freq": quant["detection_freq"],
                "quantifier_mean_relint": quant["mean_relint"],
                "quantifier_support": f"{quant['support']}/{n_total}",
                "qualifier_mz": qual["fragment_mz"] if qual else None,
                "qualifier_freq": qual["detection_freq"] if qual else None,
                "qualifier_support": f"{qual['support']}/{n_total}" if qual else None,
                "ce_range": f"{ce_lo}-{ce_hi}" if ce_lo is not None else "",
                "confidence": conf,
                "top_fragments": "; ".join(
                    f"{f['fragment_mz']}(freq {f['detection_freq']}, int {f['mean_relint']})"
                    for f in frags[:args.top_frags]),
            })
            for rank, f in enumerate(frags[:args.top_frags], 1):
                detail_rows.append({
                    "name": disp_name, "inchikey": skel, "polarity": pol, "rank": rank,
                    "fragment_mz": f["fragment_mz"], "detection_freq": f["detection_freq"],
                    "support": f["support"], "n_spectra": n_total,
                    "consensus_score": f["consensus_score"],
                    "mean_relint": f["mean_relint"], "median_relint": f["median_relint"],
                })
    con.close()

    df = pd.DataFrame(rows)
    dd = pd.DataFrame(detail_rows)
    mm = pd.DataFrame(missing)
    df.to_csv(f"{args.out}.csv", index=False)
    write_excel(f"{args.out}.xlsx", df, dd, mm)
    print(f"\n  {args.out}.csv     {len(df)} consensus transitions")
    print(f"  {args.out}.xlsx    Consensus MRM | Fragment detail | Not found")
    if not df.empty:
        print("\n  confidence tiers:")
        for tier, n in df["confidence"].value_counts().items():
            print(f"     {tier:<16} {n}")
        print("\n  polarity:")
        for pol, n in df["polarity"].value_counts().items():
            print(f"     {pol:<16} {n}")
        # A compound legitimately appears in both modes if you listed it twice,
        # once per polarity. Only flag the ones where polarity was simply missing.
        unset = set(q.loc[q["want_polarity"].isna(), "skeleton"].dropna())
        both = df.groupby("inchikey")["polarity"].nunique()
        n_both = sum(1 for k, v in both.items() if v > 1 and k in unset)
        if n_both:
            print(f"     (!) {n_both} compounds appear in BOTH modes because they "
                  f"have no polarity set in {args.input}.")
    if n_wrong_pol:
        print(f"\n  (!) {n_wrong_pol} compounds have library spectra, but not in the "
              f"polarity you asked for. Listed in the 'Not found' sheet.")


def write_excel(path, mrm, detail, missing):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    thin = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    hf = PatternFill("solid", fgColor="1F4E79")
    hfont = Font(bold=True, color="FFFFFF")
    tier_fill = {"High": PatternFill("solid", fgColor="C6EFCE"),
                 "Moderate": PatternFill("solid", fgColor="FFEB9C"),
                 "Verify-optimize": PatternFill("solid", fgColor="FFC7CE")}

    def sheet(ws, df, wrap=()):
        if df.empty:
            ws["A1"] = "No rows"; return
        ws.append(list(df.columns))
        for c in range(1, len(df.columns)+1):
            cell = ws.cell(row=1, column=c)
            cell.fill, cell.font, cell.border = hf, hfont, border
            cell.alignment = Alignment(horizontal="center", vertical="center")
        conf_idx = (list(df.columns).index("confidence")+1) if "confidence" in df.columns else None
        for _, row in df.iterrows():
            ws.append(["" if pd.isna(v) else v for v in row.tolist()])
            if conf_idx:
                val = ws.cell(row=ws.max_row, column=conf_idx).value
                if val in tier_fill:
                    for c in range(1, len(df.columns)+1):
                        ws.cell(row=ws.max_row, column=c).fill = tier_fill[val]
        for c, col in enumerate(df.columns, 1):
            L = get_column_letter(c)
            if col in wrap:
                ws.column_dimensions[L].width = 70
                for rr in range(2, ws.max_row+1):
                    ws.cell(row=rr, column=c).alignment = Alignment(wrap_text=True)
            else:
                longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
                ws.column_dimensions[L].width = min(max(longest+2, 10), 30)
        for rr in range(1, ws.max_row+1):
            for c in range(1, len(df.columns)+1):
                ws.cell(row=rr, column=c).border = border
        ws.freeze_panes = "A2"

    ws1 = wb.active; ws1.title = "Consensus MRM"; sheet(ws1, mrm, wrap=("top_fragments",))
    sheet(wb.create_sheet("Fragment detail"), detail)
    sheet(wb.create_sheet("Not found"), missing)
    wb.save(path)


if __name__ == "__main__":
    main()