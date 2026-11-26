#!/usr/bin/env python3
"""
ce_features.py -- features for the collision-energy model
=========================================================
Shared by train_ce_model.py and predict_ce.py so that training and prediction
compute features in exactly the same way. Not run on its own.

Three feature groups:

  MASS (4)       what the original model used
                   prec, prod, loss (= prec - prod), neg (1 = negative mode)

  STRUCTURE (14) computed with RDKit from the compound's SMILES, after
                 stripping counter-ions and waters
                   nF nS nN nO nCl       element counts
                   frac_F                fluorine atoms / heavy atoms
                   mw                    molecular weight
                   logp                  Crippen logP (hydrophobicity)
                   tpsa                  topological polar surface area
                   rotb                  rotatable bonds (flexibility)
                   rings                 ring count
                   arom                  aromatic atoms
                   hba                   hydrogen-bond acceptors
                   charge                net formal charge (+1 for a QAC)

  LOSS (12)      the elemental formula of the neutral that was lost
                   l_C l_H l_O l_F l_N l_S l_Cl l_Br l_P    atom counts
                   l_rdbe    rings + double bonds in the lost neutral
                   l_heavy   non-hydrogen atoms in the lost neutral
                   loss_assigned   1 if a formula was found, else 0

HOW THE LOSS FORMULA IS FOUND
  1. Work out the precursor ION's composition from the SMILES and the adduct.
     The adduct is inferred by matching the measured precursor m/z against
     [M+H]+, [M+NH4]+, [M+Na]+, [M-H]-, [M+HCOO]-, [M+Cl]-, [M-H-CO2]-, or
     [M]+ for an already-charged molecule. If NOTHING matches, the SMILES and
     the measurement describe different molecules, and the row is flagged
     'inconsistent' so its (wrong) descriptors are never used.
  2. Neutral-loss mass = precursor m/z - product m/z.
  3. Search every sub-formula of the precursor ion (you cannot lose atoms the
     ion doesn't have) for one whose exact mass matches the loss within the
     tolerance, rejecting chemically impossible formulas (negative RDBE).
  4. Keep the closest match; on a tie, the formula with fewer atoms.

  With nominal-mass product ions (one decimal place, like Agilent_CE.csv) more
  than one formula can fit, so the assignment is sometimes ambiguous. Accurate
  product masses make it unique -- that is the reason to record them.
"""
import re

import numpy as np
import pandas as pd

MASS = {"C": 12.0, "H": 1.00782503207, "D": 2.01410177812, "N": 14.0030740048,
        "O": 15.99491461956, "F": 18.99840322, "P": 30.97376163,
        "S": 31.97207100, "Cl": 34.96885268, "Br": 78.9183371,
        "I": 126.904473, "Si": 27.9769265325, "Na": 22.9897692809}
PROTON = 1.007276466812
ELECTRON = 0.000548579909

FEATURES_MASS = ["prec", "prod", "loss", "neg"]
FEATURES_STRUCT = ["nF", "nS", "nN", "nO", "nCl", "frac_F", "mw", "logp",
                   "tpsa", "rotb", "rings", "arom", "hba", "charge"]
FEATURES_LOSS = ["l_C", "l_H", "l_O", "l_F", "l_N", "l_S", "l_Cl", "l_Br",
                 "l_P", "l_rdbe", "l_heavy", "loss_assigned"]
FEATURES_FULL = FEATURES_MASS + FEATURES_STRUCT + FEATURES_LOSS

LOSS_TOL = 0.35          # Da; matches nominal-mass product ions


# ------------------------------------------------------------------ names
def name_key(s):
    """Normalise a compound name for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def name_key_nosuffix(s):
    """Same, with a trailing -1 / -2 dropped ('Monensin-1' -> 'monensin')."""
    return name_key(re.sub(r"[\s_-]+\d+\s*$", "", str(s).strip()))


def load_structures(paths):
    """Read SMILES from one or more compound files.

    Returns {name_key: smiles}. The first file listed wins on conflicts, so put
    the most trusted file first.
    """
    out, used = {}, []
    for p in paths:
        try:
            d = None
            for enc in ("utf-8-sig", "latin-1", "cp1252"):
                try:
                    d = pd.read_csv(p, encoding=enc)
                    break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
            if d is None:
                continue
        except Exception:
            continue
        nm = next((c for c in d.columns if c.strip().lower()
                   in ("name", "names", "compound", "cpd name")), None)
        sm = next((c for c in d.columns if c.strip().lower() == "smiles"), None)
        if not (nm and sm):
            continue
        n0 = len(out)
        for n, s in zip(d[nm], d[sm]):
            if pd.notna(n) and pd.notna(s) and str(s).strip():
                out.setdefault(name_key(n), str(s).strip())
        used.append((p, len(out) - n0))
    return out, used


def smiles_for(name, structures):
    """SMILES for a compound name, trying an exact key then a suffix-less key."""
    s = structures.get(name_key(name))
    if s is None:
        s = structures.get(name_key_nosuffix(name))
    return s


# ------------------------------------------------------------------ RDKit
_cache = {}


def _mol(smiles):
    """Parsed, salt-stripped molecule (cached)."""
    if smiles in _cache:
        return _cache[smiles]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles))
    if m is not None and "." in str(smiles):
        from rdkit.Chem.MolStandardize import rdMolStandardize
        m = rdMolStandardize.LargestFragmentChooser().choose(m)
    _cache[smiles] = m
    return m


def skeleton(smiles):
    """First 14 characters of the InChIKey: groups rows by molecule."""
    from rdkit import Chem
    m = _mol(smiles)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def descriptors(smiles):
    """The 14 STRUCTURE features, or None if the SMILES can't be parsed."""
    from rdkit.Chem import Descriptors, rdMolDescriptors
    m = _mol(smiles)
    if m is None:
        return None
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    n = max(len(sym), 1)
    return {
        "nF": sym.count("F"), "nS": sym.count("S"), "nN": sym.count("N"),
        "nO": sym.count("O"), "nCl": sym.count("Cl"),
        "frac_F": sym.count("F") / n,
        "mw": Descriptors.MolWt(m),
        "logp": Descriptors.MolLogP(m),
        "tpsa": Descriptors.TPSA(m),
        "rotb": Descriptors.NumRotatableBonds(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "arom": sum(1 for a in m.GetAtoms() if a.GetIsAromatic()),
        "hba": Descriptors.NumHAcceptors(m),
        "charge": sum(a.GetFormalCharge() for a in m.GetAtoms()),
    }


def _composition(mol):
    """Element counts of a molecule, with explicit H and deuterium kept apart."""
    from rdkit import Chem
    mh = Chem.AddHs(mol)
    c = {}
    for a in mh.GetAtoms():
        e = a.GetSymbol()
        if e == "H" and a.GetIsotope() == 2:
            e = "D"
        c[e] = c.get(e, 0) + 1
    return c


def _mass(comp):
    return sum(MASS.get(e, 0.0) * k for e, k in comp.items())


def ion_composition(smiles, prec_mz, negative, tol=0.6):
    """Composition of the precursor ION, inferring the adduct from its m/z.

    Returns (composition, adduct_label, ion_mz). ion_mz is the TRUE m/z of the
    ion -- normally the measured value, but see the +1 case below.
    Returns (None, None, None) if no adduct explains the measured precursor,
    which means the structure and the measurement disagree.

    Already-charged molecules (quaternary ammoniums, phosphoniums) are observed
    as the intact cation [M]+. Some methods set Q1 at cation + 1.008 instead,
    from treating the cation as neutral and adding a proton. That is accepted
    here and labelled '[M]+ (Q1 set +1)', and ion_mz is returned as the real
    cation mass so the features match what prediction will see.
    """
    m = _mol(smiles)
    if m is None:
        return None, None, None
    comp = _composition(m)
    q = sum(a.GetFormalCharge() for a in m.GetAtoms())
    M = _mass(comp)

    if q != 0:  # already an ion: [M]+ or [M]-
        mz = (M - q * ELECTRON) / abs(q)
        sign = "+" if q > 0 else "-"
        if abs(mz - prec_mz) <= tol:
            return comp, f"[M]{sign}", prec_mz
        if abs(q) == 1 and abs(mz + PROTON - prec_mz) <= tol:
            return comp, f"[M]{sign} (Q1 set +1)", mz
        return None, None, None

    def add(c, **kw):
        c = dict(c)
        for e, k in kw.items():
            c[e] = c.get(e, 0) + k
            if c[e] < 0:
                return None
        return c

    cands = ([("[M-H]-", add(comp, H=-1), M - PROTON),
              ("[M+HCOO]-", add(comp, C=1, H=1, O=2), M + 44.998201),
              ("[M+Cl]-", add(comp, Cl=1), M + 34.969402),
              # in-source decarboxylation, measured as the precursor (GenX)
              ("[M-H-CO2]-", add(comp, H=-1, C=-1, O=-2), M - PROTON - 43.989829)]
             if negative else
             [("[M+H]+", add(comp, H=1), M + PROTON),
              ("[M+NH4]+", add(comp, N=1, H=4), M + 18.033823),
              ("[M+Na]+", add(comp, Na=1), M + 22.989218)])
    best = None
    for lab, c, mz in cands:
        if c is None:
            continue
        d = abs(mz - prec_mz)
        if d <= tol and (best is None or d < best[0]):
            best = (d, c, lab, mz)
    # return the THEORETICAL m/z of the matched adduct, not the measured value:
    # a precursor recorded as 350.1 is really 350.1166, and using the exact
    # value makes the neutral-loss mass accurate to millidaltons.
    return (best[1], best[2], best[3]) if best else (None, None, None)


def _rdbe(f):
    """Rings + double bonds of a formula. Negative means impossible."""
    c = f.get("C", 0) + f.get("Si", 0)
    h = f.get("H", 0) + f.get("D", 0)
    x = sum(f.get(e, 0) for e in ("F", "Cl", "Br", "I", "Na"))
    n = f.get("N", 0) + f.get("P", 0)
    return c - (h + x) / 2 + n / 2 + 1


def assign_loss(ion_comp, loss_mass, tol=LOSS_TOL, all_matches=False):
    """Best sub-formula of the precursor ion matching the neutral-loss mass.

    Loops over the heavy elements and solves hydrogen directly, which keeps
    the search fast even for large molecules. Returns a dict, or None.
    With all_matches=True returns every formula within tolerance, closest
    first -- use it to see how ambiguous an assignment is.
    """
    if ion_comp is None or loss_mass is None or loss_mass <= 0:
        return [] if all_matches else None
    heavy = [e for e in ("I", "Br", "Cl", "S", "P", "Si", "F", "O", "N", "C", "D")
             if ion_comp.get(e, 0) > 0]
    n_h = ion_comp.get("H", 0)
    best = [None]
    found = []

    def rec(i, m, counts):
        if m - loss_mass > tol:
            return
        if i == len(heavy):
            h0 = int(round((loss_mass - m) / MASS["H"]))
            for h in (h0 - 1, h0, h0 + 1):
                if 0 <= h <= n_h:
                    err = abs(m + h * MASS["H"] - loss_mass)
                    if err <= tol:
                        f = {k: v for k, v in counts.items() if v}
                        if h:
                            f["H"] = h
                        if not f or _rdbe(f) < 0:
                            continue
                        key = (err, sum(f.values()))
                        if all_matches:
                            found.append((key, f))
                        if best[0] is None or key < best[0][0]:
                            best[0] = (key, f)
            return
        e = heavy[i]
        for k in range(ion_comp[e] + 1):
            mm = m + k * MASS[e]
            if mm - loss_mass > tol:
                break
            counts[e] = k
            rec(i + 1, mm, counts)
        counts.pop(e, None)

    rec(0, 0.0, {})
    if all_matches:
        found.sort(key=lambda kv: kv[0])
        return [f for _, f in found]
    return best[0][1] if best[0] else None


def loss_features(f):
    """The 12 LOSS features for an assigned formula (or placeholders if None)."""
    if f is None:
        out = {k: -1 for k in FEATURES_LOSS}
        out["loss_assigned"] = 0
        return out
    h = f.get("H", 0) + f.get("D", 0)
    return {
        "l_C": f.get("C", 0), "l_H": h, "l_O": f.get("O", 0),
        "l_F": f.get("F", 0), "l_N": f.get("N", 0), "l_S": f.get("S", 0),
        "l_Cl": f.get("Cl", 0), "l_Br": f.get("Br", 0), "l_P": f.get("P", 0),
        "l_rdbe": _rdbe(f),
        "l_heavy": sum(v for e, v in f.items() if e not in ("H", "D")),
        "loss_assigned": 1,
    }


def parse_formula(s):
    """'C3H9N' -> {'C':3,'H':9,'N':1}. Inverse of formula_str."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return None
    out = {}
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)):
        if el:
            out[el] = out.get(el, 0) + (int(num) if num else 1)
    return out or None


def formula_str(f):
    if not f:
        return ""
    order = ["C", "H", "D", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Na"]
    return "".join(f"{e}{f[e] if f[e] > 1 else ''}" for e in order if f.get(e))


# ------------------------------------------------------------------ rows
def build_row(prec, prod, negative, smiles=None):
    """All features for one transition.

    Returns (features_dict, info_dict). features_dict always holds the MASS
    features. STRUCTURE and LOSS features are filled when a SMILES parses.

    info['consistent'] is True only when the structure actually explains the
    measured precursor under some adduct. Only consistent rows should use the
    STRUCTURE model: an inconsistent row means the SMILES and the measurement
    describe different molecules (a labelled standard's SMILES against a
    native compound's data, a salt against its acid, a wrong polarity...), and
    its descriptors would be wrong.
    """
    prec, prod = float(prec), float(prod)
    row = {"prec": prec, "prod": prod, "loss": prec - prod,
           "neg": int(bool(negative))}
    info = {"has_structure": False, "consistent": False, "adduct": None,
            "loss_formula": "", "prec_measured": prec, "prec_used": prec}
    if not smiles or (isinstance(smiles, float) and np.isnan(smiles)):
        return row, info
    d = descriptors(smiles)
    if d is None:
        return row, info
    row.update(d)
    info["has_structure"] = True

    comp, adduct, ion_mz = ion_composition(smiles, prec, bool(negative))
    if comp is None:
        row.update(loss_features(None))
        return row, info

    # use the true ion m/z (differs from measured only for the '+1' QAC case)
    row["prec"], row["loss"] = ion_mz, ion_mz - prod
    f = assign_loss(comp, row["loss"])
    row.update(loss_features(f))
    info.update(consistent=True, adduct=adduct, loss_formula=formula_str(f),
                prec_used=ion_mz)
    return row, info
