"""Feature construction shared by CatBoost CE training and prediction."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

import ce_features as cf

RDLogger.DisableLog("rdApp.*")
ELEMENTS = ("C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Na")


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def formula_counts(value):
    out = {f"formula_{element}": 0 for element in ELEMENTS}
    if pd.isna(value):
        return out
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(value)):
        key = f"formula_{element}"
        if key in out:
            out[key] += int(count) if count else 1
    return out


def largest_mol(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms()) if fragments else mol


def structure_features(smiles):
    mol = largest_mol(smiles)
    if mol is None:
        return {}, np.zeros(256, dtype=np.uint8), None
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    desc = {
        "mol_wt": Descriptors.MolWt(mol),
        "exact_mol_wt_rdkit": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "rot_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "hetero_atoms": Lipinski.NumHeteroatoms(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
    }
    for element in ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I"):
        desc[f"atom_{element}"] = atoms.count(element)
    fp = AllChem.GetMorganGenerator(radius=2, fpSize=256).GetFingerprintAsNumPy(mol).astype(np.uint8)
    return desc, fp, mol


def _find_col(columns, *names):
    normalized = {str(c).strip().lower(): c for c in columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    for name in names:
        for low, original in normalized.items():
            if name.lower() in low:
                return original
    return None


def load_structure_lookup(paths):
    by_name, by_key = {}, {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        name_col = _find_col(frame.columns, "names", "name", "cpd name", "compound")
        smiles_col = _find_col(frame.columns, "smiles")
        key_col = _find_col(frame.columns, "inchikey")
        if smiles_col is None:
            continue
        for _, row in frame.iterrows():
            smiles = str(row[smiles_col]).strip() if pd.notna(row[smiles_col]) else ""
            if not smiles:
                continue
            if name_col and pd.notna(row[name_col]):
                by_name.setdefault(cf.name_key(row[name_col]), smiles)
                by_name.setdefault(cf.name_key_nosuffix(row[name_col]), smiles)
            if key_col and pd.notna(row[key_col]):
                by_key.setdefault(str(row[key_col]).strip().upper(), smiles)
    return by_name, by_key


def build_prediction_matrix(frame, feature_names, structure_paths, fragmentor_defaults):
    """Build the same 322-feature matrix used by benchmark_ce_models.py."""
    by_name, by_key = load_structure_lookup(structure_paths)
    c_name = _find_col(frame.columns, "name", "cpd name", "compound")
    c_key = _find_col(frame.columns, "inchikey")
    c_smi = _find_col(frame.columns, "smiles")
    c_prec = _find_col(frame.columns, "precursor_mz", "prec ion", "precursor", "prec")
    c_prod = _find_col(frame.columns, "product_mz", "prod ion", "product", "prod")
    c_pol = _find_col(frame.columns, "polarity", "polar")
    c_frag = _find_col(frame.columns, "frag (v)", "fragmentor", "frag")
    c_adduct = _find_col(frame.columns, "adduct")
    c_theo = _find_col(frame.columns, "theoretical_mz")
    c_exact = _find_col(frame.columns, "exact mass", "exact_mass")
    c_formula = _find_col(frame.columns, "formula")
    if c_prec is None or c_prod is None:
        raise ValueError("Transition table must contain precursor and product m/z columns.")

    rows, fps, audit = [], [], []
    for _, row in frame.iterrows():
        prec, prod = safe_float(row[c_prec]), safe_float(row[c_prod])
        negative = str(row[c_pol]).strip().lower().startswith("neg") if c_pol else False
        polarity = "Negative" if negative else "Positive"
        smiles = str(row[c_smi]).strip() if c_smi and pd.notna(row[c_smi]) else ""
        if not smiles and c_key and pd.notna(row[c_key]):
            smiles = by_key.get(str(row[c_key]).strip().upper(), "")
        if not smiles and c_name and pd.notna(row[c_name]):
            smiles = by_name.get(cf.name_key(row[c_name]), "") or by_name.get(cf.name_key_nosuffix(row[c_name]), "")

        structure, fp, mol = structure_features(smiles)
        cf_row, info = cf.build_row(prec, prod, negative, smiles)
        full_mol = Chem.MolFromSmiles(smiles) if smiles else None
        # PubChem formulas in the training workbook include counterions, while
        # molecular descriptors use the largest organic fragment. Preserve that
        # same distinction at inference.
        formula = row[c_formula] if c_formula and pd.notna(row[c_formula]) else (rdMolDescriptors.CalcMolFormula(full_mol) if full_mol else np.nan)
        exact = safe_float(row[c_exact]) if c_exact else (Descriptors.ExactMolWt(mol) if mol else np.nan)
        theoretical = safe_float(row[c_theo]) if c_theo else safe_float(info.get("prec_used"), prec)
        fragmentor = safe_float(row[c_frag]) if c_frag else np.nan
        if not np.isfinite(fragmentor):
            fragmentor = float(fragmentor_defaults[polarity])
        adduct = str(row[c_adduct]).strip() if c_adduct and pd.notna(row[c_adduct]) else (info.get("adduct") or "Missing")
        loss = prec - prod
        feat = {
            "prec_mz": prec, "prod_mz": prod, "neutral_loss_mz": loss,
            "product_fraction": prod / prec if prec else np.nan,
            "loss_fraction": loss / prec if prec else np.nan,
            "fragmentor_v": fragmentor, "theoretical_mz": theoretical,
            "exact_mass_source": exact,
            "prec_minus_theoretical": prec - theoretical if np.isfinite(theoretical) else np.nan,
            "prec_minus_exact_mass": prec - exact if np.isfinite(exact) else np.nan,
            "negative_mode": int(negative),
            "structure_precursor_consistent": int(bool(info.get("consistent"))),
        }
        feat.update(structure)
        feat.update(formula_counts(formula))
        for name in cf.FEATURES_LOSS:
            feat[f"loss_{name}"] = cf_row.get(name, -1)
        feat[f"adduct_{adduct}"] = 1.0
        rows.append(feat)
        fps.append(fp)
        audit.append({
            "ce_smiles_found": bool(smiles),
            "ce_structure_consistent": bool(info.get("consistent")),
            "ce_inferred_adduct": adduct,
            "neutral_loss_formula": info.get("loss_formula", ""),
            "fragmentor_v_used": fragmentor,
        })

    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    fp_frame = pd.DataFrame(np.vstack(fps), columns=[f"morgan_{i:03d}" for i in range(256)])
    features = pd.concat([features, fp_frame], axis=1)
    # Absent one-hot/fingerprint/count columns mean zero. Other absent continuous
    # columns remain NaN so the model's fitted median imputer handles them.
    for name in feature_names:
        if name not in features:
            features[name] = 0.0 if name.startswith(("adduct_", "morgan_", "formula_", "atom_")) else np.nan
    return features[list(feature_names)].astype(float), pd.DataFrame(audit)
