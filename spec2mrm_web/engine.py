"""Local browser-app adapter for the existing Spec2MRM engines."""
from contextlib import closing
import csv
from functools import lru_cache
from io import StringIO
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib

import class_rules
import consensus
import fiora_predict

# Streamlit reloads files in spec2mrm_web/ when they change, but not these
# helper modules one folder up. Reloading them whenever this engine is
# (re)loaded keeps a running server in step with edits, without a restart.
for _module in (consensus, class_rules, fiora_predict):
    importlib.reload(_module)

from class_rules import rule_polarity, rule_transitions
from consensus import (PRECURSOR_MATCH_TOL, candidate_key, confidence, library_consensus, norm_polarity, pick_quant_qual,
                       precursor_candidates)
from predict_rt_2step import prepare, resolve

EXAMPLE = b"name,smiles,polarity\nCaffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O,Positive\nAcetaminophen,CC(=O)Nc1ccc(O)cc1,Positive\nSulfamethoxazole,Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1,Positive\n"
TARGET_TEMPLATE = b"name,smiles\n"
ANCHOR_TEMPLATE = b"smiles,rt_min\n"
MAX_ROWS = 500
POLARITIES = ('Positive', 'Negative')
CE_INPUT_COLUMNS = ['name', 'inchikey', 'polarity', 'smiles', 'library_name', 'precursor_mz', 'product_mz',
                    'role', 'fragment_source', 'library_spectra', 'fragment_detection_frequency',
                    'fragment_confidence']


def read_table(raw, required):
    if not raw or len(raw) > 1_000_000:
        raise ValueError('Choose a non-empty CSV file smaller than 1 MB.')
    try:
        reader = csv.reader(StringIO(raw.decode('utf-8-sig')), strict=True)
        header = next(reader, [])
        header = [name.strip().lower() for name in header]
        header = ['name' if name == 'names' else name for name in header]
        if len(set(header)) != len(header) or not set(required).issubset(header):
            raise ValueError('Expected unique CSV headers including ' + ', '.join(required) + '.')
        records = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f'CSV line {reader.line_num} has the wrong number of fields.')
            records.append(dict(zip(header, (value.strip() for value in row))))
            if len(records) > MAX_ROWS:
                raise ValueError(f'This prototype accepts up to {MAX_ROWS} rows per CSV.')
    except UnicodeDecodeError as error:
        raise ValueError('Save the file as a UTF-8 CSV and try again.') from error
    except csv.Error as error:
        raise ValueError(f'Invalid CSV: {error}') from error
    if not records:
        raise ValueError('The CSV contains a header but no data rows.')
    return records


def molecule(smiles, label):
    mol = Chem.MolFromSmiles(smiles) if 0 < len(smiles) <= 3000 else None
    if not mol or not 0 < mol.GetNumAtoms() <= 250:
        raise ValueError(f'{label}: enter a valid SMILES with at most 250 atoms.')
    return mol


BOTH_WORDS = ('', 'both', 'any', 'auto', '+/-', '±')


def is_polarity_word(text):
    text = text.strip().lower()
    return bool(text) and (text in BOTH_WORDS or (text in '+-' or len(text) > 1) and norm_polarity(text) is not None)


def targets_from_csv(raw):
    """Validated targets, one row per (compound, polarity) to search.

    Polarity is optional. Blank or "Both" searches positive AND negative mode
    (both_modes=True); the results keep whichever modes have library spectra."""
    result, seen = [], set()
    for i, row in enumerate(read_table(raw, ('name', 'smiles')), 1):
        name = row['name']
        if not name or len(name) > 200 or name[0] in '=+@-' or any(ord(c) < 32 for c in name):
            raise ValueError(f'Compound row {i}: use a simple name of 1–200 characters without a leading =, +, @ or -.')
        text = row.get('polarity', '').strip()
        if text.lower() in BOTH_WORDS:
            pols, both = ['positive', 'negative'], True
        else:
            pol = norm_polarity(text)
            if pol is None:
                raise ValueError(f'Compound row {i}: polarity must be Positive, Negative, Both, or left blank.')
            pols, both = [pol], False
        mol = molecule(row['smiles'], f'Compound row {i}')
        key = Chem.MolToInchiKey(mol)
        if not key:
            raise ValueError(f'Compound row {i}: structure could not be assigned an InChIKey.')
        for pol in pols:
            if (key, pol) in seen:
                raise ValueError(f'Compound row {i}: this structure already occurs in the list in {pol} mode.')
            seen.add((key, pol))
            result.append(dict(name=name, smiles=Chem.MolToSmiles(mol), inchikey=key, polarity=pol.capitalize(),
                               isotope_labelled=any(a.GetIsotope() for a in mol.GetAtoms()),
                               both_modes=both, input_row=i))
    return pd.DataFrame(result)


def apply_class_rules(targets, transitions, misses):
    """Second tier: PFAS/QAC class rules for compounds without library spectra.

    Rule rows carry fragment_source 'class rule: <class>' and confidence
    'Rule-predicted'. Compounds with no applicable rule stay unmatched."""
    smiles = {(t.inchikey, t.polarity): t.smiles for t in targets.itertuples(index=False)}
    # Library compounds with a quantifier but no usable qualifier: add the
    # class-rule qualifier when it is a different ion (>= 0.7 Da away).
    if not transitions.empty:
        extra = []
        for (key, pol), group in transitions.groupby(['inchikey', 'polarity'], sort=False):
            if 'qualifier' in set(group.role) or (key, pol) not in smiles:
                continue
            result = rule_transitions(smiles[key, pol], pol.lower())
            quant = group.iloc[0]
            qual = next((r for r in (result or {}).get('rows', []) if r['role'] == 'qualifier'
                         and abs(r['product_mz'] - quant.product_mz) >= 0.7), None)
            if qual:
                row = quant.to_dict()
                row.update(role='qualifier', product_mz=qual['product_mz'],
                           fragment_source=f"class rule: {result['rule_class']} (qualifier added)",
                           fragment_confidence='Rule-predicted', fragment_detection_frequency=float('nan'),
                           product_mz_basis='exact mass of rule fragment')
                extra.append(row)
        if extra:
            transitions = pd.concat([transitions, pd.DataFrame(extra)], ignore_index=True)
            pair = transitions.inchikey + '|' + transitions.polarity
            first_seen = {p: i for i, p in reversed(list(enumerate(pair)))}
            transitions = (transitions.assign(_pair=pair.map(first_seen),
                                              _role=transitions.role.map({'quantifier': 0, 'qualifier': 1}).fillna(2))
                           .sort_values(['_pair', '_role'], kind='stable', ignore_index=True)
                           .drop(columns=['_pair', '_role']))
    if misses.empty:
        return transitions, misses
    both = {(t.inchikey, t.polarity): t.both_modes for t in targets.itertuples(index=False)}
    rows, keep = [], []
    for miss in misses.to_dict('records'):
        key = (miss['inchikey'], miss['polarity'])
        if key not in smiles or miss['reason'].startswith('Isotope'):
            keep.append(miss)
            continue
        result = rule_transitions(smiles[key], miss['polarity'].lower())
        if not result:
            other = rule_polarity(smiles[key])
            if other and not both[key]:
                miss['reason'] = (f'No library spectrum. A {other}-mode class rule exists for this structure; '
                                  f'submit it as {other.capitalize()} or Both.')
            keep.append(miss)
            continue
        for rule in result['rows']:
            rows.append(dict(name=miss['name'], inchikey=miss['inchikey'], polarity=miss['polarity'],
                             smiles=smiles[key], library_name=None, precursor_ion=result['precursor_ion'],
                             precursor_mz=result['precursor_mz'], product_mz=rule['product_mz'], role=rule['role'],
                             fragment_source=f"class rule: {result['rule_class']}", library_spectra=0,
                             duplicate_spectra_removed=0, fragment_detection_frequency=float('nan'),
                             fragment_confidence='Rule-predicted', precursor_basis='calculated from structure',
                             library_precursor_mz=float('nan'), product_mz_basis='exact mass of rule fragment',
                             rule_note=rule['rule_note']))
    if rows:
        transitions = pd.concat([transitions, pd.DataFrame(rows)], ignore_index=True)
    misses = pd.DataFrame(keep, columns=misses.columns) if keep else misses.iloc[0:0]
    return transitions, misses


def apply_fiora(targets, transitions, misses):
    """Third tier: FIORA in-silico prediction for compounds still unmatched.

    The top two predicted ions become quantifier and qualifier, labelled
    'in-silico: FIORA' with confidence 'Predicted (low)'. Returns
    (transitions, misses, error text or '')."""
    if misses.empty or not fiora_predict.available():
        return transitions, misses, '' if misses.empty else 'FIORA is not installed on this host.'
    smiles = {(t.inchikey, t.polarity): t.smiles for t in targets.itertuples(index=False)}
    both = {(t.inchikey, t.polarity): t.both_modes for t in targets.itertuples(index=False)}
    items, info, skipped = [], {}, set()
    for miss in misses.to_dict('records'):
        key = (miss['inchikey'], miss['polarity'])
        if key not in smiles or miss['reason'].startswith('Isotope'):
            continue
        # Searched in both modes: only predict [M-H]- for structures with an acidic group.
        if both[key] and miss['polarity'] == 'Negative' and not fiora_predict.ionizes_negative(smiles[key]):
            skipped.add(key)
            continue
        q = fiora_predict.query(smiles[key], miss['polarity'])
        if q:
            items.append(dict(key=key, smiles=q[0], ion=q[1], precursor=q[2]))
            info[key] = q
    try:
        predicted = fiora_predict.predict(items)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return transitions, misses, str(error)
    rows, keep = [], []
    for miss in misses.to_dict('records'):
        key = (miss['inchikey'], miss['polarity'])
        ions = predicted.get(key, [])
        if not ions:
            if key in skipped:
                miss['reason'] += ' Not predicted in negative mode: no acidic group to form [M-H]-.'
            elif key in info:
                miss['reason'] += ' FIORA predicted no usable product ion.'
            elif not miss['reason'].startswith('Isotope'):
                miss['reason'] += ' FIORA does not support this precursor type.'
            keep.append(miss)
            continue
        _, ion, precursor = info[key]
        for role, (mz, _) in zip(('quantifier', 'qualifier'), ions[:2]):
            rows.append(dict(name=miss['name'], inchikey=miss['inchikey'], polarity=miss['polarity'],
                             smiles=smiles[key], library_name=None, precursor_ion=ion,
                             precursor_mz=round(precursor, 4), product_mz=round(mz, 4), role=role,
                             fragment_source='in-silico: FIORA', library_spectra=0, duplicate_spectra_removed=0,
                             fragment_detection_frequency=float('nan'), fragment_confidence='Predicted (low)',
                             precursor_basis='calculated from structure', library_precursor_mz=float('nan'),
                             product_mz_basis='FIORA predicted fragment',
                             rule_note='In-silico prediction; confirm with a standard.'))
    if rows:
        transitions = pd.concat([transitions, pd.DataFrame(rows)], ignore_index=True)
    misses = pd.DataFrame(keep, columns=misses.columns) if keep else misses.iloc[0:0]
    return transitions, misses, ''


def resolve_both_modes(targets, transitions, misses):
    """For compounds searched in both modes, keep only the modes that matched.

    A "Both" compound with spectra in one mode is not reported as unmatched in
    the other; one with spectra in neither mode gets a single unmatched row.
    Returns (targets worth predicting RT for, misses)."""
    found = set(zip(transitions.inchikey, transitions.polarity)) if not transitions.empty else set()
    matched = {key for key, _ in found}
    hit = targets.apply(lambda r: (r.inchikey, r.polarity) in found, axis=1)
    keep = ~(targets.both_modes & ~hit & targets.inchikey.isin(matched))
    if misses.empty:
        return targets[keep], misses
    both_keys = set(targets.loc[targets.both_modes, 'inchikey'])
    single = misses[~misses.inchikey.isin(both_keys)]
    rest = misses[misses.inchikey.isin(both_keys) & ~misses.inchikey.isin(matched)]
    merged = (rest.groupby(['name', 'inchikey'], sort=False, as_index=False)
                  .agg(polarity=('polarity', ' & '.join), reason=('reason', lambda r: ' / '.join(dict.fromkeys(r)))))
    return targets[keep], pd.concat([single, merged], ignore_index=True)


def targets_from_text(text, default_polarity):
    """Pasted lines -> the same validated table as a CSV upload.

    Each line is `name, SMILES`, `name, SMILES, polarity` or just `SMILES`.
    Lines without a polarity use `default_polarity` ("Both" searches both modes). Tabs (pasted from Excel) or commas separate fields. Names may
    contain commas, because fields are taken from the right: SMILES never
    contains a comma or tab."""
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    if lines and 'smiles' in lines[0].lower():
        lines = lines[1:]  # header row copied with the data
    if not lines:
        raise ValueError('Paste at least one compound.')
    if len(lines) > MAX_ROWS:
        raise ValueError(f'This prototype accepts up to {MAX_ROWS} compounds at a time.')
    out = StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['name', 'smiles', 'polarity'])
    for i, line in enumerate(lines, 1):
        tabbed = '\t' in line
        parts = [p.strip() for p in (line.split('\t') if tabbed else line.rsplit(',', 2)) if p.strip()]
        last = parts[-1] if parts else ''
        if len(parts) >= 2 and is_polarity_word(last):
            pol = parts.pop()
        else:
            # No polarity given: the last field is the SMILES.
            pol = default_polarity
            if not tabbed:
                parts = [p.strip() for p in line.rsplit(',', 1) if p.strip()]
        smiles = parts[-1]
        name = ', '.join(parts[:-1]) or f'Compound {i}'
        if ' ' in smiles:
            raise ValueError(f'Line {i}: the SMILES "{smiles[:40]}" contains a space. '
                             'Separate name, SMILES and polarity with commas or tabs.')
        writer.writerow([name, smiles, pol])
    return targets_from_csv(out.getvalue().encode('utf-8'))


def anchors_from_csv(raw, polarity, method):
    if not method['method_id'] or method['method_id'][0] in '=+@-' or any(ord(c) < 32 for c in method['method_id']):
        raise ValueError(f'{polarity} method name must not start with =, +, @ or -.')
    items = []
    for i, row in enumerate(read_table(raw, ('smiles', 'rt_min')), 1):
        mol = molecule(row['smiles'], f'{polarity} anchor {i}')
        try:
            rt = float(row['rt_min'])
        except ValueError:
            rt = float('nan')
        if not 0 < rt <= method['run_time_min']:
            raise ValueError(f'{polarity} anchor {i}: RT must be within the run, in minutes.')
        items.append(dict(smiles=Chem.MolToSmiles(mol), rt_median_min=rt,
                          polarity=polarity, method_id=method['method_id']))
    return pd.DataFrame(items)


def supported_columns():
    base = json.loads((ROOT / 'rt_2step_config.json').read_text(encoding='utf-8'))
    root = resolve(base['report_directory']) / 'resources'
    names = []
    for table in ('hsm', 'tanaka'):
        data = pd.read_csv(root / f'{table}_database' / f'{table}_database.tsv', sep='\t')
        if 'phase type' in data:
            data = data[~data['phase type'].astype(str).str.contains('HILIC', case=False)]
        names.append(set(data.name_normalized.dropna()))
    return sorted(names[0] & names[1])


FULL_LIBRARY = ROOT / 'spectra.db'
PUBLIC_TABLE = Path(__file__).with_name('assets') / 'public_transitions.csv.gz'
BASIS_NAME = {'c': 'consensus', 'a': 'accurate-mass spectra', 'm': 'mixed-resolution average'}


def library_mode():
    """'full' = the local spectra.db; 'public' = the licence-cleared precomputed table
    (build_public_library.py). SPEC2MRM_LIBRARY=public|full forces one; otherwise
    the full database is used when present."""
    choice = os.environ.get('SPEC2MRM_LIBRARY', '').strip().lower()
    if choice in ('public', 'full'):
        return choice
    return 'full' if FULL_LIBRARY.exists() else 'public'


def public_site():
    """True on the public website: set SPEC2MRM_PUBLIC=1, or no full database present.
    The public site has no department access code, no lab RT demo and no
    automatic developer rights."""
    flag = os.environ.get('SPEC2MRM_PUBLIC', '').strip().lower()
    if flag in ('1', 'true', 'yes'):
        return True
    if flag in ('0', 'false', 'no'):
        return False
    return not FULL_LIBRARY.exists()


def demo_available():
    """The BEH C18 demo needs the lab's own standards, which are never published."""
    config = json.loads((ROOT / 'rt_2step_config.json').read_text(encoding='utf-8'))
    return bool(config.get('anchors')) and resolve(config['anchors']).is_file() and not public_site()


@lru_cache(maxsize=1)
def library_credits():
    """[(sub-library, spectra)] behind the public table, largest first ([] without the table)."""
    if not PUBLIC_TABLE.exists():
        return []
    totals = {}
    for text in public_table()[0].sources:
        for part in text.split(';'):
            name, _, count = part.rpartition(':')
            name = name.removeprefix('MoNA:')
            if name.endswith(':unattributed') or name in ('', '(none)'):
                name = 'Other GNPS/MoNA records'
            # The same library under GNPS's upper-case labels.
            name = {'MASSBANK': 'MassBank', 'MASSBANKEU': 'MassBank', 'MONA': 'MoNA', 'GNPS': 'GNPS-LIBRARY',
                    'HMDB': 'HMDB', 'RESPECT': 'ReSpect'}.get(name.upper(), name)
            totals[name] = totals.get(name, 0) + int(count or 0)
    return sorted(totals.items(), key=lambda kv: -kv[1])


def library_transitions(targets):
    return public_library_transitions(targets) if library_mode() == 'public' else full_library_transitions(targets)


@lru_cache(maxsize=1)
def public_table():
    table = pd.read_csv(PUBLIC_TABLE, keep_default_na=False,
                        dtype={'skeleton': str, 'polarity': str, 'candidates': str, 'precursor_ion': str,
                               'library_name': str, 'sources': str, 'frags': str})
    return table, table.groupby(['skeleton', 'polarity']).indices


def library_row(missing, smiles, library_name, precursor_ion, precursor_mz, library_precursor_mz, count,
                duplicates, frags, basis='calculated from structure'):
    rows = []
    for role, fragment in zip(('quantifier', 'qualifier'), pick_quant_qual(frags)):
        if fragment is None:
            continue
        rows.append(dict(**missing, smiles=smiles, library_name=library_name, precursor_ion=precursor_ion,
                         precursor_mz=precursor_mz, product_mz=fragment['fragment_mz'],
                         role=role, fragment_source='experimental library consensus',
                         library_spectra=count, duplicate_spectra_removed=duplicates,
                         fragment_detection_frequency=fragment['detection_freq'],
                         fragment_confidence=confidence(count, fragment['detection_freq']),
                         precursor_basis=basis,
                         library_precursor_mz=library_precursor_mz, product_mz_basis=fragment['mz_basis']))
    return rows


def public_library_transitions(targets):
    """Same result as the full-database search, read from the precomputed table.

    The table stores, per compound skeleton and polarity, every precursor-ion
    group computed with the structures found in the library. The group set
    computed with the same precursor candidates as the user's structure is used
    when available (identical to a live search on the cleared spectra);
    otherwise the best-supported group whose ion and m/z match one of the
    user's candidates."""
    table, index = public_table()
    rows, misses = [], []
    for item in targets.itertuples(index=False):
        missing = dict(name=item.name, inchikey=item.inchikey, polarity=item.polarity)
        if item.isotope_labelled:
            misses.append(dict(**missing, reason='Isotope-labelled structure requires isotope-specific transitions.'))
            continue
        positions = index.get((item.inchikey[:14], item.polarity))
        if positions is None:
            misses.append(dict(**missing, reason='No library spectrum with a precursor in the requested polarity.'))
            continue
        groups = table.iloc[positions]
        cands = precursor_candidates(item.smiles, item.polarity == 'Negative')
        calculated = dict(cands)
        order = {label: i for i, (label, _) in enumerate(cands)}
        same = groups[groups.candidates.eq(candidate_key(cands))]
        pick, ion = None, None
        if not same.empty:
            pick = same.sort_values('rank').iloc[0]
            ion = pick.precursor_ion
        elif cands:
            # The library wrote the structure differently (e.g. neutral acid vs
            # carboxylate, or salt vs cation). Match on m/z: first the
            # calculated precursor, then - as the live search assigns spectra -
            # the library precursor within 0.5 Da of one of the user's ions.
            for column, tol in (('precursor_mz', 0.01), ('library_precursor_mz', PRECURSOR_MATCH_TOL)):
                hits = []
                for position, mz in enumerate(groups[column]):
                    label, calc = min(cands, key=lambda c: abs(c[1] - mz))
                    if abs(calc - mz) <= tol:
                        hits.append((-groups.n_spectra.iloc[position], order[label], position, label))
                if hits:
                    _, _, position, ion = min(hits)
                    pick = groups.iloc[position]
                    break
        else:
            # No usable calculated ion (as in the live search, which then groups
            # by library adduct): best-supported group, library precursor m/z.
            pick = groups.sort_values(['rank', 'n_spectra'], ascending=[True, False]).iloc[0]
        if pick is None:
            misses.append(dict(**missing, reason='No library spectrum matches a supported precursor ion '
                                                 'for this structure, or no usable product ions.'))
            continue
        frags = [dict(fragment_mz=mz, detection_freq=freq, mz_basis=BASIS_NAME[basis])
                 for mz, freq, basis in json.loads(pick.frags)]
        precursor = round(calculated[ion], 4) if ion else pick.library_precursor_mz
        rows += library_row(missing, item.smiles, pick.library_name, ion, precursor,
                            pick.library_precursor_mz, int(pick.n_spectra), int(pick.duplicates_removed), frags,
                            basis='calculated from structure' if ion else 'library median')
    return pd.DataFrame(rows), pd.DataFrame(misses)


def full_library_transitions(targets, allowed_ids=None):
    """Live search of spectra.db; allowed_ids (optional) restricts it to those spectra."""
    rows, misses = [], []
    with closing(sqlite3.connect((ROOT / 'spectra.db').as_uri() + '?mode=ro', uri=True)) as db:
        # The existing index has separate skeleton and polarity indexes. On
        # the full database SQLite otherwise tends to scan all positive rows.
        has_skeleton_index = any(row[1] == 'idx_skeleton' for row in db.execute('PRAGMA index_list(spectra)'))
        table = 'spectra INDEXED BY idx_skeleton' if has_skeleton_index else 'spectra'
        query = ('SELECT id, name, precursor_mz, adduct, polarity, instrument_class, ce_value, peaks_json '
                 f'FROM {table} WHERE skeleton = ? AND polarity = ?')
        for item in targets.itertuples(index=False):
            name, pol = item.name, item.polarity
            missing = dict(name=name, inchikey=item.inchikey, polarity=pol)
            if item.isotope_labelled:
                misses.append(dict(**missing, reason='Isotope-labelled structure requires isotope-specific transitions.'))
                continue
            recs = db.execute(query, (item.inchikey[:14], pol.lower())).fetchall()
            usable = [r for r in recs if r[2] and math.isfinite(r[2]) and r[2] > 0
                      and (allowed_ids is None or r[0] in allowed_ids)]
            if not usable:
                misses.append(dict(**missing, reason='No library spectrum with a precursor in the requested polarity.'))
                continue
            spectra = [dict(spec_id=r[0], name=r[1], precursor_mz=r[2], adduct=r[3],
                            peaks=[(m, f) for m, f in json.loads(r[7])
                                   if math.isfinite(m) and math.isfinite(f) and 0 < m < r[2] and f > 0])
                       for r in sorted(usable)]
            # Shared with consensus.py: removes mirrored duplicate spectra, keeps
            # one calculated precursor ion, merges fragments Q3 cannot separate.
            result = library_consensus(spectra, item.smiles, pol == 'Negative')
            if result is None:
                misses.append(dict(**missing, reason='No library spectrum matches a supported precursor ion '
                                                     'for this structure, or no usable product ions.'))
                continue
            count = result['n_spectra']
            for role, fragment in zip(('quantifier', 'qualifier'), pick_quant_qual(result['frags'])):
                if fragment is None:
                    continue
                rows.append(dict(**missing, smiles=item.smiles, library_name=result['spectra'][0]['name'],
                                 precursor_ion=result['precursor_ion'],
                                 precursor_mz=result['precursor_mz'], product_mz=fragment['fragment_mz'],
                                 role=role, fragment_source='experimental library consensus',
                                 library_spectra=count, duplicate_spectra_removed=result['duplicates_removed'],
                                 fragment_detection_frequency=fragment['detection_freq'],
                                 fragment_confidence=confidence(count, fragment['detection_freq']),
                                 precursor_basis=result['precursor_basis'],
                                 library_precursor_mz=result['library_precursor_mz'],
                                 product_mz_basis=fragment['mz_basis']))
    return pd.DataFrame(rows), pd.DataFrame(misses)


def run_script(script, *args, python=None, cwd=None):
    result = subprocess.run([str(python or sys.executable), str(ROOT / script), *map(str, args)],
                            cwd=cwd or ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=180)
    if result.returncode:
        print(f'{script} failed:\n{result.stderr}', file=sys.stderr)
        raise RuntimeError(f'{script} could not complete. See the local server log for details.')


def rt_python(config):
    """The configured 2-step Python (lab PC), or this app's own Python on a server."""
    configured = config.get('python_executable')
    return resolve(configured) if configured and resolve(configured).is_file() else None


def predict_rt(targets, mode, profiles, folder):
    config = json.loads((ROOT / 'rt_2step_config.json').read_text(encoding='utf-8'))
    if mode == 'custom':
        methods = {pol: values[0] for pol, values in profiles.items()}
        anchors = pd.concat([anchors_from_csv(raw, pol, method) for pol, (method, raw) in profiles.items()])
        path = folder / 'anchors.csv'
        anchors.to_csv(path, index=False)
        config.update(methods=methods, anchors=str(path))
    selected = targets[targets.polarity.isin(config['methods'])].copy()
    if selected.empty:
        raise ValueError('No uploaded target uses a polarity with an LC profile.')
    target_path, config_path = folder / 'rt_targets.csv', folder / 'rt_config.json'
    selected.to_csv(target_path, index=False)
    config_path.write_text(json.dumps(config), encoding='utf-8')
    _, manifest = prepare(target_path, config_path, folder)
    run_script('two_step_runner.py', '--job-dir', folder,
               '--upstream', resolve(config['upstream_directory']), '--report', resolve(config['report_directory']),
               '--model', resolve(config['model']), python=rt_python(config),
               cwd=folder)
    rt = pd.read_csv(folder / 'predictions.csv')
    if len(rt) != len(selected) or not rt.predicted_rt_min.map(math.isfinite).all():
        raise RuntimeError('2-step did not return a finite result for every configured target.')
    rt['rt_status'] = 'Predicted; window not established'
    rt.loc[rt.rt_extrapolates_anchor_roi, 'rt_status'] = 'Extrapolated beyond anchor range'
    rt.loc[rt.rt_prediction_clipped, 'rt_status'] = 'Out of run range; no RT assigned'
    rt.loc[rt.rt_prediction_clipped, 'predicted_rt_min'] = float('nan')
    return rt, manifest


def generate(targets, mode='none', profiles=None, insilico=True):
    if mode not in ('none', 'demo', 'custom') or (mode == 'custom' and not profiles):
        raise ValueError('Choose at least one LC profile, or run without RT.')
    transitions, misses = library_transitions(targets)
    transitions, misses = apply_class_rules(targets, transitions, misses)
    fiora_error = ''
    if insilico:
        # Both-mode compounds already matched in one mode need no prediction in the other.
        found = set(transitions.inchikey) if not transitions.empty else set()
        both = set(targets.loc[targets.both_modes, 'inchikey'])
        skip = misses.inchikey.isin(found & both) if not misses.empty else None
        pending = misses[~skip] if skip is not None else misses
        transitions, predicted_misses, fiora_error = apply_fiora(targets, transitions, pending)
        misses = pd.concat([predicted_misses, misses[skip]], ignore_index=True) if skip is not None else predicted_misses
    rt_targets, misses = resolve_both_modes(targets, transitions, misses)
    rt, manifest, rt_error, masshunter = pd.DataFrame(), None, '', b''
    # No job input or output uses the lab pipeline's fixed output files.
    with TemporaryDirectory(prefix='spec2mrm_web_') as temporary:
        folder = Path(temporary)
        if not transitions.empty:
            fragment_path, ce_path = folder / 'fragments.csv', folder / 'ce.csv'
            # predict_ce.py finds columns by partial name ("prec", "adduct", ...),
            # so it receives only the columns it was validated with. The audit
            # columns are re-attached by row afterwards.
            extra = [c for c in transitions if c not in CE_INPUT_COLUMNS]
            transitions[CE_INPUT_COLUMNS].to_csv(fragment_path, index=False)
            run_script('predict_ce.py', '--input', fragment_path, '--model', ROOT / 'ce_catboost_model.joblib',
                       '--ce-mode', 'model', '--structures', fragment_path,
                       '--measured', folder / 'no_measured_ce.csv', '--out', ce_path, cwd=folder)
            predicted = pd.read_csv(ce_path).drop(columns='measured_ce_if_known', errors='ignore')
            if len(predicted) != len(transitions) or not predicted.product_mz.equals(transitions.product_mz):
                raise RuntimeError('CE prediction returned rows in an unexpected order.')
            if not predicted.collision_energy_eV.map(math.isfinite).all():
                raise RuntimeError('CatBoost returned a non-finite CE.')
            transitions = pd.concat([predicted, transitions[extra]], axis=1)
            # Pass only the MassHunter fields so the original exporter picks
            # fragmentor voltage, rather than the fragment-source text field.
            fragments = folder / 'export_input.csv'
            transitions[['name', 'precursor_mz', 'product_mz', 'fragmentor_v_used',
                         'collision_energy_eV', 'polarity']].to_csv(fragments, index=False)
            export = folder / 'masshunter.csv'
            run_script('export_masshunter.py', '--input', fragments, '--out', export, cwd=folder)
            masshunter = export.read_bytes()
        if mode != 'none':
            try:
                rt, manifest = predict_rt(rt_targets, mode, profiles, folder)
            except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                rt_error = str(error)
        if not transitions.empty and not rt.empty:
            columns = ['inchikey', 'polarity', 'predicted_rt_min', 'rt_roi', 'rt_anchor_count', 'rt_method_id',
                       'rt_status', 'rt_unclipped_min', 'rt_mobile_phase_ph', 'rt_extrapolates_anchor_roi', 'rt_prediction_clipped']
            transitions = transitions.merge(rt[columns], on=['inchikey', 'polarity'], how='left', validate='many_to_one')
    return dict(transitions=transitions, misses=misses, rt=rt, manifest=manifest,
                rt_error=rt_error, masshunter=masshunter, target_count=targets.input_row.nunique(),
                fiora_error=fiora_error)
