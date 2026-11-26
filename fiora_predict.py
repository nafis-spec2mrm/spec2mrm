"""FIORA in-silico MRM transitions (third tier, after library and class rules).

FIORA (Nowatzky et al., Nat. Commun. 16, 2298 (2025); MIT licence; open model
FIORA OS v1.0.0 trained on MSnLib) predicts MS/MS spectra from structure. It
runs in its own environment (.venv_fiora) and is called as a subprocess, so
the web app's environment is untouched.

For each structure the spectra predicted at several collision energies are
merged (maximum relative intensity per ion) and ranked like Spec2MRM ranks
library ions: no ions within 5 Da of or above the precursor, none below
m/z 40, no 13C/34S/37Cl/81Br copies, ions closer than 0.7 Da merged. The top
two ions become quantifier and qualifier.

Benchmark on 187 lab-measured compound/mode pairs (benchmark_insilico.py):
FIORA's top ion was a lab-monitored ion for 32% (26% for compounds absent from
public libraries). Treat every FIORA transition as a low-confidence start.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from consensus import ISOTOPE_SPACINGS, precursor_candidates

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parent
# FIORA runs from its own environment where one exists (the lab PC's
# .venv_fiora); on a server with FIORA installed alongside the app it runs from
# the app's own Python.
FIORA = ROOT / '.venv_fiora' / 'Scripts' / 'fiora-predict.exe'
CES = (10, 20, 30, 40, 50)
SUPPORTED = {'[M+H]+', '[M-H]-', '[M]+', '[M]-'}
GUARD, MIN_MZ, MERGE = 5.0, 40.0, 0.7
TOP1_HIT_ALL, TOP1_HIT_UNSEEN = 0.32, 0.26   # benchmark, see module docstring
CITATION = ('FIORA: Y. Nowatzky et al., Local neighborhood-based prediction of compound mass spectra from single '
            'fragmentation events, Nat. Commun. 16, 2298 (2025), doi:10.1038/s41467-025-57422-4 (MIT licence).')


def command():
    """Command line that starts fiora-predict, or None if FIORA is not installed."""
    if FIORA.exists():
        return [str(FIORA)]
    if importlib.util.find_spec('fiora') is not None:
        return [sys.executable, '-c', 'import sys; from fiora.cli.predict import main; sys.exit(main())']
    return None


def available():
    return command() is not None


def query(smiles, polarity):
    """(smiles for FIORA, precursor type, precursor m/z) or None if FIORA cannot handle it."""
    mol = Chem.MolFromSmiles(smiles or '')
    if mol is None or any(a.GetIsotope() for a in mol.GetAtoms()):
        return None
    if '.' in smiles:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    if Chem.GetFormalCharge(mol) < 0:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    if mol.GetNumHeavyAtoms() < 4 or mol.GetNumBonds() < 3:
        return None  # too small to fragment; FIORA also crashes on single-atom inputs
    clean = Chem.MolToSmiles(mol)
    candidates = precursor_candidates(clean, polarity.lower().startswith('neg'))
    if not candidates or candidates[0][0] not in SUPPORTED:
        return None
    ion, mz = candidates[0]
    return clean, ion, mz


ACIDIC = [Chem.MolFromSmarts(s) for s in (
    '[CX3](=O)[OX2H1,OX1-]',            # carboxylic acid
    '[SX4](=O)(=O)[OX2H1,OX1-]',        # sulfonic acid / sulfate
    '[PX4](=O)[OX2H1,OX1-]',            # phosphoric / phosphonic acid
    'c[OX2H1]',                          # phenol
    '[SX4](=O)(=O)[NX3;H1,H2]',          # sulfonamide N-H
    '[CX3](=O)[NX3H1][CX3]=O',           # imide
    'c1nn[nH]n1',                        # tetrazole
)]


def ionizes_negative(smiles):
    """True if the structure has an acidic group that can give [M-H]- (or is an anion)."""
    mol = Chem.MolFromSmiles(smiles or '')
    if mol is None:
        return False
    return Chem.GetFormalCharge(mol) < 0 or any(mol.HasSubstructMatch(p) for p in ACIDIC)


def rank(spectra, precursor):
    """Ranked [(m/z, relative intensity)] of MRM-eligible ions from spectra at several CEs."""
    best = {}
    for peaks in spectra:
        top = max((i for _, i in peaks), default=0)
        for mz, i in peaks:
            if top <= 0 or mz < MIN_MZ or mz > precursor or abs(mz - precursor) < GUARD:
                continue
            key = round(mz, 4)
            best[key] = max(best.get(key, 0), i / top)
    kept = []
    for mz, s in sorted(best.items(), key=lambda kv: -kv[1]):
        if any(abs(mz - k) < MERGE for k, _ in kept):
            continue
        if any(abs(mz - k - d) <= 0.02 for k, _ in kept for d in ISOTOPE_SPACINGS):
            continue
        kept.append((mz, s))
    return kept


def predict(items, timeout=900, _splits_left=6):
    """items: list of dict(key, smiles, ion). Returns {key: ranked ions}.

    FIORA predicts a whole batch in one process, so one structure it cannot
    handle would lose every prediction. On failure the batch is split in two
    and each half retried (a few times at most); only if that also fails is
    RuntimeError raised."""
    if not items:
        return {}
    try:
        return _predict_batch(items, timeout)
    except RuntimeError:
        if len(items) == 1 or _splits_left <= 0:
            if len(items) == 1:
                return {}  # this one structure is not predictable; skip it
            raise
        half = len(items) // 2
        out = predict(items[:half], timeout, _splits_left - 1)
        out.update(predict(items[half:], timeout, _splits_left - 1))
        return out


def _predict_batch(items, timeout):
    with tempfile.TemporaryDirectory(prefix='spec2mrm_fiora_') as folder:
        folder = Path(folder)
        rows = [dict(Name=f'q{i}_{ce}', SMILES=it['smiles'], Precursor_type=it['ion'], CE=ce, Instrument_type='HCD')
                for i, it in enumerate(items) for ce in CES]
        inp, out = folder / 'input.csv', folder / 'output.mgf'
        pd.DataFrame(rows).to_csv(inp, index=False)
        result = subprocess.run([*command(), '-i', str(inp), '-o', str(out), '--no-rt', '--no-ccs'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if result.returncode or not out.exists():
            print('FIORA failed:\n' + result.stderr[-3000:])
            raise RuntimeError('FIORA prediction could not complete.')
        spectra, title, peaks, precursor = {}, None, [], {}
        for line in out.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line == 'BEGIN IONS':
                title, peaks = None, []
            elif line.startswith('TITLE='):
                title = line[6:]
            elif line.startswith('PRECURSOR_MZ='):
                precursor[title.split('_')[0]] = float(line.split('=', 1)[1])
            elif line == 'END IONS' and title:
                spectra.setdefault(title.split('_')[0], []).append(peaks)
            elif line and line[0].isdigit():
                mz, inten = line.split()[:2]
                peaks.append((float(mz), float(inten)))
    out = {}
    for i, it in enumerate(items):
        tag = f'q{i}'
        if tag in spectra:
            out[it['key']] = rank(spectra[tag], precursor.get(tag, it.get('precursor', 1e9)))
    return out
