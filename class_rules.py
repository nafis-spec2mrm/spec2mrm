"""Class-specific MRM transition rules for PFAS and quaternary ammonium compounds.

Used for compounds with no library spectrum. Each rule encodes fragmentation
that is well established for its class and that matches transitions measured
on the lab's Agilent 6470 (Agilent_CE.csv). Product-ion m/z values are exact
masses of the fragment ions (electron mass included). Nothing here is fitted:
a compound either matches a class pattern or gets no rule.

    from class_rules import rule_transitions
    rule_transitions('OC(=O)' + 'C(F)(F)' * 6 + 'C(F)(F)F', 'negative')
    -> [{'role': 'quantifier', 'product_mz': 368.9760, ...}, ...]

Rules (negative mode unless noted):
  PFCA        [M-H]- -> [M-H-CO2]- ; CkF2k+1- (k by chain length, lab-measured)
  PFSA        [M-H]- -> FSO3- (98.96) ; SO3- (79.96)
  n:2 FTS     [M-H]- -> [M-H-HF]- ; SO3- (79.96)
  Ether PFSA  [M-H]- -> far-side perfluoroalkoxide ; FSO2- (82.96)
  Ether PFCA  [M-H]- -> [M-H-CO2]- or far-side alkoxide ; far-side anion
  FASA        [M-H]- -> NSO2- (77.97) ; SO- (47.97)
  N-alkyl FASAA [M-H]- -> CnF2n+1- ; C3F7- (168.99)
  QAC (positive, [M]+):
    benzyl type, long chain   -> [M-C7H8]+ ; C7H7+ (91.05)
    benzyl type, short groups -> C7H7+ ; C5H5+ (65.04)
    alkyl trimethyl (C>=6)    -> C6H13+ (85.10) ; C5H11+ (71.09)
    dialkyl dimethyl (C>=6)   -> [M-CnH2n]+ ; C5H11+ (71.09)
    tri/tetra-alkyl (C>=4)    -> [M-CnH2n]+ (longest chain)
    alkylpyridinium (C>=6)    -> C5H6N+ (80.05)
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

MASS = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.99491461956, 'F': 18.99840322,
        'S': 31.97207100, 'Cl': 34.96885268, 'Br': 78.9183371, 'P': 30.97376163}
E = 0.00054857990946
PROTON = MASS['H'] - E
CO2 = 12.0 + 2 * MASS['O']
HF = MASS['H'] + MASS['F']
CH2 = 12.0 + 2 * MASS['H']

ION = {  # fixed fragment ions (anions +e, cations -e)
    'FSO3-': MASS['F'] + MASS['S'] + 3 * MASS['O'] + E,
    'SO3-': MASS['S'] + 3 * MASS['O'] + E,
    'FSO2-': MASS['F'] + MASS['S'] + 2 * MASS['O'] + E,
    'NSO2-': MASS['N'] + MASS['S'] + 2 * MASS['O'] + E,
    'SO-': MASS['S'] + MASS['O'] + E,
    'C7H7+': 7 * 12.0 + 7 * MASS['H'] - E,
    'C5H5+': 5 * 12.0 + 5 * MASS['H'] - E,
    'C6H13+': 6 * 12.0 + 13 * MASS['H'] - E,
    'C5H11+': 5 * 12.0 + 11 * MASS['H'] - E,
    'C5H6N+': 5 * 12.0 + 6 * MASS['H'] + MASS['N'] - E,
}
C7H8 = 7 * 12.0 + 8 * MASS['H']


def perfluoroalkyl_anion(k):
    return 12.0 * k + MASS['F'] * (2 * k + 1) + E


# Qualifier CkF2k+1- for PFCAs, by number of perfluorinated carbons, as measured
# on the lab instrument (PFHxA 119, PFHpA/PFOA 169, PFNA 219, PFDA-PFDoDA 269,
# PFTrDA/PFTeDA 169). Chains of 3 or fewer CF carbons get no qualifier.
PFCA_QUALIFIER_K = {4: 1, 5: 2, 6: 3, 7: 3, 8: 4, 9: 5, 10: 5, 11: 5}


def _largest_fragment(smiles):
    mol = Chem.MolFromSmiles(smiles or '')
    if mol is None:
        return None
    if '.' in smiles:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    return mol


def _composition(mol, atoms=None):
    counts = {}
    for atom in mol.GetAtoms():
        if atoms is not None and atom.GetIdx() not in atoms:
            continue
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        if atom.GetTotalNumHs():
            counts['H'] = counts.get('H', 0) + atom.GetTotalNumHs()
    return counts


def _mass(counts):
    return sum(MASS[element] * n for element, n in counts.items())


def _side(mol, start, blocked):
    """Atom indices reachable from `start` without crossing atom `blocked`."""
    seen, stack = {start}, [start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j != blocked and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _is_perfluoro_carbon(atom):
    return (atom.GetSymbol() == 'C' and atom.GetTotalNumHs() == 0 and not atom.GetIsAromatic()
            and any(n.GetSymbol() == 'F' for n in atom.GetNeighbors()))


def _row(role, product, rule, note=''):
    return dict(role=role, product_mz=round(product, 4), rule=rule, rule_note=note)


# ----------------------------------------------------------------- PFAS rules
ACID_C = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
SULFONATE = Chem.MolFromSmarts('[SX4](=O)(=O)[OX2H1,OX1-]')
SULFONAMIDE_NH2 = Chem.MolFromSmarts('[SX4](=O)(=O)[NX3H2]')
FASAA = Chem.MolFromSmarts('[CX4](F)(F)[SX4](=O)(=O)[NX3]([CX4])[CH2][CX3](=O)[OX2H1,OX1-]')
FTS = Chem.MolFromSmarts('[SX4](=O)(=O)([OX2H1,OX1-])[CH2][CH2][CX4](F)(F)')


def _pfas(mol, precursor):
    counts = _composition(mol)
    elements = set(counts) - {'H'}
    n_c, n_f = counts.get('C', 0), counts.get('F', 0)
    if n_f < 3:
        return None, None

    # FASA: perfluoroalkane sulfonamide (e.g. FOSA)
    if elements == {'C', 'F', 'S', 'O', 'N'} and mol.HasSubstructMatch(SULFONAMIDE_NH2) and n_f == 2 * n_c + 1:
        # Qualifier as measured on the lab instrument: FBSA 169 (C3F7-),
        # FHxSA 219 (C4F9-), FOSA 48 (SO-).
        qualifier = perfluoroalkyl_anion(n_c - 1) if n_c <= 4 else perfluoroalkyl_anion(4) if n_c <= 7 else ION['SO-']
        return 'FASA', [_row('quantifier', ION['NSO2-'], 'FASA'), _row('qualifier', qualifier, 'FASA')]

    # N-alkyl perfluoroalkane sulfonamidoacetic acids (N-MeFOSAA, N-EtFOSAA)
    if mol.HasSubstructMatch(FASAA) and elements == {'C', 'F', 'S', 'O', 'N'}:
        k = (n_f - 1) // 2
        return 'N-alkyl FASAA', [_row('quantifier', perfluoroalkyl_anion(k), 'N-alkyl FASAA'),
                                 _row('qualifier', perfluoroalkyl_anion(3), 'N-alkyl FASAA')]

    ethers = [a for a in mol.GetAtoms() if a.GetSymbol() == 'O' and a.GetDegree() == 2
              and all(n.GetSymbol() == 'C' for n in a.GetNeighbors())]
    if not elements <= {'C', 'F', 'S', 'O', 'Cl'}:
        return None, None

    sulfonate = mol.GetSubstructMatches(SULFONATE)
    acid = mol.GetSubstructMatches(ACID_C)
    if len(sulfonate) + len(acid) != 1:
        return None, None
    head = sulfonate[0][0] if sulfonate else acid[0][0]
    other_c = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and a.GetIdx() != head]

    if not ethers:
        if sulfonate and elements == {'C', 'F', 'S', 'O'} and n_f == 2 * n_c + 1:
            return 'PFSA', [_row('quantifier', ION['FSO3-'], 'PFSA'), _row('qualifier', ION['SO3-'], 'PFSA')]
        if acid and elements == {'C', 'F', 'O'} and n_f == 2 * (n_c - 1) + 1:
            rows = [_row('quantifier', precursor - CO2, 'PFCA')]
            k = PFCA_QUALIFIER_K.get(n_c - 1, 3 if n_c - 1 >= 12 else None)
            if k:
                rows.append(_row('qualifier', perfluoroalkyl_anion(k), 'PFCA'))
            return 'PFCA', rows
        if sulfonate and elements == {'C', 'F', 'S', 'O'} and mol.HasSubstructMatch(FTS) and n_f == 2 * (n_c - 2) + 1:
            return 'n:2 FTS', [_row('quantifier', precursor - HF, 'n:2 FTS'), _row('qualifier', ION['SO3-'], 'n:2 FTS')]
        return None, None

    # Ether PFAS: every carbon except the head must be fully halogenated
    # (one CHF allowed, as in ADONA).
    if sum(1 for a in other_c if not _is_perfluoro_carbon(a)) > 1:
        return None, None
    dist = Chem.GetDistanceMatrix(mol)
    nearest = min(ethers, key=lambda o: dist[head][o.GetIdx()])
    o = nearest.GetIdx()
    near_c, far_c = sorted((n.GetIdx() for n in nearest.GetNeighbors()), key=lambda j: dist[head][j])
    far_alkoxide = _side(mol, o, near_c)                # far side including the ether O
    far_alkyl = far_alkoxide - {o}
    alkoxide_mz = _mass(_composition(mol, far_alkoxide)) + E
    alkyl_mz = _mass(_composition(mol, far_alkyl)) + E
    far_has_ether = any(e.GetIdx() in far_alkyl for e in ethers)
    if sulfonate:
        return 'Ether PFSA', [_row('quantifier', alkoxide_mz, 'Ether PFSA'),
                              _row('qualifier', ION['FSO2-'], 'Ether PFSA')]
    if far_has_ether:  # e.g. ADONA: 377 -> 251 (alkoxide), 377 -> 85 (terminal CF3O-)
        outer = max(ethers, key=lambda e: dist[head][e.GetIdx()])
        oc = sorted((n.GetIdx() for n in outer.GetNeighbors()), key=lambda j: dist[head][j])[0]
        terminal = _side(mol, outer.GetIdx(), oc)
        return 'Ether PFCA', [_row('quantifier', alkoxide_mz, 'Ether PFCA'),
                              _row('qualifier', _mass(_composition(mol, terminal)) + E, 'Ether PFCA')]
    # e.g. HFPO-DA (GenX): 329 -> 285 ([M-H-CO2]-), 329 -> 169 (C3F7-)
    return 'Ether PFCA', [_row('quantifier', precursor - CO2, 'Ether PFCA'),
                          _row('qualifier', alkyl_mz, 'Ether PFCA')]


# ------------------------------------------------------------------ QAC rules
def _alkyl_length(mol, start, nitrogen):
    """Length of an unbranched saturated C/H chain starting at `start`, else 0."""
    length, prev, cur = 0, nitrogen, start
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic() or atom.GetHybridization() != Chem.HybridizationType.SP3:
            return 0
        nxt = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != prev]
        length += 1
        if not nxt:
            return length
        if len(nxt) > 1:
            return 0
        prev, cur = cur, nxt[0]


def _qac(mol, precursor):
    charge = Chem.GetFormalCharge(mol)
    if charge != 1:
        return None, None
    cations = [a for a in mol.GetAtoms() if a.GetSymbol() == 'N' and a.GetFormalCharge() == 1]
    if len(cations) != 1:
        return None, None
    n = cations[0]
    if n.GetIsAromatic():  # alkylpyridinium
        chains = [_alkyl_length(mol, nb.GetIdx(), n.GetIdx()) for nb in n.GetNeighbors()]
        ring = mol.GetRingInfo()
        if max(chains, default=0) >= 6 and ring.NumAtomRings(n.GetIdx()) == 1 and \
                all(a.GetSymbol() in 'CNH' for a in mol.GetAtoms()):
            return 'Alkylpyridinium', [_row('quantifier', ION['C5H6N+'], 'Alkylpyridinium',
                                            'no qualifier rule; optimise a second ion on the instrument')]
        return None, None
    if n.GetDegree() != 4 or n.GetTotalNumHs():
        return None, None
    groups = []
    for nb in n.GetNeighbors():
        j = nb.GetIdx()
        if nb.GetSymbol() != 'C':
            return None, None
        if nb.GetTotalNumHs() == 2 and any(x.GetIsAromatic() and x.IsInRingSize(6) and x.GetSymbol() == 'C'
                                           for x in nb.GetNeighbors()):
            groups.append(('benzyl', 0))
            continue
        length = _alkyl_length(mol, j, n.GetIdx())
        groups.append(('methyl', 1) if length == 1 else ('alkyl', length) if length else ('other', 0))
    kinds = [g[0] for g in groups]
    long_chains = sorted((g[1] for g in groups if g[0] == 'alkyl' and g[1] >= 6), reverse=True)
    if kinds.count('benzyl') == 1:
        if long_chains:
            return 'Benzylalkylammonium', [_row('quantifier', precursor - C7H8, 'Benzylalkylammonium'),
                                           _row('qualifier', ION['C7H7+'], 'Benzylalkylammonium')]
        # The second ion varies with the other groups (lab: 72, 86, 100); no rule.
        return 'Benzylammonium', [_row('quantifier', ION['C7H7+'], 'Benzylammonium',
                                       'no qualifier rule; optimise a second ion on the instrument')]
    if 'other' in kinds or 'benzyl' in kinds:
        return None, None
    if kinds.count('methyl') == 3 and len(long_chains) == 1:
        return 'Alkyltrimethylammonium', [_row('quantifier', ION['C6H13+'], 'Alkyltrimethylammonium'),
                                          _row('qualifier', ION['C5H11+'], 'Alkyltrimethylammonium')]
    if kinds.count('methyl') == 2 and len(long_chains) == 2:
        return 'Dialkyldimethylammonium', [_row('quantifier', precursor - long_chains[0] * CH2, 'Dialkyldimethylammonium'),
                                           _row('qualifier', ION['C5H11+'], 'Dialkyldimethylammonium')]
    # Only for chains of >= 7 carbons: butyl/pentyl QACs fragment differently.
    chains = sorted((g[1] for g in groups if g[0] == 'alkyl' and g[1] >= 7), reverse=True)
    if len(chains) >= 3 and all(g[0] == 'methyl' or g[1] >= 7 for g in groups):
        return 'Tri/tetraalkylammonium', [_row('quantifier', precursor - chains[0] * CH2, 'Tri/tetraalkylammonium',
                                               'no qualifier rule; optimise a second ion on the instrument')]
    return None, None


# --------------------------------------------------------------------- public
def rule_transitions(smiles, polarity):
    """Rule-based transitions for one structure in one polarity.

    Returns dict(rule_class, precursor_mz, precursor_ion, rows) or None when no
    class rule applies. `polarity` is 'positive' or 'negative'."""
    mol = _largest_fragment(smiles)
    if mol is None or any(a.GetIsotope() for a in mol.GetAtoms()):
        return None
    counts = _composition(mol)
    if any(element not in MASS for element in counts):
        return None
    charge = Chem.GetFormalCharge(mol)
    mass = _mass(counts)
    if polarity == 'negative':
        if charge == 0:
            precursor, ion = mass - PROTON, '[M-H]-'
        elif charge == -1:
            precursor, ion = mass + E, '[M]-'
        else:
            return None
        neutral = Chem.MolFromSmiles(Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(mol)))
        rule, rows = _pfas(neutral if neutral is not None else mol, precursor)
    elif polarity == 'positive':
        if charge != 1:
            return None
        precursor, ion = mass - E, '[M]+'
        rule, rows = _qac(mol, precursor)
    else:
        return None
    if not rows:
        return None
    return dict(rule_class=rule, precursor_mz=round(precursor, 4), precursor_ion=ion, rows=rows)


def rule_polarity(smiles):
    """The polarity a class rule exists for ('negative', 'positive') or None."""
    for polarity in ('negative', 'positive'):
        if rule_transitions(smiles, polarity):
            return polarity
    return None
