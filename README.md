# Spec2MRM

Spec2MRM builds starting MRM transitions for triple-quadrupole LC-MS/MS
(written for Agilent MassHunter) from a compound **name and SMILES**, without
needing an analytical standard.

For each compound and ion mode it gives:

- **Q1/Q3 transitions**, from up to three sources, tried in this order:
  1. **Library consensus**: product ions seen most consistently across public
     experimental MS/MS spectra (GNPS, MoNA, MassBank, MSnLib and others),
     precomputed in `spec2mrm_web/assets/public_transitions.csv.gz`.
  2. **Class rules**: well-established PFAS (negative mode) and quaternary
     ammonium (positive mode) fragmentation, for compounds missing from the
     libraries.
  3. **In-silico prediction** with FIORA, for anything else. These are
     low-confidence starting points.
- **A starting collision energy** from a CatBoost model trained on measured
  Agilent 6470 settings.
- **An optional retention time** from the published 2-step model, mapped to
  your own reversed-phase LC method using standards you have measured.

Every result is a starting point: confirm transitions and optimise collision
energies on your instrument before quantitative use.

## Use it

Open the website, paste `name, SMILES` lines (or upload a CSV), click
**Generate MRM method**, and download the MassHunter CSV.

To run it on your own computer (Python 3.12):

```
pip install -r requirements.txt
streamlit run spec2mrm_web/app.py
```

`python spec2mrm_web/public_test.py` checks that the installation works.

## Credits and licences

- **Retention time: 2-step.** F. Kretschmer, E.-M. Harrieder, M. Witting and
  S. Böcker, *Nature Methods* (2026), doi:10.1038/s41592-026-03243-2. The code
  and pretrained model in `external/2-step` are used unchanged under AGPL-3.0.
- **Column tables: RepoRT** (`external/RepoRT`), CC BY-SA 4.0.
- **In-silico spectra: FIORA.** Y. Nowatzky et al., *Nat. Commun.* 16, 2298
  (2025), doi:10.1038/s41467-025-57422-4. MIT licence, installed from its
  GitHub repository.
- **Spectral libraries:** the transition table holds derived values
  (precursor ions, product-ion m/z, detection frequencies), not spectra. They
  were computed from libraries distributed by GNPS and MoNA. The website lists
  every source library and how many spectra it contributed.

Because it includes 2-step, this repository is distributed under the GNU
Affero General Public License v3.0 (`LICENSE`).

## Contact

Questions: use the Q&A page on the website, or email mh.nafis29@gmail.com.
