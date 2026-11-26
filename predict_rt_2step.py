#!/usr/bin/env python3
"""Spec2MRM adapter for the published pretrained 2-step predictor; no model training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent


def polarity(value):
    s = str(value).strip().lower()
    if s in ('positive', 'pos', '+', 'esi+', 'esi-pos'):
        return 'Positive'
    if s in ('negative', 'neg', '-', 'esi-', 'esi-neg'):
        return 'Negative'
    raise ValueError(f'Unknown polarity: {value!r}')


def find_column(frame, *names):
    columns = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name in columns:
            return columns[name]
    raise ValueError(f'Required column missing: one of {names}')


def canonical(smiles):
    # Keep isomeric information and supplied charge states. No online lookups.
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_commit(directory):
    result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def prepare(input_path, config_path, workdir):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    model = resolve(config['model'])
    if not model.is_file():
        raise ValueError(f'Published model missing: {model}')
    for db in ('hsm', 'tanaka'):
        path = resolve(config['report_directory']) / 'resources' / f'{db}_database' / f'{db}_database.tsv'
        if not path.is_file():
            raise ValueError(f'Missing column table: {path}')
        table = pd.read_csv(path, sep='\t')
        for meta in config['methods'].values():
            if meta['column_name'] not in table.name_normalized.values:
                raise ValueError(f"Column absent from {db} database: {meta['column_name']}")
    frame = pd.read_csv(input_path, encoding='utf-8-sig')
    names = find_column(frame, 'names', 'name', 'cpd name', 'compound')
    smiles = find_column(frame, 'smiles')
    pol = find_column(frame, 'polarity', 'ion_mode')
    key = next((c for c in frame if str(c).lower() == 'inchikey'), None)
    targets = pd.DataFrame({
        'input_row': range(1, len(frame) + 1), 'name': frame[names].astype(str),
        'source_smiles': frame[smiles], 'smiles': frame[smiles].map(canonical),
        'polarity': frame[pol].map(polarity),
    })
    targets['inchikey'] = frame[key] if key else targets.smiles.map(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
    if targets.empty:
        raise ValueError('No targets supplied.')
    anchors = pd.read_csv(resolve(config['anchors']), encoding='utf-8-sig')
    anchors['polarity'] = anchors.polarity.map(polarity)
    anchors['smiles'] = anchors.smiles.map(canonical)
    anchors['rt'] = pd.to_numeric(anchors.rt_median_min, errors='raise')
    if not anchors.rt.map(math.isfinite).all():
        raise ValueError('Anchor RT must be finite.')
    targets.to_csv(workdir / 'targets.csv', index=False)
    jobs = []
    for pol_name in targets.polarity.unique():
        meta = dict(config['methods'][pol_name])
        if not 0 <= float(meta['ph']) <= 14:
            raise ValueError(f'Invalid pH for {pol_name}')
        # Same geometric estimate used in RepoRT metadata_standardize.R.
        t0 = meta.get('dead_time_min')
        source = 'measured/user supplied'
        if t0 is None:
            t0 = 0.5 * meta['column_length_mm'] * meta['column_id_mm'] ** 2 / 1000 / meta['flow_ml_min']
            source = 'RepoRT geometric estimate; excludes system extra-column volume'
        threshold = float(t0) * float(meta['void_factor'])
        if not 0 <= threshold < meta['run_time_min']:
            raise ValueError(f'Invalid void threshold for {pol_name}')
        a = anchors.loc[(anchors.polarity == pol_name) &
                        (anchors.method_id == meta['method_id'])].copy()
        # Replicate observations are represented once per structure.
        a = a.groupby('smiles', as_index=False).agg(rt=('rt', 'median'))
        if len(a.loc[a.rt > threshold]) < 3:
            raise ValueError(f'Fewer than three retained anchors for {pol_name}')
        t = targets.loc[targets.polarity == pol_name]
        structures = pd.DataFrame({'smiles': pd.concat([a.smiles, t.smiles]).drop_duplicates()})
        structures['rt'] = float('nan')
        stem = pol_name.lower()
        structures.to_csv(workdir / f'{stem}_structures.tsv', sep='\t', index=False)
        a.to_csv(workdir / f'{stem}_anchors.tsv', sep='\t', index=False)
        # JSON is valid YAML. Passing threshold explicitly avoids ambiguity in
        # upstream README versus predict.py about whether t0 is doubled there.
        metadata = {'column.name': meta['column_name'],
                    'column.particle.size': meta['particle_size_um'],
                    'column.length': meta['column_length_mm'],
                    'column.id': meta['column_id_mm'],
                    'column.temperature': meta.get('column_temperature_c', 35.0),
                    'column.flowrate': meta['flow_ml_min'],
                    'column.t0': float(t0), 'eluent.A.pH': meta['ph']}
        jobs.append({'polarity': pol_name, 'stem': stem, 'metadata': metadata,
                     'method': meta, 'dead_time_min': float(t0),
                     'dead_time_source': source, 'void_threshold_min': threshold,
                     'n_anchors_total': len(a), 'n_anchors_retained': int((a.rt > threshold).sum())})
    manifest = {
        'model_name': '2-step', 'doi': '10.1038/s41592-026-03243-2',
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        # A deployed copy is not a git checkout; its config records the commits.
        'upstream_commit': config.get('upstream_commit') or git_commit(resolve(config['upstream_directory'])),
        'report_commit': config.get('report_commit') or git_commit(resolve(config['report_directory'])),
        'jobs': jobs, 'n_targets': len(targets),
        'validation': 'No model comparison or accuracy evaluation performed.',
        'interval_status': 'No validated 90%/95% intervals for this predictor.'}
    (workdir / 'job.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return config, manifest


def merge_transitions(transitions, predictions):
    left = transitions.copy()
    for col in list(left):
        if col.startswith('rt_') or col in ('predicted_rt_min', 'raw_smrt_ensemble_min'):
            left = left.drop(columns=col)
    key = find_column(left, 'inchikey')
    pol = find_column(left, 'polarity')
    left['_pol'] = left[pol].map(polarity)
    left['_key'] = left[key].astype(str).str.strip().str.upper()
    right = predictions.copy()
    right['_pol'] = right.polarity
    right['_key'] = right.inchikey.astype(str).str.strip().str.upper()
    columns = [c for c in right if c.startswith('rt_') or c == 'predicted_rt_min']
    right = right[['_key', '_pol', *columns]].drop_duplicates(['_key', '_pol'])
    merged = left.merge(right, on=['_key', '_pol'], how='left', validate='many_to_one')
    if merged.predicted_rt_min.isna().any():
        raise ValueError('Some transitions could not be matched to RT predictions; outputs not replaced.')
    return merged.drop(columns=['_key', '_pol'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='compounds.csv')
    parser.add_argument('--config', default=str(ROOT / 'rt_2step_config.json'))
    parser.add_argument('--out', default='rt_predictions.csv')
    parser.add_argument('--transitions')
    parser.add_argument('--merged-out', default='mrm_transitions_ready.csv')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--python-executable', help='Override the configured 2-step Python executable')
    args = parser.parse_args()
    workroot = ROOT / 'outputs' / 'rt_2step'
    workroot.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='run_', dir=workroot))
    config, manifest = prepare(args.input, args.config, workdir)
    print(f'Prepared {manifest["n_targets"]} compounds: {workdir}', flush=True)
    for job in manifest['jobs']:
        print(f"{job['polarity']}: pH={job['method']['ph']}; "
              f"t0={job['dead_time_min']:.4f} min ({job['dead_time_source']}); "
              f"{job['n_anchors_retained']}/{job['n_anchors_total']} anchors retained", flush=True)
    if args.prepare_only:
        return
    common = ['--job-dir', str(workdir), '--upstream', str(resolve(config['upstream_directory'])),
              '--report', str(resolve(config['report_directory'])), '--model', str(resolve(config['model']))]
    python_executable = args.python_executable or config.get('python_executable')
    if python_executable:
        command = [str(resolve(python_executable)), str(ROOT / 'two_step_runner.py'), *common]
    else:
        if not shutil.which('docker'):
            raise RuntimeError('Docker missing. Install/start Docker Desktop or use --python-executable.')
        image = config['docker_image']
        ready = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                               capture_output=True, text=True, timeout=45)
        if ready.returncode:
            raise RuntimeError('Docker engine unavailable. Start Docker Desktop, then rerun. Prepared inputs retained.')
        inspected = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                   capture_output=True, text=True, timeout=30)
        if inspected.returncode:
            raise RuntimeError(f'Image not installed. Run: docker pull {image}')
        image_id = inspected.stdout.strip()
        manifest['docker_image_id'] = image_id
        command = ['docker', 'run', '--rm', '--network', 'none',
                   '--mount', f'type=bind,source={workdir},target=/work',
                   '--mount', f'type=bind,source={resolve(config["upstream_directory"])},target=/upstream,readonly',
                   '--mount', f'type=bind,source={resolve(config["report_directory"])},target=/RepoRT,readonly',
                   '--mount', f'type=bind,source={ROOT / "two_step_runner.py"},target=/runner.py,readonly',
                   image_id, 'python', '/runner.py', '--job-dir', '/work',
                   '--upstream', '/upstream', '--report', '/RepoRT',
                   '--model', '/upstream/models/2-step0525.pt']
    subprocess.run(command, check=True)
    result = pd.read_csv(workdir / 'predictions.csv')
    if len(result) != manifest['n_targets'] or not result.predicted_rt_min.map(math.isfinite).all():
        raise ValueError('Incomplete or invalid RT predictions; outputs not replaced.')
    merged = None
    if args.transitions:
        merged = merge_transitions(pd.read_csv(args.transitions), result)
    result.to_csv(args.out, index=False)
    if merged is not None:
        merged.to_csv(args.merged_out, index=False)
        if Path(args.merged_out).resolve() == ROOT / 'mrm_transitions_ready.csv':
            audit_path = ROOT / 'production_readiness_report.json'
            audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
            audit['status'] = 'READY_FOR_INITIAL_UNSCHEDULED_ACQUISITION'
            audit['input_compound_rows'] = len(result)
            audit['transition_rows'] = len(merged)
            audit.setdefault('checks', {})['rt_predictions_complete'] = True
            audit['checks']['rt_scheduling_intervals_established'] = False
            audit['rt'] = {
                'model_type': 'published_2step_pretrained_dmpnn_with_local_anchor_mapping',
                'doi': manifest['doi'], 'model_sha256': manifest['model_sha256'],
                'upstream_commit': manifest['upstream_commit'],
                'method_ph': {j['polarity']: j['method']['ph'] for j in manifest['jobs']},
                'local_anchor_counts': {j['polarity']: j['n_anchors_total'] for j in manifest['jobs']},
                'retained_anchor_counts': {j['polarity']: j['n_anchors_retained'] for j in manifest['jobs']},
                'dead_time_source': 'RepoRT estimate unless overridden in rt_2step_config.json',
                'clipped_compounds': int(result.rt_prediction_clipped.sum()),
                'extrapolated_compounds': int(result.rt_extrapolates_anchor_roi.sum()),
                'comparison_performed': False, 'local_accuracy_evaluated': False,
                'interval_status': manifest['interval_status'], 'run_directory': str(workdir)}
            audit['warnings'] = [
                'CE is a predicted starting setting; retain its existing grouped validation provenance.',
                '2-step uses published pretrained weights and local anchor mappings; no local accuracy claim is made.',
                'RT uncertainty intervals from the previous SMRT predictor are not reused.',
                'Use unscheduled acquisition initially; clipped and extrapolated RTs require review.',
                'Dead time is estimated from column dimensions and flow, not measured.']
            audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    manifest['status'] = 'Predictions generated using published pretrained weights and local anchors.'
    (workdir / 'integration_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {args.out}; 2-step RTs have no validated scheduling intervals. Use unscheduled acquisition initially.')


if __name__ == '__main__':
    main()
