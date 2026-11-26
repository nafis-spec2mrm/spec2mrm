#!/usr/bin/env python3
"""Run official 2-step ROI inference and LAD+OLS mapping in its own environment."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    isolated_dependencies = Path(__file__).resolve().parent / '.codex_deps' / '2step'
    if isolated_dependencies.is_dir():
        sys.path.insert(0, str(isolated_dependencies))
    sys.path.insert(0, str(Path(args.upstream).resolve()))
    import numpy as np
    import pandas as pd
    import torch
    import importlib.metadata
    from predict import load_model
    from predict_rois import predict_rois
    from mapping import LADModel

    torch.set_num_threads(4)
    work = Path(args.job_dir)
    job = json.loads((work / 'job.json').read_text(encoding='utf-8'))
    targets = pd.read_csv(work / 'targets.csv')
    model = load_model(args.model, all_in_one=True)
    results, audits = [], []
    for item in job['jobs']:
        stem = item['stem']
        inferred = predict_rois(model, str(work / f'{stem}_structures.tsv'),
                                item['metadata'], args.report, batch_size=256)
        rois = inferred[['smiles', 'roi']].drop_duplicates('smiles')
        if not np.isfinite(rois.roi).all():
            raise ValueError('Published model produced non-finite ROI.')
        anchors = pd.read_csv(work / f'{stem}_anchors.tsv', sep='\t').merge(
            rois, on='smiles', how='left', validate='one_to_one')
        if anchors.roi.isna().any():
            raise ValueError('Missing anchor ROI after upstream preprocessing.')
        anchors['used_after_void_filter'] = anchors.rt > item['void_threshold_min']
        selected = anchors.loc[anchors.used_after_void_filter].copy()
        if selected.roi.nunique() < 3:
            raise ValueError('At least three distinct anchor ROIs required for quadratic mapping.')
        mapping = LADModel(selected, ols_after=True, ols_discard_if_negative=True,
                           ols_drop_mode='2*median')
        part = targets.loc[targets.polarity == item['polarity']].merge(
            rois, on='smiles', how='left', validate='many_to_one')
        if part.roi.isna().any():
            raise ValueError('Upstream preprocessing removed target structures.')
        raw = mapping.get_mapping(part.roi.to_numpy(float))
        run_max = float(item['method']['run_time_min'])
        part['predicted_rt_min'] = np.clip(raw, 0, run_max)
        part['rt_unclipped_min'] = raw
        part['rt_prediction_clipped'] = (raw < 0) | (raw > run_max)
        part['rt_roi'] = part.pop('roi')
        part['rt_model'] = '2-step (published pretrained D-MPNN)'
        part['rt_model_doi'] = job['doi']
        part['rt_method_id'] = item['method']['method_id']
        part['rt_calibration'] = 'Official 2-step LAD quadratic + eligible OLS refinement'
        part['rt_anchor_count'] = len(selected)
        part['rt_mobile_phase_ph'] = item['method']['ph']
        part['rt_dead_time_min'] = item['dead_time_min']
        part['rt_dead_time_source'] = item['dead_time_source']
        part['rt_void_threshold_min'] = item['void_threshold_min']
        part['rt_extrapolates_anchor_roi'] = ((part.rt_roi < selected.roi.min()) |
                                               (part.rt_roi > selected.roi.max()))
        part['rt_applicability'] = 'Unassessed; published model with local anchors'
        part['rt_interval_status'] = 'Not established for 2-step; use unscheduled acquisition'
        for column in ('rt_lower_90_min', 'rt_upper_90_min', 'rt_lower_95_min',
                       'rt_upper_95_min', 'rt_window_half_width_90_min'):
            part[column] = np.nan
        part['rt_acquisition_start_min'] = 0.0
        part['rt_acquisition_end_min'] = run_max
        anchors.to_csv(work / f'{stem}_anchor_audit.csv', index=False)
        audits.append({'polarity': item['polarity'], 'n_anchors_used': len(selected),
                       'coefficients': [float(c) for c in mapping.coefficients],
                       'ols_used': not bool(mapping.no_ols),
                       'ols_fallback_reason': str(mapping.no_ols_why)})
        results.append(part)
    pd.concat(results).sort_values('input_row').to_csv(work / 'predictions.csv', index=False)
    (work / 'mapping_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    packages = ['torch', 'chemprop', 'rdkit', 'numpy', 'scipy', 'scikit-learn',
                'pandas', 'pulp', 'statsmodels', 'typed-argument-parser']
    versions = {name: importlib.metadata.version(name) for name in packages}
    (work / 'runtime_versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
