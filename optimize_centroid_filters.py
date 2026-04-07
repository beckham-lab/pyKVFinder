#!/usr/bin/env python3
"""
Optuna optimization for MetalFinder centroid filter parameters.

Evaluates filter parameters across proteins with ground-truth minimized
probes. Objective: maximize F-beta (recall-weighted) score for recovering
"good metal binders" while minimizing total probes.

Three phases:
  1. Label ground truth from minimized PDBs (count hard donors around LA)
  2. Build cavity_id -> probe_number mapping per protein (one metalfinder run each)
  3. Optuna optimization over coordination + HSAB filter parameters

Usage:
  python optimize_centroid_filters.py --n-trials 100 --config metal_config.yaml
  python optimize_centroid_filters.py --n-trials 50 --min-hard-donors-gt 3 --resume
"""

import os
# Suppress OpenMP info/warning messages from pyKVFinder's C extension.
# Must be set BEFORE importing pyKVFinder (loads libomp).
os.environ.setdefault('KMP_WARNINGS', '0')
os.environ.setdefault('KMP_AFFINITY', 'quiet')

import argparse
import copy
import csv
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import optuna
import yaml

from pyKVFinder.metalfinder.cli import run_metalfinder, load_config
from pyKVFinder.metalfinder.pdb_parser import parse_pdb, parse_pdb_full


# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_STARTS_DIR = (
    "/Users/ekomp/Dropbox/BiominingLDRD/datasets/genome_mining/putida/"
    "260403_ped_cluster_centroid_probes/starts"
)
DEFAULT_FINALS_DIR = (
    "/Users/ekomp/Dropbox/BiominingLDRD/datasets/genome_mining/putida/"
    "260403_ped_cluster_centroid_probes/finals"
)
DEFAULT_APO_DIR = (
    "/Users/ekomp/Dropbox/BiominingLDRD/datasets/genome_mining/putida/putida_apo"
)

# Default donor list for residue_ca mode (the default mode)
DEFAULT_DONOR_TYPES = ['ASP', 'GLU']

# Reference list for atom mode (classical "hard" donor atom names) — not the default
DEFAULT_DONOR_TYPES_ATOM = [
    'OD1', 'OD2',   # Asp carboxylate
    'OE1', 'OE2',   # Glu carboxylate
    'OG', 'OG1',    # Ser/Thr hydroxyl
    'OH',           # Tyr hydroxyl
    'NZ',           # Lys amine
    'O',            # Backbone carbonyl (also water O, but APO CIFs have no water)
]

NON_PROTEIN_RESIDUES = {'LA', 'PRB', 'Cl-', 'WAT', 'HOH', 'H2O'}


def build_donor_candidate_mask(mode, donor_types, atom_names, residue_names):
    """Return a boolean mask selecting candidate donor atoms.

    - ``atom`` mode: atoms whose ``atom_name`` is in ``donor_types``.
    - ``residue_ca`` mode: CA atoms whose ``residue_name`` is in ``donor_types``.

    Matches the logic used by SphereDonorFilter so ground truth labeling
    and the trial filter stay in sync.
    """
    donor_set = {str(s).upper() for s in donor_types}
    atom_names_upper = np.array([str(a).upper() for a in atom_names])
    residue_names_upper = np.array([str(r).upper() for r in residue_names])
    if mode == 'atom':
        return np.isin(atom_names_upper, list(donor_set))
    elif mode == 'residue_ca':
        return (atom_names_upper == 'CA') & np.isin(residue_names_upper, list(donor_set))
    else:
        raise ValueError(f"Unknown donor mode: {mode!r}")

# Accepts both naming conventions:
#   260401: AF-X-F1-model_v4_cavity_centroids_probe_001_final.pdb
#   260403: AF-X-F1-model_v4_centroids_probe_001.pdb (starts)
#           AF-X-F1-model_v4_centroids_probe_001_final.pdb (finals)
FINALS_PATTERN = re.compile(
    r'(.+)-model_v4_(?:cavity_)?centroids_probe_(\d+)_final\.pdb'
)
STARTS_PATTERN = re.compile(
    r'(.+)-model_v4_(?:cavity_)?centroids_probe_(\d+)\.pdb'
)


# ===========================================================================
# Alignment
# ===========================================================================

def kabsch_align(ref_coords, mov_coords):
    """Compute optimal rotation + translation (Kabsch/SVD) to align mov onto ref.

    Returns R, t such that: aligned = (R @ point) + t
    """
    ref_center = ref_coords.mean(axis=0)
    mov_center = mov_coords.mean(axis=0)

    H = (mov_coords - mov_center).T @ (ref_coords - ref_center)
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    t = ref_center - R @ mov_center
    return R, t


def extract_ca_coords(pdb_file):
    """Extract CA coordinates and residue numbers from PDB/CIF.

    Returns (ca_coords (M,3), ca_resnums (M,)).
    """
    coords, atom_names, _, residues, residue_nums, _, _ = parse_pdb_full(pdb_file)

    mask = np.array([
        an == 'CA' and rn not in NON_PROTEIN_RESIDUES
        for an, rn in zip(atom_names, residues)
    ])
    return coords[mask], residue_nums[mask]


# ===========================================================================
# Phase 1: Ground truth
# ===========================================================================

def build_ground_truth(finals_dir, gt_radius, donor_mode, donor_types):
    """Label each minimized probe by donor count around LA.

    Uses the SAME candidate-atom logic as SphereDonorFilter so ground truth
    and the trial filter are predicting the same thing.

    Returns
    -------
    ground_truth : dict  {(protein_id, probe_number): {'n_donors': int, 'la_position': ndarray, 'pdb_path': str}}
    protein_ids  : list  sorted unique protein IDs
    """
    pdb_files = sorted(Path(finals_dir).glob('*_final.pdb'))
    ground_truth = {}
    protein_id_set = set()

    for pdb_file in pdb_files:
        match = FINALS_PATTERN.match(pdb_file.name)
        if not match:
            print(f"  Warning: skipping {pdb_file.name} (doesn't match pattern)")
            continue

        protein_id = match.group(1)
        probe_number = int(match.group(2))
        protein_id_set.add(protein_id)

        coords, atom_names, _, residue_names, _ = parse_pdb(str(pdb_file))

        # Find LA atom (match by atom_name; residue_name may differ
        # between finals 'LA' and starts 'PRB')
        la_mask = atom_names == 'LA'
        if not np.any(la_mask):
            print(f"  Warning: no LA atom in {pdb_file.name}")
            continue
        la_position = coords[np.where(la_mask)[0][0]]

        # Protein-only mask (exclude LA, Cl-, water)
        protein_mask = np.array([rn not in NON_PROTEIN_RESIDUES for rn in residue_names])
        p_coords = coords[protein_mask]
        p_anames = atom_names[protein_mask]
        p_rnames = residue_names[protein_mask]

        # Build candidate donor mask using same logic as the filter
        cand_mask = build_donor_candidate_mask(
            donor_mode, donor_types, p_anames, p_rnames
        )
        cand_coords = p_coords[cand_mask]

        if len(cand_coords) > 0:
            dists_cand = np.linalg.norm(cand_coords - la_position, axis=1)
            n_donors = int(np.sum(dists_cand <= gt_radius))
        else:
            n_donors = 0

        ground_truth[(protein_id, probe_number)] = {
            'n_donors': n_donors,
            'la_position': la_position,
            'pdb_path': str(pdb_file),
        }

    return ground_truth, sorted(protein_id_set)


def parse_starting_probes(starts_dir):
    """Parse starting (pre-minimization) probe PDBs to extract original LA positions.

    Each file contains protein + a single LA atom at the cavity centroid.

    Returns
    -------
    starting_positions : dict  {(protein_id, probe_number): np.ndarray (3,)}
    """
    pdb_files = sorted(Path(starts_dir).glob('*.pdb'))
    starting_positions = {}

    for pdb_file in pdb_files:
        match = STARTS_PATTERN.match(pdb_file.name)
        if not match:
            print(f"  Warning: skipping {pdb_file.name} (doesn't match pattern)")
            continue

        protein_id = match.group(1)
        probe_number = int(match.group(2))

        coords, atom_names, _, _, _ = parse_pdb(str(pdb_file))
        la_mask = atom_names == 'LA'
        if not np.any(la_mask):
            print(f"  Warning: no LA atom in starts file {pdb_file.name}")
            continue

        starting_positions[(protein_id, probe_number)] = coords[np.where(la_mask)[0][0]]

    return starting_positions


def verify_file_sets(starting_positions, ground_truth):
    """Check that starts and finals contain the same set of (protein_id, probe_number).

    Returns (is_identical, in_starts_only, in_finals_only).
    """
    starts_keys = set(starting_positions.keys())
    finals_keys = set(ground_truth.keys())

    in_starts_only = starts_keys - finals_keys
    in_finals_only = finals_keys - starts_keys
    is_identical = not in_starts_only and not in_finals_only

    return is_identical, sorted(in_starts_only), sorted(in_finals_only)


# ===========================================================================
# Phase 2: Cavity ID mapping
# ===========================================================================

def build_cavity_id_mapping(protein_ids, apo_dir, base_config):
    """Run metalfinder (no filters) to get cavity_id -> probe_number mapping.

    Returns
    -------
    mapping            : {protein_id: {cavity_id: probe_number}}
    centroid_positions : {protein_id: {probe_number: ndarray}}
    """
    config = copy.deepcopy(base_config)
    config['io']['use_cavity_centroids'] = True
    config.setdefault('distance_filter', {})['enabled'] = False
    config.setdefault('coordination_filter', {})['enabled'] = False
    config.setdefault('hsab_filter', {})['enabled'] = False
    config.setdefault('clustering', {})['enabled'] = False
    config.setdefault('output', {})['export_pdb'] = False
    config.setdefault('output', {})['export_individual_pdbs'] = False

    mapping = {}
    centroid_positions = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, pid in enumerate(protein_ids, 1):
            cif_path = os.path.join(apo_dir, f"{pid}-model_v4.cif")
            if not os.path.exists(cif_path):
                print(f"  Warning: APO CIF not found: {cif_path}")
                continue

            print(f"  [{i}/{len(protein_ids)}] {pid}...", end=" ", flush=True)
            output_prefix = os.path.join(tmpdir, f"{pid}_map")

            try:
                result = run_metalfinder(
                    pdb_file=cif_path,
                    config=config,
                    output_prefix=output_prefix,
                    save_intermediates=False,
                    verbose=False,
                )
                probes = result['probes']
                unique_ids = sorted(np.unique(probes.cavity_ids).astype(int))

                pid_mapping = {}
                pid_positions = {}
                for idx, cid in enumerate(unique_ids):
                    pnum = idx + 1
                    pid_mapping[cid] = pnum
                    pid_positions[pnum] = probes.positions[probes.cavity_ids == cid][0]

                mapping[pid] = pid_mapping
                centroid_positions[pid] = pid_positions
                print(f"{len(unique_ids)} centroids")

            except Exception as e:
                print(f"FAILED: {e}")

    return mapping, centroid_positions


# ===========================================================================
# Displacement computation
# ===========================================================================

def compute_displacements(ground_truth, starting_positions, apo_dir):
    """Compute CA-aligned displacement of LA from its starting (pre-min) position.

    The minimized PDB is aligned onto the APO CIF by CA atoms, and the
    transformed LA position is compared against the starting LA position
    (from the pre-minimization probe PDB).

    Returns {(protein_id, probe_number): displacement_angstroms}.
    """
    displacements = {}
    apo_ca_cache = {}

    for (protein_id, probe_number), gt_info in ground_truth.items():
        key = (protein_id, probe_number)

        # Skip probes that failed minimization (no minimized PDB / LA position)
        if gt_info.get('failed_minimization') or gt_info.get('pdb_path') is None:
            displacements[key] = np.nan
            continue

        # Starting (pre-min) LA position
        if key not in starting_positions:
            displacements[key] = np.nan
            continue
        original_la = starting_positions[key]

        # APO CA coords (cached)
        if protein_id not in apo_ca_cache:
            cif_path = os.path.join(apo_dir, f"{protein_id}-model_v4.cif")
            apo_ca_cache[protein_id] = (
                extract_ca_coords(cif_path) if os.path.exists(cif_path) else None
            )
        if apo_ca_cache[protein_id] is None:
            displacements[key] = np.nan
            continue
        apo_ca_coords, apo_ca_resnums = apo_ca_cache[protein_id]

        # Minimized PDB CA coords
        min_ca_coords, min_ca_resnums = extract_ca_coords(gt_info['pdb_path'])

        # Match CAs by residue number
        common = np.intersect1d(apo_ca_resnums, min_ca_resnums)
        if len(common) < 10:
            displacements[key] = np.nan
            continue

        apo_matched = np.array([
            apo_ca_coords[apo_ca_resnums == rn][0] for rn in common
        ])
        min_matched = np.array([
            min_ca_coords[min_ca_resnums == rn][0] for rn in common
        ])

        # Align minimized -> APO frame
        R, t = kabsch_align(apo_matched, min_matched)
        aligned_la = R @ gt_info['la_position'] + t
        displacements[key] = float(np.linalg.norm(aligned_la - original_la))

    return displacements


def cross_check_positions(starting_positions, centroid_positions, tolerance=0.1):
    """Cross-check that Phase 2 recomputed centroids match the starting probes.

    Returns (n_matched, n_mismatched, max_deviation).
    """
    n_matched = 0
    n_mismatched = 0
    max_dev = 0.0

    for (pid, pnum), start_pos in starting_positions.items():
        if pid not in centroid_positions or pnum not in centroid_positions[pid]:
            continue
        dev = float(np.linalg.norm(start_pos - centroid_positions[pid][pnum]))
        max_dev = max(max_dev, dev)
        if dev <= tolerance:
            n_matched += 1
        else:
            n_mismatched += 1

    return n_matched, n_mismatched, max_dev


# ===========================================================================
# Phase 3: Optimization helpers
# ===========================================================================

def compute_fbeta(tp, fp, fn, beta):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


# Optuna search-space bounds (exposed so main() can print them before running)
SEARCH_RADIUS_MIN = 5.0
SEARCH_RADIUS_MAX = 20.0
SEARCH_MIN_DONORS_MIN = 1
SEARCH_MIN_DONORS_MAX = 10


def build_trial_config(base_config, radius, min_donors, donor_mode, donor_types):
    """Deep-copy base config with trial params.

    Only the SphereDonorFilter is active. All other filters are disabled.
    """
    config = copy.deepcopy(base_config)

    config['io']['use_cavity_centroids'] = True
    config.setdefault('distance_filter', {})['enabled'] = False
    config.setdefault('coordination_filter', {})['enabled'] = False
    config.setdefault('hsab_filter', {})['enabled'] = False
    config.setdefault('clustering', {})['enabled'] = False

    config['sphere_donor_filter'] = {
        'enabled': True,
        'mode': donor_mode,
        'radius': radius,
        'min_donors': min_donors,
        'max_donors': None,
        'donor_types': list(donor_types),
    }

    config.setdefault('output', {})['export_pdb'] = False
    config['output']['export_individual_pdbs'] = False
    return config


def evaluate_trial(protein_ids, apo_dir, trial_config, cavity_id_mapping,
                   ground_truth, min_donors_gt, beta):
    """Run metalfinder on all proteins and compute confusion matrix.

    Only probes present in BOTH the ground truth and the cavity_id mapping
    are considered for TP/FP/FN/TN.
    """
    positives = {
        key for key, info in ground_truth.items()
        if info['n_donors'] >= min_donors_gt
    }
    all_gt_keys = set(ground_truth.keys())

    predicted_positives = set()
    n_surviving_total = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for pid in protein_ids:
            if pid not in cavity_id_mapping:
                continue

            cif_path = os.path.join(apo_dir, f"{pid}-model_v4.cif")
            if not os.path.exists(cif_path):
                continue

            output_prefix = os.path.join(tmpdir, f"{pid}_trial")

            try:
                result = run_metalfinder(
                    pdb_file=cif_path,
                    config=trial_config,
                    output_prefix=output_prefix,
                    save_intermediates=False,
                    verbose=False,
                )

                pid_mapping = cavity_id_mapping[pid]
                for cid in result['probes'].cavity_ids:
                    cid_int = int(cid)
                    if cid_int in pid_mapping:
                        pnum = pid_mapping[cid_int]
                        predicted_positives.add((pid, pnum))
                        n_surviving_total += 1

            except Exception:
                pass  # all probes for this protein treated as not surviving

    # Restrict to probes in ground truth
    survived_in_gt = predicted_positives & all_gt_keys
    tp = len(survived_in_gt & positives)
    fp = len(survived_in_gt - positives)
    fn = len(positives - predicted_positives)
    tn = len((all_gt_keys - positives) - predicted_positives)
    f_beta = compute_fbeta(tp, fp, fn, beta)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f_beta': f_beta,
        'n_surviving_total': n_surviving_total,
    }


def create_objective(protein_ids, apo_dir, base_config, cavity_id_mapping,
                     ground_truth, min_donors_gt, beta, donor_mode, donor_types,
                     recall_penalty=0.0):
    """Return an Optuna objective function (closure).

    When *recall_penalty* > 0 the raw F-beta score is penalized for missed
    good probes::

        score = f_beta * (1 - recall_penalty * fn / n_positives)

    At ``recall_penalty=1.0``, losing half the positives halves the score.
    """
    n_positives = sum(
        1 for info in ground_truth.values()
        if info['n_donors'] >= min_donors_gt
    )

    def objective(trial):
        radius = trial.suggest_float(
            'radius', SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX
        )
        min_donors = trial.suggest_int(
            'min_donors', SEARCH_MIN_DONORS_MIN, SEARCH_MIN_DONORS_MAX
        )

        trial_config = build_trial_config(
            base_config, radius, min_donors, donor_mode, donor_types
        )

        t0 = time.time()
        metrics = evaluate_trial(
            protein_ids, apo_dir, trial_config, cavity_id_mapping,
            ground_truth, min_donors_gt, beta,
        )
        elapsed = time.time() - t0

        score = metrics['f_beta']
        if recall_penalty > 0 and n_positives > 0:
            score *= (1.0 - recall_penalty * metrics['fn'] / n_positives)

        for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'n_surviving_total'):
            trial.set_user_attr(k, metrics[k])
        trial.set_user_attr('f_beta_raw', metrics['f_beta'])
        trial.set_user_attr('score_penalized', round(score, 6))
        trial.set_user_attr('elapsed_seconds', round(elapsed, 1))

        print(
            f"  [Trial {trial.number:3d}] "
            f"F{beta:.0f}={metrics['f_beta']:.3f} "
            f"penalized={score:.3f} | "
            f"TP={metrics['tp']:3d} FP={metrics['fp']:3d} FN={metrics['fn']:3d} | "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} | "
            f"surviving={metrics['n_surviving_total']} | {elapsed:.1f}s"
        )
        return score

    return objective


# ===========================================================================
# CSV output
# ===========================================================================

def save_ground_truth_csv(ground_truth, displacements, min_donors_gt,
                          output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'protein_id', 'probe_number', 'n_donors',
            'displacement_A', 'is_good_binder', 'failed_minimization',
        ])
        for (pid, pnum), info in sorted(ground_truth.items()):
            disp = displacements.get((pid, pnum), np.nan)
            writer.writerow([
                pid, pnum, info['n_donors'],
                f"{disp:.2f}" if not np.isnan(disp) else "NA",
                info['n_donors'] >= min_donors_gt,
                info.get('failed_minimization', False),
            ])


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize MetalFinder centroid filter parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --n-trials 100 --config metal_config.yaml
  %(prog)s --n-trials 50 --min-donors-gt 3 --beta 3.0 --resume

  # Decoupled: GT by specific atoms in tight sphere, filter by residue_ca
  %(prog)s --n-trials 100 --config metal_config.yaml \\
      --gt-mode atom --gt-donor-types OD1 OD2 OE1 OE2 --gt-radius 4.0 \\
      --filter-mode residue_ca --filter-donor-types ASP GLU
""",
    )
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument(
        '--min-donors-gt', type=int, default=2,
        help='Good binder threshold (>= N donors of the configured type). Default: 2',
    )
    parser.add_argument(
        '--beta', type=float, default=2.0,
        help='F-beta parameter (higher = more recall-weighted). Default: 2.0',
    )
    parser.add_argument(
        '--recall-penalty', type=float, default=0.0,
        help='Penalize missed good probes in the objective. '
             'score = f_beta * (1 - weight * fn/n_positives). '
             '0 = off (default), 1.0 = hard (losing half the positives halves the score).',
    )
    parser.add_argument(
        '--gt-radius', type=float, default=8.0,
        help='Radius for counting donors around the final LA (A). Default: 8.0',
    )
    parser.add_argument(
        '--donor-mode', choices=['atom', 'residue_ca'], default='residue_ca',
        help='atom: count atoms matching donor_types names; '
             'residue_ca: count CA atoms of residues matching donor_types. '
             'Default: residue_ca',
    )
    parser.add_argument(
        '--donor-types', nargs='+', default=DEFAULT_DONOR_TYPES,
        help='In atom mode: PDB atom names (e.g. OD1 OE1 NZ OH). '
             'In residue_ca mode: residue names (e.g. ASP GLU). '
             'Default: ASP GLU. '
             'Fallback for --gt-donor-types and --filter-donor-types if not set.',
    )
    # --- Decoupled GT / filter donor arguments ---
    parser.add_argument(
        '--gt-mode', choices=['atom', 'residue_ca'], default=None,
        help='Donor mode for ground truth labeling. Overrides --donor-mode for GT. '
             'Default: inherited from --donor-mode.',
    )
    parser.add_argument(
        '--gt-donor-types', nargs='+', default=None,
        help='Donor types for ground truth labeling. Overrides --donor-types for GT. '
             'Default: inherited from --donor-types.',
    )
    parser.add_argument(
        '--filter-mode', choices=['atom', 'residue_ca'], default=None,
        help='Donor mode for SphereDonorFilter optimization. Overrides --donor-mode for filter. '
             'Default: inherited from --donor-mode.',
    )
    parser.add_argument(
        '--filter-donor-types', nargs='+', default=None,
        help='Donor types for SphereDonorFilter optimization. Overrides --donor-types for filter. '
             'Default: inherited from --donor-types.',
    )
    parser.add_argument('--config', default='metal_config.yaml')
    parser.add_argument(
        '--starting-probes-dir', default=DEFAULT_STARTS_DIR,
        help='Directory with pre-minimization probe PDBs (each: protein + single LA at centroid)',
    )
    parser.add_argument(
        '--finals-dir', default=DEFAULT_FINALS_DIR,
        help='Directory with post-minimization probe PDBs',
    )
    parser.add_argument('--apo-dir', default=DEFAULT_APO_DIR)
    parser.add_argument('--db', default='optuna_sphere_donor.db')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--strict-file-check', action='store_true',
        help='Exit with error if starts and finals do not have identical file sets',
    )

    args = parser.parse_args()

    # Resolve GT vs filter donor settings (new args override old fallbacks)
    gt_mode = args.gt_mode or args.donor_mode
    gt_donor_types = args.gt_donor_types or args.donor_types
    filter_mode = args.filter_mode or args.donor_mode
    filter_donor_types = args.filter_donor_types or args.donor_types

    print("=" * 70)
    print("CENTROID FILTER PARAMETER OPTIMIZATION (SphereDonorFilter)")
    print("=" * 70)
    print(f"  GT donor mode         : {gt_mode}")
    print(f"  GT donor types        : {gt_donor_types}")
    print(f"  GT radius             : {args.gt_radius} A")
    print(f"  Good binder threshold : >= {args.min_donors_gt} donors")
    print(f"  Filter donor mode     : {filter_mode}")
    print(f"  Filter donor types    : {filter_donor_types}")
    print(f"  F-beta parameter      : {args.beta}")
    print(f"  Recall penalty weight : {args.recall_penalty}")
    print(f"  Trials                : {args.n_trials}")
    print(f"  Config                : {args.config}")
    print(f"  Starts dir            : {args.starting_probes_dir}")
    print(f"  Finals dir            : {args.finals_dir}")
    print()
    print("  Optuna search space (maximize F-beta):")
    print(f"    radius        : float in [{SEARCH_RADIUS_MIN}, {SEARCH_RADIUS_MAX}] A")
    print(f"    min_donors    : int   in [{SEARCH_MIN_DONORS_MIN}, {SEARCH_MIN_DONORS_MAX}]")
    print(f"  Fixed per run:")
    print(f"    GT:     mode={gt_mode}, types={gt_donor_types}")
    print(f"    Filter: mode={filter_mode}, types={filter_donor_types}")
    print()

    base_config = load_config(args.config)

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    print("PHASE 1: Ground truth from minimized structures")
    print("-" * 70)
    ground_truth, protein_ids = build_ground_truth(
        args.finals_dir, args.gt_radius, gt_mode, gt_donor_types,
    )

    n_good = sum(
        1 for info in ground_truth.values()
        if info['n_donors'] >= args.min_donors_gt
    )
    print(f"\n  Labeled {len(ground_truth)} probes across {len(protein_ids)} proteins")
    print(f"  Good binders (>= {args.min_donors_gt} donors): {n_good}")
    print(f"  Non-binders: {len(ground_truth) - n_good}")
    print(f"  GT binder probe ids: {sorted([k for k, v in ground_truth.items() if v['n_donors'] >= args.min_donors_gt])}")

    # Distribution
    donor_counts = [info['n_donors'] for info in ground_truth.values()]
    for n in range(max(donor_counts) + 1):
        cnt = donor_counts.count(n)
        if cnt > 0:
            print(f"    {n} donors: {cnt} probes")
    print()

    # ------------------------------------------------------------------
    # Phase 1b: Starting probes + file set verification
    # ------------------------------------------------------------------
    print("PHASE 1b: Parsing starting (pre-min) probes")
    print("-" * 70)
    starting_positions = parse_starting_probes(args.starting_probes_dir)
    print(f"  Parsed {len(starting_positions)} starting probes")

    is_identical, in_starts_only, in_finals_only = verify_file_sets(
        starting_positions, ground_truth,
    )
    if is_identical:
        print("  File sets IDENTICAL between starts and finals")
    else:
        print(f"  MISMATCH: {len(in_starts_only)} in starts only, "
              f"{len(in_finals_only)} in finals only")
        if in_starts_only:
            print("    In starts only (no minimized result):")
            for key in in_starts_only[:10]:
                print(f"      {key[0]} probe {key[1]}")
            if len(in_starts_only) > 10:
                print(f"      ... and {len(in_starts_only) - 10} more")
        if in_finals_only:
            print("    In finals only (no starting probe):")
            for key in in_finals_only[:10]:
                print(f"      {key[0]} probe {key[1]}")
            if len(in_finals_only) > 10:
                print(f"      ... and {len(in_finals_only) - 10} more")
        if args.strict_file_check:
            print("\n  --strict-file-check enabled, exiting.")
            sys.exit(1)
        print("  Proceeding with the intersection of probes.")

    # Drop probes that appear in finals but not in starts (can't match without
    # original position)
    ground_truth = {
        k: v for k, v in ground_truth.items() if k in starting_positions
    }

    # Treat probes that only exist in starts (failed minimization) as
    # NEGATIVE ground truth — n_donors = 0 so they're never "good".
    # If the filter keeps them, they count as FP.
    n_failed = 0
    for key in starting_positions:
        if key not in ground_truth:
            ground_truth[key] = {
                'n_donors': 0,
                'la_position': None,
                'pdb_path': None,
                'failed_minimization': True,
            }
            n_failed += 1

    print(f"  Ground truth probes: {len(ground_truth)} "
          f"({len(ground_truth) - n_failed} minimized + {n_failed} failed-min)")
    print()

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    print("PHASE 2: Cavity ID mapping (one metalfinder run per protein)")
    print("-" * 70)
    cavity_id_mapping, centroid_positions = build_cavity_id_mapping(
        protein_ids, args.apo_dir, base_config,
    )

    # Validate
    n_mapped = n_unmapped = 0
    for (pid, pnum) in ground_truth:
        if pid in cavity_id_mapping and pnum in cavity_id_mapping[pid].values():
            n_mapped += 1
        else:
            n_unmapped += 1

    print(f"\n  Mapped: {n_mapped} / {len(ground_truth)} ground truth probes")
    if n_unmapped:
        print(f"  Warning: {n_unmapped} probes unmapped (KVFinder params may differ)")

    # Cross-check Phase 2 recomputed centroids against starting probe positions
    n_ok, n_bad, max_dev = cross_check_positions(
        starting_positions, centroid_positions, tolerance=0.1,
    )
    print(f"  Centroid cross-check: {n_ok} match, {n_bad} deviate, "
          f"max deviation = {max_dev:.3f} A")
    if n_bad > 0:
        print("  Warning: Phase 2 centroids deviate from starting probes. "
              "KVFinder params may differ from those used to generate starts.")
    print()

    # ------------------------------------------------------------------
    # Displacements
    # ------------------------------------------------------------------
    print("Computing CA-aligned metal displacements...")
    displacements = compute_displacements(
        ground_truth, starting_positions, args.apo_dir,
    )
    valid = [d for d in displacements.values() if not np.isnan(d)]
    if valid:
        print(f"  N={len(valid)}  mean={np.mean(valid):.2f} A  "
              f"median={np.median(valid):.2f} A  "
              f"min={np.min(valid):.2f} A  max={np.max(valid):.2f} A")
    print()

    csv_path = "ground_truth_centroid_probes.csv"
    save_ground_truth_csv(ground_truth, displacements, args.min_donors_gt, csv_path)
    print(f"  Saved ground truth CSV: {csv_path}")
    print()

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    print("PHASE 3: Optuna optimization")
    print("-" * 70)

    storage = f"sqlite:///{args.db}"
    study_name = "sphere_donor_opt"

    if args.resume:
        print(f"  Resuming study from {args.db}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        print("  Creating new study")
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

    objective = create_objective(
        protein_ids, args.apo_dir, base_config, cavity_id_mapping,
        ground_truth, args.min_donors_gt, args.beta,
        filter_mode, filter_donor_types,
        recall_penalty=args.recall_penalty,
    )

    print()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"  Best trial : {study.best_trial.number}")
    print(f"  Best score : {study.best_value:.4f}"
          f"  (F{args.beta:.0f}={study.best_trial.user_attrs.get('f_beta_raw', study.best_value):.4f}"
          f", penalty weight={args.recall_penalty})")
    print()
    print("  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k:30s}: {v}")
    print()
    print("  Best trial metrics:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"    {k:30s}: {v}")

    # Best trial that kept ALL good probes (fn == 0)
    perfect_recall_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get('fn', -1) == 0
    ]
    if perfect_recall_trials:
        best_pr = max(perfect_recall_trials, key=lambda t: t.value)
        print()
        print(f"  Best trial with perfect recall (fn=0): #{best_pr.number}")
        print(f"  F{args.beta:.0f} = {best_pr.value:.4f}")
        print("  Parameters:")
        for k, v in best_pr.params.items():
            print(f"    {k:30s}: {v}")
        print("  Metrics:")
        for k, v in best_pr.user_attrs.items():
            print(f"    {k:30s}: {v}")
    else:
        print()
        print("  No trial achieved perfect recall (fn=0).")

    # Save best config
    best_config = build_trial_config(
        base_config,
        study.best_params['radius'],
        study.best_params['min_donors'],
        filter_mode,
        filter_donor_types,
    )
    output_file = 'metal_config_centroid_optimized.yaml'
    with open(output_file, 'w') as f:
        yaml.dump({'default': best_config}, f, default_flow_style=False, sort_keys=False)

    print()
    print(f"  Saved optimized config (best F{args.beta:.0f}): {output_file}")

    if perfect_recall_trials:
        pr_config = build_trial_config(
            base_config,
            best_pr.params['radius'],
            best_pr.params['min_donors'],
            filter_mode,
            filter_donor_types,
        )
        pr_output_file = 'metal_config_centroid_optimized_perfect_recall.yaml'
        with open(pr_output_file, 'w') as f:
            yaml.dump({'default': pr_config}, f, default_flow_style=False, sort_keys=False)
        print(f"  Saved optimized config (best F{args.beta:.0f} with fn=0): {pr_output_file}")

    print("  Done!")


if __name__ == '__main__':
    main()
